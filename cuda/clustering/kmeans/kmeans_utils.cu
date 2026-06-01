#include "kmeans.cuh"
#include "utils.cuh"
#include "l2norm/l2norm.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mma.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// StreamEnv Implementation
// ============================================================

void StreamEnv::allocate(int B, int Ktile) {
    cudaMalloc(&d_xnorm2,    sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_dist2, sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_idx,   sizeof(int)   * (size_t)B);
    cudaMalloc(&d_dot,        sizeof(float) * (size_t)B * (size_t)Ktile);
}

void StreamEnv::free() {
    if (d_xnorm2)    { cudaFree(d_xnorm2);    d_xnorm2 = nullptr; }
    if (d_best_dist2) { cudaFree(d_best_dist2); d_best_dist2 = nullptr; }
    if (d_best_idx)   { cudaFree(d_best_idx);   d_best_idx = nullptr; }
    if (d_dot)        { cudaFree(d_dot);        d_dot = nullptr; }
}

// ============================================================
// perform_assignment_only Implementation
// ============================================================

void perform_assignment_only(
    const KMeansCase& cfg,
    const float* h_data,
    int* d_assign,
    const float* d_centroids,
    float* d_cnorm2,
    StreamEnv& env,
    float* d_data_buffer,
    cudaStream_t stream,
    cublasHandle_t handle,
    int B,
    int Ktile,
    int n,
    int dim,
    int k
) {
    const float alpha = 1.f;
    const float beta = 0.f;

    //  centroid
    compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream);

    //
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);

        // /
        float* d_data_cur = d_data_buffer;

        // H2D  batch
        cudaMemcpyAsync(d_data_cur, h_data + (size_t)base * dim,
                       sizeof(float) * (size_t)curB * dim,
                       cudaMemcpyHostToDevice, stream);

        // xnorm2 for this batch
        compute_l2_squared_gpu(d_data_cur, env.d_xnorm2, curB, dim, L2NORM_AUTO, stream);

        // init best_dist2 = +inf, best_idx = 0
        {
            int threads = 256;
            int blocks = (curB + threads - 1) / threads;
            kernel_init_best<<<blocks, threads, 0, stream>>>(env.d_best_dist2, env.d_best_idx, curB);
        }

        // centroid  Ktile
        for (int cbase = 0; cbase < k; cbase += Ktile) {
            int curK = std::min(Ktile, k - cbase);

            const float* A = d_centroids + (size_t)cbase * dim;
            const float* Bm = d_data_cur;
            float* Cc = env.d_dot;

            // cuBLAS
            cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

            // GEMM: dotT[curK,curB] = Ccm^T[curK,dim] * Xcm[dim,curB]
            cublas_check(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    curK, curB, dim,
                    &alpha,
                    A, dim,
                    Bm, dim,
                    &beta,
                    Cc, curK
                ),
                "cublasSgemm(Ccm^T * Xcm)"
            );

            // best
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_update_best_from_dotT<<<blocks, threads, 0, stream>>>(
                    env.d_dot, env.d_xnorm2, d_cnorm2,
                    curB, curK, cbase,
                    env.d_best_idx, env.d_best_dist2
                );
            }
        }

        // assign
        cudaMemcpyAsync(d_assign + base, env.d_best_idx, sizeof(int) * (size_t)curB,
                       cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);
}

// ============================================================
// GPU Kernels Implementation
// ============================================================

/**
 * Kernel:  best_dist2 block reduce
 *  block
 */
 __global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  //  float
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //
    float val = (i < n) ? data[i] : 0.0f;

    // Block
    sdata[tid] = val;
    __syncthreads();

    //  warp shuffle + shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block 0
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int cnt = counts[c];
    if (cnt <= 0) return;  // keep old centroid
    float inv = 1.0f / (float)cnt;

    for (int col = j; col < dim; col += blockDim.x) {
        centroids[(size_t)c * dim + col] = accum[(size_t)c * dim + col] * inv;
    }
}

/**
 * Kernel: Minibatch
 * lr = 1 / (total_count + 1)
 * centroid_new = (1 - lr) * centroid_old + lr * (accum / count)
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,     // [k, dim]
    const int* __restrict__ counts,      // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int count = counts[c];
    if (count <= 0) return;  // minibatch

    // total_count
    // blockcentroid
    //
    __shared__ int old_total;
    __shared__ float lr_shared;
    if (j == 0) {
        old_total = total_counts[c];
        // lr = 1 / (old_total + 1)
        // old_total
        // "centroid"
        //
        lr_shared = 1.0f / ((float)old_total + 1.0f);
        // total_counts
        total_counts[c] = old_total + count;
    }
    __syncthreads();

    float lr = lr_shared;
    float inv_count = 1.0f / (float)count;

    // dim
    for (int col = j; col < dim; col += blockDim.x) {
        size_t idx = (size_t)c * dim + col;
        float avg = accum[idx] * inv_count;  // minibatchcentroid
        float old_val = centroids[idx];
        centroids[idx] = (1.0f - lr) * old_val + lr * avg;
    }
}

// ============================================================
// GEMM-based KMeans Kernels
// ============================================================

/**
 * Kernel: best_dist2 = INF, best_idx = 0
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    int* __restrict__ best_idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist2[i] = 3.402823e38f;  // FLT_MAX
        best_idx[i] = 0;
    }
}

/**
 * Kernel:  GEMM col-major dotT
 *
 * dotT: [curK, curB] col-major dotT[t + i*curK] = dot(centroid[cbase+t], data[i])
 *
 *
 * -
 * -  warp shuffle  warp  min
 * -
 */
__global__ void kernel_u8_to_f32_batch(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = (float)src[i];
    }
}

__global__ void kernel_u8_to_f16_batch(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2half((float)src[i]);
    }
}

__global__ void kernel_u8_to_f16_centered_batch(
    const uint8_t* __restrict__ src,
    const float* __restrict__ mean,
    __half* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int d = (int)(i % (size_t)dim);
        dst[i] = __float2half((float)src[i] - mean[d]);
    }
}

__global__ void kernel_f32_to_f16_batch(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void kernel_f32_to_f16_centered_batch(
    const float* __restrict__ src,
    const float* __restrict__ mean,
    __half* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int d = (int)(i % (size_t)dim);
        dst[i] = __float2half(src[i] - mean[d]);
    }
}

__global__ void kernel_u8_to_bf16_batch(
    const uint8_t* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2bfloat16((float)src[i]);
    }
}

__global__ void kernel_f32_to_bf16_batch(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n,
    int dim
) {
    size_t total = (size_t)n * (size_t)dim;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= curB) return;

    const float xn = xnorm2[i];
    float bestd = best_dist2[i];
    int bestc = best_idx[i];

    // centroid
    const int unroll_factor = 4;
    int t = 0;

    // centroid
    for (; t + unroll_factor <= curK; t += unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; ++u) {
            int tidx = t + u;
            float dot = dotT[tidx + (size_t)i * curK];
            float cn = cnorm2_global[cbase + tidx];
            float d2 = xn + cn - 2.f * dot;
            int cid = cbase + tidx;
            if (d2 < bestd) {
                bestd = d2;
                bestc = cid;
            }
        }
    }

    // centroid
    for (; t < curK; ++t) {
        float dot = dotT[t + (size_t)i * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    best_dist2[i] = bestd;
    best_idx[i] = bestc;
}

/**
 * Kernel:  GEMM col-major dotTWarp
 *
 *  warp shuffle  curK
 *  warp  warp  min
 */
__global__ void kernel_update_best_from_dotT_warp_point(
    const float* __restrict__ dotT,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= curB) return;

    float xn = xnorm2[warp];
    float local_best = best_dist2[warp];
    int local_idx = best_idx[warp];

    for (int t = lane; t < curK; t += 32) {
        float dot = dotT[t + (size_t)warp * curK];
        float d2 = xn + cnorm2_global[cbase + t] - 2.0f * dot;
        int cid = cbase + t;
        if (d2 < local_best || (d2 == local_best && cid < local_idx)) {
            local_best = d2;
            local_idx = cid;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_best = __shfl_down_sync(0xffffffff, local_best, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_best < local_best || (other_best == local_best && other_idx < local_idx)) {
            local_best = other_best;
            local_idx = other_idx;
        }
    }

    if (lane == 0) {
        best_dist2[warp] = local_best;
        best_idx[warp] = local_idx;
    }
}

__global__ void kernel_update_best_from_dotT_warp_point_no_xnorm(
    const float* __restrict__ dotT,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= curB) return;

    float local_best = best_score[warp];
    int local_idx = best_idx[warp];

    for (int t = lane; t < curK; t += 32) {
        float dot = dotT[t + (size_t)warp * curK];
        float score = cnorm2_global[cbase + t] - 2.0f * dot;
        int cid = cbase + t;
        if (score < local_best || (score == local_best && cid < local_idx)) {
            local_best = score;
            local_idx = cid;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_best = __shfl_down_sync(0xffffffff, local_best, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_best < local_best || (other_best == local_best && other_idx < local_idx)) {
            local_best = other_best;
            local_idx = other_idx;
        }
    }

    if (lane == 0) {
        best_score[warp] = local_best;
        best_idx[warp] = local_idx;
    }
}

__global__ void kernel_update_best_from_dotT_warp_point_no_xnorm_half(
    const __half* __restrict__ dotT,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    float dot_scale,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= curB) return;

    float local_best = best_score[warp];
    int local_idx = best_idx[warp];

    for (int t = lane; t < curK; t += 32) {
        float dot = __half2float(dotT[t + (size_t)warp * curK]) * dot_scale;
        float score = cnorm2_global[cbase + t] - 2.0f * dot;
        int cid = cbase + t;
        if (score < local_best || (score == local_best && cid < local_idx)) {
            local_best = score;
            local_idx = cid;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_best = __shfl_down_sync(0xffffffff, local_best, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_best < local_best || (other_best == local_best && other_idx < local_idx)) {
            local_best = other_best;
            local_idx = other_idx;
        }
    }

    if (lane == 0) {
        best_score[warp] = local_best;
        best_idx[warp] = local_idx;
    }
}

__device__ __forceinline__ void insert_top8_score(float score, int idx, float* s, int* id) {
    if (idx < 0) return;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (id[i] == idx) {
            if (score >= s[i]) return;
            #pragma unroll
            for (int j = i; j < 7; ++j) {
                s[j] = s[j + 1];
                id[j] = id[j + 1];
            }
            s[7] = 3.402823e38f;
            id[7] = -1;
            break;
        }
    }
    if (score > s[7] || (score == s[7] && idx >= id[7])) return;
    int pos = 7;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (score < s[i] || (score == s[i] && (id[i] < 0 || idx < id[i]))) {
            pos = i;
            break;
        }
    }
    #pragma unroll
    for (int i = 7; i > 0; --i) {
        if (i > pos) {
            s[i] = s[i - 1];
            id[i] = id[i - 1];
        }
    }
    s[pos] = score;
    id[pos] = idx;
}

__global__ void kernel_init_top8(
    float* __restrict__ top_score,
    int* __restrict__ top_idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 8) return;
    top_score[i] = 3.402823e38f;
    top_idx[i] = -1;
}

__global__ void kernel_update_top8_from_dotT_warp_point_no_xnorm_half(
    const __half* __restrict__ dotT,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    float dot_scale,
    float* __restrict__ top_score,
    int* __restrict__ top_idx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= curB) return;

    float s[8];
    int id[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = top_score[(size_t)warp * 8 + i];
        id[i] = top_idx[(size_t)warp * 8 + i];
    }

    for (int t = lane; t < curK; t += 32) {
        float dot = __half2float(dotT[t + (size_t)warp * curK]) * dot_scale;
        int cid = cbase + t;
        float score = cnorm2_global[cid] - 2.0f * dot;
        insert_top8_score(score, cid, s, id);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float os = __shfl_down_sync(0xffffffff, s[i], offset);
            int oi = __shfl_down_sync(0xffffffff, id[i], offset);
            insert_top8_score(os, oi, s, id);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            top_score[(size_t)warp * 8 + i] = s[i];
            top_idx[(size_t)warp * 8 + i] = id[i];
        }
    }
}

__global__ void kernel_refine_top8_exact_scores(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    const float* __restrict__ cnorm2_global,
    const float* __restrict__ top_score,
    const int* __restrict__ top_idx,
    int curB,
    int dim,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
) {
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= curB) return;
    float best = top_score[(size_t)point * 8 + 0];
    int best_id = top_idx[(size_t)point * 8 + 0];
    const float* x = data + (size_t)point * dim;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int cid = top_idx[(size_t)point * 8 + r];
        if (cid < 0) continue;
        const float* c = centroids + (size_t)cid * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += x[d] * c[d];
        }
        float score = cnorm2_global[cid] - 2.0f * dot;
        if (score < best || (score == best && cid < best_id)) {
            best = score;
            best_id = cid;
        }
    }
    best_score[point] = best;
    best_idx[point] = best_id;
}

__global__ void kernel_assign_wmma_argmin_tile(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
) {
    using namespace nvcuda;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int point_base = (blockIdx.x * warps_per_block + warp_id) * 16;
    if (point_base >= curB) return;

    extern __shared__ float dots_all[];
    float* dots = dots_all + warp_id * 16 * 16;

    float local_best = 3.402823e38f;
    int local_idx = 0;
    int point = point_base + lane;
    if (lane < 16 && point < curB) {
        local_best = best_dist2[point];
        local_idx = best_idx[point];
    }

    for (int kt = 0; kt < curK; kt += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        const float* a_base = centroids + (size_t)(cbase + kt) * dim;
        const float* b_base = data + (size_t)point_base * dim;
        for (int d0 = 0; d0 < dim; d0 += 8) {
            wmma::load_matrix_sync(a_frag, a_base + d0, dim);
            wmma::load_matrix_sync(b_frag, b_base + d0, dim);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        wmma::store_matrix_sync(dots, acc_frag, 16, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16 && point < curB) {
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                int cid = cbase + kt + r;
                if (kt + r < curK) {
                    float d2 = cnorm2_global[cid] - 2.0f * dots[r * 16 + lane];
                    if (d2 < local_best || (d2 == local_best && cid < local_idx)) {
                        local_best = d2;
                        local_idx = cid;
                    }
                }
            }
        }
        __syncwarp();
    }

    if (lane < 16 && point < curB) {
        best_dist2[point] = local_best;
        best_idx[point] = local_idx;
    }
}

__global__ void kernel_assign_fused_argmin64_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
) {
    using namespace nvcuda;
    constexpr int kPointTile = 64;
    constexpr int kCentTile = 64;
    constexpr int kSub = 16;
    constexpr int kWarps = 16;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarps) return;

    const int cent_quad = warp_id >> 2;
    const int point_quad = warp_id & 3;
    const int point_base = blockIdx.x * kPointTile + point_quad * kSub;
    const int cent64_base = blockIdx.y * kCentTile;
    const int cent_base = cent64_base + cent_quad * kSub;

    extern __shared__ unsigned char smem_raw[];
    float* dots_all = reinterpret_cast<float*>(smem_raw);
    float* warp_best_dist = dots_all + kWarps * kSub * kSub;
    int* warp_best_idx = reinterpret_cast<int*>(warp_best_dist + kWarps * kSub);
    float* dots = dots_all + warp_id * kSub * kSub;

    wmma::fragment<wmma::matrix_a, kSub, kSub, kSub, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int d0 = 0; d0 < dim; d0 += kSub) {
        const __half* a_ptr = centroids + (size_t)(cbase + cent_base) * dim + d0;
        const __half* b_ptr = data + (size_t)point_base * dim + d0;
        wmma::load_matrix_sync(a_frag, a_ptr, dim);
        wmma::load_matrix_sync(b_frag, b_ptr, dim);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(dots, acc_frag, kSub, wmma::mem_row_major);
    __syncthreads();

    if (lane < kSub) {
        const int point = point_base + lane;
        float best = 3.402823e38f;
        int best_idx = -1;
        if (point < curB) {
            #pragma unroll
            for (int r = 0; r < kSub; ++r) {
                const int local_cent = cent_base + r;
                if (local_cent < curK) {
                    const int cid = cbase + local_cent;
                    const float dot = dots[r * kSub + lane];
                    const float d2 = cnorm2_global[cid] - 2.0f * dot;
                    if (d2 < best || (d2 == best && cid < best_idx)) {
                        best = d2;
                        best_idx = cid;
                    }
                }
            }
        }
        warp_best_dist[warp_id * kSub + lane] = best;
        warp_best_idx[warp_id * kSub + lane] = best_idx;
    }
    __syncthreads();

    if (cent_quad == 0 && lane < kSub) {
        const int point = point_base + lane;
        if (point < curB) {
            float best = 3.402823e38f;
            int best_idx = -1;
            #pragma unroll
            for (int cq = 0; cq < 4; ++cq) {
                const int src_warp = cq * 4 + point_quad;
                const float d2 = warp_best_dist[src_warp * kSub + lane];
                const int cid = warp_best_idx[src_warp * kSub + lane];
                if (cid >= 0 && (d2 < best || (d2 == best && cid < best_idx))) {
                    best = d2;
                    best_idx = cid;
                }
            }
            const size_t out = (size_t)point * cblocks64 + blockIdx.y;
            partial_dist2[out] = best;
            partial_idx[out] = best_idx;
        }
    }
}

__global__ void kernel_assign_fused_argmin64x128_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
) {
    using namespace nvcuda;
    constexpr int kPointTile = 128;
    constexpr int kPointHalf = 64;
    constexpr int kCentTile = 64;
    constexpr int kSub = 16;
    constexpr int kWarps = 16;
    constexpr int kNGroups = 2;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarps) return;

    const int cent_quad = warp_id >> 2;
    const int point_quad = warp_id & 3;
    const int point_tile_base = blockIdx.x * kPointTile;
    const int cent64_base = blockIdx.y * kCentTile;
    const int cent_base = cent64_base + cent_quad * kSub;

    extern __shared__ unsigned char smem_raw[];
    float* dots_all = reinterpret_cast<float*>(smem_raw);
    float* warp_best_dist = dots_all + kWarps * kNGroups * kSub * kSub;
    int* warp_best_idx = reinterpret_cast<int*>(warp_best_dist + kWarps * kNGroups * kSub);
    float* dots0 = dots_all + (warp_id * kNGroups + 0) * kSub * kSub;
    float* dots1 = dots_all + (warp_id * kNGroups + 1) * kSub * kSub;

    wmma::fragment<wmma::matrix_a, kSub, kSub, kSub, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b0_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b1_frag;
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc0_frag;
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc1_frag;
    wmma::fill_fragment(acc0_frag, 0.0f);
    wmma::fill_fragment(acc1_frag, 0.0f);

    const int point_base0 = point_tile_base + point_quad * kSub;
    const int point_base1 = point_tile_base + kPointHalf + point_quad * kSub;
    for (int d0 = 0; d0 < dim; d0 += kSub) {
        const __half* a_ptr = centroids + (size_t)(cbase + cent_base) * dim + d0;
        const __half* b0_ptr = data + (size_t)point_base0 * dim + d0;
        const __half* b1_ptr = data + (size_t)point_base1 * dim + d0;
        wmma::load_matrix_sync(a_frag, a_ptr, dim);
        wmma::load_matrix_sync(b0_frag, b0_ptr, dim);
        wmma::load_matrix_sync(b1_frag, b1_ptr, dim);
        wmma::mma_sync(acc0_frag, a_frag, b0_frag, acc0_frag);
        wmma::mma_sync(acc1_frag, a_frag, b1_frag, acc1_frag);
    }

    wmma::store_matrix_sync(dots0, acc0_frag, kSub, wmma::mem_row_major);
    wmma::store_matrix_sync(dots1, acc1_frag, kSub, wmma::mem_row_major);
    __syncthreads();

    if (lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = (g == 0 ? point_base0 : point_base1) + lane;
            const float* dots = (g == 0 ? dots0 : dots1);
            float best = 3.402823e38f;
            int best_idx = -1;
            if (point < curB) {
                #pragma unroll
                for (int r = 0; r < kSub; ++r) {
                    const int local_cent = cent_base + r;
                    if (local_cent < curK) {
                        const int cid = cbase + local_cent;
                        const float dot = dots[r * kSub + lane];
                        const float d2 = cnorm2_global[cid] - 2.0f * dot;
                        if (d2 < best || (d2 == best && cid < best_idx)) {
                            best = d2;
                            best_idx = cid;
                        }
                    }
                }
            }
            const int slot = (warp_id * kNGroups + g) * kSub + lane;
            warp_best_dist[slot] = best;
            warp_best_idx[slot] = best_idx;
        }
    }
    __syncthreads();

    if (cent_quad == 0 && lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = (g == 0 ? point_base0 : point_base1) + lane;
            if (point < curB) {
                float best = 3.402823e38f;
                int best_idx = -1;
                #pragma unroll
                for (int cq = 0; cq < 4; ++cq) {
                    const int src_warp = cq * 4 + point_quad;
                    const int slot = (src_warp * kNGroups + g) * kSub + lane;
                    const float d2 = warp_best_dist[slot];
                    const int cid = warp_best_idx[slot];
                    if (cid >= 0 && (d2 < best || (d2 == best && cid < best_idx))) {
                        best = d2;
                        best_idx = cid;
                    }
                }
                const size_t out = (size_t)point * cblocks64 + blockIdx.y;
                partial_dist2[out] = best;
                partial_idx[out] = best_idx;
            }
        }
    }
}

__global__ void kernel_assign_fused_argmin64x128s_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
) {
    using namespace nvcuda;
    constexpr int kPointTile = 128;
    constexpr int kPointHalf = 64;
    constexpr int kCentTile = 64;
    constexpr int kSub = 16;
    constexpr int kWarps = 16;
    constexpr int kNGroups = 2;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarps) return;

    const int cent_quad = warp_id >> 2;
    const int point_quad = warp_id & 3;
    const int point_tile_base = blockIdx.x * kPointTile;
    const int cent64_base = blockIdx.y * kCentTile;
    const int cent_base = cent64_base + cent_quad * kSub;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* sh_cent = reinterpret_cast<__half*>(smem_raw);
    __half* sh_data = sh_cent + kCentTile * kSub;
    float* dots_all = reinterpret_cast<float*>(sh_data + kPointTile * kSub);
    float* warp_best_dist = dots_all + kWarps * kNGroups * kSub * kSub;
    int* warp_best_idx = reinterpret_cast<int*>(warp_best_dist + kWarps * kNGroups * kSub);
    float* dots0 = dots_all + (warp_id * kNGroups + 0) * kSub * kSub;
    float* dots1 = dots_all + (warp_id * kNGroups + 1) * kSub * kSub;

    wmma::fragment<wmma::matrix_a, kSub, kSub, kSub, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b0_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b1_frag;
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc0_frag;
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc1_frag;
    wmma::fill_fragment(acc0_frag, 0.0f);
    wmma::fill_fragment(acc1_frag, 0.0f);

    const int point_base0 = point_tile_base + point_quad * kSub;
    const int point_base1 = point_tile_base + kPointHalf + point_quad * kSub;
    for (int d0 = 0; d0 < dim; d0 += kSub) {
        for (int i = threadIdx.x; i < kCentTile * kSub; i += blockDim.x) {
            const int lc = i / kSub;
            const int ld = i - lc * kSub;
            const int local_cent = cent64_base + lc;
            const int cid = cbase + local_cent;
            sh_cent[i] = (local_cent < curK) ? centroids[(size_t)cid * dim + d0 + ld] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < kPointTile * kSub; i += blockDim.x) {
            const int lp = i / kSub;
            const int ld = i - lp * kSub;
            const int point = point_tile_base + lp;
            sh_data[i] = (point < curB) ? data[(size_t)point * dim + d0 + ld] : __float2half(0.0f);
        }
        __syncthreads();

        const __half* a_ptr = sh_cent + (size_t)(cent_base - cent64_base) * kSub;
        const __half* b0_ptr = sh_data + (size_t)(point_base0 - point_tile_base) * kSub;
        const __half* b1_ptr = sh_data + (size_t)(point_base1 - point_tile_base) * kSub;
        wmma::load_matrix_sync(a_frag, a_ptr, kSub);
        wmma::load_matrix_sync(b0_frag, b0_ptr, kSub);
        wmma::load_matrix_sync(b1_frag, b1_ptr, kSub);
        wmma::mma_sync(acc0_frag, a_frag, b0_frag, acc0_frag);
        wmma::mma_sync(acc1_frag, a_frag, b1_frag, acc1_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(dots0, acc0_frag, kSub, wmma::mem_row_major);
    wmma::store_matrix_sync(dots1, acc1_frag, kSub, wmma::mem_row_major);
    __syncthreads();

    if (lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = (g == 0 ? point_base0 : point_base1) + lane;
            const float* dots = (g == 0 ? dots0 : dots1);
            float best = 3.402823e38f;
            int best_idx = -1;
            if (point < curB) {
                #pragma unroll
                for (int r = 0; r < kSub; ++r) {
                    const int local_cent = cent_base - cent64_base + r;
                    if (local_cent < curK) {
                        const int cid = cbase + cent64_base + local_cent;
                        const float dot = dots[r * kSub + lane];
                        const float d2 = cnorm2_global[cid] - 2.0f * dot;
                        if (d2 < best || (d2 == best && cid < best_idx)) {
                            best = d2;
                            best_idx = cid;
                        }
                    }
                }
            }
            const int slot = (warp_id * kNGroups + g) * kSub + lane;
            warp_best_dist[slot] = best;
            warp_best_idx[slot] = best_idx;
        }
    }
    __syncthreads();

    if (cent_quad == 0 && lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = (g == 0 ? point_base0 : point_base1) + lane;
            if (point < curB) {
                float best = 3.402823e38f;
                int best_idx = -1;
                #pragma unroll
                for (int cq = 0; cq < 4; ++cq) {
                    const int src_warp = cq * 4 + point_quad;
                    const int slot = (src_warp * kNGroups + g) * kSub + lane;
                    const float d2 = warp_best_dist[slot];
                    const int cid = warp_best_idx[slot];
                    if (cid >= 0 && (d2 < best || (d2 == best && cid < best_idx))) {
                        best = d2;
                        best_idx = cid;
                    }
                }
                const size_t out = (size_t)point * cblocks64 + blockIdx.y;
                partial_dist2[out] = best;
                partial_idx[out] = best_idx;
            }
        }
    }
}

__global__ void kernel_assign_fused_argmin64x256_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
) {
    using namespace nvcuda;
    constexpr int kPointTile = 256;
    constexpr int kPointGroupStride = 64;
    constexpr int kCentTile = 64;
    constexpr int kSub = 16;
    constexpr int kWarps = 16;
    constexpr int kNGroups = 4;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarps) return;

    const int cent_quad = warp_id >> 2;
    const int point_quad = warp_id & 3;
    const int point_tile_base = blockIdx.x * kPointTile;
    const int cent64_base = blockIdx.y * kCentTile;
    const int cent_base = cent64_base + cent_quad * kSub;

    extern __shared__ unsigned char smem_raw[];
    float* dots_all = reinterpret_cast<float*>(smem_raw);
    float* warp_best_dist = dots_all + kWarps * kNGroups * kSub * kSub;
    int* warp_best_idx = reinterpret_cast<int*>(warp_best_dist + kWarps * kNGroups * kSub);
    float* dots[kNGroups];
    #pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
        dots[g] = dots_all + (warp_id * kNGroups + g) * kSub * kSub;
    }

    wmma::fragment<wmma::matrix_a, kSub, kSub, kSub, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kSub, kSub, kSub, __half, wmma::col_major> b_frag[kNGroups];
    wmma::fragment<wmma::accumulator, kSub, kSub, kSub, float> acc_frag[kNGroups];
    #pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
        wmma::fill_fragment(acc_frag[g], 0.0f);
    }

    int point_base[kNGroups];
    #pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
        point_base[g] = point_tile_base + g * kPointGroupStride + point_quad * kSub;
    }

    for (int d0 = 0; d0 < dim; d0 += kSub) {
        const __half* a_ptr = centroids + (size_t)(cbase + cent_base) * dim + d0;
        wmma::load_matrix_sync(a_frag, a_ptr, dim);
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const __half* b_ptr = data + (size_t)point_base[g] * dim + d0;
            wmma::load_matrix_sync(b_frag[g], b_ptr, dim);
            wmma::mma_sync(acc_frag[g], a_frag, b_frag[g], acc_frag[g]);
        }
    }

    #pragma unroll
    for (int g = 0; g < kNGroups; ++g) {
        wmma::store_matrix_sync(dots[g], acc_frag[g], kSub, wmma::mem_row_major);
    }
    __syncthreads();

    if (lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = point_base[g] + lane;
            float best = 3.402823e38f;
            int best_idx = -1;
            if (point < curB) {
                #pragma unroll
                for (int r = 0; r < kSub; ++r) {
                    const int local_cent = cent_base + r;
                    if (local_cent < curK) {
                        const int cid = cbase + local_cent;
                        const float dot = dots[g][r * kSub + lane];
                        const float d2 = cnorm2_global[cid] - 2.0f * dot;
                        if (d2 < best || (d2 == best && cid < best_idx)) {
                            best = d2;
                            best_idx = cid;
                        }
                    }
                }
            }
            const int slot = (warp_id * kNGroups + g) * kSub + lane;
            warp_best_dist[slot] = best;
            warp_best_idx[slot] = best_idx;
        }
    }
    __syncthreads();

    if (cent_quad == 0 && lane < kSub) {
        #pragma unroll
        for (int g = 0; g < kNGroups; ++g) {
            const int point = point_base[g] + lane;
            if (point < curB) {
                float best = 3.402823e38f;
                int best_idx = -1;
                #pragma unroll
                for (int cq = 0; cq < 4; ++cq) {
                    const int src_warp = cq * 4 + point_quad;
                    const int slot = (src_warp * kNGroups + g) * kSub + lane;
                    const float d2 = warp_best_dist[slot];
                    const int cid = warp_best_idx[slot];
                    if (cid >= 0 && (d2 < best || (d2 == best && cid < best_idx))) {
                        best = d2;
                        best_idx = cid;
                    }
                }
                const size_t out = (size_t)point * cblocks64 + blockIdx.y;
                partial_dist2[out] = best;
                partial_idx[out] = best_idx;
            }
        }
    }
}

__global__ void kernel_reduce_fused_argmin_partials(
    const float* __restrict__ partial_dist2,
    const int* __restrict__ partial_idx,
    int curB,
    int cblocks64,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= curB) return;

    float local_best = best_dist2[warp];
    int local_idx = best_idx[warp];
    const size_t base = (size_t)warp * cblocks64;
    for (int cb = lane; cb < cblocks64; cb += 32) {
        const float d2 = partial_dist2[base + cb];
        const int cid = partial_idx[base + cb];
        if (cid >= 0 && (d2 < local_best || (d2 == local_best && cid < local_idx))) {
            local_best = d2;
            local_idx = cid;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_best = __shfl_down_sync(0xffffffff, local_best, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_best < local_best || (other_best == local_best && other_idx < local_idx)) {
            local_best = other_best;
            local_idx = other_idx;
        }
    }

    if (lane == 0) {
        best_dist2[warp] = local_best;
        best_idx[warp] = local_idx;
    }
}

__global__ void kernel_update_best_from_dotT_warp_reduce(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_count = (blockDim.x + 31) / 32;
    const int data_idx = blockIdx.x * warp_count + warp_id;

    if (data_idx >= curB) return;

    const float xn = xnorm2[data_idx];
    float bestd = (lane_id == 0) ? best_dist2[data_idx] : 3.402823e38f;  // FLT_MAX
    int bestc = (lane_id == 0) ? best_idx[data_idx] : -1;

    //  curK / 32 centroid
    const int centroids_per_thread = (curK + 31) / 32;
    const int start_t = lane_id * centroids_per_thread;
    const int end_t = (start_t + centroids_per_thread < curK) ? (start_t + centroids_per_thread) : curK;

    // centroid
    for (int t = start_t; t < end_t; ++t) {
        float dot = dotT[t + (size_t)data_idx * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    // Warp shuffle  warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_d = __shfl_down_sync(0xffffffff, bestd, offset);
        int other_c = __shfl_down_sync(0xffffffff, bestc, offset);
        if (other_d < bestd) {
            bestd = other_d;
            bestc = other_c;
        }
    }

    // Lane 0
    if (lane_id == 0) {
        best_dist2[data_idx] = bestd;
        best_idx[data_idx] = bestc;
    }
}

/**
 * Kernel:  GEMM -
 *
 * 1.
 * 2.
 * 3.
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const int* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c = assign[i];
    int base = i * dim;

    //  counts
    atomicAdd(&counts[c], 1);

    //  accum
    float* acc = accum + (size_t)c * dim;

    //
    if (dim <= 4) {
        //
        if (dim >= 1) atomicAdd(&acc[0], data[base + 0]);
        if (dim >= 2) atomicAdd(&acc[1], data[base + 1]);
        if (dim >= 3) atomicAdd(&acc[2], data[base + 2]);
        if (dim >= 4) atomicAdd(&acc[3], data[base + 3]);
    } else if (dim <= 16) {
        // 4
        int j = 0;
        #pragma unroll 4
        for (; j + 4 <= dim; j += 4) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
        }
        //
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    } else {
        //
        int j = 0;
        #pragma unroll 8
        for (; j + 8 <= dim; j += 8) {
            atomicAdd(&acc[j], data[base + j]);
            atomicAdd(&acc[j+1], data[base + j+1]);
            atomicAdd(&acc[j+2], data[base + j+2]);
            atomicAdd(&acc[j+3], data[base + j+3]);
            atomicAdd(&acc[j+4], data[base + j+4]);
            atomicAdd(&acc[j+5], data[base + j+5]);
            atomicAdd(&acc[j+6], data[base + j+6]);
            atomicAdd(&acc[j+7], data[base + j+7]);
        }
        //
        for (; j < dim; ++j) {
            atomicAdd(&acc[j], data[base + j]);
        }
    }
}

// ============================================================
// Vector Reordering Kernels
// ============================================================

/**
 * Kernel: cluster
 * cluster
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] cluster
    int* __restrict__ cluster_counts,    // [k] cluster
    int n, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int cluster_id = assign[idx];
    if (cluster_id >= 0 && cluster_id < k) {
        // cluster
        int pos = atomicAdd(&cluster_counts[cluster_id], 1);
        cluster_indices[idx] = pos;
    } else {
        cluster_indices[idx] = -1;  // cluster ID
    }
}

/**
 * Kernel: Exclusive scan ()
 *  offsets[i] = sum(counts[0..i-1])
 * counts = [2, 3, 1]offsets = [0, 2, 5]
 *
 * kernel
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
) {
    //
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
}

/**
 * Kernel: clustercluster
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int cluster_id = assign[idx];
    int cluster_idx = cluster_indices[idx];

    if (cluster_id >= 0 && cluster_idx >= 0) {
        //
        int out_pos = cluster_offsets[cluster_id] + cluster_idx;

        //
        for (int d = 0; d < dim; ++d) {
            data_out[(size_t)out_pos * dim + d] = data_in[(size_t)idx * dim + d];
        }
    }
}
