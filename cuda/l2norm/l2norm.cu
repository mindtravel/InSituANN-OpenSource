/**
 * l2norm_optimized.cu
 *  L2  kernel
 *
 *
 * 1.  (vectorized load)
 * 2. Warp-level
 * 3.
 * 4.  __ldg()
 * 5.
 */

#include "l2norm.cuh"
#include <device_launch_parameters.h>
#include <cmath>
#include "pch.h"

/*
 * :  + warp-level
 *
 *
 * -  __ldg()
 * -  n_dim <= 32 warp
 * -  32 < n_dim <= 128 warp-level  +
 * -  n_dim > 128
 */
__global__ void l2_norm_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;  // lane within warp
    const int warp_id = tid / 32;
    const int n_warps = (blockDim.x + 31) / 32;

    if (bid >= n_batch) return;

    const float* vec_ptr = vectors + bid * n_dim;

    /* 1:  <= 32 warp-level  */
    if (n_dim <= 32) {
        /*  warp  */
        if (warp_id == 0) {
            float sum = 0.0f;

            /*  lane  */
            if (lane < n_dim) {
                /*  __ldg()  */
                float val = __ldg(&vec_ptr[lane]);
                sum = val * val;
            }

            /* Warp-level  shuffle  */
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            /* Lane 0  */
            if (lane == 0) {
                vector_l2_norm[bid] = sqrtf(sum);
            }
        }
        return;
    }

    /* 2: 32 < n_dim <= 128 warp-level  +  */
    if (n_dim <= 128) {
        extern __shared__ float sdata[];
        float sum = 0.0f;

        /*  */
        const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < n_dim) {
                float val = __ldg(&vec_ptr[idx]);
                sum += val * val;
            }
        }

        /* Warp-level  */
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        /*  warp  */
        if (lane == 0) {
            sdata[warp_id] = sum;
        }
        __syncthreads();

        /* Warp 0  warp  */
        if (warp_id == 0) {
            sum = (lane < n_warps) ? sdata[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                vector_l2_norm[bid] = sqrtf(sum);
            }
        }
        return;
    }

    /* 3: n_dim > 128 warp-level  + 2 */
    extern __shared__ float sdata[];

    /*  */
    float sum = 0.0f;
    const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;

    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < n_dim) {
            /*  __ldg()  */
            float val = __ldg(&vec_ptr[idx]);
            sum += val * val;
        }
    }

    /* Warp-level  */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /*  warp  */
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    /* Warp 0  warp  */
    if (warp_id == 0) {
        sum = (lane < n_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
    }
}

/*
 * 2:
 *
 *
 * -  (<=32): warp shuffle
 * -  (33-256): warp shuffle + shared memory
 * -  (>256):
 */
__global__ void l2_norm_kernel_optimized_v2(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    const int n_warps = (blockDim.x + 31) / 32;

    if (bid >= n_batch) return;

    const float* vec_ptr = vectors + bid * n_dim;

    /*  */
    float sum = 0.0f;
    const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;

    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < n_dim) {
            /*  __ldg()  */
            float val = __ldg(&vec_ptr[idx]);
            sum += val * val;
        }
    }

    /* Warp-level  */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /*  warp  */
    if (n_warps == 1 || n_dim <= 32) {
        /*  warp  */
        if (warp_id == 0 && lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
        return;
    }

    /*  warp  */
    extern __shared__ float sdata[];
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    /* Warp 0  */
    if (warp_id == 0) {
        sum = (lane < n_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
    }
}

/*
 * Device float4 L2FMA
 *
 *
 * -  4  float
 * - FMA
 * - devicekernel
 *
 * @param vec_ptr
 * @param n_dim
 * @param tid ID
 * @param block_dim block
 * @return
 */
__device__ __forceinline__ float compute_l2_norm_float4_device(
    const float* __restrict__ vec_ptr,
    int n_dim,
    int tid,
    int block_dim)
{
    const float4* vec_ptr4 = reinterpret_cast<const float4*>(vec_ptr);
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    const int n_warps = (block_dim + 31) / 32;

    float sum = 0.0f;
    const int vec4_count = n_dim / 4;
    const int remainder = n_dim % 4;

    /*  float4FMA */
    #pragma unroll 4
    for (int i = 0; i < vec4_count; i += block_dim) {
        int idx = i + tid;
        if (idx < vec4_count) {
            /*  __ldg()  */
            float4 val4 = __ldg(&vec_ptr4[idx]);
            /* FMA */
            sum += val4.x * val4.x + val4.y * val4.y +
       val4.z * val4.z + val4.w * val4.w;
        }
    }

    /* 4FMA */
    if (remainder > 0) {
        int idx = vec4_count * 4 + tid;
        if (idx < n_dim) {
            float val = __ldg(&vec_ptr[idx]);
            sum = fmaf(val, val, sum);
        }
    }

    /* Warp-level  */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    /*  warp  */
    if (n_warps > 1) {
        extern __shared__ float sdata[];
        if (lane == 0) {
            sdata[warp_id] = sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            sum = (lane < n_warps) ? sdata[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
        }
    }

    return sum;
}

/*
 * Kernel float4 FMA
 * device
 */
__global__ void l2_norm_kernel_float4(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid / 32;

    if (bid >= n_batch) return;

    const float* vec_ptr = vectors + bid * n_dim;

    /* device */
    float sum = compute_l2_norm_float4_device(vec_ptr, n_dim, tid, blockDim.x);

    /*  */
    if (lane == 0 && warp_id == 0) {
        vector_l2_norm[bid] = sqrtf(sum);
    }
}

/*
*
* Args:
*   vectors:
*   vetcor_suqared_sum: l2 norm
*   n_dim:
*   n_batch:
*
* sqrt
*
*/
__global__ void l2_norm_kernel_basic(
    float *vectors,
    float *vector_l2_norm,
    int n_batch,
    int n_dim
)
{
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_dim = blockDim.x;

    //
    float square = 0.0f;
    if (tid < n_dim) {
        int idx = bid * n_dim + tid;
        square = vectors[idx] * vectors[idx];
    }

    //
    if (tid < block_dim) {
        shared_mem[tid] = (tid < n_dim) ? square : 0.0f;
    }
    __syncthreads();

    // block_sizen_dim
    // n_dimblock_dim
    int reduce_size = block_dim;
    if (reduce_size > n_dim) {
        reduce_size = n_dim;
    }

    for (int s = 1; s < reduce_size; s *= 2) {
        __syncthreads();
        if (tid % (2 * s) == 0 && (tid + s) < reduce_size) {
            shared_mem[tid] += shared_mem[tid + s];
        }
    }

    // L2
    if (tid == 0) {
        vector_l2_norm[bid] = sqrtf(shared_mem[0]);
    }
}

/**
 * L2host
 *
 *
 */
void compute_l2_norm_gpu(
    const float* vectors,
    float* vector_l2_norm,
    int n_batch,
    int n_dim,
    L2NormVersion version,
    cudaStream_t stream)
{
    /* kernel */
    const int block_size = 256;  /* block */
    const int grid_size = n_batch;

    /* kernel */
    const int shared_mem_size = (n_dim > 32) ?
        ((block_size + 31) / 32) * sizeof(float) : 0;

    /*  */
    if (version != L2NORM_AUTO) {
        switch (version) {
            case L2NORM_BASIC: {
                /* basic kernelblockn_dim */
                /* min(1024, max(n_dim, 256)) */
                const int basic_block_size = (n_dim <= 256) ? 256 : ((n_dim <= 1024) ? n_dim : 256);
                const int basic_shared_mem = basic_block_size * sizeof(float);
                if (stream) {
                    l2_norm_kernel_basic<<<grid_size, basic_block_size,
                        basic_shared_mem, stream>>>(
                        const_cast<float*>(vectors), vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_basic<<<grid_size, basic_block_size,
                        basic_shared_mem>>>(
                        const_cast<float*>(vectors), vector_l2_norm, n_batch, n_dim);
                }
                break;
            }

            case L2NORM_OPTIMIZED:
                if (stream) {
                    l2_norm_kernel<<<grid_size, block_size,
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel<<<grid_size, block_size,
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;

            case L2NORM_OPTIMIZED_V2:
                if (stream) {
                    l2_norm_kernel_optimized_v2<<<grid_size, block_size,
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_optimized_v2<<<grid_size, block_size,
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;

            case L2NORM_OPTIMIZED_V3:
                if (stream) {
                    l2_norm_kernel_float4<<<grid_size, block_size,
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_float4<<<grid_size, block_size,
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;

            default:
                /*  */
                version = L2NORM_AUTO;
                break;
        }

        if (version != L2NORM_AUTO) {
            return;
        }
    }

    /*  */
    /* 1: dim4float4 */
    if (n_dim >= 128 && (n_dim % 4 == 0)) {
        if (stream) {
            l2_norm_kernel_float4<<<grid_size, block_size,
                shared_mem_size, stream>>>(
                vectors, vector_l2_norm, n_batch, n_dim);
        } else {
            l2_norm_kernel_float4<<<grid_size, block_size,
                shared_mem_size>>>(
                vectors, vector_l2_norm, n_batch, n_dim);
        }
        return;
    }

    /* 2: 2 */
    /* dim */
    if (stream) {
        l2_norm_kernel_optimized_v2<<<grid_size, block_size,
            shared_mem_size, stream>>>(
            vectors, vector_l2_norm, n_batch, n_dim);
    } else {
        l2_norm_kernel_optimized_v2<<<grid_size, block_size,
            shared_mem_size>>>(
            vectors, vector_l2_norm, n_batch, n_dim);
    }
}
