#include "kmeans.cuh"
#include "utils.cuh"
#include "l2norm/l2norm.cuh"
#include "cudatimer.h"
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define ENABLE_CUDA_TIMING 0 /*CUDATimer*/

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU KMeans Minibatch Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_minibatch(
    const KMeansCase& cfg,
    const float* h_data,             // [n, dim] row-major
    int* d_assign,                   // [n] (minibatchassign)
    float* d_centroids,              // [k, dim] (in/out)
    float* h_objective                // sum dist2 ()
) {
    CUDATimer timer_total("gpu_kmeans_minibatch: Total Time", ENABLE_CUDA_TIMING);
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== Minibatch  ======
    const int MINIBATCH_SIZE = 1 << 15;  // minibatch32K
    // centroid
    const int Ktile = 4096;              // GEMMcentroid

    // ====== buffer ======
    float* d_cnorm2 = nullptr;     // [k]

    // minibatch
    const int M = MINIBATCH_SIZE;
    float* d_minibatch_data[2] = {nullptr, nullptr};  // [M, dim]
    cudaStream_t stream[2];                           // CUDA
    //
    cudaEvent_t evt_h2d_done[2];   // H2D
    cudaEvent_t evt_compute_done[2];  //

    // StreamEnv buffer
    StreamEnv stream_env[2];

    // d_centroids
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    //  buffer
    stream_env[0].allocate(M, Ktile);
    stream_env[1].allocate(M, Ktile);

    // minibatch
    cudaMalloc(&d_minibatch_data[0], sizeof(float) * (size_t)M * dim);
    cudaMalloc(&d_minibatch_data[1], sizeof(float) * (size_t)M * dim);
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    //
    cudaEventCreate(&evt_h2d_done[0]);
    cudaEventCreate(&evt_h2d_done[1]);
    cudaEventCreate(&evt_compute_done[0]);
    cudaEventCreate(&evt_compute_done[1]);

    // Minibatchcentroid
    float* d_minibatch_accum = nullptr;  // [k, dim]
    int*   d_minibatch_counts = nullptr; // [k]
    cudaMalloc(&d_minibatch_accum, sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_minibatch_counts, sizeof(int) * (size_t)k);

    // GPU
    int* d_total_counts = nullptr;  // [k]
    cudaMalloc(&d_total_counts, sizeof(int) * (size_t)k);
    cudaMemset(d_total_counts, 0, sizeof(int) * (size_t)k);

    // GPU  objective  float
    float* d_objective_sum = nullptr;
    if (h_objective) {
        cudaMalloc(&d_objective_sum, sizeof(float));
    }

    // minibatch
    std::mt19937 rng(cfg.seed);


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ======  ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // ====== Minibatch ======
    int cur_buf = 0;

    // n < Mminibatch
    const int actual_minibatch_size = (n < M) ? n : M;

    // minibatch
    int first_minibatch_start = 0;
    if (n > M) {
        std::uniform_int_distribution<int> dist_idx(0, n - M);
        first_minibatch_start = dist_idx(rng);
        if (first_minibatch_start + M > n) {
            first_minibatch_start = n - M;
        }
    }
    //  n <= M
    cudaMemcpyAsync(d_minibatch_data[cur_buf], h_data + (size_t)first_minibatch_start * dim,
                   sizeof(float) * (size_t)actual_minibatch_size * dim,
                   cudaMemcpyHostToDevice, stream[cur_buf]);
    cudaEventRecord(evt_h2d_done[cur_buf], stream[cur_buf]);

    for (int it = 0; it < cfg.minibatch_iters; ++it) {
        CUDATimer timer_iter("gpu_kmeans_minibatch: Iteration " + std::to_string(it), ENABLE_CUDA_TIMING);

        // minibatch
        cudaStreamWaitEvent(stream[cur_buf], evt_h2d_done[cur_buf], 0);

        // minibatch
        int next_buf = 1 - cur_buf;
        if (it + 1 < cfg.minibatch_iters && n > M) {
            std::uniform_int_distribution<int> dist_idx(0, n - M);
            int next_minibatch_start = dist_idx(rng);
            if (next_minibatch_start + M > n) {
                next_minibatch_start = n - M;
            }
            // next_buf
            cudaStreamWaitEvent(stream[next_buf], evt_compute_done[next_buf], 0);
            cudaMemcpyAsync(d_minibatch_data[next_buf], h_data + (size_t)next_minibatch_start * dim,
                           sizeof(float) * (size_t)actual_minibatch_size * dim,
                           cudaMemcpyHostToDevice, stream[next_buf]);
            cudaEventRecord(evt_h2d_done[next_buf], stream[next_buf]);
        }

        float* d_data_cur = d_minibatch_data[cur_buf];

        {
            CUDATimer timer_init("  Iter " + std::to_string(it) + ": Initialize (compute centroid norms, clear accum/counts)", ENABLE_CUDA_TIMING);
            // stream[cur_buf]centroid
            compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[cur_buf]);

            // minibatch
            cudaMemsetAsync(d_minibatch_accum, 0, sizeof(float) * (size_t)k * dim, stream[cur_buf]);
            cudaMemsetAsync(d_minibatch_counts, 0, sizeof(int) * (size_t)k, stream[cur_buf]);

            // objective
            if (h_objective && d_objective_sum) {
                cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[cur_buf]);
            }
        }

        {
            CUDATimer timer_minibatch("  Iter " + std::to_string(it) + ": Process Minibatch", ENABLE_CUDA_TIMING);
            //  StreamEnv buffer
            StreamEnv& cur_env = stream_env[cur_buf];

            // minibatchxnorm2
            compute_l2_squared_gpu(d_data_cur, cur_env.d_xnorm2, actual_minibatch_size, dim, L2NORM_AUTO, stream[cur_buf]);

            // best_dist2best_idx
            {
                int threads = 256;
                int blocks = (actual_minibatch_size + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(cur_env.d_best_dist2, cur_env.d_best_idx, actual_minibatch_size);
            }

            // ====== centroid ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);

                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data_cur;
                float* Cc = cur_env.d_dot;

                // cuBLAS
                cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");

                // GEMM: dotT[curK, actual_M] = centroids[curK, dim]^T * minibatch[actual_M, dim]
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        curK, actual_minibatch_size, dim,
                        &alpha,
                        A, dim,
                        Bm, dim,
                        &beta,
                        Cc, curK
                    ),
                    "cublasSgemm"
                );

                // best
                {
                    int threads = 256;
                    int blocks = (actual_minibatch_size + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                        cur_env.d_dot, cur_env.d_xnorm2, d_cnorm2,
                        actual_minibatch_size, curK, cbase,
                        cur_env.d_best_idx, cur_env.d_best_dist2
                    );
                }
            }

            // minibatchassign
            //  block size
            {
                int threads = 512;  //  512
                int blocks = (actual_minibatch_size + threads - 1) / threads;
                blocks = std::min(blocks, 65535);
                kernel_accum_from_assign<<<blocks, threads, 0, stream[cur_buf]>>>(
                    d_data_cur, actual_minibatch_size, dim,
                    cur_env.d_best_idx, d_minibatch_accum, d_minibatch_counts
                );
            }
        }

        {
            CUDATimer timer_update("  Iter " + std::to_string(it) + ": Update Centroids (Minibatch)", ENABLE_CUDA_TIMING);
            // GPUcentroidshostD2H/H2D
            // Minibatch
            // minibatch k-means
            // centroid_new = centroid_old + lr * (centroid_batch - centroid_old)
            //  lr = 1 / (total_count + 1)total_count
            // centroid_new = (1 - lr) * centroid_old + lr * centroid_batch
            {
                // GPU kernelcentroidshost
                kernel_update_centroids_minibatch<<<k, 256, 0, stream[cur_buf]>>>(
                    d_centroids, d_minibatch_accum, d_minibatch_counts, d_total_counts, k, dim);
            }
        }

        // minibatchminibatch
        cudaEventRecord(evt_compute_done[cur_buf], stream[cur_buf]);

        //
        if (it + 1 < cfg.minibatch_iters) {
            cur_buf = next_buf;
        }
    }
    cudaDeviceSynchronize();

    // // objectiveCPU
    // if (h_objective && d_objective_sum) {
    //     CUDATimer timer_objective("gpu_kmeans_minibatch: Compute Objective (All Data)", ENABLE_CUDA_TIMING);
    //     cudaMemset(d_objective_sum, 0, sizeof(float));

    //     // centroid
    //     compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);

    //     // objective
    //     const int B = 1 << 15;  // 32K per batch
    //     for (int base = 0; base < n; base += B) {
    //         int curB = std::min(B, n - base);

    //         // batch
    //         cudaMemcpyAsync(d_minibatch_data[0], h_data + (size_t)base * dim,
    //                        sizeof(float) * (size_t)curB * dim,
    //                        cudaMemcpyHostToDevice, stream[0]);

    //         // xnorm2
    //         compute_l2_squared_gpu(d_minibatch_data[0], d_xnorm2, curB, dim, L2NORM_AUTO, stream[0]);

    //         // best
    //         {
    //             int threads = 256;
    //             int blocks = (curB + threads - 1) / threads;
    //             kernel_init_best<<<blocks, threads, 0, stream[0]>>>(d_best_dist2, d_best_idx, curB);
    //         }

    //         // centroid
    //         for (int cbase = 0; cbase < k; cbase += Ktile) {
    //             int curK = std::min(Ktile, k - cbase);

    //             const float* A = d_centroids + (size_t)cbase * dim;
    //             const float* Bm = d_minibatch_data[0];
    //             float* Cc = d_dot;

    //             cublas_check(cublasSetStream(handle, stream[0]), "cublasSetStream");

    //             cublas_check(
    //                 cublasSgemm(
    //                     handle,
    //                     CUBLAS_OP_T, CUBLAS_OP_N,
    //                     curK, curB, dim,
    //                     &alpha,
    //                     A, dim,
    //                     Bm, dim,
    //                     &beta,
    //                     Cc, curK
    //                 ),
    //                 "cublasSgemm"
    //             );

    //             {
    //                 int threads = 256;
    //                 int blocks = (curB + threads - 1) / threads;
    //                 kernel_update_best_from_dotT<<<blocks, threads, 0, stream[0]>>>(
    //                     d_dot, d_xnorm2, d_cnorm2,
    //                     curB, curK, cbase,
    //                     d_best_idx, d_best_dist2
    //                 );
    //             }
    //         }

    //         // objective
    //         {
    //             int reduce_threads = 256;
    //             int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
    //             int reduce_shmem = reduce_threads * sizeof(float);
    //             kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[0]>>>(
    //                 d_best_dist2, d_objective_sum, curB);
    //         }
    //     }

    //     //
    //     cudaStreamSynchronize(stream[0]);

    //     // objective
    //     float obj_val = 0.0f;
    //     cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
    //     *h_objective = obj_val;
    //     COUT_ENDL("minibatch final obj=", obj_val);
    // }

    // ======  ======
    // Minibatch
    if (d_assign) {
        CUDATimer timer_final_assign("gpu_kmeans_minibatch: Final Assignment (All Points)", ENABLE_CUDA_TIMING);

        //  stream_env[0]  assignment
        //  d_minibatch_data[0] /
        perform_assignment_only(
            cfg, h_data, d_assign, d_centroids, d_cnorm2,
            stream_env[0], d_minibatch_data[0], stream[0], handle, MINIBATCH_SIZE, Ktile, n, dim, k
        );
    }

    cublasDestroy(handle);

    //
    cudaFree(d_minibatch_data[0]);
    cudaFree(d_minibatch_data[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaEventDestroy(evt_h2d_done[0]);
    cudaEventDestroy(evt_h2d_done[1]);
    cudaEventDestroy(evt_compute_done[0]);
    cudaEventDestroy(evt_compute_done[1]);

    if (d_objective_sum) {
        cudaFree(d_objective_sum);
    }

    //  StreamEnv buffer
    stream_env[0].free();
    stream_env[1].free();

    // d_centroids
    cudaFree(d_cnorm2);
    cudaFree(d_minibatch_accum);
    cudaFree(d_minibatch_counts);
    cudaFree(d_total_counts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_minibatch: %s\n", cudaGetErrorString(err));
    }
}