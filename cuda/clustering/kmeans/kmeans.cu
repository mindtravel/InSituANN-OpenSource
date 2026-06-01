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
#include <thread>
#include <chrono>

#define ENABLE_CUDA_TIMING 0 /*CUDATimer*/

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU KMeans Runner Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,             // [n, dim] row-major
    int* d_assign,                   // [n]
    float* d_centroids,              // [k, dim] (in/out)
    float* h_objective                // sum dist2
) {
    CUDATimer timer_total("gpu_kmeans_lloyd: Total Time", ENABLE_CUDA_TIMING);
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== batch & ktile ======
    //  B  Ktile  d_dot
    // d_dot = B * Ktile <= 1GB ( 256M floats)
    const int Ktile = 4096;  // GEMMcentroid81924096
    const int B = 1 << 15;   // 32K256K32K
    // d_dot = 32K * 4K = 128M floats  512MB

    // ====== buffer ======
    float* d_accum = nullptr;      // [k, dim]
    int*   d_counts = nullptr;     // [k]
    float* d_cnorm2 = nullptr;     // [k]

    // GPU
    float* d_data_buf[2] = {nullptr, nullptr};  //
    cudaStream_t stream[2];                      // CUDA
    cudaEvent_t event[2];                        //

    // StreamEnv buffer
    StreamEnv stream_env[2];

    // d_centroids
    cudaMalloc(&d_accum,     sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_counts,    sizeof(int)   * (size_t)k);
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    //  buffer
    stream_env[0].allocate(B, Ktile);
    stream_env[1].allocate(B, Ktile);

    //
    cudaMalloc(&d_data_buf[0], sizeof(float) * (size_t)B * dim);
    cudaMalloc(&d_data_buf[1], sizeof(float) * (size_t)B * dim);
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaEventCreate(&event[0]);
    cudaEventCreate(&event[1]);

    // GPU  objective  float
    float* d_objective_sum = nullptr;
    if (h_objective) {
        cudaMalloc(&d_objective_sum, sizeof(float));
    }


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ======  ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // 01
    int cur_buf = 0;

    // ====== iteration ======
    for (int it = 0; it < cfg.iters; ++it) {
        CUDATimer timer_iter("gpu_kmeans_lloyd: Iteration " + std::to_string(it), ENABLE_CUDA_TIMING);
        // clear accum & counts
        // batch
        {
            CUDATimer timer_init("  Iter " + std::to_string(it) + ": Initialize (clear accum/counts, compute centroid norms)", ENABLE_CUDA_TIMING);
            cudaMemsetAsync(d_accum, 0, sizeof(float) * (size_t)k * dim, stream[0]);
            cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream[0]);
            //  stream[0]batch
            compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);
        }

        // ======  ======
        cur_buf = 0;

        //  objective
        if (h_objective && d_objective_sum) {
            cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[0]);
        }

        //  stream[0] accumcentroid
        // batch d_cnorm2  d_accum
        cudaEvent_t init_event;
        cudaEventCreate(&init_event);
        cudaEventRecord(init_event, stream[0]);

        // batch
        int first_base = 0;
        int first_curB = std::min(B, n - first_base);
        //  d_cnorm2  d_accum
        cudaStreamWaitEvent(stream[cur_buf], init_event, 0);
        cudaMemcpyAsync(d_data_buf[cur_buf], h_data + (size_t)first_base * dim,
                       sizeof(float) * (size_t)first_curB * dim,
                       cudaMemcpyHostToDevice, stream[cur_buf]);
        //
        cudaEventRecord(event[cur_buf], stream[cur_buf]);

        {
            CUDATimer timer_batch("  Iter " + std::to_string(it) + ": Process All Batches", ENABLE_CUDA_TIMING);
        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);
            float* d_data_cur = d_data_buf[cur_buf];

            // batch
            // batch
            cudaStreamWaitEvent(stream[cur_buf], event[cur_buf], 0);

            // batch
            //
            int next_base = base + B;
            int next_buf = 1 - cur_buf;

            //  StreamEnv buffer
            StreamEnv& cur_env = stream_env[cur_buf];

            // xnorm2 for this batch
            compute_l2_squared_gpu(d_data_cur, cur_env.d_xnorm2, curB, dim, L2NORM_AUTO, stream[cur_buf]);

            // init best_dist2 = +inf, best_idx = 0
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(cur_env.d_best_dist2, cur_env.d_best_idx, curB);
            }

            // ====== centroid  Ktile ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);

                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data_cur;
                float* Cc = cur_env.d_dot;

                // cuBLAS
                cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");

                // Treat row-major C[curK,dim] as col-major Ccm[dim,curK]
                // Treat row-major X[curB,dim] as col-major Xcm[dim,curB]
                // Compute dotT[curK,curB] (col-major) = Ccm^T[curK,dim] * Xcm[dim,curB]
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,     // A^T * B
                        curK, curB, dim,              // m=curK, n=curB, k=dim
                        &alpha,
                        A, dim,                       // A is Ccm with shape [dim,curK], lda=dim
                        Bm, dim,                      // B is Xcm with shape [dim,curB], ldb=dim
                        &beta,
                        Cc, curK                     // C is [curK,curB] col-major, ldc=curK
                    ),
                    "cublasSgemm(Ccm^T * Xcm)"
                );


                // best
                {
                    int threads = 256;
                    int blocks = (curB + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                        cur_env.d_dot, cur_env.d_xnorm2, d_cnorm2,
                        curB, curK, cbase,
                        cur_env.d_best_idx, cur_env.d_best_dist2
                    );
                }
            }

            // assign
            cudaMemcpyAsync(d_assign + base, cur_env.d_best_idx, sizeof(int) * (size_t)curB,
                           cudaMemcpyDeviceToDevice, stream[cur_buf]);

            // accum + counts
            //  block size  kernel
            {
                int threads = 512;  //  512
                int blocks = (curB + threads - 1) / threads;
                //  block
                blocks = std::min(blocks, 65535);
                kernel_accum_from_assign<<<blocks, threads, 0, stream[cur_buf]>>>(
                    d_data_cur, curB, dim,
                    cur_env.d_best_idx, d_accum, d_counts
                );
            }

            // objective GPU  reduce host
            if (h_objective && d_objective_sum) {
                int reduce_threads = 256;
                int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
                int reduce_shmem = reduce_threads * sizeof(float);
                kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[cur_buf]>>>(
                    cur_env.d_best_dist2, d_objective_sum, curB);
            }

            // batchbatch
            cudaEventRecord(event[cur_buf], stream[cur_buf]);

            // batchbatch
            if (next_base < n) {
                int next_curB = std::min(B, n - next_base);
                // batch
                cudaStreamWaitEvent(stream[next_buf], event[cur_buf], 0);
                cudaMemcpyAsync(d_data_buf[next_buf], h_data + (size_t)next_base * dim,
                               sizeof(float) * (size_t)next_curB * dim,
                               cudaMemcpyHostToDevice, stream[next_buf]);
                //
                cudaEventRecord(event[next_buf], stream[next_buf]);
            }

            //
            cur_buf = next_buf;
        }
        }

        // batch
        // batchaccum
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);

        // update centroids: one block per centroid stream[0]
        {
            CUDATimer timer_update("  Iter " + std::to_string(it) + ": Update Centroids", ENABLE_CUDA_TIMING);
        kernel_update_centroids<<<k, 256, 0, stream[0]>>>(d_centroids, d_accum, d_counts, k, dim);

        //  centroid  stream[0]
        cudaStreamSynchronize(stream[0]);
        }

        //  GPU  objective
        if (h_objective && d_objective_sum) {
            float obj_val = 0.0f;
            cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
            *h_objective = obj_val;
            printf("iter=%d, obj=%f\n", it, obj_val);
        }

        //
        cudaEventDestroy(init_event);
    }

    // ====== Final Assignment Pass ======
    // 2
    // d_centroids  d_assign
    //  assignment d_assign  d_centroids
    {
        CUDATimer timer_final("gpu_kmeans_lloyd: Final Assignment Pass", ENABLE_CUDA_TIMING);
        //  stream[0]  stream_env[0]  assignment
        //  d_data_buf[0] /
        perform_assignment_only(
            cfg, h_data, d_assign, d_centroids, d_cnorm2,
            stream_env[0], d_data_buf[0], stream[0], handle, B, Ktile, n, dim, k
        );
    }

    cublasDestroy(handle);

    //
    cudaFree(d_data_buf[0]);
    cudaFree(d_data_buf[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);

    //  StreamEnv buffer
    stream_env[0].free();
    stream_env[1].free();

    if (d_objective_sum) {
        cudaFree(d_objective_sum);
    }

    // d_centroids
    cudaFree(d_accum);
    cudaFree(d_counts);
    cudaFree(d_cnorm2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_lloyd: %s\n", cudaGetErrorString(err));
    }
}


// ============================================================
// Multi-GPU KMeans Lloyd
// ============================================================

void gpu_kmeans_lloyd_multigpu(
    const KMeansCase& cfg,
    const float* h_data,
    int* d_assign,         // on device 0
    float* d_centroids,    // on device 0
    float* h_objective
) {
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus < 2) {
        gpu_kmeans_lloyd(cfg, h_data, d_assign, d_centroids, h_objective);
        return;
    }
    if (n_gpus > 8) n_gpus = 8;

    const int n = cfg.n, dim = cfg.dim, k = cfg.k;
    const int B = 1 << 15;
    const int Ktile = 4096;
    const float alpha = 1.f, beta = 0.f;

    std::printf("[MGPU-KMEANS] n=%d k=%d dim=%d iters=%d gpus=%d\n",
                n, k, dim, cfg.iters, n_gpus);

    struct GPUCtx {
        int dev;
        int v_start, v_end;
        float* d_centroids;
        float* d_cnorm2;
        float* d_accum;
        int*   d_counts;
        float* d_data_buf[2];
        StreamEnv env[2];
        cudaStream_t stream[2];
        cudaEvent_t event[2];
        cublasHandle_t handle;
        float* d_obj_sum;
        int*   d_assign_local;
    };

    std::vector<GPUCtx> ctx(n_gpus);
    std::vector<float> h_centroids((size_t)k * dim);
    std::vector<float> h_accum_total((size_t)k * dim);
    std::vector<int>   h_counts_total(k);

    cudaSetDevice(0);
    cudaMemcpy(h_centroids.data(), d_centroids, (size_t)k * dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int g = 0; g < n_gpus; g++) {
        auto& c = ctx[g];
        c.dev = g;
        c.v_start = (int)((long long)n * g / n_gpus);
        c.v_end   = (int)((long long)n * (g + 1) / n_gpus);
        int local_n = c.v_end - c.v_start;

        cudaSetDevice(g);
        cudaMalloc(&c.d_centroids, (size_t)k * dim * sizeof(float));
        cudaMalloc(&c.d_cnorm2,    (size_t)k * sizeof(float));
        cudaMalloc(&c.d_accum,     (size_t)k * dim * sizeof(float));
        cudaMalloc(&c.d_counts,    (size_t)k * sizeof(int));
        cudaMalloc(&c.d_data_buf[0], (size_t)B * dim * sizeof(float));
        cudaMalloc(&c.d_data_buf[1], (size_t)B * dim * sizeof(float));
        cudaMalloc(&c.d_assign_local, (size_t)local_n * sizeof(int));
        c.d_obj_sum = nullptr;
        if (h_objective) cudaMalloc(&c.d_obj_sum, sizeof(float));
        c.env[0].allocate(B, Ktile);
        c.env[1].allocate(B, Ktile);
        cudaStreamCreate(&c.stream[0]);
        cudaStreamCreate(&c.stream[1]);
        cudaEventCreate(&c.event[0]);
        cudaEventCreate(&c.event[1]);
        cublasCreate(&c.handle);
        cublasSetMathMode(c.handle, CUBLAS_TF32_TENSOR_OP_MATH);

        cudaMemcpy(c.d_centroids, h_centroids.data(), (size_t)k * dim * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    for (int it = 0; it < cfg.iters; it++) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        /* Broadcast centroids to all GPUs */
        cudaSetDevice(0);
        cudaMemcpy(h_centroids.data(), d_centroids, (size_t)k * dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMemcpy(ctx[g].d_centroids, h_centroids.data(),
                       (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice);
            compute_l2_squared_gpu(ctx[g].d_centroids, ctx[g].d_cnorm2, k, dim,
                                   L2NORM_AUTO, ctx[g].stream[0]);
            cudaMemsetAsync(ctx[g].d_accum, 0, (size_t)k * dim * sizeof(float), ctx[g].stream[0]);
            cudaMemsetAsync(ctx[g].d_counts, 0, (size_t)k * sizeof(int), ctx[g].stream[0]);
            if (ctx[g].d_obj_sum)
                cudaMemsetAsync(ctx[g].d_obj_sum, 0, sizeof(float), ctx[g].stream[0]);
            cudaStreamSynchronize(ctx[g].stream[0]);
        }

        /* Parallel: each GPU processes its chunk */
        std::vector<std::thread> threads;
        for (int g = 0; g < n_gpus; g++) {
            threads.emplace_back([&, g]() {
                auto& c = ctx[g];
                cudaSetDevice(g);
                int local_n = c.v_end - c.v_start;
                const float* local_data = h_data + (size_t)c.v_start * dim;
                int cur_buf = 0;

                for (int base = 0; base < local_n; base += B) {
                    int curB = std::min(B, local_n - base);
                    float* d_cur = c.d_data_buf[cur_buf];
                    StreamEnv& se = c.env[cur_buf];

                    cudaMemcpyAsync(d_cur, local_data + (size_t)base * dim,
                                   (size_t)curB * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, c.stream[cur_buf]);

                    compute_l2_squared_gpu(d_cur, se.d_xnorm2, curB, dim,
                                          L2NORM_AUTO, c.stream[cur_buf]);
                    {
                        int thr = 256, blk = (curB + thr - 1) / thr;
                        kernel_init_best<<<blk, thr, 0, c.stream[cur_buf]>>>(
                            se.d_best_dist2, se.d_best_idx, curB);
                    }
                    for (int cbase = 0; cbase < k; cbase += Ktile) {
                        int curK = std::min(Ktile, k - cbase);
                        cublas_check(cublasSetStream(c.handle, c.stream[cur_buf]), "set");
                        cublas_check(cublasSgemm(c.handle,
                            CUBLAS_OP_T, CUBLAS_OP_N, curK, curB, dim, &alpha,
                            c.d_centroids + (size_t)cbase * dim, dim,
                            d_cur, dim, &beta, se.d_dot, curK), "sgemm");
                        {
                            int thr = 256, blk = (curB + thr - 1) / thr;
                            kernel_update_best_from_dotT<<<blk, thr, 0, c.stream[cur_buf]>>>(
                                se.d_dot, se.d_xnorm2, c.d_cnorm2, curB, curK, cbase,
                                se.d_best_idx, se.d_best_dist2);
                        }
                    }
                    cudaMemcpyAsync(c.d_assign_local + base, se.d_best_idx,
                                   curB * sizeof(int), cudaMemcpyDeviceToDevice,
                                   c.stream[cur_buf]);
                    {
                        int thr = 512, blk = std::min((curB + thr - 1) / thr, 65535);
                        kernel_accum_from_assign<<<blk, thr, 0, c.stream[cur_buf]>>>(
                            d_cur, curB, dim, se.d_best_idx, c.d_accum, c.d_counts);
                    }
                    if (c.d_obj_sum) {
                        int rt = 256, rb = (curB + rt - 1) / rt;
                        kernel_reduce_sum<<<rb, rt, rt * sizeof(float), c.stream[cur_buf]>>>(
                            se.d_best_dist2, c.d_obj_sum, curB);
                    }
                    cur_buf = 1 - cur_buf;
                }
                cudaStreamSynchronize(c.stream[0]);
                cudaStreamSynchronize(c.stream[1]);
            });
        }
        for (auto& t : threads) t.join();

        /* Reduce accum + counts to host, then update centroids on GPU 0 */
        std::memset(h_accum_total.data(), 0, (size_t)k * dim * sizeof(float));
        std::memset(h_counts_total.data(), 0, (size_t)k * sizeof(int));

        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            std::vector<float> h_acc((size_t)k * dim);
            std::vector<int>   h_cnt(k);
            cudaMemcpy(h_acc.data(), ctx[g].d_accum, (size_t)k * dim * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cnt.data(), ctx[g].d_counts, k * sizeof(int),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < (size_t)k * dim; i++) h_accum_total[i] += h_acc[i];
            for (int i = 0; i < k; i++) h_counts_total[i] += h_cnt[i];
        }

        /* Update centroids on GPU 0 */
        cudaSetDevice(0);
        cudaMemcpy(ctx[0].d_accum, h_accum_total.data(),
                   (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx[0].d_counts, h_counts_total.data(),
                   k * sizeof(int), cudaMemcpyHostToDevice);
        kernel_update_centroids<<<k, 256>>>(d_centroids, ctx[0].d_accum,
                                             ctx[0].d_counts, k, dim);
        cudaDeviceSynchronize();

        /* Objective */
        float obj = 0;
        if (h_objective) {
            for (int g = 0; g < n_gpus; g++) {
                float local_obj = 0;
                cudaSetDevice(g);
                cudaMemcpy(&local_obj, ctx[g].d_obj_sum, sizeof(float),
                           cudaMemcpyDeviceToHost);
                obj += local_obj;
            }
            *h_objective = obj;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
        std::printf("iter=%d, obj=%f, time=%.1fs\n", it, obj, iter_ms / 1000.0);
    }

    /* Final assignment pass with updated centroids  multi-GPU */
    std::printf("[MGPU-KMEANS] Final assignment (8-GPU)...\n");
    {
        auto fa_start = std::chrono::high_resolution_clock::now();
        /* Broadcast final centroids */
        cudaSetDevice(0);
        cudaMemcpy(h_centroids.data(), d_centroids, (size_t)k * dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMemcpy(ctx[g].d_centroids, h_centroids.data(),
                       (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice);
            compute_l2_squared_gpu(ctx[g].d_centroids, ctx[g].d_cnorm2, k, dim,
                                   L2NORM_AUTO, ctx[g].stream[0]);
            cudaStreamSynchronize(ctx[g].stream[0]);
        }
        /* Each GPU re-assigns its chunk */
        std::vector<std::thread> fa_threads;
        for (int g = 0; g < n_gpus; g++) {
            fa_threads.emplace_back([&, g]() {
                auto& c = ctx[g];
                cudaSetDevice(g);
                int local_n = c.v_end - c.v_start;
                const float* local_data = h_data + (size_t)c.v_start * dim;
                int cur_buf = 0;
                for (int base = 0; base < local_n; base += B) {
                    int curB = std::min(B, local_n - base);
                    float* d_cur = c.d_data_buf[cur_buf];
                    StreamEnv& se = c.env[cur_buf];
                    cudaMemcpyAsync(d_cur, local_data + (size_t)base * dim,
                                   (size_t)curB * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, c.stream[cur_buf]);
                    compute_l2_squared_gpu(d_cur, se.d_xnorm2, curB, dim,
                                          L2NORM_AUTO, c.stream[cur_buf]);
                    { int thr=256, blk=(curB+thr-1)/thr;
                      kernel_init_best<<<blk,thr,0,c.stream[cur_buf]>>>(
                          se.d_best_dist2, se.d_best_idx, curB); }
                    for (int cbase = 0; cbase < k; cbase += Ktile) {
                        int curK = std::min(Ktile, k - cbase);
                        cublas_check(cublasSetStream(c.handle, c.stream[cur_buf]), "set");
                        cublas_check(cublasSgemm(c.handle,
                            CUBLAS_OP_T, CUBLAS_OP_N, curK, curB, dim, &alpha,
                            c.d_centroids + (size_t)cbase * dim, dim,
                            d_cur, dim, &beta, se.d_dot, curK), "sgemm");
                        { int thr=256, blk=(curB+thr-1)/thr;
                          kernel_update_best_from_dotT<<<blk,thr,0,c.stream[cur_buf]>>>(
                              se.d_dot, se.d_xnorm2, c.d_cnorm2, curB, curK, cbase,
                              se.d_best_idx, se.d_best_dist2); }
                    }
                    cudaMemcpyAsync(c.d_assign_local + base, se.d_best_idx,
                                   curB * sizeof(int), cudaMemcpyDeviceToDevice,
                                   c.stream[cur_buf]);
                    cur_buf = 1 - cur_buf;
                }
                cudaStreamSynchronize(c.stream[0]);
                cudaStreamSynchronize(c.stream[1]);
            });
        }
        for (auto& t : fa_threads) t.join();
        auto fa_end = std::chrono::high_resolution_clock::now();
        double fa_ms = std::chrono::duration<double, std::milli>(fa_end - fa_start).count();
        std::printf("[MGPU-KMEANS] Final assignment done: %.1fs\n", fa_ms / 1000.0);
    }

    /* Copy assignments to d_assign on GPU 0 */
    cudaSetDevice(0);
    for (int g = 0; g < n_gpus; g++) {
        auto& c = ctx[g];
        int local_n = c.v_end - c.v_start;
        if (g == 0) {
            cudaMemcpy(d_assign + c.v_start, c.d_assign_local,
                       local_n * sizeof(int), cudaMemcpyDeviceToDevice);
        } else {
            std::vector<int> h_tmp(local_n);
            cudaSetDevice(g);
            cudaMemcpy(h_tmp.data(), c.d_assign_local, local_n * sizeof(int),
                       cudaMemcpyDeviceToHost);
            cudaSetDevice(0);
            cudaMemcpy(d_assign + c.v_start, h_tmp.data(),
                       local_n * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    /* Cleanup */
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(ctx[g].d_centroids);
        cudaFree(ctx[g].d_cnorm2);
        cudaFree(ctx[g].d_accum);
        cudaFree(ctx[g].d_counts);
        cudaFree(ctx[g].d_data_buf[0]);
        cudaFree(ctx[g].d_data_buf[1]);
        cudaFree(ctx[g].d_assign_local);
        if (ctx[g].d_obj_sum) cudaFree(ctx[g].d_obj_sum);
        ctx[g].env[0].free(); ctx[g].env[1].free();
        cudaStreamDestroy(ctx[g].stream[0]);
        cudaStreamDestroy(ctx[g].stream[1]);
        cudaEventDestroy(ctx[g].event[0]);
        cudaEventDestroy(ctx[g].event[1]);
        cublasDestroy(ctx[g].handle);
    }
    cudaSetDevice(0);
    std::printf("[MGPU-KMEANS] done\n");
}


// ============================================================
// CPU Initialization Functions Implementation
// ============================================================

__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
) {
    std::mt19937 rng(cfg.seed);

    //  [0, 1, 2, ..., n-1]
    std::vector<int> indices(cfg.n);
    for (int i = 0; i < cfg.n; ++i) {
        indices[i] = i;
    }

    // Fisher-Yates  k
    //  k  k
    for (int i = 0; i < cfg.k && i < cfg.n; ++i) {
        std::uniform_int_distribution<int> dist(i, cfg.n - 1);
        int j = dist(rng);
        std::swap(indices[i], indices[j]);
    }

    //  k
    for (int c = 0; c < cfg.k; ++c) {
        int idx = indices[c];
        std::memcpy(out_centroids + (size_t)c * cfg.dim,
                    data + (size_t)idx * cfg.dim,
                    sizeof(float) * cfg.dim);
    }
}
