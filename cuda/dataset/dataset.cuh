#ifndef CLUSTER_DATASET_CUH
#define CLUSTER_DATASET_CUH

#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"
#include "l2norm/l2norm.cuh"
#include "cpu_array_utils.h"
#include "numa_utils.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>
#include <thread>
#include <malloc.h>
/**
 * Cluster
 *  ClusterInfo
 *  ClusteringInterface
 */
struct ClusterDataset {
    float* reordered_data;      // [n_total_vectors, vector_dim]
    int* reordered_indices;     // [n_total_vectors] reordered_data
    float* centroids;           // [cluster_info.k, vector_dim] CPU
    ClusterInfo cluster_info;   // clusteroffsetscountskcluster
    int n_total_vectors;        //
    int vector_dim;             //

    /**
     *
     */
    void init_with_clustering(
        ClusteringInterface* clusterer,
        const float* h_data,
        int n_total_vectors,
        int vector_dim,
        const ClusteringConfig& cfg
    ) {
        clusterer->fit(h_data, n_total_vectors, vector_dim, cfg);
        ClusteringResult res = clusterer->get_result();
        _copy_from_clustering_result(res);
    }

    /**
     * ClusterDatasetK-means KMeansClustering + init_with_clustering
     */
    void init_with_kmeans(
        int n_total_vectors,
        int vector_dim,
        int n_clusters,
        float* h_objective,
        float* h_data = nullptr,
        int kmeans_iters = 20,
        bool use_minibatch = false,
        DistanceType distance_mode = COSINE_DISTANCE,
        unsigned int seed = 1234,
        int batch_size = (1 << 20),
        int device_id = 0
    ) {
        (void)h_objective;
        this->n_total_vectors = n_total_vectors;
        this->vector_dim = vector_dim;
        size_t data_size = (size_t)n_total_vectors * vector_dim;
        float* h_data_ptr = nullptr;
        bool need_free = false;
        if (!h_data) {
            need_free = true;
            h_data_ptr = (float*)memalign(64, data_size * sizeof(float));
            init_array_multithreaded(h_data_ptr, data_size, seed, -1.0f, 1.0f);
        } else {
            h_data_ptr = h_data;
        }
        ClusteringConfig cfg;
        cfg.n_clusters = n_clusters;
        cfg.max_iters = kmeans_iters;
        cfg.distance_mode = distance_mode;
        cfg.seed = seed;
        cfg.use_minibatch = use_minibatch;
        cfg.batch_size = batch_size;
        cfg.device_id = device_id;
        KMeansClustering kmeans;
        init_with_clustering(&kmeans, h_data_ptr, n_total_vectors, vector_dim, cfg);
        if (need_free) std::free(h_data_ptr);
    }

    /**
     * Subset KMeans: train on first train_n vectors, assign ALL n vectors, reorder all.
     * Dramatically faster than full KMeans on 1B vectors.
     */
    void init_with_subset_kmeans(
        int n, int dim, int nlist, float* h_objective,
        float* h_base, int train_n, int kmeans_iters,
        DistanceType dist = L2_DISTANCE, int device_id = 0
    ) {
        this->n_total_vectors = n;
        this->vector_dim = dim;

        std::printf("[SUBSET-KMEANS] train_n=%d (of %d)  nlist=%d iters=%d dev=%d\n",
                    train_n, n, nlist, kmeans_iters, device_id);

        /* --- Phase 1: KMeans on subset  get centroids --- */
        ClusterDataset sub;
        sub.init_with_kmeans(train_n, dim, nlist, h_objective, h_base,
                             kmeans_iters, false, dist, 1234, 1 << 20, device_id);
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        std::memcpy(centroids, sub.centroids, (size_t)nlist * dim * sizeof(float));
        sub.release();
        std::printf("[SUBSET-KMEANS] Phase 1 done: centroids trained\n");

        /* --- Phase 2: GPU-assisted assignment of ALL n vectors --- */
        std::printf("[SUBSET-KMEANS] Phase 2: assigning %d vectors ...\n", n);
        std::vector<int32_t> assign(n);
        {
            int n_gpus = 0;
            cudaGetDeviceCount(&n_gpus);
            if (n_gpus > 8) n_gpus = 8;
            if (n_gpus < 1) n_gpus = 1;
            if (const char* env_gpus = std::getenv("IVFT_SUBSET_ASSIGN_GPUS")) {
                int requested = std::atoi(env_gpus);
                if (requested > 0) n_gpus = std::min(n_gpus, requested);
            }
            std::printf("[SUBSET-KMEANS] Phase 2 uses %d GPU(s)\n", n_gpus);
            const int B = 1 << 15;
            const int Ktile = 4096;

            auto gpu_worker = [&](int gpu, int v_start, int v_end) {
                cudaSetDevice(gpu);
                float* d_centroids = nullptr;
                cudaMalloc(&d_centroids, (size_t)nlist * dim * sizeof(float));
                cudaMemcpy(d_centroids, centroids, (size_t)nlist * dim * sizeof(float),
                           cudaMemcpyHostToDevice);

                float* d_cnorm2 = nullptr;
                cudaMalloc(&d_cnorm2, (size_t)nlist * sizeof(float));
                compute_l2_squared_gpu(d_centroids, d_cnorm2, nlist, dim, L2NORM_AUTO, 0);
                cudaDeviceSynchronize();

                float* d_data = nullptr;
                float* d_xnorm2 = nullptr;
                float* d_dot = nullptr;
                float* d_best_dist2 = nullptr;
                int*   d_best_idx = nullptr;
                cudaMalloc(&d_data,       (size_t)B * dim * sizeof(float));
                cudaMalloc(&d_xnorm2,     (size_t)B * sizeof(float));
                cudaMalloc(&d_dot,        (size_t)B * Ktile * sizeof(float));
                cudaMalloc(&d_best_dist2, (size_t)B * sizeof(float));
                cudaMalloc(&d_best_idx,   (size_t)B * sizeof(int));

                cublasHandle_t handle;
                cublasCreate(&handle);
                cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
                const float alpha = 1.f, beta = 0.f;

                for (int base = v_start; base < v_end; base += B) {
                    int curB = std::min(B, v_end - base);
                    cudaMemcpy(d_data, h_base + (size_t)base * dim,
                               (size_t)curB * dim * sizeof(float), cudaMemcpyHostToDevice);
                    compute_l2_squared_gpu(d_data, d_xnorm2, curB, dim, L2NORM_AUTO, 0);
                    {
                        int thr = 256, blk = (curB + thr - 1) / thr;
                        kernel_init_best<<<blk, thr>>>(d_best_dist2, d_best_idx, curB);
                    }
                    for (int cbase = 0; cbase < nlist; cbase += Ktile) {
                        int curK = std::min(Ktile, nlist - cbase);
                        cublasSetStream(handle, 0);
                        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    curK, curB, dim, &alpha,
                                    d_centroids + (size_t)cbase * dim, dim,
                                    d_data, dim, &beta, d_dot, curK);
                        {
                            int thr = 256, blk = (curB + thr - 1) / thr;
                            kernel_update_best_from_dotT<<<blk, thr>>>(
                                d_dot, d_xnorm2, d_cnorm2, curB, curK, cbase,
                                d_best_idx, d_best_dist2);
                        }
                    }
                    cudaMemcpy(assign.data() + base, d_best_idx, curB * sizeof(int),
                               cudaMemcpyDeviceToHost);
                    if (gpu == 0 && (base - v_start) % (B * 100) == 0)
                        std::printf("  assign GPU0: %d / %d (%.1f%%)\n",
                                    base - v_start, v_end - v_start,
                                    100.0 * (base - v_start) / std::max(1, v_end - v_start));
                }
                cublasDestroy(handle);
                cudaFree(d_centroids); cudaFree(d_cnorm2);
                cudaFree(d_data); cudaFree(d_xnorm2); cudaFree(d_dot);
                cudaFree(d_best_dist2); cudaFree(d_best_idx);
            };

            std::vector<std::thread> threads;
            for (int g = 0; g < n_gpus; ++g) {
                int v_start = (int)((long long)n * g / n_gpus);
                int v_end   = (int)((long long)n * (g + 1) / n_gpus);
                threads.emplace_back(gpu_worker, g, v_start, v_end);
            }
            for (auto& t : threads) t.join();
        }
        std::printf("[SUBSET-KMEANS] Phase 2 done: all vectors assigned\n");

        /* --- Phase 3: count, offset, reorder (same as init_from_dump) --- */
        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c - 1] + cluster_info.counts[c - 1];
        reordered_data    = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            int pos = write_pos[c]++;
            std::memcpy(reordered_data + (size_t)pos * dim,
                        h_base + (size_t)i * dim, dim * sizeof(float));
            reordered_indices[pos] = i;
        }
        std::printf("[SUBSET-KMEANS] Phase 3 done: reordered %d vectors\n", n);
    }

    /**
     * Centroids-only mode: load centroids, multi-GPU assign, reorder.
     * Reads base data from u8bin file on-the-fly to avoid 2 float32 memory.
     * Peak memory: reordered_data (ndim4) + assign (n4) + centroids.
     */
    void init_from_centroids_only(
        const char* centroid_path, const char* u8bin_path,
        int n, int dim, int nlist
    ) {
        this->n_total_vectors = n;
        this->vector_dim = dim;

        FILE* fc = std::fopen(centroid_path, "rb");
        if (!fc) throw std::runtime_error(std::string("cannot open ") + centroid_path);
        int32_t file_nlist, file_dim;
        std::fread(&file_nlist, 4, 1, fc);
        std::fread(&file_dim, 4, 1, fc);
        if (file_nlist != nlist || file_dim != dim) {
            std::fclose(fc);
            throw std::runtime_error("centroid file header mismatch");
        }
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        std::fread(centroids, sizeof(float), (size_t)nlist * dim, fc);
        std::fclose(fc);
        std::printf("[CENT-ONLY] loaded centroids: %s (nlist=%d dim=%d)\n",
                    centroid_path, nlist, dim);

        /* Multi-GPU assignment: read u8bin in batches, convert to fp32 on-the-fly */
        int n_gpus = 0;
        cudaGetDeviceCount(&n_gpus);
        if (n_gpus > 8) n_gpus = 8;
        if (n_gpus < 1) n_gpus = 1;
        std::printf("[CENT-ONLY] GPU assignment of %d vectors on %d GPUs from %s\n",
                    n, n_gpus, u8bin_path);
        std::vector<int32_t> assign(n);
        {
            const int B = 1 << 15;
            const int Ktile = 4096;

            auto gpu_worker = [&](int gpu, int v_start, int v_end) {
                cudaSetDevice(gpu);
                float* d_cent = nullptr;
                cudaMalloc(&d_cent, (size_t)nlist * dim * sizeof(float));
                cudaMemcpy(d_cent, centroids, (size_t)nlist * dim * sizeof(float),
                           cudaMemcpyHostToDevice);
                float* d_cnorm2 = nullptr;
                cudaMalloc(&d_cnorm2, (size_t)nlist * sizeof(float));
                compute_l2_squared_gpu(d_cent, d_cnorm2, nlist, dim, L2NORM_AUTO, 0);
                cudaDeviceSynchronize();

                float* d_data = nullptr; float* d_xnorm2 = nullptr;
                float* d_dot = nullptr; float* d_best_dist2 = nullptr;
                int* d_best_idx = nullptr;
                cudaMalloc(&d_data,       (size_t)B * dim * sizeof(float));
                cudaMalloc(&d_xnorm2,     (size_t)B * sizeof(float));
                cudaMalloc(&d_dot,        (size_t)B * Ktile * sizeof(float));
                cudaMalloc(&d_best_dist2, (size_t)B * sizeof(float));
                cudaMalloc(&d_best_idx,   (size_t)B * sizeof(int));

                std::vector<uint8_t> u8buf((size_t)B * dim);
                std::vector<float>   f32buf((size_t)B * dim);

                FILE* fp = std::fopen(u8bin_path, "rb");
                if (!fp) { std::fprintf(stderr, "GPU%d: cannot open %s\n", gpu, u8bin_path); return; }
                std::fseek(fp, 8 + (size_t)v_start * dim, SEEK_SET);

                cublasHandle_t handle;
                cublasCreate(&handle);
                cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
                const float alpha = 1.f, beta = 0.f;

                for (int base = v_start; base < v_end; base += B) {
                    int curB = std::min(B, v_end - base);
                    size_t cnt = (size_t)curB * dim;
                    std::fread(u8buf.data(), 1, cnt, fp);
                    for (size_t j = 0; j < cnt; ++j) f32buf[j] = (float)u8buf[j];
                    cudaMemcpy(d_data, f32buf.data(), cnt * sizeof(float),
                               cudaMemcpyHostToDevice);
                    compute_l2_squared_gpu(d_data, d_xnorm2, curB, dim, L2NORM_AUTO, 0);
                    { int thr=256, blk=(curB+thr-1)/thr;
                      kernel_init_best<<<blk,thr>>>(d_best_dist2, d_best_idx, curB); }
                    for (int cbase = 0; cbase < nlist; cbase += Ktile) {
                        int curK = std::min(Ktile, nlist - cbase);
                        cublasSetStream(handle, 0);
                        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    curK, curB, dim, &alpha,
                                    d_cent + (size_t)cbase * dim, dim,
                                    d_data, dim, &beta, d_dot, curK);
                        { int thr=256, blk=(curB+thr-1)/thr;
                          kernel_update_best_from_dotT<<<blk,thr>>>(
                              d_dot, d_xnorm2, d_cnorm2, curB, curK, cbase,
                              d_best_idx, d_best_dist2); }
                    }
                    cudaMemcpy(assign.data() + base, d_best_idx, curB * sizeof(int),
                               cudaMemcpyDeviceToHost);
                    if (gpu == 0 && (base - v_start) % (B * 100) == 0)
                        std::printf("  assign GPU0: %d / %d (%.1f%%)\n",
                                    base - v_start, v_end - v_start,
                                    100.0*(base - v_start)/(v_end - v_start));
                }
                std::fclose(fp);
                cublasDestroy(handle);
                cudaFree(d_cent); cudaFree(d_cnorm2);
                cudaFree(d_data); cudaFree(d_xnorm2); cudaFree(d_dot);
                cudaFree(d_best_dist2); cudaFree(d_best_idx);
            };

            std::vector<std::thread> threads;
            for (int g = 0; g < n_gpus; ++g) {
                int v_start = (int)((long long)n * g / n_gpus);
                int v_end   = (int)((long long)n * (g + 1) / n_gpus);
                threads.emplace_back(gpu_worker, g, v_start, v_end);
            }
            for (auto& t : threads) t.join();
        }
        std::printf("[CENT-ONLY] assignment done\n");

        /* Reorder: read u8bin again, convert to fp32, scatter to reordered_data */
        std::printf("[CENT-ONLY] reordering %d vectors (streaming from %s)...\n", n, u8bin_path);
        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c-1] + cluster_info.counts[c-1];

        reordered_data    = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];

        FILE* fp = std::fopen(u8bin_path, "rb");
        std::fseek(fp, 8, SEEK_SET);
        std::vector<uint8_t> row_u8(dim);
        for (int i = 0; i < n; ++i) {
            std::fread(row_u8.data(), 1, dim, fp);
            int c = assign[i];
            int pos = write_pos[c]++;
            float* dst = reordered_data + (size_t)pos * dim;
            for (int d = 0; d < dim; ++d) dst[d] = (float)row_u8[d];
            reordered_indices[pos] = i;
            if (i % 100000000 == 0)
                std::printf("  reorder: %d / %d (%.0f%%)\n", i, n, 100.0*i/n);
        }
        std::fclose(fp);
        std::printf("[CENT-ONLY] reorder done: %d vectors\n", n);
    }

    /**
     * Load pre-computed centroids + assignments from dump files and reorder
     * base vectors accordingly, skipping KMeans entirely.
     *
     * Centroids file format: [nlist:int32][dim:int32][float32 nlist*dim]
     * Assign file format:    [n:int64][int32 n]   (cluster id per base vector)
     */
    /**
     *  u8bin  base assign  reordered_data
     *  fp32  1B scale  h_base
     *  h_base + reordered_data  ~1 TB
     */
    void init_from_dump_stream_u8(
        const char* centroid_path,
        const char* assign_path,
        const char* u8bin_path,
        int n, int dim, int nlist
    ) {
        this->n_total_vectors = n;
        this->vector_dim = dim;

        FILE* fc = std::fopen(centroid_path, "rb");
        if (!fc) throw std::runtime_error(std::string("cannot open ") + centroid_path);
        int32_t file_nlist, file_dim;
        std::fread(&file_nlist, 4, 1, fc);
        std::fread(&file_dim, 4, 1, fc);
        if (file_nlist != nlist || file_dim != dim) {
            std::fclose(fc);
            throw std::runtime_error("centroid file header mismatch");
        }
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        std::fread(centroids, sizeof(float), (size_t)nlist * dim, fc);
        std::fclose(fc);

        FILE* fa = std::fopen(assign_path, "rb");
        if (!fa) throw std::runtime_error(std::string("cannot open ") + assign_path);
        int64_t file_n;
        std::fread(&file_n, 8, 1, fa);
        if ((int64_t)n != file_n) {
            std::fclose(fa);
            throw std::runtime_error("assign file n mismatch");
        }
        std::vector<int32_t> assign(n);
        std::fread(assign.data(), sizeof(int32_t), n, fa);
        std::fclose(fa);

        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c - 1] + cluster_info.counts[c - 1];

        reordered_data    = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        if (!reordered_data || !reordered_indices) {
            throw std::runtime_error("[init_from_dump_stream_u8] OOM allocating reordered_data");
        }
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];

        /*  u8bin CHUNK u8fp32  scatter  reordered_data */
        FILE* fb = std::fopen(u8bin_path, "rb");
        if (!fb) throw std::runtime_error(std::string("cannot open ") + u8bin_path);
        uint32_t hdr[2];
        std::fread(hdr, 4, 2, fb);
        if ((int)hdr[1] != dim) {
            std::fclose(fb);
            throw std::runtime_error("u8bin dim header mismatch");
        }
        const size_t CHUNK = 1 << 16;  /* 64K vectors  8 MB for dim=128 */
        std::vector<uint8_t> buf((size_t)CHUNK * (size_t)dim);
        int64_t idx = 0;
        int64_t pct_step = std::max<int64_t>(1, (int64_t)n / 20);
        int64_t next_pct = pct_step;
        std::printf("[STREAM-DUMP] reordering %d vectors from %s (chunk=%zu)...\n",
                    n, u8bin_path, CHUNK);
        while (idx < n) {
            int64_t nread = std::min((int64_t)CHUNK, (int64_t)n - idx);
            size_t bytes = (size_t)nread * (size_t)dim;
            size_t got = std::fread(buf.data(), 1, bytes, fb);
            if (got != bytes) {
                std::fclose(fb);
                throw std::runtime_error("u8bin short read");
            }
            for (int64_t k = 0; k < nread; ++k) {
                int i = (int)(idx + k);
                int c = assign[i];
                int pos = write_pos[c]++;
                float* dst = reordered_data + (size_t)pos * (size_t)dim;
                const uint8_t* src = buf.data() + (size_t)k * (size_t)dim;
                for (int d = 0; d < dim; ++d) dst[d] = (float)src[d];
                reordered_indices[pos] = i;
            }
            idx += nread;
            if (idx >= next_pct) {
                std::printf("  [STREAM-DUMP] %lld / %d (%.0f%%)\n",
                            (long long)idx, n, 100.0 * idx / n);
                next_pct += pct_step;
            }
        }
        std::fclose(fb);

        std::printf("[STREAM-DUMP] done: centroids=%s assigns=%s nlist=%d n=%d dim=%d\n",
                    centroid_path, assign_path, nlist, n, dim);
    }

    void init_from_dump_stream_u8_direct(
        const char* centroid_path,
        const char* assign_path,
        const char* u8bin_path,
        int n, int dim, int nlist,
        std::vector<uint8_t>& reordered_u8
    ) {
        this->reordered_data = nullptr;
        this->reordered_indices = nullptr;
        this->centroids = nullptr;
        this->cluster_info.k = 0;
        this->cluster_info.counts = nullptr;
        this->cluster_info.offsets = nullptr;
        this->n_total_vectors = n;
        this->vector_dim = dim;

        FILE* fc = std::fopen(centroid_path, "rb");
        if (!fc) throw std::runtime_error(std::string("cannot open ") + centroid_path);
        int32_t file_nlist, file_dim;
        std::fread(&file_nlist, 4, 1, fc);
        std::fread(&file_dim, 4, 1, fc);
        if (file_nlist != nlist || file_dim != dim) {
            std::fclose(fc);
            throw std::runtime_error("centroid file header mismatch");
        }
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        if (!centroids) {
            std::fclose(fc);
            throw std::runtime_error("[init_from_dump_stream_u8_direct] OOM allocating centroids");
        }
        std::fread(centroids, sizeof(float), (size_t)nlist * dim, fc);
        std::fclose(fc);

        FILE* fa = std::fopen(assign_path, "rb");
        if (!fa) throw std::runtime_error(std::string("cannot open ") + assign_path);
        int64_t file_n;
        std::fread(&file_n, 8, 1, fa);
        if ((int64_t)n != file_n) {
            std::fclose(fa);
            throw std::runtime_error("assign file n mismatch");
        }
        std::vector<int32_t> assign(n);
        std::fread(assign.data(), sizeof(int32_t), n, fa);
        std::fclose(fa);

        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        if (!cluster_info.counts || !cluster_info.offsets) {
            throw std::runtime_error("[init_from_dump_stream_u8_direct] OOM allocating cluster metadata");
        }
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c - 1] + cluster_info.counts[c - 1];

        reordered_u8.resize((size_t)n * (size_t)dim);
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        if (reordered_u8.empty() || !reordered_indices) {
            throw std::runtime_error("[init_from_dump_stream_u8_direct] OOM allocating u8 reordered data");
        }
        ivftensor::numa::place_cluster_major_memory(
            reordered_u8.data(),
            (size_t)dim,
            cluster_info.counts,
            nlist,
            "SIFT reordered_u8");
        ivftensor::numa::place_cluster_major_memory(
            reordered_indices,
            sizeof(int),
            cluster_info.counts,
            nlist,
            "SIFT reordered_indices");
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];

        FILE* fb = std::fopen(u8bin_path, "rb");
        if (!fb) throw std::runtime_error(std::string("cannot open ") + u8bin_path);
        uint32_t hdr[2];
        std::fread(hdr, 4, 2, fb);
        if ((int)hdr[1] != dim) {
            std::fclose(fb);
            throw std::runtime_error("u8bin dim header mismatch");
        }
        const size_t CHUNK = 1 << 16;
        std::vector<uint8_t> buf((size_t)CHUNK * (size_t)dim);
        int64_t idx = 0;
        int64_t pct_step = std::max<int64_t>(1, (int64_t)n / 20);
        int64_t next_pct = pct_step;
        std::printf("[STREAM-U8-DIRECT] reordering %d vectors from %s (chunk=%zu)...\n",
                    n, u8bin_path, CHUNK);
        while (idx < n) {
            int64_t nread = std::min((int64_t)CHUNK, (int64_t)n - idx);
            size_t bytes = (size_t)nread * (size_t)dim;
            size_t got = std::fread(buf.data(), 1, bytes, fb);
            if (got != bytes) {
                std::fclose(fb);
                throw std::runtime_error("u8bin short read");
            }
            for (int64_t k = 0; k < nread; ++k) {
                int i = (int)(idx + k);
                int c = assign[i];
                int pos = write_pos[c]++;
                std::memcpy(reordered_u8.data() + (size_t)pos * (size_t)dim,
                            buf.data() + (size_t)k * (size_t)dim,
                            (size_t)dim);
                reordered_indices[pos] = i;
            }
            idx += nread;
            if (idx >= next_pct) {
                std::printf("  [STREAM-U8-DIRECT] %lld / %d (%.0f%%)\n",
                            (long long)idx, n, 100.0 * idx / n);
                next_pct += pct_step;
            }
        }
        std::fclose(fb);

        std::printf("[STREAM-U8-DIRECT] done: centroids=%s assigns=%s nlist=%d n=%d dim=%d\n",
                    centroid_path, assign_path, nlist, n, dim);
    }

    void init_from_dump_stream_fbin(
        const char* centroid_path,
        const char* assign_path,
        const char* fbin_path,
        int n, int dim, int nlist
    ) {
        this->n_total_vectors = n;
        this->vector_dim = dim;

        FILE* fc = std::fopen(centroid_path, "rb");
        if (!fc) throw std::runtime_error(std::string("cannot open ") + centroid_path);
        int32_t file_nlist, file_dim;
        std::fread(&file_nlist, 4, 1, fc);
        std::fread(&file_dim, 4, 1, fc);
        if (file_nlist != nlist || file_dim != dim) {
            std::fclose(fc);
            throw std::runtime_error("centroid file header mismatch");
        }
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        std::fread(centroids, sizeof(float), (size_t)nlist * dim, fc);
        std::fclose(fc);

        FILE* fa = std::fopen(assign_path, "rb");
        if (!fa) throw std::runtime_error(std::string("cannot open ") + assign_path);
        int64_t file_n;
        std::fread(&file_n, 8, 1, fa);
        if ((int64_t)n != file_n) {
            std::fclose(fa);
            throw std::runtime_error("assign file n mismatch");
        }
        std::vector<int32_t> assign(n);
        std::fread(assign.data(), sizeof(int32_t), n, fa);
        std::fclose(fa);

        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c - 1] + cluster_info.counts[c - 1];

        reordered_data    = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        if (!reordered_data || !reordered_indices) {
            throw std::runtime_error("[init_from_dump_stream_fbin] OOM allocating reordered_data");
        }
        ivftensor::numa::place_cluster_major_memory(
            reordered_data,
            (size_t)dim * sizeof(float),
            cluster_info.counts,
            nlist,
            "fp32 reordered_data");
        ivftensor::numa::place_cluster_major_memory(
            reordered_indices,
            sizeof(int),
            cluster_info.counts,
            nlist,
            "fp32 reordered_indices");
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];

        FILE* fb = std::fopen(fbin_path, "rb");
        if (!fb) throw std::runtime_error(std::string("cannot open ") + fbin_path);
        uint32_t hdr[2];
        std::fread(hdr, 4, 2, fb);
        if ((int)hdr[1] != dim) {
            std::fclose(fb);
            throw std::runtime_error("fbin dim header mismatch");
        }
        const size_t CHUNK = 1 << 16;
        std::vector<float> buf((size_t)CHUNK * (size_t)dim);
        int64_t idx = 0;
        int64_t pct_step = std::max<int64_t>(1, (int64_t)n / 20);
        int64_t next_pct = pct_step;
        std::printf("[STREAM-FBIN] reordering %d vectors from %s (chunk=%zu)...\n",
                    n, fbin_path, CHUNK);
        while (idx < n) {
            int64_t nread = std::min((int64_t)CHUNK, (int64_t)n - idx);
            size_t elems = (size_t)nread * (size_t)dim;
            if (std::fread(buf.data(), sizeof(float), elems, fb) != elems) {
                std::fclose(fb);
                throw std::runtime_error("fbin short read");
            }
            for (int64_t k = 0; k < nread; ++k) {
                int i = (int)(idx + k);
                int c = assign[i];
                int pos = write_pos[c]++;
                std::memcpy(reordered_data + (size_t)pos * (size_t)dim,
                            buf.data() + (size_t)k * (size_t)dim,
                            (size_t)dim * sizeof(float));
                reordered_indices[pos] = i;
            }
            idx += nread;
            if (idx >= next_pct) {
                std::printf("  [STREAM-FBIN] %lld / %d (%.0f%%)\n",
                            (long long)idx, n, 100.0 * idx / n);
                next_pct += pct_step;
            }
        }
        std::fclose(fb);
        std::printf("[STREAM-FBIN] done: centroids=%s assigns=%s nlist=%d n=%d dim=%d\n",
                    centroid_path, assign_path, nlist, n, dim);
    }

    void init_from_dump(
        const char* centroid_path,
        const char* assign_path,
        const float* h_base,
        int n, int dim, int nlist
    ) {
        this->n_total_vectors = n;
        this->vector_dim = dim;

        // --- load centroids ---
        FILE* fc = std::fopen(centroid_path, "rb");
        if (!fc) throw std::runtime_error(std::string("cannot open ") + centroid_path);
        int32_t file_nlist, file_dim;
        std::fread(&file_nlist, 4, 1, fc);
        std::fread(&file_dim, 4, 1, fc);
        if (file_nlist != nlist || file_dim != dim) {
            std::fclose(fc);
            throw std::runtime_error("centroid file header mismatch");
        }
        centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
        std::fread(centroids, sizeof(float), (size_t)nlist * dim, fc);
        std::fclose(fc);

        // --- load assignments ---
        FILE* fa = std::fopen(assign_path, "rb");
        if (!fa) throw std::runtime_error(std::string("cannot open ") + assign_path);
        int64_t file_n;
        std::fread(&file_n, 8, 1, fa);
        if ((int64_t)n != file_n) {
            std::fclose(fa);
            throw std::runtime_error("assign file n mismatch");
        }
        std::vector<int32_t> assign(n);
        std::fread(assign.data(), sizeof(int32_t), n, fa);
        std::fclose(fa);

        // --- count per cluster ---
        cluster_info.k = nlist;
        cluster_info.counts  = (int*)memalign(64, (size_t)nlist * sizeof(int));
        cluster_info.offsets = (long long*)memalign(64, (size_t)nlist * sizeof(long long));
        std::memset(cluster_info.counts, 0, (size_t)nlist * sizeof(int));
        for (int i = 0; i < n; ++i) cluster_info.counts[assign[i]]++;

        // --- compute offsets (prefix sum) ---
        cluster_info.offsets[0] = 0;
        for (int c = 1; c < nlist; ++c)
            cluster_info.offsets[c] = cluster_info.offsets[c - 1] + cluster_info.counts[c - 1];

        // --- reorder base vectors by cluster ---
        reordered_data    = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
        std::vector<int> write_pos(nlist);
        for (int c = 0; c < nlist; ++c) write_pos[c] = (int)cluster_info.offsets[c];
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            int pos = write_pos[c]++;
            std::memcpy(reordered_data + (size_t)pos * dim,
                        h_base + (size_t)i * dim,
                        dim * sizeof(float));
            reordered_indices[pos] = i;
        }

        std::printf("[LOAD-DUMP] centroids=%s assigns=%s  nlist=%d n=%d dim=%d\n",
                    centroid_path, assign_path, nlist, n, dim);
    }

private:
    void _copy_from_clustering_result(const ClusteringResult& res) {
        n_total_vectors = res.n_vectors;
        vector_dim = res.dim;
        const int k = res.cluster_info ? res.cluster_info->k : 0;
        const size_t data_size = (size_t)n_total_vectors * vector_dim;
        const size_t centroid_size = (size_t)k * vector_dim;
        reordered_data = (float*)memalign(64, data_size * sizeof(float));
        reordered_indices = (int*)memalign(64, (size_t)n_total_vectors * sizeof(int));
        centroids = (float*)memalign(64, centroid_size * sizeof(float));
        if (!reordered_data || !reordered_indices || !centroids) {
            if (reordered_data) std::free(reordered_data);
            if (reordered_indices) std::free(reordered_indices);
            if (centroids) std::free(centroids);
            throw std::bad_alloc();
        }
        std::memcpy(reordered_data, res.reordered_data, data_size * sizeof(float));
        std::memcpy(reordered_indices, res.reordered_indices, (size_t)n_total_vectors * sizeof(int));
        std::memcpy(centroids, res.centroids, centroid_size * sizeof(float));
        cluster_info.k = k;
        cluster_info.offsets = (long long*)memalign(64, (size_t)k * sizeof(long long));
        cluster_info.counts = (int*)memalign(64, (size_t)k * sizeof(int));
        if (!cluster_info.offsets || !cluster_info.counts) {
            release();
            throw std::bad_alloc();
        }
        std::memcpy(cluster_info.offsets, res.cluster_info->offsets, (size_t)k * sizeof(long long));
        std::memcpy(cluster_info.counts, res.cluster_info->counts, (size_t)k * sizeof(int));
    }

public:
    /**
     * ClusterDataset
     */
    void release() {
        //
        if (reordered_data) {
            std::free(reordered_data);
            reordered_data = nullptr;
        }

        //
        if (reordered_indices) {
            std::free(reordered_indices);
            reordered_indices = nullptr;
        }

        // centroids
        if (centroids) {
            std::free(centroids);
            centroids = nullptr;
        }

        //  ClusterInfo
        if (cluster_info.offsets) {
            std::free(cluster_info.offsets);
            cluster_info.offsets = nullptr;
        }
        if (cluster_info.counts) {
            std::free(cluster_info.counts);
            cluster_info.counts = nullptr;
        }

        cluster_info.k = 0;
        n_total_vectors = 0;
        vector_dim = 0;
    }

    /**
     * ClusterDataset
     */
    bool is_valid() const {
        return reordered_data != nullptr &&
               reordered_indices != nullptr &&
               centroids != nullptr &&
               cluster_info.offsets != nullptr &&
               cluster_info.counts != nullptr &&
               cluster_info.k > 0 &&
               n_total_vectors > 0 &&
               vector_dim > 0;
    }

    /**
     * cluster
     */
    int get_n_clusters() const {
        return cluster_info.k;
    }
};

#endif // CLUSTER_DATASET_CUH
