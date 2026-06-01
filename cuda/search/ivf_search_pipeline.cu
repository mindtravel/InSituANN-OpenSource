
#include "pch.h"
#include "search/ivf_search.cuh"
#include "search/coarse/fusion_dist_topk.cuh"
#include "search/fine/indexed_gemm.cuh"
#include "search/topk/warpsort_utils.cuh"
#include "search/topk/warpsort_topk.cu"
#include "cudatimer.h"
#include "l2norm/l2norm.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define ENABLE_CUDA_TIMING 0

using namespace INSITUANN::warpsort_utils;
using namespace INSITUANN::warpsort_topk;

/**
 * GPU
 * 6G
 */
struct PipelinePersistentData {
    //
    float* d_cluster_vectors = nullptr;  // cluster
    float** d_cluster_vector_ptr = nullptr;  // cluster
    float* d_cluster_vector_norm = nullptr;  // L2

    // Cluster
    int* d_probe_vector_offset = nullptr;  // [n_total_clusters + 1]
    int* d_probe_vector_count = nullptr;  // [n_total_clusters]

    //
    float* d_cluster_centers = nullptr;
    float* d_cluster_centers_norm = nullptr;

    //
    int n_total_clusters = 0;
    int n_total_vectors = 0;
    int n_dim = 0;

    //
    size_t total_data_size_bytes = 0;

    //
    bool initialized = false;

    /**
     * 6G
     *
     * @param n_total_vectors
     * @param n_dim
     * @param n_total_clusters 0
     */
    bool can_persist(size_t n_total_vectors, int n_dim, size_t n_total_clusters = 0) {
        //  n_total_clusters cluster  sqrt(n_total_vectors)
        if (n_total_clusters == 0) {
            n_total_clusters = static_cast<size_t>(std::sqrt(static_cast<double>(n_total_vectors)));
            if (n_total_clusters == 0) n_total_clusters = 1;
        }

        size_t vector_data_size = n_total_vectors * n_dim * sizeof(float);
        size_t norm_data_size = n_total_vectors * sizeof(float);
        size_t cluster_center_size = n_total_clusters * n_dim * sizeof(float);
        size_t cluster_center_norm_size = n_total_clusters * sizeof(float);
        size_t metadata_size = (n_total_clusters + 1) * sizeof(int) + n_total_clusters * sizeof(int) + n_total_clusters * sizeof(float*);
        total_data_size_bytes = vector_data_size + norm_data_size + cluster_center_size + cluster_center_norm_size + metadata_size;
        return total_data_size_bytes < (6ULL * 1024 * 1024 * 1024);  // 6GB
    }

    /**
     *
     */
    void initialize(int* cluster_size,
                    float*** cluster_vectors,
                    float** cluster_center_data,
                    int n_total_clusters,
                    int n_total_vectors,
                    int n_dim,
                    cudaStream_t stream = 0) {
        if (initialized) {
            //
            if (this->n_total_clusters != n_total_clusters ||
                this->n_total_vectors != n_total_vectors ||
                this->n_dim != n_dim) {
                //
                cleanup();
            } else {
                //
                return;
            }
        }

        this->n_total_clusters = n_total_clusters;
        this->n_total_vectors = n_total_vectors;
        this->n_dim = n_dim;

        // clusterCPU
        d_cluster_vector_ptr = (float**)malloc(n_total_clusters * sizeof(float*));

        // GPU
        cudaMalloc(&d_cluster_vectors, n_total_vectors * n_dim * sizeof(float));
        cudaMalloc(&d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMalloc(&d_probe_vector_count, n_total_clusters * sizeof(int));
        cudaMalloc(&d_cluster_centers, n_total_clusters * n_dim * sizeof(float));
        cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
        cudaMalloc(&d_cluster_centers_norm, n_total_clusters * sizeof(float));
        CHECK_CUDA_ERRORS;

        // cluster_sizeGPUoffset
        cudaMemcpyAsync(d_probe_vector_count, cluster_size,
                       n_total_clusters * sizeof(int), cudaMemcpyHostToDevice, stream);
        compute_prefix_sum(d_probe_vector_count, d_probe_vector_offset, n_total_clusters, stream);

        // clusterCPU
        // cluster_sizeCPUoffsetGPU
        int* probe_vector_offset_host = (int*)malloc((n_total_clusters + 1) * sizeof(int));
        probe_vector_offset_host[0] = 0;
        for (int i = 0; i < n_total_clusters; ++i) {
            probe_vector_offset_host[i + 1] = probe_vector_offset_host[i] + cluster_size[i];
        }

        // clusterGPU
        // streamCUDA
        for (int i = 0; i < n_total_clusters; ++i) {
            float* cluster_start = d_cluster_vectors + probe_vector_offset_host[i] * n_dim;
            cudaMemcpyAsync(cluster_start, cluster_vectors[i][0],
                            cluster_size[i] * n_dim * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            d_cluster_vector_ptr[i] = cluster_start;  // GPU
        }

        // cluster
        cudaMemcpyAsync(d_cluster_centers, cluster_center_data[0],
                       n_total_clusters * n_dim * sizeof(float),
                       cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        // L2stream
        // streamCUDA
        //  L2
        compute_l2_norm_gpu(d_cluster_vectors, d_cluster_vector_norm, n_total_vectors, n_dim, L2NORM_AUTO, stream);
        compute_l2_norm_gpu(d_cluster_centers, d_cluster_centers_norm, n_total_clusters, n_dim, L2NORM_AUTO, stream);

        free(probe_vector_offset_host);

        // initialized
        //  stream  stream (0)
        //  stream  stream
        if (stream == 0 || stream == nullptr) {
            //  stream
            initialized = true;
        }
        // initialized  stream
    }

    /**
     *  stream
     */
    void mark_initialized() {
        initialized = true;
    }

    /**
     *
     */
    void cleanup() {
        if (d_cluster_vectors != nullptr) {
            cudaFree(d_cluster_vectors);
            d_cluster_vectors = nullptr;
        }
        if (d_cluster_vector_ptr != nullptr) {
            free(d_cluster_vector_ptr);
            d_cluster_vector_ptr = nullptr;
        }
        if (d_cluster_vector_norm != nullptr) {
            cudaFree(d_cluster_vector_norm);
            d_cluster_vector_norm = nullptr;
        }
        if (d_probe_vector_offset != nullptr) {
            cudaFree(d_probe_vector_offset);
            d_probe_vector_offset = nullptr;
        }
        if (d_probe_vector_count != nullptr) {
            cudaFree(d_probe_vector_count);
            d_probe_vector_count = nullptr;
        }
        if (d_cluster_centers != nullptr) {
            cudaFree(d_cluster_centers);
            d_cluster_centers = nullptr;
        }
        if (d_cluster_centers_norm != nullptr) {
            cudaFree(d_cluster_centers_norm);
            d_cluster_centers_norm = nullptr;
        }
        //
        initialized = false;
        n_total_clusters = 0;
        n_total_vectors = 0;
        n_dim = 0;
        total_data_size_bytes = 0;
    }
};

//
static PipelinePersistentData g_persistent_data;

/**
 *
 */
bool initialize_persistent_data(
    int* cluster_size,
    float*** cluster_vectors,
    float** cluster_center_data,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim)
{
    //
    if (!g_persistent_data.can_persist(n_total_vectors, n_dim, n_total_clusters)) {
        return false;
    }

    // stream
    cudaStream_t init_stream;
    cudaStreamCreate(&init_stream);

    //
    g_persistent_data.initialize(cluster_size, cluster_vectors, cluster_center_data,
                                 n_total_clusters, n_total_vectors, n_dim, init_stream);

    //
    g_persistent_data.mark_initialized();

    cudaStreamDestroy(init_stream);
    CHECK_CUDA_ERRORS;

    return true;
}

/**
 *
 */
void cleanup_persistent_data()
{
    g_persistent_data.cleanup();
}

/**
 *
 *
 *
 * 1. queryquery norm
 * 2.
 *
 * 6GGPU
 *
 * @param coarse_index  [n_query, n_probes]nullptr
 * @param coarse_dist  [n_query, n_probes]nullptr
 */
void ivf_search_pipeline(float** query_batch,
                           int* cluster_size,
                           float*** cluster_vectors,
                           float** cluster_center_data,
                           int* initial_indices,
                           float** topk_dist,
                           int** topk_index,
                           int* n_isnull,

                           int n_query,
                           int n_dim,
                           int n_total_clusters,
                           int n_total_vectors,
                           int n_probes,
                           int k,
                           int distance_mode,
                           int** coarse_index = nullptr,  // [n_query, n_probes]
                           float** coarse_dist = nullptr  // [n_query, n_probes]
                        )
{

    if (n_query <= 0 || n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        printf("[ERROR] Invalid parameters: n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
               n_query, n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid ivf_search_pipeline configuration");
    }
    if (!cluster_size || !cluster_vectors) {
        throw std::invalid_argument("cluster metadata is null");
    }

    if (!cluster_center_data) {
        throw std::invalid_argument("cluster_center_data must not be null for coarse search");
    }
    if (n_probes <= 0 || n_probes > n_total_clusters) {
        throw std::invalid_argument("invalid n_probes");
    }

    // CUDA streams
    cudaStream_t data_stream, compute_stream;
    cudaStreamCreate(&data_stream);
    cudaStreamCreate(&compute_stream);

    //
    cudaEvent_t data_ready_event;
    cudaEventCreate(&data_ready_event);

    //
    bool use_persistent = g_persistent_data.can_persist(n_total_vectors, n_dim, n_total_clusters);

    //
    float* d_queries = nullptr;
    float* d_query_norm = nullptr;
    int* d_topk_index = nullptr;
    float* d_topk_dist = nullptr;
    int* d_top_nprobe_index = nullptr;
    float* d_top_nprobe_dist = nullptr;
    float* d_inner_product = nullptr;

    dim3 queryDim(n_query);
    dim3 dataDim(n_total_clusters);
    dim3 vectorDim(n_dim);
    dim3 probeDim(n_probes);

    // ==================================================================
    // 1data_stream
    // ==================================================================
    {
        CUDATimer timer("Pipeline Stage 1: Data Preparation");

        // query
        cudaMalloc(&d_queries, n_query * n_dim * sizeof(float));
        cudaMalloc(&d_query_norm, n_query * sizeof(float));
        cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
        cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
        cudaMalloc(&d_top_nprobe_index, n_query * n_probes * sizeof(int));
        CHECK_CUDA_ERRORS;

        // query norm0compute_l2_norm_gpu
        cudaMemsetAsync(d_query_norm, 0, n_query * sizeof(float), data_stream);

        //
        //
        if (use_persistent) {
            //
            bool need_reinit = !g_persistent_data.initialized ||
                              g_persistent_data.n_total_clusters != n_total_clusters ||
                              g_persistent_data.n_total_vectors != n_total_vectors ||
                              g_persistent_data.n_dim != n_dim;

            if (need_reinit) {
                // data_stream
                // initialize()  cleanup()
                g_persistent_data.initialize(cluster_size, cluster_vectors, cluster_center_data,
                                            n_total_clusters, n_total_vectors, n_dim, data_stream);
                //
                //  query  norm  n_dim
            }
        } else if (!use_persistent) {
            // TODO:
            //
            throw std::runtime_error("Non-persistent mode not implemented yet");
        }

        // queryGPU
        // query_batch[0] query
        // query_batch  malloc_vector_list
        //  query_batch[0]  query
        //
        //  query_batch  n_dim
        //  n_dim  query_batch
        //
        // query_batch[0]  [query0, query1, ..., queryN]
        //  query  n_dim * sizeof(float)
        // query_batch[i] = query_batch[0] + i * n_dim * sizeof(float)
        //
        // d_queries
        // d_queries[0 ... n_dim-1] = query0
        // d_queries[n_dim ... 2*n_dim-1] = query1
        // ...
        // d_queries[i*n_dim ... (i+1)*n_dim-1] = query_i
        cudaMemcpyAsync(d_queries, query_batch[0],
                       n_query * n_dim * sizeof(float),
                       cudaMemcpyHostToDevice, data_stream);

        // query normdata_stream
        // compute_l2_norm_gpu  query  L2
        // d_query_norm[i] = ||d_queries[i * n_dim : (i+1) * n_dim]||_2
        //  n_dim norm
        //
        //  n_dim  query_batch  n_dim
        // -  n_dim >  n_dim query
        // -  n_dim <  n_dim query
        compute_l2_norm_gpu(d_queries, d_query_norm, n_query, n_dim, L2NORM_AUTO, data_stream);

        cudaStreamSynchronize(data_stream);
        CHECK_CUDA_ERRORS;

        //
        // data_streamqueryquery norm
        cudaEventRecord(data_ready_event, data_stream);
    }

    // ==================================================================
    // 2compute_stream
    // ==================================================================

    // stream
    cudaStreamWaitEvent(compute_stream, data_ready_event, 0);

    // Step 1:
    {
        CUDATimer timer("Pipeline Stage 2: Coarse Search");

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasHandle_t handle;
        int* d_index = nullptr;
        bool need_free_index = false;

        cudaMalloc(&d_inner_product, n_query * n_total_clusters * sizeof(float));
        cudaMalloc(&d_top_nprobe_dist, n_query * n_probes * sizeof(float));
            cublasCreate(&handle);
        CHECK_CUDA_ERRORS;

        //
        if (initial_indices != nullptr) {
            // GPU
            cudaMalloc(&d_index, n_query * n_total_clusters * sizeof(int));
            cudaMemcpyAsync(d_index, initial_indices,
                           n_query * n_total_clusters * sizeof(int),
                           cudaMemcpyHostToDevice, compute_stream);
            need_free_index = true;
        } else {
            //
            cudaMalloc(&d_index, n_query * n_total_clusters * sizeof(int));
            need_free_index = true;
            dim3 block_dim((n_total_clusters < 256) ? n_total_clusters : 256);
        generate_sequence_indices_kernel<<<queryDim, block_dim, 0, compute_stream>>>(
                d_index, n_query, n_total_clusters);
        }

        // fill kernelthrust::fill
        dim3 fill_block(256);
        int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
        dim3 fill_grid(fill_grid_size);
        fill_kernel<<<fill_grid, fill_block, 0, compute_stream>>>(
            d_top_nprobe_dist,
            FLT_MAX,
            n_query * n_probes
        );

        // compute_stream
        cublasSetStream(handle, compute_stream);
            cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n_total_clusters, n_query, n_dim,
                &alpha,
            g_persistent_data.d_cluster_centers, n_dim,
                d_queries, n_dim,
                &beta,
                d_inner_product, n_total_clusters
            );

        //  + topkcompute_stream
        if(distance_mode == COSINE_DISTANCE){
            INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                d_query_norm, g_persistent_data.d_cluster_centers_norm, d_inner_product, d_index,
                n_query, n_total_clusters, n_probes,
                d_top_nprobe_dist, d_top_nprobe_index,
                true /* select min */,
                compute_stream  // compute_stream
            );
        }
        else if(distance_mode == L2_DISTANCE){
            INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                d_query_norm, g_persistent_data.d_cluster_centers_norm, d_inner_product, d_index,
                n_query, n_total_clusters, n_probes,
                d_top_nprobe_dist, d_top_nprobe_index,
            true /* select min */,
            compute_stream  // compute_stream
            );
        }

        // CPU
        if (coarse_index != nullptr && coarse_dist != nullptr) {
            cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;
            cudaMemcpyAsync(coarse_index[0], d_top_nprobe_index,
                           n_query * n_probes * sizeof(int),
                           cudaMemcpyDeviceToHost, compute_stream);
            cudaMemcpyAsync(coarse_dist[0], d_top_nprobe_dist,
                           n_query * n_probes * sizeof(float),
                           cudaMemcpyDeviceToHost, compute_stream);
            cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;
        }

        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        //
        cublasDestroy(handle);
        // d_cluster_centers
        cudaFree(d_inner_product);
        if (need_free_index && d_index != nullptr) {
            cudaFree(d_index);
        }
        // d_top_nprobe_dist  d_top_nprobe_index
        //  Step 2  Step 2
        CHECK_CUDA_ERRORS;
    }

    // Step 2: entry
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;
    int n_entry = 0;
    constexpr int kQueriesPerBlock = 8;

    {
        CUDATimer timer("Pipeline Stage 2: Build Entry Data");

        // clusterquery
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemsetAsync(d_cluster_query_count, 0, n_total_clusters * sizeof(int), compute_stream);

        count_cluster_queries_kernel<<<queryDim, probeDim, 0, compute_stream>>>(
            d_top_nprobe_index,
            d_cluster_query_count,
            n_query,
            n_probes,
            n_total_clusters
        );

        // offset
        cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        compute_prefix_sum(d_cluster_query_count, d_cluster_query_offset, n_total_clusters, compute_stream);

        int total_entries = 0;
        cudaMemcpyAsync(&total_entries, d_cluster_query_offset + n_total_clusters,
                       sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        // cluster-query
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpyAsync(d_cluster_write_pos, d_cluster_query_offset,
                       n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream);

        cudaMalloc(&d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&d_cluster_query_probe_indices, total_entries * sizeof(int));

        build_cluster_query_mapping_kernel<<<queryDim, probeDim, 0, compute_stream>>>(
            d_top_nprobe_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            n_probes,
            n_total_clusters
        );

        // entry
        int* d_entry_count_per_cluster = nullptr;
        cudaMalloc(&d_entry_count_per_cluster, n_total_clusters * sizeof(int));

        dim3 clusterDim(n_total_clusters);
        dim3 blockDim_entry(1);
        count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry, 0, compute_stream>>>(
            d_cluster_query_offset,
            d_entry_count_per_cluster,
            n_total_clusters,
            kQueriesPerBlock
        );

        // entry offset
        int* d_entry_offset = nullptr;
        cudaMalloc(&d_entry_offset, (n_total_clusters + 1) * sizeof(int));
        compute_prefix_sum(d_entry_count_per_cluster, d_entry_offset, n_total_clusters, compute_stream);

        cudaMemcpyAsync(&n_entry, d_entry_offset + n_total_clusters,
                       sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        // entry
        int* d_entry_query_offset = nullptr;
        cudaMalloc(&d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMemcpyAsync(d_entry_query_offset, d_cluster_query_offset,
                       (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream);

        if (n_entry > 0) {
            cudaMalloc(&d_entry_cluster_id, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_start, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_count, n_entry * sizeof(int));
            cudaMalloc(&d_entry_queries, total_entries * sizeof(int));
            cudaMalloc(&d_entry_probe_indices, total_entries * sizeof(int));

            build_entry_data_kernel<<<clusterDim, blockDim_entry, 0, compute_stream>>>(
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_query_probe_indices,
                d_entry_offset,
                d_entry_query_offset,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                n_total_clusters,
                kQueriesPerBlock
            );
        }

        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        //
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);
        cudaFree(d_entry_count_per_cluster);
        cudaFree(d_entry_offset);
        cudaFree(d_entry_query_offset);
        // d_top_nprobe_index  d_top_nprobe_dist  Step 2
        //
    }

    // query
    //
    cudaStreamWaitEvent(compute_stream, data_ready_event, 0);

    // Step 3:
    {
        CUDATimer timer("Pipeline Stage 2: Fine Search");

        int capacity = 32;
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);

        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;
        cudaMalloc(&d_topk_dist_candidate, n_query * n_probes * k * sizeof(float));
        cudaMalloc(&d_topk_index_candidate, n_query * n_probes * k * sizeof(int));

        dim3 init_block(512);
        int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
        dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block, 0, compute_stream>>>(
                d_topk_dist_candidate,
                d_topk_index_candidate,
                n_query * n_probes * k
            );

        dim3 block(kQueriesPerBlock * 32);
        if (n_entry > 0) {
            if(distance_mode == COSINE_DISTANCE){
                if (capacity <= 32) {
                    launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else if (capacity <= 64) {
                    launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else {
                    launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                }
            }
            else if(distance_mode == L2_DISTANCE){
                if (capacity <= 32) {
                    launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else if (capacity <= 64) {
                    launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else {
                    launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                        block, n_dim, d_queries,
                        g_persistent_data.d_cluster_vectors,
                        g_persistent_data.d_probe_vector_offset,
                        g_persistent_data.d_probe_vector_count,
                        d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                        d_entry_queries, d_entry_probe_indices,
                        d_query_norm, g_persistent_data.d_cluster_vector_norm,
                        n_entry, n_probes, k,
                        d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                }
            }
        }

        //  kernel
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        // top-k
        // d_topk_dist_candidate  d_topk_index_candidate  [n_query][n_probes][k]
        // select_k  [n_query][n_probes * k]
        // select_k 0  n_probes * k - 1
            select_k<float, int>(
                d_topk_dist_candidate, n_query, n_probes * k, k,
            d_topk_dist, d_topk_index, true, compute_stream
            );

        //  select_k
        cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;

        //  select_k
        //  d_topk_index  CPU
        #ifdef DEBUG_MAP_INDICES
        int* h_topk_index_debug = (int*)malloc(n_query * k * sizeof(int));
        cudaMemcpy(h_topk_index_debug, d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        int max_candidates = n_probes * k;
        for (int i = 0; i < n_query * k; ++i) {
            if (h_topk_index_debug[i] < 0 || h_topk_index_debug[i] >= max_candidates) {
                printf("[DEBUG] Invalid candidate_pos at idx=%d: %d (max=%d)\n",
                       i, h_topk_index_debug[i], max_candidates);
            }
        }
        free(h_topk_index_debug);
        #endif

        //
        // d_topk_index
        // d_topk_index_candidate  [n_query][n_probes][k] [n_query][n_probes * k]
            dim3 map_block(256);
            dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
        map_candidate_indices_kernel<<<map_grid, map_block, 0, compute_stream>>>(
            d_topk_index_candidate,  //  [n_query][n_probes * k]
            d_topk_index,            //  [n_query][k]
                n_query,
                n_probes,
                k
            );

        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

        cudaFree(d_topk_dist_candidate);
        cudaFree(d_topk_index_candidate);
    }

    // CPU
    cudaMemcpyAsync(topk_dist[0], d_topk_dist,
                   n_query * k * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
    cudaMemcpyAsync(topk_index[0], d_topk_index,
                   n_query * k * sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;

    //
    cudaFree(d_queries);
    cudaFree(d_query_norm);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);
    //
    if (d_top_nprobe_index != nullptr) cudaFree(d_top_nprobe_index);
    if (d_top_nprobe_dist != nullptr) cudaFree(d_top_nprobe_dist);
    if (d_entry_cluster_id != nullptr) cudaFree(d_entry_cluster_id);
    if (d_entry_query_start != nullptr) cudaFree(d_entry_query_start);
    if (d_entry_query_count != nullptr) cudaFree(d_entry_query_count);
    if (d_entry_queries != nullptr) cudaFree(d_entry_queries);
    if (d_entry_probe_indices != nullptr) cudaFree(d_entry_probe_indices);

    // streams
    cudaEventDestroy(data_ready_event);
    cudaStreamDestroy(data_stream);
    cudaStreamDestroy(compute_stream);

    CHECK_CUDA_ERRORS;
}