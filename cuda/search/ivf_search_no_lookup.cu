/* limits.hThrustCHAR_MIN */
#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
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
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define ENABLE_CUDA_TIMING 0

using namespace INSITUANN::warpsort_utils;
using namespace INSITUANN::warpsort_topk;

void ivf_search_no_lookup(float* d_query_batch,
                           int* d_cluster_size,
                           float* d_cluster_vectors,
                           float* d_cluster_centers,
                           int* d_initial_indices,
                           float* d_topk_dist,
                           int* d_topk_index,
                           int n_query,
                           int n_dim,
                           int n_total_clusters,
                           int n_total_vectors,
                           int n_probes,
                           int k,
                           int distance_mode,
                           int** h_coarse_index,  // [n_query, n_probes] host
                           float** h_coarse_dist  // [n_query, n_probes] host
                        ) {

    // fprintf(stderr, "ivf_search: , n_query=%d, n_dim=%d, n_total_clusters=%d, n_total_vectors=%d, n_probes=%d, k=%d\n",
    //         n_query, n_dim, n_total_clusters, n_total_vectors, n_probes, k);

    if (n_query <= 0 || n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        fprintf(stderr, "[ERROR] ivf_search_no_lookup:  - n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
               n_query, n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid ivf_search_no_lookup configuration");
    }
    if (!d_cluster_size || !d_cluster_vectors || !d_cluster_centers || !d_query_batch) {
        fprintf(stderr, "[ERROR] ivf_search_no_lookup: devicenull\n");
        throw std::invalid_argument("input device pointers must not be null");
    }
    if (!d_topk_dist || !d_topk_index) {
        fprintf(stderr, "[ERROR] ivf_search_no_lookup: devicenull\n");
        throw std::invalid_argument("output device pointers must not be null");
    }
    if (n_probes <= 0 || n_probes > n_total_clusters) {
        fprintf(stderr, "[ERROR] ivf_search_no_lookup: n_probes - n_probes=%d, n_total_clusters=%d\n",
               n_probes, n_total_clusters);
        throw std::invalid_argument("invalid n_probes");
    }

    //  device
    float* d_queries = d_query_batch;
    float* d_cluster_vectors_ptr = d_cluster_vectors;
    float* d_cluster_centers_ptr = d_cluster_centers;

    float *d_cluster_centers_norm = nullptr;
    float *d_query_norm = nullptr;
    float *d_cluster_vector_norm = nullptr;

    int *d_top_nprobe_index = nullptr;
    float *d_top_nprobe_dist = nullptr;
    float *d_inner_product = nullptr;
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;

    dim3 queryDim(n_query);
    dim3 dataDim(n_total_clusters);
    dim3 vectorDim(n_dim);
    dim3 probeDim(n_probes);

    {
        CUDATimer timer("Step 0: Data Preparation", ENABLE_CUDA_TIMING);
        // fprintf(stderr, "ivf_search: Step 0 - \n");

        // GPUprobe_vector_offset
        cudaMalloc(&d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
        d_probe_vector_count = d_cluster_size;  //  device

        // GPUoffset
        compute_prefix_sum(d_probe_vector_count, d_probe_vector_offset, n_total_clusters, 0);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;

        cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*queryl2 Norm*/
        cudaMalloc(&d_cluster_centers_norm, n_total_clusters * sizeof(float)); /*datal2 Norm*/
        CHECK_CUDA_ERRORS;

        cudaMalloc(&d_top_nprobe_index, n_query * n_probes * sizeof(int));/*top n_probes*/
        CHECK_CUDA_ERRORS;
        cudaDeviceSynchronize();

        compute_l2_norm_gpu(d_cluster_vectors_ptr, d_cluster_vector_norm, n_total_vectors, n_dim);
        compute_l2_norm_gpu(d_queries, d_query_norm, n_query, n_dim);
        compute_l2_norm_gpu(d_cluster_centers_ptr, d_cluster_centers_norm, n_total_clusters, n_dim);

        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

        // ------------------------------------------------------------------
        // Step 1.  warpsort  query -> cluster mapping
        // ------------------------------------------------------------------
    // data_index  cuda_cos_topk_warpsort  CUDA kernel  [0, 1, 2, ..., n_total_clusters-1]

    {
        CUDATimer timer("Step 1: Coarse Search (cuda_cos_topk_warpsort)", ENABLE_CUDA_TIMING);
        float alpha = 1.0f;
        float beta = 0.0f;

        // cuBLAS
        cublasHandle_t handle;

        //
        {
            CUDATimer timer("Step 1: GPU Memory Allocation", ENABLE_CUDA_TIMING);

            cudaMalloc(&d_inner_product, n_query * n_total_clusters * sizeof(float));/*querydata*/
            cudaMalloc(&d_top_nprobe_dist, n_query * n_probes * sizeof(float));/*topk*/

            cublasCreate(&handle);
        }

        //
        {
            // COUT_ENDL("begin data transfer");

            CUDATimer timer_trans1("Step 1: H2D Data Transfer", ENABLE_CUDA_TIMING);


            /* fill kernelthrust::fill */
            dim3 fill_block(256);
            int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
            dim3 fill_grid(fill_grid_size);
            fill_kernel<<<fill_grid, fill_block>>>(
                d_top_nprobe_dist,
                FLT_MAX,
                n_query * n_probes
            );
            // cudaMemset((void*)d_top_nprobe_dist, (int)0xEF, n_query * k * sizeof(float)) /*memset*/
            // table_cuda_2D("topk cos distance", d_top_nprobe_dist, n_query, k);
            // COUT_ENDL("finish data transfer");
        }

        /*  */
        {
            CUDATimer timer("Step 1: Kernel Execution: matrix multiply", ENABLE_CUDA_TIMING);

            /**
            * cuBLAS
            * cuBLASleading dimension
            * */
            cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n_total_clusters, n_query, n_dim,
                &alpha,
                d_cluster_centers_ptr, n_dim,
                d_queries, n_dim,
                &beta,
                d_inner_product, n_total_clusters
            );

            cudaDeviceSynchronize();
        }

        {
            CUDATimer timer("Step 1: Kernel Execution: cos + topk", ENABLE_CUDA_TIMING);

            if(distance_mode == COSINE_DISTANCE){
                INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                    d_query_norm, d_cluster_centers_norm, d_inner_product, d_initial_indices,
                    n_query, n_total_clusters, n_probes,  //  n_probes  cluster
                    d_top_nprobe_dist, d_top_nprobe_index,
                    true, // select min
                    0  //
                );
            }
            else if(distance_mode == L2_DISTANCE){
                INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                    d_query_norm, d_cluster_centers_norm, d_inner_product, d_initial_indices,
                    n_query, n_total_clusters, n_probes,  //  n_probes  cluster
                    d_top_nprobe_dist, d_top_nprobe_index,
                    true, // select min
                    0  //
                );
            }
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;

            // CPU
            if (h_coarse_index != nullptr && h_coarse_dist != nullptr) {
                cudaMemcpy(h_coarse_index[0], d_top_nprobe_index,
                           n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_coarse_dist[0], d_top_nprobe_dist,
                           n_query * n_probes * sizeof(float), cudaMemcpyDeviceToHost);
                CHECK_CUDA_ERRORS;
            }

            //
            if(false){
                int* h_top_nprobe_index = (int*)malloc(n_query * n_probes * sizeof(int));
                float* h_top_nprobe_dist = (float*)malloc(n_query * n_probes * sizeof(float));
                cudaMemcpy(h_top_nprobe_index, d_top_nprobe_index,
                           n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_top_nprobe_dist, d_top_nprobe_dist,
                           n_query * n_probes * sizeof(float), cudaMemcpyDeviceToHost);
                CHECK_CUDA_ERRORS;

                printf("[DEBUG GPU Coarse] Query 0 coarse results (top %d clusters):\n", n_probes);
                for (int p = 0; p < n_probes; ++p) {
                    printf("  Probe %d: cluster_id=%d, dist=%.6f\n",
                           p, h_top_nprobe_index[p], h_top_nprobe_dist[p]);
                }
                if (n_query > 1) {
                    printf("[DEBUG GPU Coarse] Query 1 coarse results (top %d clusters):\n", n_probes);
                    for (int p = 0; p < n_probes; ++p) {
                        printf("  Probe %d: cluster_id=%d, dist=%.6f\n",
                               p, h_top_nprobe_index[n_probes + p], h_top_nprobe_dist[n_probes + p]);
                    }
                }

                free(h_top_nprobe_index);
                free(h_top_nprobe_dist);
            }
        }


        {
            CUDATimer timer("Step 1: GPU Memory Free", false, ENABLE_CUDA_TIMING);
            cublasDestroy(handle);
            // d_cluster_centers_ptr
            cudaFree(d_inner_product);
            cudaFree(d_cluster_centers_norm);
            // d_top_nprobe_dist
            //  Step 2
            if (h_coarse_index == nullptr || h_coarse_dist == nullptr) {
                cudaFree(d_top_nprobe_dist);
            }
        }

        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS
    }

    // ------------------------------------------------------------------
    // Step 2.  querycluster  entry v5 entry-based
    // ------------------------------------------------------------------
    //  Step 1  Step 2
    //  cluster-query CSR entry
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;

    // EntryGPU
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;
    int n_entry = 0;
    constexpr int kQueriesPerBlock = 8;

    {
        CUDATimer timer("Step 2: Build entry data (GPU)", ENABLE_CUDA_TIMING);

        // GPUclusterquery
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemset(d_cluster_query_count, 0, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;

        count_cluster_queries_kernel<<<queryDim, probeDim>>>(
            d_top_nprobe_index,
            d_cluster_query_count,
            n_query,
            n_probes,
            n_total_clusters
        );
        CHECK_CUDA_ERRORS;

        // GPUCSRoffset
        cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;

        compute_prefix_sum(d_cluster_query_count, d_cluster_query_offset, n_total_clusters, 0);
        CHECK_CUDA_ERRORS;

        // GPU
        int total_entries = 0;
        cudaMemcpy(&total_entries, d_cluster_query_offset + n_total_clusters,
                   sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;

        // offset
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpy(d_cluster_write_pos, d_cluster_query_offset,
                   n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS;

        // GPUcluster-query
        cudaMalloc(&d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&d_cluster_query_probe_indices, total_entries * sizeof(int));
        CHECK_CUDA_ERRORS;

        // GPUCSRcluster-query
        build_cluster_query_mapping_kernel<<<queryDim, probeDim>>>(
            d_top_nprobe_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            n_probes,
            n_total_clusters
        );
        CHECK_CUDA_ERRORS;

        // GPUclusterentry
        int* d_entry_count_per_cluster = nullptr;
        cudaMalloc(&d_entry_count_per_cluster, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;

        dim3 clusterDim(n_total_clusters);
        dim3 blockDim_entry(1);
        count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry>>>(
            d_cluster_query_offset,
            d_entry_count_per_cluster,
            n_total_clusters,
            kQueriesPerBlock
        );
        CHECK_CUDA_ERRORS;

        // entryoffset
        int* d_entry_offset = nullptr;
        cudaMalloc(&d_entry_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;

        compute_prefix_sum(d_entry_count_per_cluster, d_entry_offset, n_total_clusters, 0);
        CHECK_CUDA_ERRORS;

        // entryGPU
        cudaMemcpy(&n_entry, d_entry_offset + n_total_clusters,
                   sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;

        // clusterentry queries
        //  d_cluster_query_offsetclusterqueryentry
        int* d_entry_query_offset = nullptr;
        cudaMalloc(&d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMemcpy(d_entry_query_offset, d_cluster_query_offset,
                   (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERRORS;

        // entry
        if (n_entry > 0) {
            cudaMalloc(&d_entry_cluster_id, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_start, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_count, n_entry * sizeof(int));

            // querytotal_entriesentry-queryquery
            cudaMalloc(&d_entry_queries, total_entries * sizeof(int));
            cudaMalloc(&d_entry_probe_indices, total_entries * sizeof(int));
            CHECK_CUDA_ERRORS;

            // GPUentry
            build_entry_data_kernel<<<clusterDim, blockDim_entry>>>(
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
            CHECK_CUDA_ERRORS;
        }

        //
        cudaFree(d_entry_query_offset);

        //
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);
        cudaFree(d_entry_count_per_cluster);
        cudaFree(d_entry_offset);
        cudaFree(d_top_nprobe_index);
        CHECK_CUDA_ERRORS;
    }


        // ------------------------------------------------------------------
    // Step 3.  v5 entry-based
        // ------------------------------------------------------------------
    {
        CUDATimer timer("Step 3: Fine Search (v5 entry-based)", ENABLE_CUDA_TIMING);

        int capacity = 32;
        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;

        // kernel launch
        // v5 entry-basedblockentrycluster + query
        dim3 block(kQueriesPerBlock * 32);  // 8warpwarp 32

        {
            CUDATimer timer("Init Invalid Values Kernel", ENABLE_CUDA_TIMING);

            // Capacity2 > k
            while (capacity < k) capacity <<= 1;
            capacity = std::min(capacity, kMaxCapacity);

            CHECK_CUDA_ERRORS;

            // query [n_query][n_probes][k]
            cudaMalloc(&d_topk_dist_candidate, n_query * n_probes * k * sizeof(float));
            cudaMalloc(&d_topk_index_candidate, n_query * n_probes * k * sizeof(int));

            // FLT_MAX  -1
            dim3 init_block(512);
            int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
            dim3 init_grid(init_grid_size);
            init_invalid_values_kernel<<<init_grid, init_block>>>(
                d_topk_dist_candidate,
                d_topk_index_candidate,
                n_query * n_probes * k
            );
            CHECK_CUDA_ERRORS;
        }

        {
            CUDATimer timer("Indexed Inner Product with TopK Kernel (v5 entry-based)", ENABLE_CUDA_TIMING);

            if (n_entry != 0) {
                if(distance_mode == COSINE_DISTANCE){
                    // capacitykernel
                    if (capacity <= 32) {
                        launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    } else if (capacity <= 64) {
                        launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    } else {
                        launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    }
                }
                else if(distance_mode == L2_DISTANCE){
                    // capacitykernel
                    if (capacity <= 32) {
                        launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    } else if (capacity <= 64) {
                        launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    } else {
                        launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                            block,
                            n_dim,
                            d_queries,
                            d_cluster_vectors_ptr,
                            d_probe_vector_offset,
                            d_probe_vector_count,
                            d_entry_cluster_id,
                            d_entry_query_start,
                            d_entry_query_count,
                            d_entry_queries,
                            d_entry_probe_indices,
                            d_query_norm,
                            d_cluster_vector_norm,
                            n_entry,
                            n_probes,
                            k,
                            d_topk_dist_candidate,  // query [n_query][n_probes][k]
                            d_topk_index_candidate,  // query [n_query][n_probes][k]
                            0
                        );
                    }
                }
            }
            CHECK_CUDA_ERRORS;
        }

        //  [n_query][n_probes][k]  [n_query][k]
        // GPUCPU-GPU
        {
            CUDATimer timer("Reduce probe results to query top-k", ENABLE_CUDA_TIMING);

            select_k<float, int>(
                d_topk_dist_candidate, n_query, n_probes * k, k,
                d_topk_dist, d_topk_index, true, 0
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;

            // 2.
            // select_k
            dim3 map_block(256);
            dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
            map_candidate_indices_kernel<<<map_grid, map_block>>>(
                d_topk_index_candidate,  //
                d_topk_index,
                n_query,
                n_probes,
                k
            );
            CHECK_CUDA_ERRORS;

            //
            cudaFree(d_topk_dist_candidate);
            cudaFree(d_topk_index_candidate);
        }

        // kernel
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;

        cudaFree(d_cluster_vector_norm);
        CHECK_CUDA_ERRORS;
    }

    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);

    // entry
    if (d_entry_cluster_id != nullptr) {
        cudaFree(d_entry_cluster_id);
    }
    if (d_entry_query_start != nullptr) {
        cudaFree(d_entry_query_start);
    }
    if (d_entry_query_count != nullptr) {
        cudaFree(d_entry_query_count);
    }
    if (d_entry_queries != nullptr) {
        cudaFree(d_entry_queries);
    }
    if (d_entry_probe_indices != nullptr) {
        cudaFree(d_entry_probe_indices);
    }

    cudaFree(d_probe_vector_offset);
    // d_probe_vector_count  d_cluster_size
    // d_queries  d_query_batch
    cudaFree(d_query_norm);

    // d_top_nprobe_dist  Step 1
    // d_top_nprobe_dist  Step 1
    //

    CHECK_CUDA_ERRORS;
}
