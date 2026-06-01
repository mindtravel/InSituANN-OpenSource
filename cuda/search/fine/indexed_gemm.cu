#include "indexed_gemm.cuh"
#include "pch.h"
// ============================================================================
// Indexed Inner Product Kernel for Fine Screening
// ============================================================================


/**
 * kernel
 *
 * blockclusterclusterquerycluster
 * INSITUANN
 *
 * max_candidates_per_querymax_cluster_vector_count
 *
 */
__global__ void indexed_inner_product_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_vector_index,  // cluster
    const int* __restrict__ d_cluster_vector_num,
    float* __restrict__ d_inner_product,
    int* __restrict__ d_index,
    int* __restrict__ d_query_count,  // query
    const int* __restrict__ d_num_samples,  // query [n_query]
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int tol_vector,
    int max_candidates_per_query  // query
) {
    const int cluster_idx = blockIdx.x;
    if (cluster_idx >= distinct_cluster_count) return;

    const int thread_idx = threadIdx.x;
    const int block_dim = blockDim.x;

    // clusterquery
    // offsetoffset distinct_cluster_count + 1
    // count = offset[i+1] - offset[i]
    int query_start = d_cluster_query_offset[cluster_idx];
    int query_end = d_cluster_query_offset[cluster_idx + 1];
    int query_count = query_end - query_start;

    if (query_count <= 0) return;

    // cluster
    int vector_start_idx = d_cluster_vector_index[cluster_idx];  // cluster
    int vector_count = d_cluster_vector_num[cluster_idx];

    //
    if (vector_start_idx < 0 || vector_count <= 0 ||
        vector_start_idx + vector_count > tol_vector) {
        return;
    }

    // querycluster
    // clusterquery
    for (int q = 0; q < query_count; q++) {
        int query_idx = d_cluster_query_data[query_start + q];

        //
        if (query_idx < 0 || query_idx >= n_query) continue;

        // cluster
        for (int vec_idx = thread_idx; vec_idx < vector_count; vec_idx += block_dim) {
            int global_vec_idx = vector_start_idx + vec_idx;  //

            //
            if (global_vec_idx < 0 || global_vec_idx >= tol_vector) continue;

            //
            float dot_product = 0.0f;
            #pragma unroll 4
            for (int dim = 0; dim < n_dim; dim++) {
                dot_product += d_query_group[query_idx * n_dim + dim] *
                              d_cluster_vector[global_vec_idx * n_dim + dim];
            }

            // query
            int pos = atomicAdd(&d_query_count[query_idx], 1);

            // query
            int actual_num_samples = d_num_samples[query_idx];
            if (pos < actual_num_samples && pos < max_candidates_per_query) {
                int output_idx = query_idx * max_candidates_per_query + pos;
                d_inner_product[output_idx] = dot_product;
                d_index[output_idx] = global_vec_idx;  //
            }
        }
    }
}