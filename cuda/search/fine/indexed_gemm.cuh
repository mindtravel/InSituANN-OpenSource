#ifndef INDEXED_GEMM_CUH
#define INDEXED_GEMM_CUH

/**
 * kernel
 *
 * INSITUANN
 *
 * max_candidates_per_querymax_cluster_vector_count
 *
 * d_num_samplesquery
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
);

/**
 * Launchentry-basedblockentrycluster + query
 *
 * @tparam Capacity warp-sort queue
 * @tparam Ascending
 * @tparam QueriesPerBlock entryquery8
 *
 * Entry-based
 * - grid = n_entryquerycluster
 * - blockentrycluster + query
 * - queryclusterblock
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_cos_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry] entrycluster_id
    int* __restrict__ d_entry_query_start,  // [n_entry] entryquery
    int* __restrict__ d_entry_query_count,  // [n_entry] entryquery
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries] entryquery
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries] entryprobe_indices
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobe
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_l2_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry] entrycluster_id
    int* __restrict__ d_entry_query_start,  // [n_entry] entryquery
    int* __restrict__ d_entry_query_count,  // [n_entry] entryquery
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries] entryquery
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries] entryprobe_indices
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobe
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);
#endif
