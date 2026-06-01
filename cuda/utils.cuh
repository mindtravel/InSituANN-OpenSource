/**
 * CUDA Utility Kernels
 *
 * Common utility kernels used across multiple modules in INSITUANN.
 */

#ifndef INSITUANN_CUDA_UTILS_CUH
#define INSITUANN_CUDA_UTILS_CUH

void _check_cuda_last_error(const char *file, int line);
#define CHECK_CUDA_ERRORS _check_cuda_last_error(__FILE__, __LINE__);/*CUDA*/

/**
 * Kernel:
 * query [0, 1, 2, ..., n_batch-1]
 *
 *
 * - gridDim.x = n_query (blockquery)
 * - blockDim.x = min(256, n_batch) (block)
 * -  n_batch > blockDim.x
 */
__global__ void generate_sequence_indices_kernel(
    int* d_index,
    int n_query,
    int n_batch
);

/**
 * Kernel:  [0, 1, 2, ..., n-1]
 * query
 */
__global__ void generate_single_sequence_indices_kernel(
    int* d_index,
    int n
);

/**
 * Host:
 * GPU [0, 1, 2, ..., n-1]
 *
 * @param d_index  [n]
 * @param n
 */
void generate_sequence_indices(
    int* d_index,
    int n
);

/**
 * Kernel: IVFcluster
 * query [0, 1, 2, ..., n_total_clusters-1]
 *
 *  batch_search_pipeline  d_initial_indices  nullptr
 * GPUCPU-GPU
 *
 *
 * -  grid
 * - total = n_query * n_total_clusters
 * - query
 */
__global__ void generate_sequential_indices_kernel(
    int* d_indices,  // [n_query * n_total_clusters]
    int n_query,     // query
    int n_total_clusters  // cluster
);

/**
 * Kernel: clusterquery
 *
 *
 * - gridDim.x = n_query (blockquery)
 * - blockDim.x = k (blockquerykprobe)
 * - queryprobe
 *
 * clusterquery
 */
__global__ void count_cluster_queries_kernel(
    const int* d_topk_index,  // [n_query * k] querytopk cluster
    int* d_cluster_query_count,  // [n_total_clusters] clusterquery
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: cluster-queryCSR
 *
 *
 * - gridDim.x = n_query (blockquery)
 * - blockDim.x = k (blockquerykprobe)
 * - queryprobe
 *
 * cluster-query
 */
__global__ void build_cluster_query_mapping_kernel(
    const int* d_topk_index,  // [n_query * k]
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_cluster_query_data,  // [total_entries] CSRdataquery_id
    int* d_cluster_query_probe_indices,  // [total_entries] probequery
    int* d_cluster_write_pos,  // [n_total_clusters] cluster
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: FLT_MAX  -1
 *
 *  top-k  FLT_MAX -1
 *
 *
 * -  grid
 * - total_size = n_query * n_probes * k
 */
__global__ void init_invalid_values_kernel(
    float* __restrict__ d_topk_dist_probe,  // [total_size] -  FLT_MAX
    int* __restrict__ d_topk_index_probe,  // [total_size] -  -1
    int total_size  //
);

/**
 * Kernel:  entry  slot  base_offsetproducer-consumer
 *  >= 0  offset-1
 */
__global__ void add_base_offset_to_block_slots_kernel(
    const int* d_entry_queries,
    const int* d_entry_probe_indices,
    const int* d_entry_query_start,
    const int* d_entry_query_count,
    int entry_start,
    int entry_count,
    int base_offset,
    int* d_topk_index,
    int n_probe_slots,
    int k
);

/**
 * Kernel:
 *
 *  select_k
 *
 *
 * -  grid top-k
 * - total = n_query * k
 */
__global__ void map_candidate_indices_kernel(
    const int* __restrict__ d_candidate_indices,  // [n_query][n_probes * k]
    int* __restrict__ d_topk_index,  // [n_query][k] -
    int n_query,
    int n_probes,
    int k
);

/**
 * Kernel:
 *
 *  thrust::fill
 */
__global__ void fill_kernel(
    float* __restrict__ d_data,
    float value,
    int n);

/**
 * Kernel:
 *
 *  thrust::fill
 */
__global__ void fill_int_kernel(
    int* __restrict__ d_data,
    int value,
    int n);

/**
 * Kernel:  -
 */
__global__ void inclusive_scan_block_kernel(
    const int* __restrict__ d_input,  // [n]
    int* __restrict__ d_output,  // [n]
    int* __restrict__ d_block_sums,  // [num_blocks] block
    int n);

/**
 * Kernel:  -
 */
__global__ void inclusive_scan_add_block_sums_kernel(
    int* __restrict__ d_output,  // [n]
    const int* __restrict__ d_block_prefix_sums,  // [num_blocks] block
    int n);

/**
 * Kernel: inclusive scan- block
 *
 * Thrustinclusive_scan
 * kernelblockblock
 */
__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,  // [n]
    int* __restrict__ d_output,  // [n]
    int n);

/**
 * Host: CSRoffset
 *
 *  exclusive prefix sumoffset[0] = 0, offset[i+1] = offset[i] + count[i]
 *
 * inclusive_scan kernelThrust
 *
 * @param d_count  [n]GPU
 * @param d_offset  [n+1]GPUoffset[0] = 0
 * @param n
 * @param stream CUDA0
 */
void compute_prefix_sum(
    const int* d_count,  // [n]
    int* d_offset,  // [n+1] offset[0] = 0
    int n,  //
    cudaStream_t stream = 0
);

/**
 * Kernel: clusterentryentry
 *
 *
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_entry_count_per_cluster,  // [n_total_clusters] clusterentry
    int n_total_clusters,
    int kQueriesPerBlock);

/**
 * Kernel: entryentry
 *
 *
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 */
__global__ void build_entry_data_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    const int* d_cluster_query_data,  // [total_entries] CSRdataquery_id
    const int* d_cluster_query_probe_indices,  // [total_entries] probequery
    const int* d_entry_offset,  // [n_total_clusters + 1] entryoffsetCSR
    const int* d_entry_query_offset,  // [n_total_clusters + 1] clusterd_entry_queries
    int* d_entry_cluster_id,  // [n_entry] entrycluster_id
    int* d_entry_query_start,  // [n_entry] entryqueryd_entry_queries
    int* d_entry_query_count,  // [n_entry] entryquery
    int* d_entry_queries,  // [total_queries_in_entries] entryquery
    int* d_entry_probe_indices,  // [total_queries_in_entries] entryprobe_indices
    int n_total_clusters,
    int kQueriesPerBlock);

/**
 * Kernel:  -
 *
 *
 * -  grid top-k
 * - total = n_query * k
 *
 * @param d_reordered_indices  device  [n_total_vectors]
 * @param d_reordered_index_in  device  [n_query * k]
 * @param d_original_index_out  device  [n_query * k]
 * @param n_query
 * @param k                     top-k
 */
__global__ void lookup_original_indices_kernel(
    const int* __restrict__ d_reordered_indices,
    const int* __restrict__ d_reordered_index_in,
    int* __restrict__ d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
);

/**
 *  ivf_search
 *
 * @param d_reordered_indices  device  [n_total_vectors]
 * @param d_reordered_index_in  device  [n_query * k]
 * @param d_original_index_out  device  [n_query * k]
 * @param n_query
 * @param k                     top-k
 * @param n_total_vectors
 */
void ivf_lookup_reordered_to_original(
    const int* d_reordered_indices,
    const int* d_reordered_index_in,
    int* d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
);

#endif // INSITUANN_CUDA_UTILS_CUH
