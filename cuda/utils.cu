/**
 * CUDA Utility Kernels Implementation
 *
 * Common utility kernels used across multiple modules in INSITUANN.
 */

#include "utils.cuh"
#include <cfloat>
#include <iomanip>
#include <algorithm>

void _check_cuda_last_error(const char *file, int line)
{
    //  cudaGetLastError()
    //
    //
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        //
        fprintf(stderr, "[CUDA Last Error]: %s ---- Location: %s:%d\n",
                cudaGetErrorString(err), file, line);

        //
        cudaDeviceReset(); // CUDA
        exit(EXIT_FAILURE);
    }
}

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
    int n_batch)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // stride loop
    for (int idx = tid; idx < n_batch; idx += block_size) {
        d_index[query_id * n_batch + idx] = idx;
    }
}

/**
 * Kernel:  [0, 1, 2, ..., n-1]
 * query
 */
__global__ void generate_single_sequence_indices_kernel(
    int* d_index,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_index[idx] = idx;
    }
}

/**
 * Host:
 * GPU [0, 1, 2, ..., n-1]
 *
 * @param d_index  [n]
 * @param n
 */
void generate_sequence_indices(
    int* d_index,
    int n)
{
    if (!d_index || n <= 0) {
        fprintf(stderr, "[ERROR] generate_sequence_indices: invalid parameters - d_index=%p, n=%d\n",
               (void*)d_index, n);
        return;
    }

    //  cudaMalloc
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] generate_sequence_indices: previous CUDA error: %s\n",
               cudaGetErrorString(err));
    }

    //  kernel
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    //  grid_size  CUDA  65535
    if (grid_size > 65535) {
        fprintf(stderr, "[ERROR] generate_sequence_indices: grid_size %d too large (n=%d, block_size=%d)\n",
               grid_size, n, block_size);
        return;
    }

    if (grid_size <= 0) {
        fprintf(stderr, "[ERROR] generate_sequence_indices: invalid grid_size %d (n=%d)\n", grid_size, n);
        return;
    }

    generate_single_sequence_indices_kernel<<<grid_size, block_size>>>(
        d_index, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] generate_sequence_indices: kernel launch failed: %s (grid_size=%d, block_size=%d, n=%d)\n",
               cudaGetErrorString(err), grid_size, block_size, n);
        return;
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
}

/**
 * Kernel: IVFcluster
 * query [0, 1, 2, ..., n_total_clusters-1]
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * n_total_clusters;
    if (idx < total) {
        int query_idx = idx / n_total_clusters;
        int cluster_idx = idx % n_total_clusters;
        d_indices[idx] = cluster_idx;  // query
    }
}

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
    int n_total_clusters)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;

    const int rank = threadIdx.x;
    if (rank >= k) return;

    int cluster_id = d_topk_index[query_id * k + rank];

    // cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        atomicAdd(&d_cluster_query_count[cluster_id], 1);
    }
}

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
    int n_total_clusters)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;

    const int rank = threadIdx.x;
    if (rank >= k) return;

    int cluster_id = d_topk_index[query_id * k + rank];

    // cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        //
        int write_pos = atomicAdd(&d_cluster_write_pos[cluster_id], 1);
        d_cluster_query_data[write_pos] = query_id;
        d_cluster_query_probe_indices[write_pos] = rank;  // rankprobe_index_in_query
    }
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    d_topk_dist_probe[idx] = FLT_MAX;
    d_topk_index_probe[idx] = -1;
}

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
) {
    const int entry_id = blockIdx.x;
    if (entry_id >= entry_count) return;
    int q_start = d_entry_query_start[entry_start + entry_id];
    int q_count = d_entry_query_count[entry_start + entry_id];
    for (int i = threadIdx.x; i < q_count * k; i += blockDim.x) {
        int qi = i / k, ki = i % k;
        int eq_idx = q_start + qi;
        int query_id = d_entry_queries[eq_idx];
        int probe_idx = d_entry_probe_indices[eq_idx];
        int pos = (query_id * n_probe_slots + probe_idx) * k + ki;
        int v = d_topk_index[pos];
        if (v >= 0) d_topk_index[pos] = v + base_offset;
    }
}

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
) {
    int max_candidates_per_query = n_probes * k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * k;
    if (idx >= total) return;

    int query_id = idx / k;

    int candidate_pos = d_topk_index[idx];
    if (candidate_pos >= 0 && candidate_pos < max_candidates_per_query) {
        int original_idx = d_candidate_indices[query_id * max_candidates_per_query + candidate_pos];
        d_topk_index[idx] = original_idx;
    } else {
        d_topk_index[idx] = -1;
    }
}

/**
 * Kernel:
 *
 *
 * -  grid
 * -  thrust::fill
 */
__global__ void fill_kernel(
    float* __restrict__ d_data,
    float value,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_data[idx] = value;
}

/**
 * Kernel:
 *
 *
 * -  grid
 * -  thrust::fill
 */
__global__ void fill_int_kernel(
    int* __restrict__ d_data,
    int value,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_data[idx] = value;
}

/**
 * Kernel: Block-level inclusive scan (Hillis-Steele)
 *
 * block
 * blockblock
 */
__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,  // [n]
    int* __restrict__ d_output,       // [n]
    int* __restrict__ d_block_sums,   // [n_blocks] block
    int n)                            //
{
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    //
    if (idx < n) {
        s_data[tid] = d_input[idx];
    } else {
        s_data[tid] = 0;
    }
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            s_data[tid] += s_data[tid - stride];
        }
        __syncthreads();
    }

    // block
    if (tid == blockDim.x - 1 && d_block_sums != nullptr) {
        d_block_sums[bid] = s_data[tid];
    }

    //
    if (idx < n) {
        d_output[idx] = s_data[tid];
    }
}

/**
 * Kernel: block
 *
 * blockblock
 */
__global__ void merge_scan_blocks_kernel(
    const int* __restrict__ d_block_sums,  // [n_blocks] block
    int* __restrict__ d_output,            // [n]
    int n,                                  //
    int block_size)                         // block
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int block_id = idx / block_size;
    if (block_id > 0) {
        // block
        int prefix_sum = 0;
        for (int i = 0; i < block_id; i++) {
            prefix_sum += d_block_sums[i];
        }
        d_output[idx] += prefix_sum;
    }
}

/**
 * Host: CSRoffset
 *
 *  inclusive prefix sumoffset[i+1] = offset[i] + count[i]
 *  offset[0] = 0 exclusive scan
 *
 * CUDAThrust
 */
void compute_prefix_sum(
    const int* d_count,  // [n]
    int* d_offset,       // [n+1]
    int n,               //
    cudaStream_t stream)
{
    // 1.  offset[0]  0
    cudaMemsetAsync(d_offset, 0, sizeof(int), stream);

    if (n <= 0) return;

    // 2.  inclusive scan offset[1..n]
    const int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    int shared_mem_size = block_size * sizeof(int);

    if (n_blocks == 1) {
        // blockblock sums
        inclusive_scan_kernel<<<1, block_size, shared_mem_size, stream>>>(
            d_count,
            d_offset + 1,  //  offset[1..n]
            nullptr,       // block sums
            n
        );
    } else {
        // block
        // 1blockscanblock
        int* d_block_sums = nullptr;
        cudaMalloc(&d_block_sums, n_blocks * sizeof(int));

        inclusive_scan_kernel<<<n_blocks, block_size, shared_mem_size, stream>>>(
            d_count,
            d_offset + 1,
            d_block_sums,
            n
        );

        // 2blocksblockblock
        merge_scan_blocks_kernel<<<n_blocks, block_size, 0, stream>>>(
            d_block_sums,
            d_offset + 1,
            n,
            block_size
        );

        cudaFree(d_block_sums);
    }
}

/**
 * Kernel: clusterentryentry
 *
 *
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 *
 * clusterqueryentrykQueriesPerBlockquery
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSRoffset
    int* d_entry_count_per_cluster,  // [n_total_clusters] clusterentry
    int n_total_clusters,
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;

    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;

    // clusterentrykQueriesPerBlockquery
    int n_entries = (n_queries > 0) ? (n_queries + kQueriesPerBlock - 1) / kQueriesPerBlock : 0;
    d_entry_count_per_cluster[cluster_id] = n_entries;
}

/**
 * Kernel: entryentry
 *
 *
 * - gridDim.x = n_total_clusters (blockcluster)
 * - blockDim.x = 1 ()
 *
 * clusterqueryentryentry
 *
 * d_entry_queries  entry  query entry
 *  cluster  d_entry_queries
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
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;

    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;

    if (n_queries == 0) return;  // querycluster

    int entry_start = d_entry_offset[cluster_id];
    int entry_idx = entry_start;

    // clusterentry queries
    int cluster_entry_query_start = d_entry_query_offset[cluster_id];
    int current_query_offset = cluster_entry_query_start;

    // queryentrykQueriesPerBlock
    for (int batch_start = 0; batch_start < n_queries; batch_start += kQueriesPerBlock) {
        int batch_size = min(kQueriesPerBlock, n_queries - batch_start);

        // entry
        d_entry_cluster_id[entry_idx] = cluster_id;
        d_entry_query_start[entry_idx] = current_query_offset;
        d_entry_query_count[entry_idx] = batch_size;

        // entryqueryprobe_indices
        for (int i = 0; i < batch_size; ++i) {
            int query_idx = query_start + batch_start + i;
            d_entry_queries[current_query_offset + i] = d_cluster_query_data[query_idx];
            d_entry_probe_indices[current_query_offset + i] = d_cluster_query_probe_indices[query_idx];
        }

        current_query_offset += batch_size;
        entry_idx++;
    }
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * k;

    if (idx < total) {
        int reordered_idx = d_reordered_index_in[idx];
        if (reordered_idx >= 0 && reordered_idx < n_total_vectors) {
            d_original_index_out[idx] = d_reordered_indices[reordered_idx];
        } else {
            d_original_index_out[idx] = -1;
        }
    }
}

/**
 *
 */
void ivf_lookup_reordered_to_original(
    const int* d_reordered_indices,
    const int* d_reordered_index_in,
    int* d_original_index_out,
    int n_query,
    int k,
    int n_total_vectors
) {
    if (!d_reordered_indices || !d_reordered_index_in || !d_original_index_out) {
        fprintf(stderr, "[ERROR] ivf_lookup_reordered_to_original: null\n");
        throw std::invalid_argument("input pointers must not be null");
    }

    if (n_query <= 0 || k <= 0) {
        fprintf(stderr, "[ERROR] ivf_lookup_reordered_to_original:  - n_query=%d, k=%d\n", n_query, k);
        throw std::invalid_argument("invalid parameters");
    }

    //  GPU kernel
    dim3 block_size(256);
    dim3 grid_size((n_query * k + block_size.x - 1) / block_size.x);

    lookup_original_indices_kernel<<<grid_size, block_size>>>(
        d_reordered_indices,
        d_reordered_index_in,
        d_original_index_out,
        n_query,
        k,
        n_total_vectors
    );

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
}
