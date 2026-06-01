#include "indexed_gemm.cuh"
#include "inner_product_utils.cuh"
#include "pch.h"
#include "search/topk/warpsort_utils.cuh"
#include "search/topk/warpsort.cuh"
#include <cfloat>
#include <stdint.h>
#include <vector>
#include <algorithm>

using namespace INSITUANN::warpsort_utils;
using namespace INSITUANN::warpsort;

/**
 * v5Entry-based
 * entryclusterquery84
 * gridn_entryblockentry
 */

/**
 * EntryGPUentry
 * d_entry_cluster_id: entrycluster_id [n_entry]
 * d_entry_query_start: entryquery [n_entry]
 * d_entry_query_count: entryquery [n_entry]
 * d_entry_queries: entryquery[total_queries_in_entries]
 * d_entry_probe_indices: entry-queryprobequery [total_queries_in_entries]
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1)
void indexed_inner_product_with_l2_topk_kernel(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobeprobe_index_in_query
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,  // [n_query][n_probes][k]
    int* __restrict__ d_topk_index   // [n_query][n_probes][k]
) {
    __shared__ float s_query_norm[QueriesPerBlock];

    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;

    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int vector_count = d_probe_vector_count[cluster_id];

    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];
    // const int entry_query_end = entry_query_start + entry_query_count;

    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = laneId();

    if (local_query_idx >= entry_query_count) return;

    const int entry_query_idx = entry_query_start + local_query_idx;

    //  entry_query_idx
    //  d_entry_queries  d_entry_probe_indices  total_entries
    //  entry_query_idx  [entry_query_start, entry_query_start + entry_query_count)
    const int query_global_id = d_entry_queries[entry_query_idx];
    const int probe_index_in_query = d_entry_probe_indices[entry_query_idx];

    //  query_global_id
    //  n_query kernel
    //  query_global_id

    if (lane == 0) {
        s_query_norm[local_query_idx] = d_query_norm[query_global_id];
    }
    __syncthreads();

    //  query norm
    if (s_query_norm[local_query_idx] < 1e-6f) return;

    const float* query_global_ptr = d_query_group + query_global_id * n_dim;
    const bool query_ptr_aligned = (reinterpret_cast<uintptr_t>(query_global_ptr) & (sizeof(float4) - 1)) == 0;
    const bool data_ptr_aligned = (reinterpret_cast<uintptr_t>(d_cluster_vector) & (sizeof(float4) - 1)) == 0;
    const bool prefer_vec4 = query_ptr_aligned && data_ptr_aligned && ((n_dim & 3) == 0);

    using WarpSortBase = INSITUANN::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();

    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;

    if (prefer_vec4) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_vec4_aligned(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float query_norm = s_query_norm[local_query_idx];
                    float l2_distance = query_norm*query_norm + data_norm*data_norm - 2.0f * dot_product;
                    queue.add(l2_distance, vec_idx);
                }
            }
        }
    } else {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_accumulate(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float query_norm = s_query_norm[local_query_idx];
                    float l2_distance = query_norm*query_norm + data_norm*data_norm - 2.0f * dot_product;
                    queue.add(l2_distance, vec_idx);
                }
            }
        }
    }
    __syncwarp();

    queue.done();
    __syncwarp();

    // [n_query][n_probes][k]
    // probe
    float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
    int* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;
    queue.store(row_dist, row_idx);
}

/**
 * Launch entry-based
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_l2_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry
    int n_probes,  // queryprobe
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream) {

    dim3 grid(n_entry, 1, 1);

    //  generic
    indexed_inner_product_with_l2_topk_kernel<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
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
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

//
// QueriesPerBlock=1
template void launch_indexed_inner_product_with_l2_topk_kernel<64, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_l2_topk_kernel<128, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_l2_topk_kernel<256, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

// QueriesPerBlock=8
template void launch_indexed_inner_product_with_l2_topk_kernel<64, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_l2_topk_kernel<128, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_l2_topk_kernel<256, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);
