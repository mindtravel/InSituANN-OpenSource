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
#include "dataset/dataset.cuh"
#include "cudatimer.h"
#include "l2norm/l2norm.cuh"
#include "utils.cuh"

#if defined(USE_NVTX) && USE_NVTX
#include <nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP()      ((void)0)
#endif

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

void ivf_search(
                        float* d_query_batch,
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
    DistanceType distance_mode,
    int** h_coarse_index,
    float** h_coarse_dist,
    const int* d_reordered_indices,
    cudaStream_t stream,
    const float* h_cluster_vectors,
    const int* h_cluster_sizes
                        ) {
    if (n_query <= 0 || n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        fprintf(stderr, "[ERROR] ivf_search:  - n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
               n_query, n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid ivf_search configuration");
    }
    if (!d_cluster_size || !d_cluster_vectors || !d_cluster_centers || !d_query_batch) {
        fprintf(stderr, "[ERROR] ivf_search: devicenull\n");
        throw std::invalid_argument("input device pointers must not be null");
    }
    if (!d_topk_dist || !d_topk_index) {
        fprintf(stderr, "[ERROR] ivf_search: devicenull\n");
        throw std::invalid_argument("output device pointers must not be null");
    }
    if (n_probes <= 0 || n_probes > n_total_clusters) {
        fprintf(stderr, "[ERROR] ivf_search: n_probes - n_probes=%d, n_total_clusters=%d\n",
               n_probes, n_total_clusters);
        throw std::invalid_argument("invalid n_probes");
    }

    IVFLookupContext ctx = {};
    ctx.d_query_batch = d_query_batch;
    ctx.d_cluster_size = d_cluster_size;
    ctx.d_cluster_vectors = d_cluster_vectors;
    ctx.d_cluster_centers = d_cluster_centers;
    ctx.d_initial_indices = d_initial_indices;
    ctx.d_topk_dist = d_topk_dist;
    ctx.d_topk_index = d_topk_index;
    ctx.n_query = n_query;
    ctx.n_dim = n_dim;
    ctx.n_total_clusters = n_total_clusters;
    ctx.n_total_vectors = n_total_vectors;
    ctx.n_probes = n_probes;
    ctx.k = k;
    ctx.distance_mode = distance_mode;
    ctx.h_coarse_index = h_coarse_index;
    ctx.h_coarse_dist = h_coarse_dist;
    ctx.d_reordered_indices = d_reordered_indices;
    ctx.stream = stream;
    ctx.h_cluster_vectors = h_cluster_vectors;
    ctx.h_cluster_sizes = h_cluster_sizes;
    ctx.n_cluster = 0;
    ctx.h_cluster_to_block_offset = nullptr;
    ctx.max_probe_slots = 0;
    ctx.d_initial_indices_internal = nullptr;
    ctx.need_free_initial_indices = false;

    if (d_initial_indices == nullptr) {
        cudaMalloc(&ctx.d_initial_indices_internal, n_query * n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;
        dim3 block(256);
        dim3 grid((n_query * n_total_clusters + block.x - 1) / block.x);
        generate_sequential_indices_kernel<<<grid, block, 0, stream>>>(
            ctx.d_initial_indices_internal, n_query, n_total_clusters);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        ctx.d_initial_indices = ctx.d_initial_indices_internal;
        ctx.need_free_initial_indices = true;
    }

    ivf_search_lookup_stage0(&ctx);
    ivf_search_lookup_stage1(&ctx);
    ivf_search_lookup_expand_and_upload(&ctx);
    ivf_search_lookup_stage2(&ctx);
    ivf_search_lookup_stage3(&ctx);
    ivf_search_lookup_stage4(&ctx);

    if (ctx.need_free_initial_indices && ctx.d_initial_indices_internal != nullptr) {
        cudaFree(ctx.d_initial_indices_internal);
    }
                CHECK_CUDA_ERRORS;
            }

void ivf_search_lookup_blocks(
    float* d_query_batch,
    int* d_block_sizes,
    float* d_block_vectors,
    float* d_cluster_centers,
    int* d_initial_indices,
    float* d_topk_dist,
    int* d_topk_index,
    int n_query,
    int n_dim,
    int n_cluster,
    int n_balanced,
    int n_total_vectors,
    int n_probes,
    int k,
    DistanceType distance_mode,
    const int* h_cluster_to_block_offset,
    const float* h_block_vectors,
    const int* h_block_sizes,
    const int* d_reordered_indices,
    cudaStream_t stream
) {
    if (n_query <= 0 || n_dim <= 0 || n_cluster <= 0 || n_balanced <= 0 || k <= 0) {
        throw std::invalid_argument("invalid ivf_search_lookup_blocks configuration");
    }
    if (!d_block_sizes || !d_block_vectors || !d_cluster_centers || !d_query_batch || !h_cluster_to_block_offset) {
        throw std::invalid_argument("input pointers must not be null");
    }
    if (!d_topk_dist || !d_topk_index) {
        throw std::invalid_argument("output pointers must not be null");
    }

    int max_blocks_per_cluster = 0;
    for (int c = 0; c < n_cluster; ++c) {
        int nb = h_cluster_to_block_offset[c + 1] - h_cluster_to_block_offset[c];
        if (nb > max_blocks_per_cluster) max_blocks_per_cluster = nb;
    }
    int max_probe_slots = n_probes * max_blocks_per_cluster;
    if (max_probe_slots > 1024) max_probe_slots = 1024;

    IVFLookupContext ctx = {};
    ctx.d_query_batch = d_query_batch;
    ctx.d_cluster_size = d_block_sizes;
    ctx.d_cluster_vectors = d_block_vectors;
    ctx.d_cluster_centers = d_cluster_centers;
    ctx.d_initial_indices = d_initial_indices;
    ctx.d_topk_dist = d_topk_dist;
    ctx.d_topk_index = d_topk_index;
    ctx.n_query = n_query;
    ctx.n_dim = n_dim;
    ctx.n_total_clusters = n_balanced;
    ctx.n_total_vectors = n_total_vectors;
    ctx.n_probes = n_probes;
    ctx.k = k;
    ctx.distance_mode = distance_mode;
    ctx.h_coarse_index = nullptr;
    ctx.h_coarse_dist = nullptr;
    ctx.d_reordered_indices = d_reordered_indices;
    ctx.stream = stream;
    ctx.h_cluster_vectors = h_block_vectors;
    ctx.h_cluster_sizes = h_block_sizes;
    ctx.n_cluster = n_cluster;
    ctx.h_cluster_to_block_offset = h_cluster_to_block_offset;
    ctx.max_probe_slots = max_probe_slots;
    ctx.d_initial_indices_internal = nullptr;
    ctx.need_free_initial_indices = false;

    if (d_initial_indices == nullptr) {
        cudaMalloc(&ctx.d_initial_indices_internal, n_query * n_cluster * sizeof(int));
        CHECK_CUDA_ERRORS;
        dim3 block(256);
        dim3 grid((n_query * n_cluster + block.x - 1) / block.x);
        generate_sequential_indices_kernel<<<grid, block, 0, stream>>>(
            ctx.d_initial_indices_internal, n_query, n_cluster);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        ctx.d_initial_indices = ctx.d_initial_indices_internal;
        ctx.need_free_initial_indices = true;
    }

    ivf_search_lookup_stage0(&ctx);
    ivf_search_lookup_stage1(&ctx);
    ivf_search_lookup_expand_and_upload(&ctx);
    ivf_search_lookup_stage2(&ctx);
    ivf_search_lookup_stage3(&ctx);
    ivf_search_lookup_stage4(&ctx);

    if (ctx.need_free_initial_indices && ctx.d_initial_indices_internal != nullptr) {
        cudaFree(ctx.d_initial_indices_internal);
    }
    CHECK_CUDA_ERRORS;
}
