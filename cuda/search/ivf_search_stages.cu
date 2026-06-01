/*  limits.h Thrust  CHAR_MIN  */
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

#if defined(USE_NVTX) && USE_NVTX
#include <nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP()      ((void)0)
#endif

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define ENABLE_CUDA_TIMING 0
#define IVF_VALIDATE_BCS_MAPPING 1  /*  0  cluster-block  */
#define IVF_FORCE_SERIAL_PC 0      /*  1  producer-consumer  */
#define IVF_FORCE_ONESHOT_BCS 0    /*  1  producer-consumer */
#define IVF_PC_DEBUG_BUFFER 0     /*  1  PC  buffer  */
#define IVF_PC_DEBUG_DUP_ANALYSIS 1  /*  1  block */

using namespace INSITUANN::warpsort_utils;
using namespace INSITUANN::warpsort_topk;

#if IVF_PC_DEBUG_DUP_ANALYSIS
static std::vector<int> s_debug_slot_block;  /* [n_query * probe_slots]  */
static int s_debug_probe_slots = 0;
#endif

// ---------- Stage0: prefix sumL2 norm ----------
void ivf_search_lookup_stage0(IVFLookupContext* ctx) {
    int n_query = ctx->n_query, n_dim = ctx->n_dim, n_total_clusters = ctx->n_total_clusters;
    int n_total_vectors = ctx->n_total_vectors, n_probes = ctx->n_probes;
    cudaStream_t stream = ctx->stream;
    int* d_cluster_size = ctx->d_cluster_size;
    float* d_cluster_vectors = ctx->d_cluster_vectors;
    float* d_query_batch = ctx->d_query_batch;
    float* d_cluster_centers = ctx->d_cluster_centers;
    bool block_mode = (ctx->n_cluster > 0 && ctx->h_cluster_to_block_offset != nullptr);
    int n_coarse = block_mode ? ctx->n_cluster : n_total_clusters;
    int probe_slots = block_mode ? ctx->max_probe_slots : n_probes;

    ctx->d_probe_vector_offset = nullptr;
    ctx->d_cluster_vector_norm = nullptr;
    ctx->d_query_norm = nullptr;
    ctx->d_cluster_centers_norm = nullptr;
    ctx->d_top_nprobe_index = nullptr;

    NVTX_PUSH("Stage0_Prepare");
    {
        CUDATimer timer("Step 0: Data Preparation", ENABLE_CUDA_TIMING);
        cudaMalloc(&ctx->d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
        compute_prefix_sum(d_cluster_size, ctx->d_probe_vector_offset, n_total_clusters, stream);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;

        cudaMalloc(&ctx->d_cluster_vector_norm, n_total_vectors * sizeof(float));
        cudaMalloc(&ctx->d_query_norm, n_query * sizeof(float));
        cudaMalloc(&ctx->d_cluster_centers_norm, n_coarse * sizeof(float));
        cudaMalloc(&ctx->d_top_nprobe_index, n_query * probe_slots * sizeof(int));
        CHECK_CUDA_ERRORS;
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;

        compute_l2_norm_gpu(d_query_batch, ctx->d_query_norm, n_query, n_dim, L2NORM_AUTO, stream);
        compute_l2_norm_gpu(d_cluster_centers, ctx->d_cluster_centers_norm, n_coarse, n_dim, L2NORM_AUTO, stream);
        if (ctx->h_cluster_vectors == nullptr || ctx->h_cluster_sizes == nullptr) {
            compute_l2_norm_gpu(d_cluster_vectors, ctx->d_cluster_vector_norm, n_total_vectors, n_dim, L2NORM_AUTO, stream);
        }
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
    }
    NVTX_POP();
}

// ----------  clusterblock / ----------
void ivf_search_lookup_expand_and_upload(IVFLookupContext* ctx) {
    const int n_query = ctx->n_query;
    const int n_probes = ctx->n_probes;
    const int n_dim = ctx->n_dim;
    const int n_total_clusters = ctx->n_total_clusters;
    cudaStream_t stream = ctx->stream;
    bool block_mode = (ctx->n_cluster > 0 && ctx->h_cluster_to_block_offset != nullptr);

    /*  producer-consumer  */
    ctx->d_entry_offset = nullptr;
    ctx->stream_upload = nullptr;
    ctx->d_double_buffer[0] = ctx->d_double_buffer[1] = nullptr;
    ctx->d_double_buffer_norm[0] = ctx->d_double_buffer_norm[1] = nullptr;
    ctx->d_probe_offset_block = ctx->d_probe_count_block = nullptr;
    ctx->d_entry_cluster_id_zero = nullptr;
    ctx->h_probe_block_ids = nullptr;
    ctx->n_probe_blocks = 0;
    ctx->max_block_size = 0;

    if (block_mode) {
        std::vector<int> h_cluster_ids(n_query * n_probes);
        cudaMemcpy(h_cluster_ids.data(), ctx->d_top_nprobe_index, n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;

        std::vector<int> h_block_ids(n_query * ctx->max_probe_slots, -1);
        for (int q = 0; q < n_query; ++q) {
            int slot = 0;
            for (int p = 0; p < n_probes && slot < ctx->max_probe_slots; ++p) {
                int cid = h_cluster_ids[q * n_probes + p];
                if (cid < 0 || cid >= ctx->n_cluster) continue;
                int b_start = ctx->h_cluster_to_block_offset[cid];
                int b_end = ctx->h_cluster_to_block_offset[cid + 1];
                for (int b = b_start; b < b_end && slot < ctx->max_probe_slots; ++b, ++slot)
                    h_block_ids[q * ctx->max_probe_slots + slot] = b;
            }
        }
#if IVF_VALIDATE_BCS_MAPPING
        /* Check 2:  query  block id  */
        for (int q = 0; q < n_query; ++q) {
            std::unordered_set<int> seen;
            for (int s = 0; s < ctx->max_probe_slots; ++s) {
                int bid = h_block_ids[q * ctx->max_probe_slots + s];
                if (bid < 0) break;
                if (!seen.insert(bid).second) {
                    throw std::runtime_error("[IVF BCS] expand: duplicate block_id " + std::to_string(bid) + " for query " + std::to_string(q));
                }
            }
        }
        /* Check 3: clusterblock  */
        const int* off = ctx->h_cluster_to_block_offset;
        if (off[0] != 0 || off[ctx->n_cluster] != n_total_clusters) {
            throw std::runtime_error("[IVF BCS] cluster_to_block_offset: invalid bounds");
        }
        for (int c = 0; c < ctx->n_cluster; ++c) {
            if (off[c + 1] < off[c] || off[c] < 0 || off[c + 1] > n_total_clusters) {
                throw std::runtime_error("[IVF BCS] cluster_to_block_offset: invalid range for cluster " + std::to_string(c));
            }
        }
#endif
        cudaMemcpy(ctx->d_top_nprobe_index, h_block_ids.data(), n_query * ctx->max_probe_slots * sizeof(int), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;
    }

    if (ctx->h_cluster_vectors == nullptr || ctx->h_cluster_sizes == nullptr) return;

    int n_units = n_total_clusters;
    std::vector<int> h_offsets(n_units + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < n_units; ++i)
        h_offsets[i + 1] = h_offsets[i] + ctx->h_cluster_sizes[i];

    int n_slots = block_mode ? ctx->max_probe_slots : n_probes;
    std::vector<int> h_probe_ids(n_query * n_slots);
    cudaMemcpy(h_probe_ids.data(), ctx->d_top_nprobe_index, n_query * n_slots * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    std::unordered_set<int> probe_ids;
    int max_block = 0;
    for (int i = 0; i < n_query * n_slots; ++i) {
        int id = h_probe_ids[i];
        if (id >= 0 && id < n_units) {
            probe_ids.insert(id);
            int sz = ctx->h_cluster_sizes[id];
            if (sz > max_block) max_block = sz;
        }
    }
    ctx->max_block_size = max_block;

    /* -2  BCS  buffer 2  probe block */
    bool use_pc = (probe_ids.size() >= 2 && max_block > 0 && max_block <= 4096);
#if IVF_FORCE_ONESHOT_BCS
    use_pc = false;  /*  producer-consumer  */
    fprintf(stderr, "[IVF BCS] IVF_FORCE_ONESHOT_BCS=1: using one-shot upload (no producer-consumer)\n");
#endif
    if (use_pc) {
        std::vector<int> bid_vec(probe_ids.begin(), probe_ids.end());
        std::sort(bid_vec.begin(), bid_vec.end());
        ctx->n_probe_blocks = static_cast<int>(bid_vec.size());
        ctx->h_probe_block_ids = (int*)malloc(ctx->n_probe_blocks * sizeof(int));
        for (int i = 0; i < ctx->n_probe_blocks; ++i)
            ctx->h_probe_block_ids[i] = bid_vec[i];
        cudaStreamCreate(&ctx->stream_upload);
        size_t buf_floats = (size_t)max_block * n_dim;
        size_t buf_norm = (size_t)max_block;
        cudaMalloc(&ctx->d_double_buffer[0], buf_floats * sizeof(float));
        cudaMalloc(&ctx->d_double_buffer[1], buf_floats * sizeof(float));
        cudaMalloc(&ctx->d_double_buffer_norm[0], buf_norm * sizeof(float));
        cudaMalloc(&ctx->d_double_buffer_norm[1], buf_norm * sizeof(float));
        cudaMalloc(&ctx->d_probe_offset_block, 2 * sizeof(int));
        cudaMalloc(&ctx->d_probe_count_block, sizeof(int));
        return;  /*  Stage3  producer-consumer d_entry_cluster_id_zero  Stage3  */
    }

    /*  */
    for (int bid : probe_ids) {
        const int start_vec = h_offsets[bid];
        const int count_vec = ctx->h_cluster_sizes[bid];
        if (count_vec <= 0) continue;
        const size_t num_floats = (size_t)count_vec * n_dim;
        cudaMemcpyAsync(
            ctx->d_cluster_vectors + (size_t)start_vec * n_dim,
            ctx->h_cluster_vectors + (size_t)start_vec * n_dim,
            num_floats * sizeof(float),
            cudaMemcpyHostToDevice,
            stream);
        CHECK_CUDA_ERRORS;
        compute_l2_norm_gpu(
            ctx->d_cluster_vectors + (size_t)start_vec * n_dim,
            ctx->d_cluster_vector_norm + start_vec,
            count_vec, n_dim, L2NORM_AUTO, stream);
    }
    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;
}

// ---------- Stage1: Sgemm + warpsort topk ----------
void ivf_search_lookup_stage1(IVFLookupContext* ctx) {
    int n_query = ctx->n_query, n_dim = ctx->n_dim, n_probes = ctx->n_probes;
    cudaStream_t stream = ctx->stream;
    float* d_query_batch = ctx->d_query_batch;
    float* d_cluster_centers = ctx->d_cluster_centers;
    float* d_inner_product = nullptr;
    float* d_top_nprobe_dist = nullptr;
    bool block_mode = (ctx->n_cluster > 0 && ctx->h_cluster_to_block_offset != nullptr);
    int n_coarse = block_mode ? ctx->n_cluster : ctx->n_total_clusters;

    NVTX_PUSH("Stage1_Coarse");
    {
        CUDATimer timer("Step 1: Coarse Search (cuda_cos_topk_warpsort)", ENABLE_CUDA_TIMING);
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasHandle_t handle;
        {
            CUDATimer timer_alloc("Step 1: GPU Memory Allocation", ENABLE_CUDA_TIMING);
            cudaMalloc(&d_inner_product, n_query * n_coarse * sizeof(float));
            cudaMalloc(&d_top_nprobe_dist, n_query * n_probes * sizeof(float));
            cublasCreate(&handle);
            cublasSetStream(handle, stream);
        }
        {
            dim3 fill_block(256);
            dim3 fill_grid((n_query * n_probes + fill_block.x - 1) / fill_block.x);
            fill_kernel<<<fill_grid, fill_block, 0, stream>>>(
                d_top_nprobe_dist, FLT_MAX, n_query * n_probes);
        }
        {
            CUDATimer timer_gemm("Step 1: Kernel Execution: matrix multiply", ENABLE_CUDA_TIMING);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n_coarse, n_query, n_dim,
                &alpha, d_cluster_centers, n_dim, d_query_batch, n_dim,
                &beta, d_inner_product, n_coarse);
            cudaStreamSynchronize(stream);
        }
        {
            CUDATimer timer_topk("Step 1: Kernel Execution: cos + topk", ENABLE_CUDA_TIMING);
            if (ctx->distance_mode == COSINE_DISTANCE) {
                INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                    ctx->d_query_norm, ctx->d_cluster_centers_norm, d_inner_product, ctx->d_initial_indices,
                    n_query, n_coarse, n_probes,
                    d_top_nprobe_dist, ctx->d_top_nprobe_index, true, stream);
            } else if (ctx->distance_mode == L2_DISTANCE) {
                INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                    ctx->d_query_norm, ctx->d_cluster_centers_norm, d_inner_product, ctx->d_initial_indices,
                    n_query, n_coarse, n_probes,
                    d_top_nprobe_dist, ctx->d_top_nprobe_index, true, stream);
            }
            cudaStreamSynchronize(stream);
            CHECK_CUDA_ERRORS;
            if (ctx->h_coarse_index != nullptr && ctx->h_coarse_dist != nullptr) {
                cudaMemcpyAsync(ctx->h_coarse_index[0], ctx->d_top_nprobe_index,
                    n_query * n_probes * sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(ctx->h_coarse_dist[0], d_top_nprobe_dist,
                    n_query * n_probes * sizeof(float), cudaMemcpyDeviceToHost, stream);
                CHECK_CUDA_ERRORS;
            }
        }
        {
            cublasDestroy(handle);
            cudaFree(d_inner_product);
            cudaFree(ctx->d_cluster_centers_norm);
            ctx->d_cluster_centers_norm = nullptr;
            if (ctx->h_coarse_index == nullptr || ctx->h_coarse_dist == nullptr) {
                cudaFree(d_top_nprobe_dist);
            }
        }
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
    }
    NVTX_POP();
}

// ---------- Stage2:  entry querycluster/block  entry ----------
void ivf_search_lookup_stage2(IVFLookupContext* ctx) {
    constexpr int kQueriesPerBlock = 8;
    int n_query = ctx->n_query, n_total_clusters = ctx->n_total_clusters, n_probes = ctx->n_probes;
    cudaStream_t stream = ctx->stream;
    bool block_mode = (ctx->n_cluster > 0 && ctx->h_cluster_to_block_offset != nullptr);
    int probe_slots = block_mode ? ctx->max_probe_slots : n_probes;
    dim3 queryDim(n_query);
    dim3 probeDim(probe_slots);

    ctx->d_cluster_query_offset = nullptr;
    ctx->d_cluster_query_data = nullptr;
    ctx->d_cluster_query_probe_indices = nullptr;
    ctx->d_entry_cluster_id = nullptr;
    ctx->d_entry_query_start = nullptr;
    ctx->d_entry_query_count = nullptr;
    ctx->d_entry_queries = nullptr;
    ctx->d_entry_probe_indices = nullptr;
    ctx->n_entry = 0;

    NVTX_PUSH("Stage2_Entry");
    {
        CUDATimer timer("Step 2: Build entry data (GPU)", ENABLE_CUDA_TIMING);
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemset(d_cluster_query_count, 0, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;
        count_cluster_queries_kernel<<<queryDim, probeDim, 0, stream>>>(
            ctx->d_top_nprobe_index, d_cluster_query_count, n_query, probe_slots, n_total_clusters);
        CHECK_CUDA_ERRORS;
        cudaMalloc(&ctx->d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;
        compute_prefix_sum(d_cluster_query_count, ctx->d_cluster_query_offset, n_total_clusters, stream);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        int total_entries = 0;
        cudaMemcpy(&total_entries, ctx->d_cluster_query_offset + n_total_clusters, sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpyAsync(d_cluster_write_pos, ctx->d_cluster_query_offset, n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        CHECK_CUDA_ERRORS;
        cudaMalloc(&ctx->d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&ctx->d_cluster_query_probe_indices, total_entries * sizeof(int));
        CHECK_CUDA_ERRORS;
        build_cluster_query_mapping_kernel<<<queryDim, probeDim, 0, stream>>>(
            ctx->d_top_nprobe_index, ctx->d_cluster_query_offset, ctx->d_cluster_query_data, ctx->d_cluster_query_probe_indices,
            d_cluster_write_pos, n_query, probe_slots, n_total_clusters);
        CHECK_CUDA_ERRORS;
#if IVF_VALIDATE_BCS_MAPPING
        /* Check 4: entry  probe_idx  d_top_nprobe_index  block  slot  */
        if (block_mode && total_entries > 0) {
            cudaStreamSynchronize(stream);
            std::vector<int> h_top_index(n_query * probe_slots);
            std::vector<int> h_query_offset(n_total_clusters + 1);
            std::vector<int> h_query_data(total_entries);
            std::vector<int> h_probe_idx(total_entries);
            cudaMemcpy(h_top_index.data(), ctx->d_top_nprobe_index, n_query * probe_slots * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_query_offset.data(), ctx->d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_query_data.data(), ctx->d_cluster_query_data, total_entries * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_probe_idx.data(), ctx->d_cluster_query_probe_indices, total_entries * sizeof(int), cudaMemcpyDeviceToHost);
            for (int bid = 0; bid < n_total_clusters; ++bid) {
                int start = h_query_offset[bid], end = h_query_offset[bid + 1];
                for (int i = start; i < end; ++i) {
                    int q = h_query_data[i];
                    int slot = h_probe_idx[i];
                    if (q < 0 || q >= n_query || slot < 0 || slot >= probe_slots) {
                        throw std::runtime_error("[IVF BCS] cluster_query: invalid query_id or probe_idx");
                    }
                    int block_at_slot = h_top_index[q * probe_slots + slot];
                    if (block_at_slot != bid) {
                        throw std::runtime_error("[IVF BCS] cluster_query: block " + std::to_string(bid) + " at slot " + std::to_string(slot) + " for query " + std::to_string(q) + " but d_top_nprobe_index has " + std::to_string(block_at_slot));
                    }
                }
            }
        }
#endif
        dim3 clusterDim(n_total_clusters);
        dim3 blockDim_entry(1);
        int* d_entry_count_per_cluster = nullptr;
        cudaMalloc(&d_entry_count_per_cluster, n_total_clusters * sizeof(int));
        CHECK_CUDA_ERRORS;
        count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
            ctx->d_cluster_query_offset, d_entry_count_per_cluster, n_total_clusters, kQueriesPerBlock);
        CHECK_CUDA_ERRORS;
        int* d_entry_offset = nullptr;
        cudaMalloc(&d_entry_offset, (n_total_clusters + 1) * sizeof(int));
        CHECK_CUDA_ERRORS;
        compute_prefix_sum(d_entry_count_per_cluster, d_entry_offset, n_total_clusters, stream);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        cudaMemcpy(&ctx->n_entry, d_entry_offset + n_total_clusters, sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        int* d_entry_query_offset = nullptr;
        cudaMalloc(&d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMemcpyAsync(d_entry_query_offset, ctx->d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        CHECK_CUDA_ERRORS;
        if (ctx->n_entry > 0) {
            cudaMalloc(&ctx->d_entry_cluster_id, ctx->n_entry * sizeof(int));
            cudaMalloc(&ctx->d_entry_query_start, ctx->n_entry * sizeof(int));
            cudaMalloc(&ctx->d_entry_query_count, ctx->n_entry * sizeof(int));
            cudaMalloc(&ctx->d_entry_queries, total_entries * sizeof(int));
            cudaMalloc(&ctx->d_entry_probe_indices, total_entries * sizeof(int));
            CHECK_CUDA_ERRORS;
            build_entry_data_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
                ctx->d_cluster_query_offset, ctx->d_cluster_query_data, ctx->d_cluster_query_probe_indices,
                d_entry_offset, d_entry_query_offset, ctx->d_entry_cluster_id, ctx->d_entry_query_start,
                ctx->d_entry_query_count, ctx->d_entry_queries, ctx->d_entry_probe_indices, n_total_clusters, kQueriesPerBlock);
            CHECK_CUDA_ERRORS;
        }
        cudaFree(d_entry_query_offset);
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);
        cudaFree(d_entry_count_per_cluster);
        if (ctx->h_probe_block_ids == nullptr) {
            cudaFree(d_entry_offset);
        } else {
            ctx->d_entry_offset = d_entry_offset;  /* producer-consumer  */
#if IVF_PC_DEBUG_DUP_ANALYSIS
            s_debug_probe_slots = probe_slots;
            s_debug_slot_block.resize(n_query * probe_slots);
            cudaMemcpy(s_debug_slot_block.data(), ctx->d_top_nprobe_index, (size_t)n_query * probe_slots * sizeof(int), cudaMemcpyDeviceToHost);
#endif
        }
        cudaFree(ctx->d_top_nprobe_index);
        ctx->d_top_nprobe_index = nullptr;
        CHECK_CUDA_ERRORS;
    }
    NVTX_POP();
}

// ---------- Stage3: entry-based fine search +  topk ----------
void ivf_search_lookup_stage3(IVFLookupContext* ctx) {
    constexpr int kQueriesPerBlock = 8;
    int n_query = ctx->n_query, n_dim = ctx->n_dim, n_probes = ctx->n_probes, k = ctx->k;
    int n_entry = ctx->n_entry;
    cudaStream_t stream = ctx->stream;
    int* d_probe_vector_count = ctx->d_cluster_size;
    bool block_mode = (ctx->n_cluster > 0 && ctx->h_cluster_to_block_offset != nullptr);
    int n_probe_slots = block_mode ? ctx->max_probe_slots : n_probes;
    int n_total_clusters = ctx->n_total_clusters;

    NVTX_PUSH("Stage3_Fine");
    {
        CUDATimer timer("Step 3: Fine Search (v5 entry-based)", ENABLE_CUDA_TIMING);
        int capacity = 32;
        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;
        dim3 block_dim(kQueriesPerBlock * 32);

        {
            CUDATimer timer_init("Init Invalid Values Kernel", ENABLE_CUDA_TIMING);
            while (capacity < k) capacity <<= 1;
            capacity = std::min(capacity, kMaxCapacity);
            CHECK_CUDA_ERRORS;
            cudaMalloc(&d_topk_dist_candidate, n_query * n_probe_slots * k * sizeof(float));
            cudaMalloc(&d_topk_index_candidate, n_query * n_probe_slots * k * sizeof(int));
            dim3 init_block(512);
            dim3 init_grid((n_query * n_probe_slots * k + init_block.x - 1) / init_block.x);
            init_invalid_values_kernel<<<init_grid, init_block, 0, stream>>>(
                d_topk_dist_candidate, d_topk_index_candidate, n_query * n_probe_slots * k);
            CHECK_CUDA_ERRORS;
        }

        /* - */
        if (ctx->h_probe_block_ids != nullptr && ctx->n_probe_blocks > 0 && ctx->d_entry_offset != nullptr) {
            std::vector<int> h_offsets(n_total_clusters + 1);
            h_offsets[0] = 0;
            for (int i = 0; i < n_total_clusters; ++i)
                h_offsets[i + 1] = h_offsets[i] + ctx->h_cluster_sizes[i];
            std::vector<int> h_entry_offset(n_total_clusters + 1);
            cudaMemcpy(h_entry_offset.data(), ctx->d_entry_offset, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERRORS;

            if (ctx->d_entry_cluster_id_zero == nullptr) {
                cudaMalloc(&ctx->d_entry_cluster_id_zero, std::max(n_entry, 1) * sizeof(int));
                cudaMemset(ctx->d_entry_cluster_id_zero, 0, std::max(n_entry, 1) * sizeof(int));
            }
            cudaStream_t stream_up = ctx->stream_upload;
            cudaEvent_t ev_upload, ev_compute[2];
            cudaEventCreate(&ev_upload);
            cudaEventCreate(&ev_compute[0]);
            cudaEventCreate(&ev_compute[1]);

            for (int i = 0; i < ctx->n_probe_blocks; ++i) {
                /*  upload i i>0  */
                if (i > 0) cudaStreamWaitEvent(stream, ev_upload, 0);
                /*  block i  buf i%2  compute i-2 buf i%2  compute i-2  */
                if (i >= 2) cudaStreamWaitEvent(stream_up, ev_compute[(i - 2) % 2], 0);
                int bid = ctx->h_probe_block_ids[i];
                int buf_id = i % 2;
                int block_size = ctx->h_cluster_sizes[bid];
                int entry_start = h_entry_offset[bid];
                int entry_count = h_entry_offset[bid + 1] - entry_start;
                if (block_size <= 0 || entry_count <= 0) continue;

                int base_offset = h_offsets[bid];
                size_t num_floats = (size_t)block_size * n_dim;
                float* d_buf = ctx->d_double_buffer[buf_id];
                float* d_buf_norm = ctx->d_double_buffer_norm[buf_id];

                /*  block i  buffer[buf_id] */
                cudaMemcpyAsync(d_buf, ctx->h_cluster_vectors + (size_t)base_offset * n_dim,
                    num_floats * sizeof(float), cudaMemcpyHostToDevice, stream_up);
                compute_l2_norm_gpu(d_buf, d_buf_norm, block_size, n_dim, L2NORM_AUTO, stream_up);
                cudaEventRecord(ev_upload, stream_up);
                cudaStreamWaitEvent(stream, ev_upload, 0);  /*  */
#if IVF_FORCE_SERIAL_PC
                cudaStreamSynchronize(stream_up);  /*  */
#endif
#if IVF_PC_DEBUG_BUFFER
                {
                    cudaStreamSynchronize(stream_up);
                    const int check_dim = std::min(4, n_dim);
                    std::vector<float> h_buf(check_dim);
                    std::vector<float> h_exp(check_dim);
                    cudaMemcpy(h_buf.data(), d_buf, check_dim * sizeof(float), cudaMemcpyDeviceToHost);
                    for (int d = 0; d < check_dim; ++d)
                        h_exp[d] = ctx->h_cluster_vectors[(size_t)base_offset * n_dim + d];
                    for (int d = 0; d < check_dim; ++d) {
                        if (std::fabs(h_buf[d] - h_exp[d]) > 1e-4f) {
                            fprintf(stderr, "[IVF PC DEBUG] block bid=%d buf_id=%d: d_buf[%d]=%.6f != expected %.6f (base_offset=%d)\n",
                                    bid, buf_id, d, h_buf[d], h_exp[d], base_offset);
                            throw std::runtime_error("[IVF PC] buffer content mismatch - reading wrong block data");
                        }
                    }
                }
#endif

                int h_off[2] = {0, block_size};
                int h_cnt[1] = {block_size};
                cudaMemcpy(ctx->d_probe_offset_block, h_off, 2 * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(ctx->d_probe_count_block, h_cnt, sizeof(int), cudaMemcpyHostToDevice);

                if (ctx->distance_mode == L2_DISTANCE && capacity <= 32) {
                    launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                        block_dim, n_dim, ctx->d_query_batch, d_buf,
                        ctx->d_probe_offset_block, ctx->d_probe_count_block,
                        ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                        ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                        ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                        d_topk_dist_candidate, d_topk_index_candidate, stream);
                } else if (ctx->distance_mode == L2_DISTANCE && capacity <= 64) {
                    launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                        block_dim, n_dim, ctx->d_query_batch, d_buf,
                        ctx->d_probe_offset_block, ctx->d_probe_count_block,
                        ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                        ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                        ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                        d_topk_dist_candidate, d_topk_index_candidate, stream);
                } else if (ctx->distance_mode == L2_DISTANCE) {
                    launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                        block_dim, n_dim, ctx->d_query_batch, d_buf,
                        ctx->d_probe_offset_block, ctx->d_probe_count_block,
                        ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                        ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                        ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                        d_topk_dist_candidate, d_topk_index_candidate, stream);
                } else if (ctx->distance_mode == COSINE_DISTANCE) {
                    if (capacity <= 32)
                        launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                            block_dim, n_dim, ctx->d_query_batch, d_buf,
                            ctx->d_probe_offset_block, ctx->d_probe_count_block,
                            ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                            ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                    else if (capacity <= 64)
                        launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                            block_dim, n_dim, ctx->d_query_batch, d_buf,
                            ctx->d_probe_offset_block, ctx->d_probe_count_block,
                            ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                            ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                    else
                        launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                            block_dim, n_dim, ctx->d_query_batch, d_buf,
                            ctx->d_probe_offset_block, ctx->d_probe_count_block,
                            ctx->d_entry_cluster_id_zero, ctx->d_entry_query_start + entry_start,
                            ctx->d_entry_query_count + entry_start, ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, d_buf_norm, entry_count, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                }
                add_base_offset_to_block_slots_kernel<<<entry_count, 256, 0, stream>>>(
                    ctx->d_entry_queries, ctx->d_entry_probe_indices,
                    ctx->d_entry_query_start, ctx->d_entry_query_count,
                    entry_start, entry_count, base_offset,
                    d_topk_index_candidate, n_probe_slots, k);
#if IVF_FORCE_SERIAL_PC
                cudaStreamSynchronize(stream);  /* +add_base_offset  */
#endif
                /*  upload i+1  buf (i+1)%2 compute i  compute i-1  buffer */
                if (i + 1 < ctx->n_probe_blocks) {
                    int bid_next = ctx->h_probe_block_ids[i + 1];
                    int buf_next = (i + 1) % 2;
                    int block_size_next = ctx->h_cluster_sizes[bid_next];
                    int base_next = h_offsets[bid_next];
                    if (block_size_next > 0) {
                        if (i >= 1) cudaStreamWaitEvent(stream_up, ev_compute[(i - 1) % 2], 0);  /*  compute i-1  buf (i+1)%2 */
                        size_t num_next = (size_t)block_size_next * n_dim;
                        cudaMemcpyAsync(ctx->d_double_buffer[buf_next],
                            ctx->h_cluster_vectors + (size_t)base_next * n_dim,
                            num_next * sizeof(float), cudaMemcpyHostToDevice, stream_up);
                        compute_l2_norm_gpu(ctx->d_double_buffer[buf_next],
                            ctx->d_double_buffer_norm[buf_next], block_size_next, n_dim, L2NORM_AUTO, stream_up);
                        cudaEventRecord(ev_upload, stream_up);
                    }
                }
                cudaEventRecord(ev_compute[i % 2], stream);
            }
            cudaStreamSynchronize(stream_up);
            cudaStreamSynchronize(stream);
            cudaEventDestroy(ev_upload);
            cudaEventDestroy(ev_compute[0]);
            cudaEventDestroy(ev_compute[1]);
        } else {
        /*  */
        if (n_entry != 0) {
                if (ctx->distance_mode == COSINE_DISTANCE) {
                    // capacitykernel
                    if (capacity <= 32) {
                        launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                    } else if (capacity <= 64) {
                        launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                    } else {
                        launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k,
                            d_topk_dist_candidate, d_topk_index_candidate, stream);
                    }
                } else if (ctx->distance_mode == L2_DISTANCE) {
                    // capacitykernel
                    if (capacity <= 32) {
                        launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k, d_topk_dist_candidate, d_topk_index_candidate, stream);
                    } else if (capacity <= 64) {
                        launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k, d_topk_dist_candidate, d_topk_index_candidate, stream);
                    } else {
                        launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                            block_dim, ctx->n_dim,
                            ctx->d_query_batch, ctx->d_cluster_vectors,
                            ctx->d_probe_vector_offset, d_probe_vector_count,
                            ctx->d_entry_cluster_id, ctx->d_entry_query_start, ctx->d_entry_query_count,
                            ctx->d_entry_queries, ctx->d_entry_probe_indices,
                            ctx->d_query_norm, ctx->d_cluster_vector_norm,
                            n_entry, n_probe_slots, k, d_topk_dist_candidate, d_topk_index_candidate, stream);
                    }
                }
            }
            CHECK_CUDA_ERRORS;
        }

        {
            CUDATimer timer_red("Reduce probe results to query top-k", ENABLE_CUDA_TIMING);
#if IVF_VALIDATE_BCS_MAPPING
            /* Check 5: select_k  d_topk_index_candidate  */
            if (block_mode && n_query > 0 && n_probe_slots * k > 0) {
                cudaStreamSynchronize(stream);
                int ncand = n_probe_slots * k;
                std::vector<int> h_idx(n_query * ncand);
                std::vector<float> h_dist(n_query * ncand);
                cudaMemcpy(h_idx.data(), d_topk_index_candidate, (size_t)n_query * ncand * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_dist.data(), d_topk_dist_candidate, (size_t)n_query * ncand * sizeof(float), cudaMemcpyDeviceToHost);
                int dup_count = 0;
                const int max_report = 5;
                for (int q = 0; q < n_query && dup_count < max_report; ++q) {
                    std::unordered_map<int, std::vector<std::tuple<int, int, float>>> idx_to_pos;
                    for (int p = 0; p < ncand; ++p) {
                        int idx = h_idx[q * ncand + p];
                        float d = h_dist[q * ncand + p];
                        /*  padding dist=inf  idx=-1  WarpSort  */
                        if (idx >= 0 && std::isfinite(d)) {
                            int slot = p / k, ki = p % k;
                            idx_to_pos[idx].emplace_back(slot, ki, d);
                        }
                    }
                    for (const auto& kv : idx_to_pos) {
                        if (kv.second.size() > 1) {
                            dup_count++;
                            std::string msg = "[IVF BCS] duplicate reordered index " + std::to_string(kv.first) + " for query " + std::to_string(q) + " at ";
                            for (size_t i = 0; i < kv.second.size(); ++i) {
                                int slot, ki; float d;
                                std::tie(slot, ki, d) = kv.second[i];
                                msg += "(slot=" + std::to_string(slot) + ",ki=" + std::to_string(ki) + ",dist=" + std::to_string(d) + ")" + (i + 1 < kv.second.size() ? "; " : "");
                            }
                            fprintf(stderr, "%s\n", msg.c_str());
#if IVF_PC_DEBUG_DUP_ANALYSIS
                            if (ctx->h_cluster_sizes && s_debug_probe_slots > 0 && !s_debug_slot_block.empty()) {
                                std::vector<long long> ho(ctx->n_total_clusters + 1);
                                ho[0] = 0;
                                for (int b = 0; b < ctx->n_total_clusters; ++b)
                                    ho[b + 1] = ho[b] + ctx->h_cluster_sizes[b];
                                int owner = -1;
                                for (int b = 0; b < ctx->n_total_clusters; ++b) {
                                    if (kv.first >= ho[b] && kv.first < ho[b + 1]) { owner = b; break; }
                                }
                                fprintf(stderr, "  -> index %d belongs to block %d; slot->block: ", kv.first, owner);
                                for (size_t i = 0; i < kv.second.size(); ++i) {
                                    int slot = std::get<0>(kv.second[i]);
                                    int blk = (q * s_debug_probe_slots + slot < (int)s_debug_slot_block.size())
                                        ? s_debug_slot_block[q * s_debug_probe_slots + slot] : -999;
                                    fprintf(stderr, "s%d->b%d%s", slot, blk, (i + 1 < kv.second.size()) ? ", " : "");
                                    if (blk >= 0 && blk != owner)
                                        fprintf(stderr, "[WRONG]");
                                }
                                fprintf(stderr, "\n");
                            }
#endif
                            if (dup_count >= max_report) break;
                        }
                    }
                }
                if (dup_count > 0)
                    fprintf(stderr, "[IVF BCS] found %d+ queries with duplicate indices in d_topk_index_candidate (before select_k)\n", dup_count);
            }
#endif
            select_k<float, int>(
                d_topk_dist_candidate, n_query, n_probe_slots * k, k,
                ctx->d_topk_dist, ctx->d_topk_index, true, stream);
            cudaStreamSynchronize(stream);
            CHECK_CUDA_ERRORS;
            dim3 map_block(256);
            dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
            map_candidate_indices_kernel<<<map_grid, map_block, 0, stream>>>(
                d_topk_index_candidate, ctx->d_topk_index, n_query, n_probe_slots, k);
            CHECK_CUDA_ERRORS;
            cudaFree(d_topk_dist_candidate);
            cudaFree(d_topk_index_candidate);
        }
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        cudaFree(ctx->d_cluster_vector_norm);
        ctx->d_cluster_vector_norm = nullptr;
        CHECK_CUDA_ERRORS;
    }
    NVTX_POP();

    cudaFree(ctx->d_cluster_query_offset);
    cudaFree(ctx->d_cluster_query_data);
    cudaFree(ctx->d_cluster_query_probe_indices);
    if (ctx->d_entry_cluster_id != nullptr) cudaFree(ctx->d_entry_cluster_id);
    if (ctx->d_entry_query_start != nullptr) cudaFree(ctx->d_entry_query_start);
    if (ctx->d_entry_query_count != nullptr) cudaFree(ctx->d_entry_query_count);
    if (ctx->d_entry_queries != nullptr) cudaFree(ctx->d_entry_queries);
    if (ctx->d_entry_probe_indices != nullptr) cudaFree(ctx->d_entry_probe_indices);
    cudaFree(ctx->d_probe_vector_offset);
    cudaFree(ctx->d_query_norm);
    /* producer-consumer  */
    if (ctx->h_probe_block_ids != nullptr) {
        free(ctx->h_probe_block_ids);
        ctx->h_probe_block_ids = nullptr;
    }
    if (ctx->stream_upload != nullptr) {
        cudaStreamDestroy(ctx->stream_upload);
        ctx->stream_upload = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (ctx->d_double_buffer[i] != nullptr) { cudaFree(ctx->d_double_buffer[i]); ctx->d_double_buffer[i] = nullptr; }
        if (ctx->d_double_buffer_norm[i] != nullptr) { cudaFree(ctx->d_double_buffer_norm[i]); ctx->d_double_buffer_norm[i] = nullptr; }
    }
    if (ctx->d_probe_offset_block != nullptr) { cudaFree(ctx->d_probe_offset_block); ctx->d_probe_offset_block = nullptr; }
    if (ctx->d_probe_count_block != nullptr) { cudaFree(ctx->d_probe_count_block); ctx->d_probe_count_block = nullptr; }
    if (ctx->d_entry_cluster_id_zero != nullptr) { cudaFree(ctx->d_entry_cluster_id_zero); ctx->d_entry_cluster_id_zero = nullptr; }
    if (ctx->d_entry_offset != nullptr) { cudaFree(ctx->d_entry_offset); ctx->d_entry_offset = nullptr; }
}

// ---------- Stage4:    ----------
void ivf_search_lookup_stage4(IVFLookupContext* ctx) {
    if (ctx->d_reordered_indices == nullptr) return;
    int n_query = ctx->n_query, k = ctx->k;
    cudaStream_t stream = ctx->stream;

    NVTX_PUSH("Stage4_Lookup");
    {
        CUDATimer timer("Step 4: Lookup original indices", ENABLE_CUDA_TIMING);
        int* d_original_index = nullptr;
        cudaMalloc(&d_original_index, n_query * k * sizeof(int));
        CHECK_CUDA_ERRORS;
        lookup_original_indices_kernel<<<dim3((n_query * k + 255) / 256), dim3(256), 0, stream>>>(
            ctx->d_reordered_indices, ctx->d_topk_index, d_original_index, n_query, k, ctx->n_total_vectors);
        cudaStreamSynchronize(stream);
        CHECK_CUDA_ERRORS;
        cudaMemcpyAsync(ctx->d_topk_index, d_original_index, n_query * k * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        CHECK_CUDA_ERRORS;
        cudaFree(d_original_index);
    }
    NVTX_POP();
}
