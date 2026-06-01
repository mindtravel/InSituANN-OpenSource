/* ivf_search_separated.cu */

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
#include "l2norm/l2norm.cuh"
#include "utils.cuh"
#include <vector>
#include <algorithm>
#include <cfloat>

using namespace INSITUANN::warpsort_utils;
using namespace INSITUANN::warpsort_topk;

// ---------------------------------------------------------
//  ivfscanbatch.h
//  CUDA  PostgreSQL
// ---------------------------------------------------------

/**
 *  (Dataset & Clusters)
 *  ivfscanbatch.h  IVFIndexContext
 */
struct IVFIndexContext {
    float* d_cluster_vectors;
    float* d_cluster_vector_norm;
    int* d_probe_vector_offset;
    int* d_probe_vector_count;
    float* d_cluster_centers;
    float* d_cluster_centers_norm;
    int n_total_clusters;
    int n_total_vectors;
    int n_dim;
    bool is_initialized;
};

/**
 * BatchStream
 *  ivfscanbatch.h  IVFQueryBatchContext
 */
struct IVFQueryBatchContext {
    void* stream;                    /* cudaStream_t */
    void* data_ready_event;          /* cudaEvent_t */
    void* compute_done_event;         /* cudaEvent_t */
    float* d_queries;
    float* d_query_norm;
    float* d_inner_product;
    float* d_top_nprobe_dist;
    int* d_top_nprobe_index;
    int* d_index_seq;
    int* d_cluster_query_count;
    int* d_cluster_query_offset;
    int* d_cluster_query_data;
    int* d_cluster_query_probe_indices;
    int* d_cluster_write_pos;
    int* d_entry_count_per_cluster;
    int* d_entry_offset;
    int* d_entry_query_offset;
    int* d_entry_cluster_id;
    int* d_entry_query_start;
    int* d_entry_query_count;
    int* d_entry_queries;
    int* d_entry_probe_indices;
    float* d_topk_dist_candidate;
    int* d_topk_index_candidate;
    float* d_topk_dist;
    int* d_topk_index;
    int max_n_query;
    int n_dim;
    int max_n_probes;
    int max_k;
    int n_total_clusters;
};

//  void*  CUDA
static inline cudaStream_t get_stream(IVFQueryBatchContext* ctx) {
    return static_cast<cudaStream_t>(ctx->stream);
}

static inline cudaEvent_t get_data_ready_event(IVFQueryBatchContext* ctx) {
    return static_cast<cudaEvent_t>(ctx->data_ready_event);
}

static inline cudaEvent_t get_compute_done_event(IVFQueryBatchContext* ctx) {
    return static_cast<cudaEvent_t>(ctx->compute_done_event);
}

// ---------------------------------------------------------
// 1. Index  ()
// ---------------------------------------------------------

void* ivf_create_index_context() {
    return new IVFIndexContext{0};
}

void ivf_destroy_index_context(void* ctx_ptr) {
    if (!ctx_ptr) return;
    IVFIndexContext* ctx = (IVFIndexContext*)ctx_ptr;

    //
    if (ctx->d_cluster_vectors) cudaFree(ctx->d_cluster_vectors);
    if (ctx->d_cluster_vector_norm) cudaFree(ctx->d_cluster_vector_norm);
    if (ctx->d_probe_vector_offset) cudaFree(ctx->d_probe_vector_offset);
    if (ctx->d_probe_vector_count) cudaFree(ctx->d_probe_vector_count);
    if (ctx->d_cluster_centers) cudaFree(ctx->d_cluster_centers);
    if (ctx->d_cluster_centers_norm) cudaFree(ctx->d_cluster_centers_norm);

    delete ctx;
}

//  GPU ( Stage 0)
int ivf_load_dataset(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;
    if (ctx->is_initialized) return 0; //

    ctx->n_total_clusters = n_total_clusters;
    ctx->n_total_vectors = n_total_vectors;
    ctx->n_dim = n_dim;

    // 1.
    cudaMalloc(&ctx->d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_cluster_vector_norm, n_total_vectors * sizeof(float));
    cudaMalloc(&ctx->d_cluster_centers_norm, n_total_clusters * sizeof(float));

    // 2.  device
    ctx->d_cluster_vectors = d_cluster_vectors;
    ctx->d_probe_vector_count = d_cluster_size;
    ctx->d_cluster_centers = d_cluster_centers;

    // 3. Offset
    compute_prefix_sum(ctx->d_probe_vector_count, ctx->d_probe_vector_offset, n_total_clusters, 0);

    // 4.  Norm
    compute_l2_norm_gpu(ctx->d_cluster_vectors, ctx->d_cluster_vector_norm, n_total_vectors, n_dim, L2NORM_AUTO, 0);
    compute_l2_norm_gpu(ctx->d_cluster_centers, ctx->d_cluster_centers_norm, n_total_clusters, n_dim, L2NORM_AUTO, 0);

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    ctx->is_initialized = true;
    return 1;
}

// ---------------------------------------------------------
//  Build
// ---------------------------------------------------------

/**
 *  GPU
 */
void ivf_init_streaming_upload(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;

    ctx->n_total_clusters = n_total_clusters;
    ctx->n_total_vectors = n_total_vectors;
    ctx->n_dim = n_dim;

    //
    cudaMalloc(&ctx->d_cluster_vectors, (size_t)n_total_vectors * n_dim * sizeof(float));
    cudaMalloc(&ctx->d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_probe_vector_count, n_total_clusters * sizeof(int));

    //  offset  0
    int zero = 0;
    cudaMemcpy(ctx->d_probe_vector_offset, &zero, sizeof(int), cudaMemcpyHostToDevice);

    //
    ctx->is_initialized = false;
    CHECK_CUDA_ERRORS;
}

/**
 *  Cluster  (Append Mode)
 *
 * @param idx_ctx_ptr
 * @param cluster_id  ID
 * @param host_vector_data CPU  = count * dim
 * @param count  cluster
 * @param start_offset_idx  cluster
 */
void ivf_append_cluster_data(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;

    if (count <= 0 || host_vector_data == nullptr) {
        //  cluster count  offset
        int zero = 0;
        cudaMemcpy(ctx->d_probe_vector_count + cluster_id, &zero, sizeof(int), cudaMemcpyHostToDevice);
        // offset[cluster_id]  cluster_id  offset[cluster_id + 1]
        // cluster_id = 0 offset[0]  0
        cudaMemcpy(ctx->d_probe_vector_offset + cluster_id, &start_offset_idx, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    size_t dim_size = sizeof(float) * ctx->n_dim;

    // 1.  GPU
    float* d_dest = ctx->d_cluster_vectors + (size_t)start_offset_idx * ctx->n_dim;

    // 2.
    cudaMemcpy(d_dest, host_vector_data, count * dim_size, cudaMemcpyHostToDevice);

    // 3.  (Count  Offset)  GPU
    cudaMemcpy(ctx->d_probe_vector_count + cluster_id, &count, sizeof(int), cudaMemcpyHostToDevice);
    // offset[cluster_id]  cluster_id  offset[cluster_id + 1]
    // cluster_id = 0 offset[0]  0
    cudaMemcpy(ctx->d_probe_vector_offset + cluster_id, &start_offset_idx, sizeof(int), cudaMemcpyHostToDevice);

    CHECK_CUDA_ERRORS;
}

/**
 *  Offset Norm
 *
 * @param idx_ctx_ptr
 * @param center_data_flat  = n_total_clusters * dim
 * @param total_vectors_check
 */
void ivf_finalize_streaming_upload(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;

    //  offset (total count)
    cudaMemcpy(ctx->d_probe_vector_offset + ctx->n_total_clusters, &total_vectors_check, sizeof(int), cudaMemcpyHostToDevice);

    //
    cudaMalloc(&ctx->d_cluster_centers, ctx->n_total_clusters * ctx->n_dim * sizeof(float));
    cudaMemcpy(ctx->d_cluster_centers, center_data_flat, ctx->n_total_clusters * ctx->n_dim * sizeof(float), cudaMemcpyHostToDevice);

    //  Norm
    cudaMalloc(&ctx->d_cluster_vector_norm, ctx->n_total_vectors * sizeof(float));
    cudaMalloc(&ctx->d_cluster_centers_norm, ctx->n_total_clusters * sizeof(float));

    //  Norm GPU
    compute_l2_norm_gpu(ctx->d_cluster_vectors, ctx->d_cluster_vector_norm, ctx->n_total_vectors, ctx->n_dim, L2NORM_AUTO, 0);
    compute_l2_norm_gpu(ctx->d_cluster_centers, ctx->d_cluster_centers_norm, ctx->n_total_clusters, ctx->n_dim, L2NORM_AUTO, 0);

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    ctx->is_initialized = true;
}

// ---------------------------------------------------------
// 2. Query Batch
// ---------------------------------------------------------

void* ivf_create_batch_context(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters) {
    IVFQueryBatchContext* ctx = new IVFQueryBatchContext();

    ctx->max_n_query = max_n_query;
    ctx->n_dim = n_dim;
    ctx->max_n_probes = max_n_probes;
    ctx->max_k = max_k;
    ctx->n_total_clusters = n_total_clusters;

    cudaStream_t stream;
    cudaEvent_t data_ready_event;
    cudaEvent_t compute_done_event;

    cudaStreamCreate(&stream);
    cudaEventCreate(&data_ready_event);
    cudaEventCreate(&compute_done_event);

    ctx->stream = stream;
    ctx->data_ready_event = data_ready_event;
    ctx->compute_done_event = compute_done_event;

    // malloc
    cudaMalloc(&ctx->d_queries, max_n_query * n_dim * sizeof(float));
    cudaMalloc(&ctx->d_query_norm, max_n_query * sizeof(float));

    //
    cudaMalloc(&ctx->d_topk_dist, max_n_query * max_k * sizeof(float));
    cudaMalloc(&ctx->d_topk_index, max_n_query * max_k * sizeof(int));
    cudaMalloc(&ctx->d_top_nprobe_index, max_n_query * max_n_probes * sizeof(int));
    cudaMalloc(&ctx->d_top_nprobe_dist, max_n_query * max_n_probes * sizeof(float));
    cudaMalloc(&ctx->d_inner_product, max_n_query * n_total_clusters * sizeof(float));

    //
    cudaMalloc(&ctx->d_index_seq, max_n_query * n_total_clusters * sizeof(int));

    // Entry
    cudaMalloc(&ctx->d_cluster_query_count, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_entry_count_per_cluster, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_entry_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_cluster_write_pos, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));

    //  query * probes
    size_t max_entries = max_n_query * max_n_probes;
    cudaMalloc(&ctx->d_cluster_query_data, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_cluster_query_probe_indices, max_entries * sizeof(int));

    // Entry Arrays
    cudaMalloc(&ctx->d_entry_cluster_id, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_start, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_count, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_queries, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_probe_indices, max_entries * sizeof(int));

    // Fine Search Candidates
    cudaMalloc(&ctx->d_topk_dist_candidate, max_n_query * max_n_probes * max_k * sizeof(float));
    cudaMalloc(&ctx->d_topk_index_candidate, max_n_query * max_n_probes * max_k * sizeof(int));

    CHECK_CUDA_ERRORS;

    return ctx;
}

void ivf_destroy_batch_context(void* ctx_ptr) {
    if(!ctx_ptr) return;
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)ctx_ptr;

    cudaStreamDestroy(get_stream(ctx));
    cudaEventDestroy(get_data_ready_event(ctx));
    cudaEventDestroy(get_compute_done_event(ctx));

    cudaFree(ctx->d_queries);
    cudaFree(ctx->d_query_norm);
    cudaFree(ctx->d_topk_dist);
    cudaFree(ctx->d_topk_index);
    cudaFree(ctx->d_top_nprobe_index);
    cudaFree(ctx->d_top_nprobe_dist);
    cudaFree(ctx->d_inner_product);
    cudaFree(ctx->d_index_seq);

    cudaFree(ctx->d_cluster_query_count);
    cudaFree(ctx->d_cluster_query_offset);
    cudaFree(ctx->d_entry_count_per_cluster);
    cudaFree(ctx->d_entry_offset);
    cudaFree(ctx->d_cluster_write_pos);
    cudaFree(ctx->d_entry_query_offset);

    cudaFree(ctx->d_cluster_query_data);
    cudaFree(ctx->d_cluster_query_probe_indices);
    cudaFree(ctx->d_entry_cluster_id);
    cudaFree(ctx->d_entry_query_start);
    cudaFree(ctx->d_entry_query_count);
    cudaFree(ctx->d_entry_queries);
    cudaFree(ctx->d_entry_probe_indices);
    cudaFree(ctx->d_topk_dist_candidate);
    cudaFree(ctx->d_topk_index_candidate);

    delete ctx;
}

// ---------------------------------------------------------
// 3. Pipeline
// ---------------------------------------------------------

/**
 *  1:  (Preprocessing)
 * QueryGPUNorm
 * DMAGPU Compute
 */
void ivf_pipeline_stage1_prepare(
    void* batch_ctx_ptr,
    float* query_batch_host, //  [n_query * n_dim]
    int n_query
) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;

    if (n_query > ctx->max_n_query) {
        fprintf(stderr, "[Error] Batch size exceeds capacity: %d > %d\n", n_query, ctx->max_n_query);
        return;
    }

    cudaStream_t stream = get_stream(ctx);

    // 1.  Query
    cudaMemcpyAsync(ctx->d_queries, query_batch_host,
                    n_query * ctx->n_dim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 2.  Query Norm
    compute_l2_norm_gpu(ctx->d_queries, ctx->d_query_norm, n_query, ctx->n_dim, L2NORM_AUTO, stream);

    // 3.
    cudaEventRecord(get_data_ready_event(ctx), stream);
}

/**
 *  2:  (Compute)
 *
 *
 */
void ivf_pipeline_stage2_compute(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
) {
    /*  */
    if (!batch_ctx_ptr) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: batch_ctx_ptr is NULL");
    }
    if (!idx_ctx_ptr) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: idx_ctx_ptr is NULL");
    }

    IVFQueryBatchContext* q_ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    IVFIndexContext* idx_ctx = (IVFIndexContext*)idx_ctx_ptr;

    /*  */
    if (!idx_ctx->is_initialized) {
        throw std::runtime_error("ivf_pipeline_stage2_compute:  (is_initialized=false)");
    }

    /*  GPU  */
    if (!idx_ctx->d_cluster_centers) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_centers is NULL");
    }
    if (!idx_ctx->d_cluster_centers_norm) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_centers_norm is NULL");
    }
    if (!idx_ctx->d_cluster_vectors) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_vectors is NULL");
    }
    if (!idx_ctx->d_cluster_vector_norm) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_vector_norm is NULL");
    }
    if (!idx_ctx->d_probe_vector_offset) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_probe_vector_offset is NULL");
    }
    if (!idx_ctx->d_probe_vector_count) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_probe_vector_count is NULL");
    }

    /*  */
    if (idx_ctx->n_total_clusters <= 0) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_total_clusters <= 0");
    }
    if (idx_ctx->n_dim <= 0) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_dim <= 0");
    }
    if (n_probes > idx_ctx->n_total_clusters) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_probes > n_total_clusters");
    }

    /*  CUDA  */
    CHECK_CUDA_ERRORS;

    cudaStream_t stream = get_stream(q_ctx);

    // ---------------- Coarse Search ----------------
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    //  (0,1,2...)
    dim3 queryDim(n_query);
    dim3 block_dim(std::min(idx_ctx->n_total_clusters, 256));
    generate_sequence_indices_kernel<<<queryDim, block_dim, 0, stream>>>(
        q_ctx->d_index_seq, n_query, idx_ctx->n_total_clusters);

    //
    dim3 fill_block(256);
    int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
    dim3 fill_grid(fill_grid_size);
    fill_kernel<<<fill_grid, fill_block, 0, stream>>>(
        q_ctx->d_top_nprobe_dist, FLT_MAX, n_query * n_probes);

    //  Inner Product (Query x Centers)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                idx_ctx->n_total_clusters, n_query, idx_ctx->n_dim,
                &alpha,
                idx_ctx->d_cluster_centers, idx_ctx->n_dim,
                q_ctx->d_queries, idx_ctx->n_dim,
                &beta,
                q_ctx->d_inner_product, idx_ctx->n_total_clusters);

    // TopK Cosine
    INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
        q_ctx->d_query_norm, idx_ctx->d_cluster_centers_norm, q_ctx->d_inner_product, q_ctx->d_index_seq,
        n_query, idx_ctx->n_total_clusters, n_probes,
        q_ctx->d_top_nprobe_dist, q_ctx->d_top_nprobe_index,
        true, stream
    );

    cublasDestroy(handle);

    // ---------------- Build Entry Data ----------------
    //  Cluster Count
    cudaMemsetAsync(q_ctx->d_cluster_query_count, 0, idx_ctx->n_total_clusters * sizeof(int), stream);

    //  Query
    dim3 probeDim(n_probes);
    count_cluster_queries_kernel<<<queryDim, probeDim, 0, stream>>>(
        q_ctx->d_top_nprobe_index, q_ctx->d_cluster_query_count, n_query, n_probes, idx_ctx->n_total_clusters
    );

    //  offset
    compute_prefix_sum(q_ctx->d_cluster_query_count, q_ctx->d_cluster_query_offset, idx_ctx->n_total_clusters, stream);

    //  Cluster -> Query
    cudaMemcpyAsync(q_ctx->d_cluster_write_pos, q_ctx->d_cluster_query_offset,
                    idx_ctx->n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    build_cluster_query_mapping_kernel<<<queryDim, probeDim, 0, stream>>>(
        q_ctx->d_top_nprobe_index, q_ctx->d_cluster_query_offset,
        q_ctx->d_cluster_query_data, q_ctx->d_cluster_query_probe_indices,
        q_ctx->d_cluster_write_pos, n_query, n_probes, idx_ctx->n_total_clusters
    );

    //  Entry
    constexpr int kQueriesPerBlock = 8;
    dim3 clusterDim(idx_ctx->n_total_clusters);
    dim3 blockDim_entry(1);
    count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
        q_ctx->d_cluster_query_offset, q_ctx->d_entry_count_per_cluster, idx_ctx->n_total_clusters, kQueriesPerBlock
    );

    compute_prefix_sum(q_ctx->d_entry_count_per_cluster, q_ctx->d_entry_offset, idx_ctx->n_total_clusters, stream);

    //  Entry
    int n_entry = 0;
    cudaMemcpyAsync(&n_entry, q_ctx->d_entry_offset + idx_ctx->n_total_clusters, sizeof(int), cudaMemcpyDeviceToHost, stream);

    //  n_entry Host
    cudaStreamSynchronize(stream);

    if (n_entry > 0) {
        //  Entry
        cudaMemcpyAsync(q_ctx->d_entry_query_offset, q_ctx->d_cluster_query_offset,
                        (idx_ctx->n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        build_entry_data_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
            q_ctx->d_cluster_query_offset, q_ctx->d_cluster_query_data, q_ctx->d_cluster_query_probe_indices,
            q_ctx->d_entry_offset, q_ctx->d_entry_query_offset,
            q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
            q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
            idx_ctx->n_total_clusters, kQueriesPerBlock
        );

        // ---------------- Fine Search ----------------
        //
        dim3 init_block(512);
        int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
        dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block, 0, stream>>>(
            q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, n_query * n_probes * k
        );

        //  Kernel
        int capacity = 32;
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);

        dim3 block(kQueriesPerBlock * 32);

        // capacitykernel
        if (distance_mode == COSINE_DISTANCE){
            if (capacity <= 32) {
                launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else if (capacity <= 64) {
                launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else {
                launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            }
        }
        else if(distance_mode == L2_DISTANCE){
            if (capacity <= 32) {
                launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else if (capacity <= 64) {
                launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else {
                launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            }
        }


        // ---------------- Selection & Mapping ----------------
        select_k<float, int>(q_ctx->d_topk_dist_candidate, n_query, n_probes * k, k,
                             q_ctx->d_topk_dist, q_ctx->d_topk_index, true, stream);

        dim3 map_block(256);
        dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
        map_candidate_indices_kernel<<<map_grid, map_block, 0, stream>>>(
            q_ctx->d_topk_index_candidate, q_ctx->d_topk_index, n_query, n_probes, k
        );
    }

    //
    cudaEventRecord(get_compute_done_event(q_ctx), stream);
}

/**
 *  (Download)
 * CPU
 */
void ivf_pipeline_get_results(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    cudaStream_t stream = get_stream(ctx);

    cudaMemcpyAsync(topk_dist, ctx->d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(topk_index, ctx->d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // INSITUANN
    cudaStreamSynchronize(stream);
}

/**
 *  (Wait)
 *  Host
 */
void ivf_pipeline_sync_batch(void* batch_ctx_ptr) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    cudaStream_t stream = get_stream(ctx);
    cudaStreamSynchronize(stream);
}
