#ifndef INSITUANN_CUDA_IVF_SEARCH_CUH
#define INSITUANN_CUDA_IVF_SEARCH_CUH

#include "pch.h"

/**
 *
 *
 * 6GGPU
 *
 *
 * @param cluster_size           cluster  n_total_cluster
 * @param cluster_vectors        cluster  [n_total_cluster]
 * @param cluster_center_data    [n_total_cluster]
 * @param n_total_clusters
 * @param n_total_vectors
 * @param n_dim
 * @return true false
 */
bool initialize_persistent_data(
    int* cluster_size,
    float*** cluster_vectors,
    float** cluster_center_data,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 *
 */
void cleanup_persistent_data();

/**
 *  cluster
 *
 * @param d_query_batch         device query
 * @param d_cluster_size         device  cluster
 * @param d_cluster_vectors      device  cluster
 * @param d_cluster_centers      device
 * @param d_initial_indices      device nullptr
 * @param d_topk_dist            device  top-k
 * @param d_topk_index           device  top-k cluster
 * @param n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k, distance_mode
 * @param h_coarse_index, h_coarse_dist
 */
void ivf_search_no_lookup(
    float* d_query_batch,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int* d_initial_indices,
    float* d_topk_dist,
    int* d_topk_index,
    int n_query,
    int n_dim,
    int n_total_cluster,
    int n_total_vectors,
    int n_probes,
    int k,
    int distance_mode,
    int** h_coarse_index = nullptr,
    float** h_coarse_dist = nullptr
);

/**
 *
 *
 * @param d_query_batch         device query
 * @param d_cluster_size         device  cluster
 * @param d_cluster_vectors      device  cluster
 * @param d_cluster_centers      device
 * @param d_initial_indices      device nullptr
 * @param d_topk_dist            device  top-k
 * @param d_topk_index           device  top-k  d_reordered_indices
 * @param n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k, distance_mode
 * @param h_coarse_index, h_coarse_dist
 * @param d_reordered_indices    [n_total_vectors]
 * @param stream, h_cluster_vectors, h_cluster_sizes
 */
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
    int n_total_cluster,
    int n_total_vectors,
    int n_probes,
    int k,
    DistanceType distance_mode,
    int** h_coarse_index = nullptr,  // [n_query, n_probes] host
    float** h_coarse_dist = nullptr,  // [n_query, n_probes] host
    const int* d_reordered_indices = nullptr,  // [n_total_vectors]
    cudaStream_t stream = 0,  // 0 =
    const float* h_cluster_vectors = nullptr,  // host  cluster
    const int* h_cluster_sizes = nullptr       // host  cluster  [n_total_cluster] h_cluster_vectors
);

/**
 * Block  lookup cluster block
 *  clusterblock block  sizes/vectors/centers
 */
void ivf_search_lookup_blocks(
    float* d_query_batch,
    int* d_block_sizes,         // [n_balanced]  block
    float* d_block_vectors,     // block
    float* d_cluster_centers,   // [n_cluster, n_dim]
    int* d_initial_indices,
    float* d_topk_dist,
    int* d_topk_index,
    int n_query,
    int n_dim,
    int n_cluster,              //  cluster
    int n_balanced,             // block
    int n_total_vectors,
    int n_probes,
    int k,
    DistanceType distance_mode,
    const int* h_cluster_to_block_offset,  // [n_cluster+1]
    const float* h_block_vectors,
    const int* h_block_sizes,
    const int* d_reordered_indices,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------
// Lookup
// ---------------------------------------------------------

/**  stage  Stage1  cluster  Stage2/3/4 */
struct IVFLookupContext {
    float* d_query_batch;
    int* d_cluster_size;       /* cluster  cluster block  block  */
    float* d_cluster_vectors;
    float* d_cluster_centers;   /* cluster [n_total_clusters,dim]block  [n_cluster,dim] */
    int* d_initial_indices;
    float* d_topk_dist;
    int* d_topk_index;
    int n_query, n_dim, n_total_clusters, n_total_vectors, n_probes, k;
    DistanceType distance_mode;
    int** h_coarse_index;
    float** h_coarse_dist;
    const int* d_reordered_indices;
    cudaStream_t stream;
    const float* h_cluster_vectors;
    const int* h_cluster_sizes;
    /* block  cluster clusterblock  [n_cluster+1]max_probe_slots  query  block  */
    int n_cluster;
    const int* h_cluster_to_block_offset;
    int max_probe_slots;
    /* Stage0  */
    int* d_probe_vector_offset;
    float* d_cluster_vector_norm;
    float* d_query_norm;
    float* d_cluster_centers_norm;
    int* d_top_nprobe_index;    /* cluster =cluster idblock  expand =block id */
    int* d_initial_indices_internal;
    bool need_free_initial_indices;
    /* Stage2  */
    int* d_cluster_query_offset;
    int* d_cluster_query_data;
    int* d_cluster_query_probe_indices;
    int* d_entry_cluster_id;
    int* d_entry_query_start;
    int* d_entry_query_count;
    int* d_entry_queries;
    int* d_entry_probe_indices;
    int n_entry;
    /* - (2BCS)stream_uploadprobe block  */
    int* d_entry_offset;           /* [n_total_clusters+1]  block  entry Stage2  */
    cudaStream_t stream_upload;    /*  stream  */
    float* d_double_buffer[2];    /* 2BCSn_dim  */
    float* d_double_buffer_norm[2];/* 2BCS  norm */
    int* d_probe_offset_block;     /* [0, block_size]  block  */
    int* d_probe_count_block;     /* [block_size]  block  */
    int* d_entry_cluster_id_zero;  /*  0 block kernel  */
    int* h_probe_block_ids;        /* probe  block id  free */
    int n_probe_blocks;
    int max_block_size;            /* BCS  max(cluster_size) */
};

void ivf_search_lookup_stage0(IVFLookupContext* ctx);
void ivf_search_lookup_stage1(IVFLookupContext* ctx);
/** Stage1 block  cluster id  block idcluster  probe  cluster */
void ivf_search_lookup_expand_and_upload(IVFLookupContext* ctx);
void ivf_search_lookup_stage2(IVFLookupContext* ctx);
void ivf_search_lookup_stage3(IVFLookupContext* ctx);
void ivf_search_lookup_stage4(IVFLookupContext* ctx);

// ---------------------------------------------------------
//
// ---------------------------------------------------------

/**
 *
 *
 * @return
 */
extern "C" void* ivf_create_index_context();

/**
 *
 *
 * @param ctx_ptr
 */
extern "C" void ivf_destroy_index_context(void* ctx_ptr);

/**
 *  GPUStage 0
 *
 *  device
 *
 * @param idx_ctx_ptr
 * @param d_cluster_size device  cluster  [n_total_clusters]
 * @param d_cluster_vectors device  cluster  [n_total_vectors * n_dim]
 * @param d_cluster_centers device  [n_total_clusters * n_dim]
 * @param n_total_clusters
 * @param n_total_vectors
 * @param n_dim
 * @return 1 0
 */
extern "C" int ivf_load_dataset(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 * BatchStream
 *
 * BufferContext
 *
 * @param max_n_query
 * @param n_dim
 * @param max_n_probes probe
 * @param max_k top-k
 * @param n_total_clusters
 * @return
 */
extern "C" void* ivf_create_batch_context(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters);

/**
 *
 *
 * @param ctx_ptr
 */
extern "C" void ivf_destroy_batch_context(void* ctx_ptr);

/**
 *  1:  (Preprocessing)
 *
 * QueryGPUNorm
 * DMAGPU Compute
 *
 * @param batch_ctx_ptr
 * @param query_batch_host CPU query  [n_query * n_dim]
 * @param n_query
 */
extern "C" void ivf_pipeline_stage1_prepare(
    void* batch_ctx_ptr,
    float* query_batch_host,
    int n_query
);

/**
 *  2:  (Compute)
 *
 *
 *
 *
 * @param batch_ctx_ptr
 * @param idx_ctx_ptr
 * @param n_query
 * @param n_probes querycluster
 * @param k top-k
 */
extern "C" void ivf_pipeline_stage2_compute(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
);

/**
 *  (Download)
 *
 * CPU
 *
 * @param batch_ctx_ptr
 * @param topk_dist CPU  top-k  [n_query * k]
 * @param topk_index CPU  top-k  [n_query * k]
 * @param n_query
 * @param k top-k
 */
extern "C" void ivf_pipeline_get_results(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
);

/**
 *  (Wait)
 *
 *  Host
 *
 * @param batch_ctx_ptr
 */
extern "C" void ivf_pipeline_sync_batch(void* batch_ctx_ptr);

/**
 *  Build
 */

/**
 *  GPU
 */
extern void ivf_init_streaming_upload(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 *  Cluster  (Append Mode)
 */
extern void ivf_append_cluster_data(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
);

/**
 *  Offset Norm
 */
extern void ivf_finalize_streaming_upload(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
);

#endif  // INSITUANN_CUDA_IVF_SEARCH_CUH
