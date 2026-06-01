#ifndef FUSION_DIST_TOPK_CUH
#define FUSION_DIST_TOPK_CUH

#include "pch.h"

/**
 * +  topk
 **/
__global__ void fusion_cos_topk_warpsort_kernel(
    float* d_query_norm,
    float* d_data_norm,
    float* d_inner_product,
    int* d_index,
    int* topk_index,
    float* topk_dist,
    int n_query,
    int n_batch,
    int k
);

__global__ void fusion_l2_topk_warpsort_kernel(
    float* d_query_norm,
    float* d_data_norm,
    float* d_inner_product,
    int* d_index,
    int* topk_index,
    float* topk_dist,
    int n_query,
    int n_batch,
    int k
);

/**
 * +  topk
 **/
void cuda_cos_topk_warpsort(
    const float** h_query_vector_group,       /*query*/
    const float** h_data_vector_group,        /*data*/
    int** topk_index,                   /* [n_querys * n_probes]*/
    float** topk_cos_dist,              /* [n_querys * n_probes]*/
    int n_query,                        /*query*/
    int n_batch,                        /*datan_total_clusters*/
    int n_dim,                          /**/
    int k                               /* n_probes*/
);

/**
 * INSITUANN::fusion_dist_topk_warpsort
 * GPU
 */
namespace INSITUANN {
namespace fusion_dist_topk_warpsort {

/**
 *  top-k
 *
 * GPU
 *
 * @param[in] d_query_norm queryL2 [batch_size]
 * @param[in] d_data_norm dataL2 [len]
 * @param[in] d_inner_product  [batch_size, len]
 * @param[in] d_index  [batch_size, len]
 * @param[in] batch_size
 * @param[in] len
 * @param[in] k
 * @param[out] output_vals  top-k  [batch_size, k]
 * @param[out] output_idx  top-k  [batch_size, k]
 * @param[in] select_min  true  k  k
 * @param[in] stream CUDA0
 * @return cudaError_t CUDA
 */
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
);

/**
 *  top-k
 *
 * GPU
 *
 * @param[in] d_query_norm queryL2 [batch_size]
 * @param[in] d_data_norm dataL2 [len]
 * @param[in] d_inner_product  [batch_size, len]
 * @param[in] d_index  [batch_size, len]
 * @param[in] batch_size
 * @param[in] len
 * @param[in] k
 * @param[out] output_vals  top-k  [batch_size, k]
 * @param[out] output_idx  top-k  [batch_size, k]
 * @param[in] select_min  true  k  k
 * @param[in] stream CUDA0
 * @return cudaError_t CUDA
 */
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
);

} // namespace fusion_dist_topk_warpsort
} // namespace INSITUANN

/**
 * kernel
 **/
inline size_t get_shared_memory_size(int k) {
    return (k * sizeof(float) + k * sizeof(int));  /*  +  */
}

/**
 * Entry-basedtop-k
 *
 *
 * -  block  entry cluster +  query84
 * - grid = n_entryqueryclusterblock
 * - max_queries_per_probegridentry
 * - cluster
 *
 * @param d_query_group query [n_query * n_dim]
 * @param d_cluster_vector [n_total_vectors * n_dim]
 * @param d_probe_vector_offset probed_cluster_vector [n_probes]
 * @param d_probe_vector_count probe [n_probes]
 * @param d_probe_queries probequeryCSR[total_queries]
 * @param d_probe_query_offsets probequeryCSR[n_probes + 1]
 * @param d_probe_query_probe_indices probe-queryprobequery [total_queries_in_probes]
 * @param d_query_norm queryl2norm [n_query]
 * @param d_cluster_vector_norm l2norm [n_total_vectors]
 * @param d_topk_index [out] querytopk [n_query][k]
 * @param d_topk_dist [out] querytopk [n_query][k]
 *
 * @param n_query query
 * @param n_total_clusters cluster
 * @param n_probes probe
 * @param n_dim
 * @param k topk
 */
void cuda_cos_topk_warpsort_fine(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_probe_vector_offset,
    int* d_probe_vector_count,
    int* d_probe_queries,
    int* d_probe_query_offsets,
    int* d_probe_query_probe_indices,
    float* d_query_norm,
    float* d_cluster_vector_norm,
    int* d_topk_index,
    float* d_topk_dist,

    float** candidate_dist,
    int** candidate_index,

    int n_query,
    int n_total_clusters,
    int n_probes,
    int n_dim,
    int k
);
#endif