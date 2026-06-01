#ifndef IVFTENSOR_CPU_FINE_PQ_RESID_H
#define IVFTENSOR_CPU_FINE_PQ_RESID_H

/**
 * Residual Product Quantization (IVFPQ-style) fine kernel.
 *
 *  fine
 *   Stage 1 (PQ scan)
 *     - per (query, probed cluster c):
 *         q_resid = q - centroids[c]
 *         LUT[m][k] = ||q_resid_seg_m - codebook[m][k]||   (m=0..M-1, k=0..K-1)
 *         for each base vec in cluster c:
 *             pq = pq_codes[vec]   (M bytes)
 *             dist_approx = _m LUT[m][pq[m]]
 *             push_topN(dist_approx, vec)
 *   Stage 2 (rerank)
 *     - per query topN    u8  exact L2   topK
 *
 *
 *   - centroids       : [n_total_clusters, dim]        float32   IVF centroid coarse
 *   - codebook        : [M, K, d_sub]                  float32   residual PQ M*d_sub=dimK=256
 *   - pq_codes        : [N, M]                         uint8      cluster-reorder  h_base_u8
 *   - h_base_u8       : [N, dim]                       uint8     u8 rerank
 *   - h_base_fp32     : [N, dim]                       float32   float rerank  nullptr
 *   - h_query_u8      : [n_query, dim]                 uint8     u8 rerank
 *   - h_query_fp32    : [n_query, dim]                 float32   LUT
 *   - h_coarse_cluster_ids : [n_query, n_probes]       int32     coarse
 *   - h_cluster_offsets    : [n_total_clusters+1]      int64      cluster  reorder
 *   - h_cluster_counts     : [n_total_clusters]        int32      cluster
 *
 *  wall-clock
 */

#include <cstdint>

#ifdef __cplusplus
namespace ivftensor {
namespace cpu_fine {

/**
 * Residual PQ + rerank fine kernel.
 *
 * @param h_base_u8        [N, dim] uint8     rerank cluster reorder
 * @param h_cluster_offsets[n_total_clusters+1]  cluster  reorder
 * @param h_cluster_counts [n_total_clusters]     cluster
 * @param h_pq_codes       [N, M] uint8    residual PQ codes h_base_u8
 * @param h_codebook       [M, K, d_sub] float32   residual PQ
 * @param h_centroids      [n_total_clusters, dim] float32   IVF centroids q - c
 * @param h_query_u8       [n_query, dim] uint8    rerank
 * @param h_query_fp32     [n_query, dim] float32  LUT
 * @param h_base_fp32      [N, dim] float32  float rerank  nullptr  u8 rerank
 * @param use_float_rerank  0  Stage 2 exact rerank  fp32 base/query
 * @param h_coarse_cluster_ids [n_query, n_probes] int32
 * @param n_query
 * @param dim
 * @param n_total_clusters
 * @param n_probes
 * @param topk              top-k 10
 * @param rerank_n         Stage 1 N  topk 50~500
 * @param pq_M             PQ = dim / d_sub 16
 * @param pq_K              code  256  256
 * @param num_threads      OpenMP 0 = max
 * @param[out] h_topk_local_idx [n_query, topk] cluster-reorder  index
 * @param[out] h_topk_dist      [n_query, topk] exact L2^2 (float)
 * @param[out] out_stage1_ms     nullptrStage 1 ms
 * @param[out] out_stage2_ms     nullptrStage 2 rerank ms
 * @return  Stage 1  FMA  Roofline
 */
long long cpu_fine_kernel_pq_resid(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_pq_codes,
    const float* h_codebook,
    const float* h_centroids,
    const uint8_t* h_query_u8,
    const float* h_query_fp32,
    const float* h_base_fp32,
    int use_float_rerank,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int rerank_n,
    int pq_M,
    int pq_K,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    double* out_stage1_ms,
    double* out_stage2_ms
);

}  // namespace cpu_fine
}  // namespace ivftensor
#endif  /* __cplusplus */

#endif  /* IVFTENSOR_CPU_FINE_PQ_RESID_H */
