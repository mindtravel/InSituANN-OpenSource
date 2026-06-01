#ifndef IVFTENSOR_CPU_FINE_U8_H
#define IVFTENSOR_CPU_FINE_U8_H

/**
 * B1uint8  AVX-512 fine kernel  ( V3-uint8 / v3_u8)
 *
 *
 * ----
 * SIFT / DEEP  uint8 V3 (cpu_fine_kernels_v3.cpp)  fp32
 * fine  memory-bound DRAM  base  dim*4
 *  75%  u8fp32  padding
 *
 *  kernel  uint8  base / query widen  int16
 *  `_mm512_madd_epi16(diff, diff)`  ~2 ops/cycle
 *
 *   (q-b)^2  =  _k (q[k] - b[k])^2
 *            =  widen u8i16, subtract, madd_epi16, add to i32 acc
 *
 *  V3
 *   - uint8_t  float
 *   -  base dim  dim*4   ( DRAM  1/4)
 *   -  SIFT/DEEP  uint8
 *
 * topk_dist  float int32  float
 * dim  1024  max distance < 2^24
 */

#include <cstdint>

#ifdef __cplusplus
namespace ivftensor {
namespace cpu_fine {

/**
 * Q=4 query-tile uint8 fine kernel ( V3  CPU ).
 *
 *
 *   - h_base_u8 / h_query_u8  cluster reorder V3  reordered_data
 *   - h_cluster_offsets / h_cluster_counts  cluster / V3
 *   - h_coarse_cluster_ids  GPU  D2H  host
 *
 * parallel over clusters (schedule dynamic)
 *           per-thread  topk heap merge  topk
 *
 *  (i16 mul + i32 add)  Roofline
 */
long long cpu_fine_kernel_v3_u8(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query_u8,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int num_threads,
    int* h_topk_local_idx,     /* [n_query, topk] */
    float* h_topk_dist         /* [n_query, topk] float(L2^2)  */
);

long long cpu_fine_kernel_v3_u8_touched(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query_u8,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist
);

}  // namespace cpu_fine
}  // namespace ivftensor
#endif  /* __cplusplus */

#endif  /* IVFTENSOR_CPU_FINE_U8_H */
