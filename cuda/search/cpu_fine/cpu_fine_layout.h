#ifndef IVFTENSOR_CPU_FINE_LAYOUT_H
#define IVFTENSOR_CPU_FINE_LAYOUT_H

/**
 * CPU Fine Searchrow-major  AoSoA
 *
 * AoSoA<kTile> per cluster
 *    cluster  M  cluster  pad  M' = ceil(M / kTile) * kTile
 *   cluster  AoSoA M' * dim  float
 *   groups  [dim * kTile] groups = M' / kTile
 *     group g  dim  "lane" lane  kTile  float kTile
 *
 *
 *    1  query dim  q[d] kTile  float lane
 *    `_mm512_fmadd_ps(diff, diff, acc)` kTile  gather
 *
 * Padding
 *   M  kTile  group  [M % kTile .. kTile)  padding
 *    kernel  `valid_in_last_group = M - (groups-1)*kTile`
 *    padding  padding  0 push  heap
 */

#include <cstddef>
#include <cstdint>

#include "search/cpu_fine/cpu_fine.h"

namespace ivftensor {
namespace cpu_fine {

/**
 *  AoSoA  float  cluster  float
 *
 * @param h_cluster_counts  [n_total_clusters]
 * @param n_total_clusters  cluster
 * @param dim
 * @param tile              AoSoA tile
 * @param aosoa_offsets_out [n_total_clusters+1]  cluster  float-
 * @return  float
 */
long long plan_aosoa_layout(
    const int* h_cluster_counts,
    int n_total_clusters,
    int dim,
    int tile,
    long long* aosoa_offsets_out
);

/**
 *  row-major base cluster  AoSoA<tile>
 *  OpenMP  per cluster
 *
 * @param h_base_rowmajor    [n_total_vectors, dim] row-major
 * @param h_cluster_offsets  [n_total_clusters+1]   row-major  cluster vec index
 * @param h_cluster_counts   [n_total_clusters]      cluster
 * @param aosoa_offsets      [n_total_clusters+1]    plan_aosoa_layout
 * @param n_total_clusters
 * @param dim
 * @param tile
 * @param h_base_aosoa_out    AoSoA  = aosoa_offsets[n_total_clusters]
 * @param num_threads        0 = OpenMP
 */
void convert_rowmajor_to_aosoa(
    const float* h_base_rowmajor,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const long long* aosoa_offsets,
    int n_total_clusters,
    int dim,
    int tile,
    float* h_base_aosoa_out,
    int num_threads
);

}  // namespace cpu_fine
}  // namespace ivftensor

#endif  /* IVFTENSOR_CPU_FINE_LAYOUT_H */
