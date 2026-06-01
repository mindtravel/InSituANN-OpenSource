#include "search/cpu_fine/cpu_fine_layout.h"

#include <omp.h>

#include <algorithm>
#include <cstring>

namespace ivftensor {
namespace cpu_fine {

long long plan_aosoa_layout(
    const int* h_cluster_counts,
    int n_total_clusters,
    int dim,
    int tile,
    long long* aosoa_offsets_out
) {
    aosoa_offsets_out[0] = 0;
    for (int c = 0; c < n_total_clusters; ++c) {
        int cnt = h_cluster_counts[c];
        int groups = (cnt + tile - 1) / tile;
        long long bytes_float = (long long)groups * (long long)tile * (long long)dim;
        aosoa_offsets_out[c + 1] = aosoa_offsets_out[c] + bytes_float;
    }
    return aosoa_offsets_out[n_total_clusters];
}

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
) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    /* padding  */
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int c = 0; c < n_total_clusters; ++c) {
        long long s = aosoa_offsets[c];
        long long e = aosoa_offsets[c + 1];
        if (e > s) {
            std::memset(h_base_aosoa_out + s, 0, (size_t)(e - s) * sizeof(float));
        }
    }

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int c = 0; c < n_total_clusters; ++c) {
        int cnt = h_cluster_counts[c];
        if (cnt <= 0) continue;

        const float* src_cluster = h_base_rowmajor + h_cluster_offsets[c] * (long long)dim;
        float* dst_cluster = h_base_aosoa_out + aosoa_offsets[c];

        int groups = (cnt + tile - 1) / tile;
        for (int g = 0; g < groups; ++g) {
            int tile_begin = g * tile;
            int tile_end   = std::min(tile_begin + tile, cnt);
            float* dst_group = dst_cluster + (long long)g * tile * dim;
            /*  dim  lane lane  tile  */
            for (int d = 0; d < dim; ++d) {
                float* lane = dst_group + (long long)d * tile;
                for (int t = 0; t < tile_end - tile_begin; ++t) {
                    lane[t] = src_cluster[(long long)(tile_begin + t) * dim + d];
                }
                /* lane[tile_end-tile_begin .. tile)  0memset  */
            }
        }
    }
}

}  // namespace cpu_fine
}  // namespace ivftensor
