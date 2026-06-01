/**
 * V5V4 + AoSoA<kTile>
 *
 * cputo_do_list.md 3.4
 *   GPU  InterleavedcoalescedCPU  AoSoA `loadu`  lane
 *   AoSoA<kTile=16>
 *     -  q[d]1  `_mm512_set1_ps`
 *     -  kTile=16 " d " float1  `_mm512_load_ps`
 *     - fma  kTile  base vector  kTile  L2
 *    gather transpositioncache line  100%
 *
 *  template<int kTile> AVX2tile=8/NEONtile=4
 *
 * Parallelismper-query outerOpenMP inner  AoSoA-group
 *   V5  per-query outer V3/V4  + per-thread heap
 *   AoSoA  kTile  base  1  SIMD  query tiling
 *   Q  kTile  dim = 4  16  128 = 8KB  L1
 *    memory bandwidth
 *
 * -O3 -march=native -mavx512f -mfma -fopenmp
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"
#include "search/cpu_fine/cpu_fine_layout.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define IVFTENSOR_V5_HAS_AVX512 1
#else
#define IVFTENSOR_V5_HAS_AVX512 0
#endif

namespace ivftensor {
namespace cpu_fine {

/** AoSoA  kernel query vs  groupkTile  kTile  */
#if IVFTENSOR_V5_HAS_AVX512

template <int kTile>
static inline void l2_sq_v5_group(
    const float* __restrict__ q,
    const float* __restrict__ group_base,   /*  group [dim, kTile]  */
    int dim,
    float* __restrict__ out_dists            /* [kTile] */
);

/* tile=16  __m512  kTile  */
template <>
inline void l2_sq_v5_group<16>(
    const float* __restrict__ q,
    const float* __restrict__ group_base,
    int dim,
    float* __restrict__ out_dists
) {
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < dim; ++d) {
        __m512 q_broad = _mm512_set1_ps(q[d]);
        __m512 v_lane  = _mm512_loadu_ps(group_base + (size_t)d * 16);
        __m512 diff    = _mm512_sub_ps(q_broad, v_lane);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    _mm512_storeu_ps(out_dists, acc);
}

#else  /* AVX-512  */

template <int kTile>
static inline void l2_sq_v5_group(
    const float* q, const float* group_base, int dim, float* out_dists
) {
    for (int t = 0; t < kTile; ++t) out_dists[t] = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float qd = q[d];
        const float* lane = group_base + (size_t)d * kTile;
        for (int t = 0; t < kTile; ++t) {
            float diff = qd - lane[t];
            out_dists[t] += diff * diff;
        }
    }
}

#endif

long long cpu_fine_kernel_v5(
    const float* /*h_base*/,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const float* h_query,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist
) {
    (void)n_total_clusters;
    (void)distance_mode;
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    if (!h_base_aosoa || !h_aosoa_offsets) {
        /*  AoSoA V5 */
        std::memset(h_topk_local_idx, 0xff, (size_t)n_query * topk * sizeof(int));
        for (size_t i = 0; i < (size_t)n_query * topk; ++i) {
            h_topk_dist[i] = std::numeric_limits<float>::infinity();
        }
        return 0;
    }

    constexpr int TILE = kTile;  /*  cpu_fine.h */
    long long total_fma = 0;

    #pragma omp parallel for schedule(dynamic, 32) num_threads(num_threads) reduction(+:total_fma)
    for (int qi = 0; qi < n_query; ++qi) {
        const float* q = h_query + (size_t)qi * dim;

        float dist_buf[1024];
        int   idx_buf[1024];
        TopKHeap heap;
        heap.init(topk, dist_buf, idx_buf);

        long long local_fma = 0;
        alignas(64) float group_dists[TILE];

        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid < 0) continue;
            int count = h_cluster_counts[cid];
            if (count <= 0) continue;
            long long rm_base = h_cluster_offsets[cid];   /* row-major base idx */
            long long ao_base = h_aosoa_offsets[cid];     /* AoSoA float offset */

            int groups = (count + TILE - 1) / TILE;
            for (int g = 0; g < groups; ++g) {
                const float* group_ptr = h_base_aosoa + ao_base
                                       + (long long)g * TILE * dim;
                l2_sq_v5_group<TILE>(q, group_ptr, dim, group_dists);

                int tile_begin = g * TILE;
                int tile_end = std::min(tile_begin + TILE, count);
                for (int t = 0; t < tile_end - tile_begin; ++t) {
                    int gidx = (int)(rm_base + tile_begin + t);
                    heap.push(group_dists[t], gidx);
                }
            }
            local_fma += (long long)count * dim;
        }

        finalize_topk(heap,
                      h_topk_dist + (size_t)qi * topk,
                      h_topk_local_idx + (size_t)qi * topk,
                      topk);
        total_fma += local_fma;
    }

    return total_fma;
}

const char* variant_name_v5() { return "V5_aosoa16_avx512"; }

}  // namespace cpu_fine
}  // namespace ivftensor
