/**
 * V0 FP32
 *
 * Reference scalar path.
 * /" C++  CPU "
 *
 *  CMakeLists.txt
 *   -O0 -fno-tree-vectorize -fno-tree-slp-vectorize
 *
 *
 *    for  FP32  SIMD unroll
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace ivftensor {
namespace cpu_fine {

static float l2_sq_v0(const float* q, const float* v, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = q[i] - v[i];
        s += d * d;
    }
    return s;
}

static float cos_dist_v0(const float* q, const float* v, int dim) {
    float dot = 0.0f, qn = 0.0f, vn = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += q[i] * v[i];
        qn  += q[i] * q[i];
        vn  += v[i] * v[i];
    }
    float denom = qn * vn;
    if (denom < 1e-24f) return 1.0f;
    float cs = dot / std::sqrt(denom);
    if (cs > 1.0f) cs = 1.0f;
    if (cs < -1.0f) cs = -1.0f;
    return 1.0f - cs;
}

long long cpu_fine_kernel_v0(
    const float* h_base,
    const float* /*h_base_aosoa*/,
    const long long* /*h_aosoa_offsets*/,
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
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    long long total_fma = 0;

    #pragma omp parallel for schedule(dynamic, 32) num_threads(num_threads) reduction(+:total_fma)
    for (int qi = 0; qi < n_query; ++qi) {
        const float* q = h_query + (size_t)qi * (size_t)dim;

        float dist_buf[1024];
        int   idx_buf[1024];
        TopKHeap heap;
        heap.init(topk, dist_buf, idx_buf);

        long long local_fma = 0;

        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid < 0) continue;
            long long offset = h_cluster_offsets[cid];
            int count = h_cluster_counts[cid];

            for (int vi = 0; vi < count; ++vi) {
                const float* v = h_base + ((size_t)offset + vi) * (size_t)dim;
                float d = (distance_mode == 0) ? l2_sq_v0(q, v, dim)
                                               : cos_dist_v0(q, v, dim);
                heap.push(d, (int)(offset + vi));
            }
            local_fma += (long long)count * (long long)dim;
        }

        finalize_topk(heap,
                      h_topk_dist + (size_t)qi * (size_t)topk,
                      h_topk_local_idx + (size_t)qi * (size_t)topk,
                      topk);
        total_fma += local_fma;
    }

    return total_fma;
}

const char* variant_name_v0() { return "V0_scalar_O0"; }

}  // namespace cpu_fine
}  // namespace ivftensor
