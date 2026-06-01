/**
 * V2 AVX-512 + FMA intrinsics
 *
 *  V1
 *   -  _mm512_* intrinsics  16-wide FP32 SIMD
 *   - 4 acc0..acc3 FMA  FMA
 *   -  64 dim=128  2 dim=768  12
 *   -  reduce_add
 *
 * -O3 -march=native -mavx512f -mfma
 *
 * dim  16 SIFT=128, BERT=768
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define IVFTENSOR_HAS_AVX512 1
#else
#define IVFTENSOR_HAS_AVX512 0
#endif

namespace ivftensor {
namespace cpu_fine {

#if IVFTENSOR_HAS_AVX512

static inline float l2_sq_v2(const float* __restrict__ q,
                             const float* __restrict__ v, int dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 64 <= dim; i += 64) {
        __m512 q0 = _mm512_loadu_ps(q + i +  0);
        __m512 v0 = _mm512_loadu_ps(v + i +  0);
        __m512 d0 = _mm512_sub_ps(q0, v0);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);

        __m512 q1 = _mm512_loadu_ps(q + i + 16);
        __m512 v1 = _mm512_loadu_ps(v + i + 16);
        __m512 d1 = _mm512_sub_ps(q1, v1);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);

        __m512 q2 = _mm512_loadu_ps(q + i + 32);
        __m512 v2 = _mm512_loadu_ps(v + i + 32);
        __m512 d2 = _mm512_sub_ps(q2, v2);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);

        __m512 q3 = _mm512_loadu_ps(q + i + 48);
        __m512 v3 = _mm512_loadu_ps(v + i + 48);
        __m512 d3 = _mm512_sub_ps(q3, v3);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 16 <= dim; i += 16) {
        __m512 qq = _mm512_loadu_ps(q + i);
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 dd = _mm512_sub_ps(qq, vv);
        acc0 = _mm512_fmadd_ps(dd, dd, acc0);
    }

    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1),
                               _mm512_add_ps(acc2, acc3));
    return _mm512_reduce_add_ps(acc);
}

#else  /* fallback    CPU  AVX-512V2  V1  */

static inline float l2_sq_v2(const float* __restrict__ q,
                             const float* __restrict__ v, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = q[i] - v[i];
        s += d * d;
    }
    return s;
}

#endif

/** cosine  V2  V1 P1  SIFT  L2 cos */
static inline float cos_dist_v2(const float* q, const float* v, int dim) {
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

long long cpu_fine_kernel_v2(
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
                float d = (distance_mode == 0) ? l2_sq_v2(q, v, dim)
                                               : cos_dist_v2(q, v, dim);
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

long long cpu_fine_kernel_v2_touched(
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
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    const int max_qp = n_query * n_probes;
    std::vector<int> touched_clusters;
    touched_clusters.reserve((size_t)max_qp);
    std::vector<std::vector<int>> cluster_query_lists;
    cluster_query_lists.reserve((size_t)max_qp);
    std::unordered_map<int, int> cluster_pos;
    cluster_pos.reserve((size_t)max_qp * 2 + 1);

    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid < 0 || cid >= n_total_clusters) continue;
            auto it = cluster_pos.find(cid);
            int pos;
            if (it == cluster_pos.end()) {
                pos = (int)touched_clusters.size();
                cluster_pos.emplace(cid, pos);
                touched_clusters.push_back(cid);
                cluster_query_lists.emplace_back();
            } else {
                pos = it->second;
            }
            cluster_query_lists[(size_t)pos].push_back(qi);
        }
    }

    std::vector<int> order(touched_clusters.size());
    for (int i = 0; i < (int)order.size(); ++i) order[(size_t)i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return touched_clusters[(size_t)a] < touched_clusters[(size_t)b];
    });

    const int T = num_threads;
    std::vector<float> th_dists((size_t)T * n_query * topk);
    std::vector<int> th_idxs((size_t)T * n_query * topk);
    std::vector<int> th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);

    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        float* my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int* my_idxs = th_idxs.data() + (size_t)tid * n_query * topk;
        int* my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            TopKHeap h;
            h.init(topk, my_dists + (size_t)qid * topk,
                   my_idxs + (size_t)qid * topk);
            if (!my_touched[qid]) {
                my_touched[qid] = 1;
                my_sizes[qid] = 0;
            }
            h.size = my_sizes[qid];
            h.push(d, gidx);
            my_sizes[qid] = h.size;
        };

        #pragma omp for schedule(dynamic, 1)
        for (int oi = 0; oi < (int)order.size(); ++oi) {
            int pos = order[(size_t)oi];
            int cid = touched_clusters[(size_t)pos];
            long long offset = h_cluster_offsets[cid];
            int count = h_cluster_counts[cid];
            if (count <= 0) continue;

            const std::vector<int>& qlist = cluster_query_lists[(size_t)pos];
            for (int qid : qlist) {
                const float* q = h_query + (size_t)qid * (size_t)dim;
                for (int vi = 0; vi < count; ++vi) {
                    const float* v = h_base + ((size_t)offset + vi) * (size_t)dim;
                    float d = (distance_mode == 0) ? l2_sq_v2(q, v, dim)
                                                   : cos_dist_v2(q, v, dim);
                    push_local(qid, d, (int)(offset + vi));
                }
                total_fma += (long long)count * (long long)dim;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 64) num_threads(T)
    for (int qi = 0; qi < n_query; ++qi) {
        float out_dist[1024];
        int out_idx[1024];
        TopKHeap g;
        g.init(topk, out_dist, out_idx);

        for (int t = 0; t < T; ++t) {
            if (!th_touched[(size_t)t * n_query + qi]) continue;
            int sz = th_sizes[(size_t)t * n_query + qi];
            const float* td = th_dists.data() + ((size_t)t * n_query + qi) * topk;
            const int* ti = th_idxs.data() + ((size_t)t * n_query + qi) * topk;
            for (int j = 0; j < sz; ++j) g.push(td[j], ti[j]);
        }
        finalize_topk(g,
                      h_topk_dist + (size_t)qi * topk,
                      h_topk_local_idx + (size_t)qi * topk,
                      topk);
    }

    return total_fma;
}

const char* variant_name_v2() { return "V2_avx512_fma_4acc"; }

}  // namespace cpu_fine
}  // namespace ivftensor
