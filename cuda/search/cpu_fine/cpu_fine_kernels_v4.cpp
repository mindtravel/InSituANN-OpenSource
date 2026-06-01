/**
 * V4V3 + Software Prefetching
 *
 *  V3  Q=4 query-tile kernel  `_mm_prefetch`
 *   -  (base vectors)  compute kernel  P
 *     cache line  L1D_MM_HINT_T0
 *   - prefetch distance  8  base vector 8 * dim*4 SIFT=4KB
 *      200 cycle  DRAM latency
 *   - prefetch  prefetch /cache-line  0.3 cycle
 *      DRAM  prefetch
 *          L1
 *     ""
 *
 * -O3 -march=native -mavx512f -mfma -fopenmp
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define IVFTENSOR_V4_HAS_AVX512 1
#else
#define IVFTENSOR_V4_HAS_AVX512 0
#endif

namespace ivftensor {
namespace cpu_fine {

/** prefetch distance base vector 8 dim=128  4KB
 *   ~200 cycle  DRAM  */
static constexpr int kPrefetchAhead = 8;

#if IVFTENSOR_V4_HAS_AVX512

static inline void prefetch_vec_T0(const float* p, int dim) {
    /*  64  prefetchdim=128  8  */
    const char* cp = reinterpret_cast<const char*>(p);
    for (int b = 0; b < dim * (int)sizeof(float); b += 64) {
        _mm_prefetch(cp + b, _MM_HINT_T0);
    }
}

static inline void l2_sq_v4_q4(
    const float* __restrict__ q0,
    const float* __restrict__ q1,
    const float* __restrict__ q2,
    const float* __restrict__ q3,
    const float* __restrict__ v,
    int dim,
    float& out0, float& out1, float& out2, float& out3
) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 qq0 = _mm512_loadu_ps(q0 + i);
        __m512 qq1 = _mm512_loadu_ps(q1 + i);
        __m512 qq2 = _mm512_loadu_ps(q2 + i);
        __m512 qq3 = _mm512_loadu_ps(q3 + i);
        __m512 d0 = _mm512_sub_ps(qq0, vv);
        __m512 d1 = _mm512_sub_ps(qq1, vv);
        __m512 d2 = _mm512_sub_ps(qq2, vv);
        __m512 d3 = _mm512_sub_ps(qq3, vv);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    out0 = _mm512_reduce_add_ps(acc0);
    out1 = _mm512_reduce_add_ps(acc1);
    out2 = _mm512_reduce_add_ps(acc2);
    out3 = _mm512_reduce_add_ps(acc3);
    for (; i < dim; ++i) {
        float a0 = q0[i] - v[i]; out0 += a0 * a0;
        float a1 = q1[i] - v[i]; out1 += a1 * a1;
        float a2 = q2[i] - v[i]; out2 += a2 * a2;
        float a3 = q3[i] - v[i]; out3 += a3 * a3;
    }
}

static inline float l2_sq_v4_q1(const float* q, const float* v, int dim) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m512 vv0 = _mm512_loadu_ps(v + i);
        __m512 qq0 = _mm512_loadu_ps(q + i);
        __m512 d0 = _mm512_sub_ps(qq0, vv0);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        __m512 vv1 = _mm512_loadu_ps(v + i + 16);
        __m512 qq1 = _mm512_loadu_ps(q + i + 16);
        __m512 d1 = _mm512_sub_ps(qq1, vv1);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 16 <= dim; i += 16) {
        __m512 vv = _mm512_loadu_ps(v + i);
        __m512 qq = _mm512_loadu_ps(q + i);
        __m512 d  = _mm512_sub_ps(qq, vv);
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

#else

static inline void prefetch_vec_T0(const float*, int) { /* no-op */ }
static inline void l2_sq_v4_q4(const float* q0, const float* q1,
                               const float* q2, const float* q3,
                               const float* v, int dim,
                               float& o0, float& o1, float& o2, float& o3) {
    o0 = o1 = o2 = o3 = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float a0 = q0[i] - v[i]; o0 += a0 * a0;
        float a1 = q1[i] - v[i]; o1 += a1 * a1;
        float a2 = q2[i] - v[i]; o2 += a2 * a2;
        float a3 = q3[i] - v[i]; o3 += a3 * a3;
    }
}
static inline float l2_sq_v4_q1(const float* q, const float* v, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = q[i] - v[i]; s += d * d; }
    return s;
}

#endif

struct LocalTopKV4 {
    int size;
    float* dists;
    int*   idxs;
    static inline void sift_up(float* d, int* i, int pos) {
        while (pos > 0) {
            int p = (pos - 1) >> 1;
            if (d[p] >= d[pos]) break;
            float td = d[pos]; d[pos] = d[p]; d[p] = td;
            int   ti = i[pos]; i[pos] = i[p]; i[p] = ti;
            pos = p;
        }
    }
    static inline void sift_down(float* d, int* i, int n, int pos) {
        while (true) {
            int l = 2 * pos + 1, r = 2 * pos + 2, largest = pos;
            if (l < n && d[l] > d[largest]) largest = l;
            if (r < n && d[r] > d[largest]) largest = r;
            if (largest == pos) break;
            float td = d[pos]; d[pos] = d[largest]; d[largest] = td;
            int   ti = i[pos]; i[pos] = i[largest]; i[largest] = ti;
            pos = largest;
        }
    }
    inline void push(float dist, int idx, int cap) {
        if (size < cap) {
            dists[size] = dist; idxs[size] = idx; ++size;
            sift_up(dists, idxs, size - 1);
        } else if (dist < dists[0]) {
            dists[0] = dist; idxs[0] = idx;
            sift_down(dists, idxs, size, 0);
        }
    }
};

long long cpu_fine_kernel_v4(
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
    (void)distance_mode;
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    /* ----  ---- */
    std::vector<int> ccount(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid >= 0 && cid < n_total_clusters) ccount[cid]++;
        }
    }
    std::vector<int> coff(n_total_clusters + 1, 0);
    for (int c = 0; c < n_total_clusters; ++c) coff[c + 1] = coff[c] + ccount[c];
    std::vector<int> clist(coff[n_total_clusters]);
    {
        std::vector<int> wpos(n_total_clusters, 0);
        for (int qi = 0; qi < n_query; ++qi) {
            for (int pi = 0; pi < n_probes; ++pi) {
                int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
                if (cid >= 0 && cid < n_total_clusters) {
                    clist[coff[cid] + wpos[cid]++] = qi;
                }
            }
        }
    }

    int T = num_threads;
    std::vector<float>   th_dists((size_t)T * n_query * topk);
    std::vector<int>     th_idxs ((size_t)T * n_query * topk);
    std::vector<int>     th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);

    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        float* my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*   my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*   my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopKV4 h;
            h.dists = my_dists + (size_t)qid * topk;
            h.idxs  = my_idxs  + (size_t)qid * topk;
            if (!my_touched[qid]) { my_touched[qid] = 1; my_sizes[qid] = 0; }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        #pragma omp for schedule(dynamic, 1)
        for (int cid = 0; cid < n_total_clusters; ++cid) {
            int qs = coff[cid];
            int qe = coff[cid + 1];
            int qn = qe - qs;
            if (qn == 0) continue;
            long long base_off = h_cluster_offsets[cid];
            int count = h_cluster_counts[cid];
            if (count <= 0) continue;
            const int* qlist = clist.data() + qs;

            /*  cluster  kPrefetchAhead  base vector  L1D */
            int pf_end = std::min(kPrefetchAhead, count);
            for (int vi = 0; vi < pf_end; ++vi) {
                prefetch_vec_T0(h_base + ((size_t)base_off + vi) * dim, dim);
            }

            int qi = 0;
            for (; qi + 4 <= qn; qi += 4) {
                int q0 = qlist[qi + 0], q1 = qlist[qi + 1];
                int q2 = qlist[qi + 2], q3 = qlist[qi + 3];
                const float* pq0 = h_query + (size_t)q0 * dim;
                const float* pq1 = h_query + (size_t)q1 * dim;
                const float* pq2 = h_query + (size_t)q2 * dim;
                const float* pq3 = h_query + (size_t)q3 * dim;

                for (int vi = 0; vi < count; ++vi) {
                    /*  vi + kPrefetchAhead */
                    int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < count) {
                        prefetch_vec_T0(h_base + ((size_t)base_off + pf_vi) * dim, dim);
                    }

                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d0, d1, d2, d3;
                    l2_sq_v4_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    int gidx = (int)(base_off + vi);
                    push_local(q0, d0, gidx);
                    push_local(q1, d1, gidx);
                    push_local(q2, d2, gidx);
                    push_local(q3, d3, gidx);
                }
                total_fma += (long long)count * dim * 4;
            }
            for (; qi < qn; ++qi) {
                int qid = qlist[qi];
                const float* q = h_query + (size_t)qid * dim;
                for (int vi = 0; vi < count; ++vi) {
                    int pf_vi = vi + kPrefetchAhead;
                    if (pf_vi < count) {
                        prefetch_vec_T0(h_base + ((size_t)base_off + pf_vi) * dim, dim);
                    }
                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d = l2_sq_v4_q1(q, v, dim);
                    push_local(qid, d, (int)(base_off + vi));
                }
                total_fma += (long long)count * dim;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 64) num_threads(T)
    for (int qi = 0; qi < n_query; ++qi) {
        float out_dist[1024];
        int   out_idx[1024];
        TopKHeap g;
        g.init(topk, out_dist, out_idx);
        for (int t = 0; t < T; ++t) {
            if (!th_touched[(size_t)t * n_query + qi]) continue;
            int sz = th_sizes[(size_t)t * n_query + qi];
            const float* td = th_dists.data() + ((size_t)t * n_query + qi) * topk;
            const int*   ti = th_idxs.data()  + ((size_t)t * n_query + qi) * topk;
            for (int j = 0; j < sz; ++j) g.push(td[j], ti[j]);
        }
        finalize_topk(g,
                      h_topk_dist + (size_t)qi * topk,
                      h_topk_local_idx + (size_t)qi * topk,
                      topk);
    }

    return total_fma;
}

const char* variant_name_v4() { return "V4_v3_plus_prefetch"; }

}  // namespace cpu_fine
}  // namespace ivftensor
