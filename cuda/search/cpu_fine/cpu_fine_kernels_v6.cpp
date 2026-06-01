/**
 * V6: V3  + V5 AoSoA  (best-of-both)
 *
 *
 *   - per-cluster V3 cluster  probing queries
 *   - query-tile of Q=4V3  reuse  base  4 query
 *   -  AoSoA<kTile=16>  SIMD load  16  base vector
 *           avoiding horizontal reductions until the very end
 *
 *  outer iter cluster  tile-of-16-base  query-tile-4
 *   - For d in [0, dim):
 *     - v_lane = load_16_floats(aosoa[tile_base + d*16])  // 16 lanes = 16 base
 *     - for each q in (q0..q3):
 *       - qd = broadcast(h_query[q*dim + d])
 *       - diff = qd - v_lane
 *       - acc_q = fma(diff, diff, acc_q)   // 16 independent L2^2 for this q
 *   - After all dims: acc_q is a 16-wide zmm of 16 L2^2 values
 *      16  16  base index  query  local heap
 *
 *
 *   -  base tile (16  dim  4B = 16  512 B = 8 KB)  4  query reuse
 *     = 1 load * 4 query  16 lane  dim  2 FLOP = 16384 FLOPs
 *   - Arithmetic intensity = 16384 / 8192 = 2.0 FLOPs/byte V3
 *   -  V6  horizontal reduction (_mm512_reduce_add_ps)
 *      query  gather/
 *
 *  convert_rowmajor_to_aosoa()  h_base_aosoa
 *  h_aosoa_offsets[n_total_clusters+1]
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define IVFTENSOR_V6_HAS_AVX512 1
#else
#define IVFTENSOR_V6_HAS_AVX512 0
#endif

namespace ivftensor {
namespace cpu_fine {

/* ======================================================================
 * l2_sq_tile16_q4 1  AoSoA tile (16 base  dim)  4  query
 *                    16  4 = 64  L2^2
 *   base_aosoa [dim][16]dim  16 lane
 *   out_q0, out_q1, out_q2, out_q3  16-wide  L2^2__m512 reg
 * ====================================================================== */

#if IVFTENSOR_V6_HAS_AVX512
static inline void l2_sq_v6_tile16_q4(
    const float* __restrict__ q0,
    const float* __restrict__ q1,
    const float* __restrict__ q2,
    const float* __restrict__ q3,
    const float* __restrict__ base_tile,  /* size = dim * 16 */
    int dim,
    __m512& out0, __m512& out1, __m512& out2, __m512& out3
) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    /*  iter  dimload  16-lane basebroadcast 4  query scalar */
    for (int d = 0; d < dim; ++d) {
        __m512 v16 = _mm512_load_ps(base_tile + (size_t)d * 16);
        __m512 qb0 = _mm512_set1_ps(q0[d]);
        __m512 qb1 = _mm512_set1_ps(q1[d]);
        __m512 qb2 = _mm512_set1_ps(q2[d]);
        __m512 qb3 = _mm512_set1_ps(q3[d]);
        __m512 diff0 = _mm512_sub_ps(qb0, v16);
        __m512 diff1 = _mm512_sub_ps(qb1, v16);
        __m512 diff2 = _mm512_sub_ps(qb2, v16);
        __m512 diff3 = _mm512_sub_ps(qb3, v16);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);
        acc1 = _mm512_fmadd_ps(diff1, diff1, acc1);
        acc2 = _mm512_fmadd_ps(diff2, diff2, acc2);
        acc3 = _mm512_fmadd_ps(diff3, diff3, acc3);
    }
    out0 = acc0; out1 = acc1; out2 = acc2; out3 = acc3;
}

/*  query  query_tile  */
static inline void l2_sq_v6_tile16_q1(
    const float* __restrict__ q0,
    const float* __restrict__ base_tile,
    int dim,
    __m512& out0
) {
    __m512 acc0 = _mm512_setzero_ps();
    for (int d = 0; d < dim; ++d) {
        __m512 v16 = _mm512_load_ps(base_tile + (size_t)d * 16);
        __m512 qb0 = _mm512_set1_ps(q0[d]);
        __m512 diff0 = _mm512_sub_ps(qb0, v16);
        acc0 = _mm512_fmadd_ps(diff0, diff0, acc0);
    }
    out0 = acc0;
}
#endif

/* Per-thread  TopK heap V3  */
struct LocalTopK6 {
    int size;
    float* dists;
    int*   idxs;
    static inline void sift_up(float* d, int* i, int pos) {
        while (pos > 0) {
            int parent = (pos - 1) >> 1;
            if (d[parent] >= d[pos]) break;
            float td = d[pos]; d[pos] = d[parent]; d[parent] = td;
            int   ti = i[pos]; i[pos] = i[parent]; i[parent] = ti;
            pos = parent;
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

long long cpu_fine_kernel_v6(
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
    (void)distance_mode;
    if (num_threads <= 0) num_threads = omp_get_max_threads();

#if !IVFTENSOR_V6_HAS_AVX512
    /*  AVX-512 fallback  V3 */
    extern long long cpu_fine_kernel_v3(
        const float*, const float*, const long long*,
        const long long*, const int*, const float*, const int*,
        int, int, int, int, int, int, int, int*, float*);
    (void)h_base_aosoa; (void)h_aosoa_offsets;
    /* h_base  */
    return 0;   /* V6  AVX-512  0 FLOP */
#else

    /* Step 1. cluster  queries */
    std::vector<int> cluster_query_count(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid >= 0 && cid < n_total_clusters) cluster_query_count[cid]++;
        }
    }
    std::vector<int> cluster_query_offset(n_total_clusters + 1, 0);
    for (int c = 0; c < n_total_clusters; ++c) {
        cluster_query_offset[c + 1] = cluster_query_offset[c] + cluster_query_count[c];
    }
    int total_qp = cluster_query_offset[n_total_clusters];
    std::vector<int> cluster_query_list(total_qp);
    {
        std::vector<int> wpos(n_total_clusters, 0);
        for (int qi = 0; qi < n_query; ++qi) {
            for (int pi = 0; pi < n_probes; ++pi) {
                int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
                if (cid >= 0 && cid < n_total_clusters) {
                    int pos = cluster_query_offset[cid] + wpos[cid]++;
                    cluster_query_list[pos] = qi;
                }
            }
        }
    }

    /* Step 2. per-thread heaps */
    int T = num_threads;
    std::vector<float> th_dists((size_t)T * n_query * topk);
    std::vector<int>   th_idxs((size_t)T * n_query * topk);
    std::vector<int>   th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);

    /* Step 3. parallel over clusters */
    long long total_fma = 0;

    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        float* my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*   my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*   my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        /* 16-lane  __m512 distance  float[16] */
        alignas(64) float dist_buf[kTile];

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopK6 h;
            h.dists = my_dists + (size_t)qid * topk;
            h.idxs  = my_idxs  + (size_t)qid * topk;
            if (!my_touched[qid]) { my_touched[qid] = 1; my_sizes[qid] = 0; }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        #pragma omp for schedule(dynamic, 1)
        for (int cid = 0; cid < n_total_clusters; ++cid) {
            int qs = cluster_query_offset[cid];
            int qe = cluster_query_offset[cid + 1];
            int qcount = qe - qs;
            if (qcount == 0) continue;

            int count = h_cluster_counts[cid];
            if (count <= 0) continue;

            long long aosoa_off = h_aosoa_offsets[cid];
            long long base_off  = h_cluster_offsets[cid];
            const int* qlist = cluster_query_list.data() + qs;

            int n_tiles = (count + kTile - 1) / kTile;

            /*  tile of 4 queries */
            int qi = 0;
            for (; qi + 4 <= qcount; qi += 4) {
                int qid0 = qlist[qi + 0];
                int qid1 = qlist[qi + 1];
                int qid2 = qlist[qi + 2];
                int qid3 = qlist[qi + 3];
                const float* pq0 = h_query + (size_t)qid0 * dim;
                const float* pq1 = h_query + (size_t)qid1 * dim;
                const float* pq2 = h_query + (size_t)qid2 * dim;
                const float* pq3 = h_query + (size_t)qid3 * dim;

                for (int t = 0; t < n_tiles; ++t) {
                    const float* tile_ptr = h_base_aosoa + aosoa_off + (size_t)t * dim * kTile;
                    __m512 acc0, acc1, acc2, acc3;
                    l2_sq_v6_tile16_q4(pq0, pq1, pq2, pq3, tile_ptr, dim,
                                       acc0, acc1, acc2, acc3);

                    int tile_start = t * kTile;
                    int tile_end   = tile_start + kTile;
                    if (tile_end > count) tile_end = count;

                    /*  query  16  */
                    _mm512_store_ps(dist_buf, acc0);
                    for (int lane = 0; lane < tile_end - tile_start; ++lane) {
                        int gidx = (int)(base_off + tile_start + lane);
                        push_local(qid0, dist_buf[lane], gidx);
                    }
                    _mm512_store_ps(dist_buf, acc1);
                    for (int lane = 0; lane < tile_end - tile_start; ++lane) {
                        int gidx = (int)(base_off + tile_start + lane);
                        push_local(qid1, dist_buf[lane], gidx);
                    }
                    _mm512_store_ps(dist_buf, acc2);
                    for (int lane = 0; lane < tile_end - tile_start; ++lane) {
                        int gidx = (int)(base_off + tile_start + lane);
                        push_local(qid2, dist_buf[lane], gidx);
                    }
                    _mm512_store_ps(dist_buf, acc3);
                    for (int lane = 0; lane < tile_end - tile_start; ++lane) {
                        int gidx = (int)(base_off + tile_start + lane);
                        push_local(qid3, dist_buf[lane], gidx);
                    }
                }
                /* FMA  tile  dim  4 query  FMA FMA 16-lane */
                total_fma += (long long)n_tiles * dim * 4 * kTile;
            }
            /*  query */
            for (; qi < qcount; ++qi) {
                int qid = qlist[qi];
                const float* q = h_query + (size_t)qid * dim;
                for (int t = 0; t < n_tiles; ++t) {
                    const float* tile_ptr = h_base_aosoa + aosoa_off + (size_t)t * dim * kTile;
                    __m512 acc0;
                    l2_sq_v6_tile16_q1(q, tile_ptr, dim, acc0);
                    _mm512_store_ps(dist_buf, acc0);
                    int tile_start = t * kTile;
                    int tile_end   = tile_start + kTile;
                    if (tile_end > count) tile_end = count;
                    for (int lane = 0; lane < tile_end - tile_start; ++lane) {
                        int gidx = (int)(base_off + tile_start + lane);
                        push_local(qid, dist_buf[lane], gidx);
                    }
                }
                total_fma += (long long)n_tiles * dim * kTile;
            }
        }
    }

    /* Step 4. merge per-thread heaps */
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
#endif /* IVFTENSOR_V6_HAS_AVX512 */
}

}  // namespace cpu_fine
}  // namespace ivftensor
