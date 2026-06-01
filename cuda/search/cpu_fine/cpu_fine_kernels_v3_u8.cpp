/**
 * B1V3-uint8 AVX-512 fine kernel
 *
 *  mirror cpu_fine_kernels_v3.cppQ=4 query tiling, per-thread heap,
 * parallel over cluster fp32 L2 kernel  uint8  i16  i32
 *
 *  32  u8  AVX
 *   load      32  u8   __m256i
 *   widen      __m512i (32  i16)
 *   subtract  diff = q_i16 - b_i16
 *   madd      __m512i prod = _mm512_madd_epi16(diff, diff)   // 16  i32
 *   add       acc = _mm512_add_epi32(acc, prod)
 *
 *  `_mm512_reduce_add_epi32`  int32 cast  float  topk heap
 *
 * -O3 -march=native -mavx512f -mavx512bw -fopenmp
 *     `_mm512_madd_epi16`  AVX-512 BW EPYC / Zen4 / Ice Lake
 */

#include "search/cpu_fine/cpu_fine_u8.h"
#include "search/cpu_fine/cpu_fine_common.h"
#include "numa_utils.h"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define IVFTENSOR_V3U8_HAS_AVX512BW 1
#else
#define IVFTENSOR_V3U8_HAS_AVX512BW 0
#endif

namespace ivftensor {
namespace cpu_fine {

/* ======================================================================
 * Q=4 Query Tile 4  query vs 1  base vector  L2^2int32
 * ====================================================================== */

#if IVFTENSOR_V3U8_HAS_AVX512BW

static inline void l2_sq_v3u8_q4(
    const uint8_t* __restrict__ q0,
    const uint8_t* __restrict__ q1,
    const uint8_t* __restrict__ q2,
    const uint8_t* __restrict__ q3,
    const uint8_t* __restrict__ v,
    int dim,
    int32_t& out0, int32_t& out1, int32_t& out2, int32_t& out3
) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256i b8  = _mm256_loadu_si256((const __m256i*)(v  + i));
        __m256i q08 = _mm256_loadu_si256((const __m256i*)(q0 + i));
        __m256i q18 = _mm256_loadu_si256((const __m256i*)(q1 + i));
        __m256i q28 = _mm256_loadu_si256((const __m256i*)(q2 + i));
        __m256i q38 = _mm256_loadu_si256((const __m256i*)(q3 + i));
        __m512i bi  = _mm512_cvtepu8_epi16(b8);
        __m512i q0i = _mm512_cvtepu8_epi16(q08);
        __m512i q1i = _mm512_cvtepu8_epi16(q18);
        __m512i q2i = _mm512_cvtepu8_epi16(q28);
        __m512i q3i = _mm512_cvtepu8_epi16(q38);
        __m512i d0  = _mm512_sub_epi16(q0i, bi);
        __m512i d1  = _mm512_sub_epi16(q1i, bi);
        __m512i d2  = _mm512_sub_epi16(q2i, bi);
        __m512i d3  = _mm512_sub_epi16(q3i, bi);
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));
        acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));
        acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));
    }
    out0 = _mm512_reduce_add_epi32(acc0);
    out1 = _mm512_reduce_add_epi32(acc1);
    out2 = _mm512_reduce_add_epi32(acc2);
    out3 = _mm512_reduce_add_epi32(acc3);

    /* Tail (dim % 32)SIFT-128 / DEEP-96  */
    for (; i < dim; ++i) {
        int s0 = (int)q0[i] - (int)v[i]; out0 += s0 * s0;
        int s1 = (int)q1[i] - (int)v[i]; out1 += s1 * s1;
        int s2 = (int)q2[i] - (int)v[i]; out2 += s2 * s2;
        int s3 = (int)q3[i] - (int)v[i]; out3 += s3 * s3;
    }
}

static inline int32_t l2_sq_v3u8_q1(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ v, int dim
) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= dim; i += 64) {
        __m256i b8a = _mm256_loadu_si256((const __m256i*)(v + i +  0));
        __m256i b8b = _mm256_loadu_si256((const __m256i*)(v + i + 32));
        __m256i q8a = _mm256_loadu_si256((const __m256i*)(q + i +  0));
        __m256i q8b = _mm256_loadu_si256((const __m256i*)(q + i + 32));
        __m512i da = _mm512_sub_epi16(_mm512_cvtepu8_epi16(q8a),
                                      _mm512_cvtepu8_epi16(b8a));
        __m512i db = _mm512_sub_epi16(_mm512_cvtepu8_epi16(q8b),
                                      _mm512_cvtepu8_epi16(b8b));
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(da, da));
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(db, db));
    }
    for (; i + 32 <= dim; i += 32) {
        __m256i b8 = _mm256_loadu_si256((const __m256i*)(v + i));
        __m256i q8 = _mm256_loadu_si256((const __m256i*)(q + i));
        __m512i dd = _mm512_sub_epi16(_mm512_cvtepu8_epi16(q8),
                                      _mm512_cvtepu8_epi16(b8));
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(dd, dd));
    }
    int32_t s = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
    for (; i < dim; ++i) {
        int d = (int)q[i] - (int)v[i];
        s += d * d;
    }
    return s;
}

#else  /* AVX-512 BW scalar fallback */

static inline void l2_sq_v3u8_q4(
    const uint8_t* q0, const uint8_t* q1, const uint8_t* q2, const uint8_t* q3,
    const uint8_t* v, int dim,
    int32_t& o0, int32_t& o1, int32_t& o2, int32_t& o3
) {
    o0 = o1 = o2 = o3 = 0;
    for (int i = 0; i < dim; ++i) {
        int a0 = (int)q0[i] - (int)v[i]; o0 += a0 * a0;
        int a1 = (int)q1[i] - (int)v[i]; o1 += a1 * a1;
        int a2 = (int)q2[i] - (int)v[i]; o2 += a2 * a2;
        int a3 = (int)q3[i] - (int)v[i]; o3 += a3 * a3;
    }
}

static inline int32_t l2_sq_v3u8_q1(const uint8_t* q, const uint8_t* v, int dim) {
    int32_t s = 0;
    for (int i = 0; i < dim; ++i) {
        int d = (int)q[i] - (int)v[i];
        s += d * d;
    }
    return s;
}

#endif  /* AVX-512 BW */


/* ======================================================================
 * Per-thread  TopKHeap
 * ====================================================================== */

struct LocalTopKU8 {
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
            dists[size] = dist;
            idxs[size]  = idx;
            ++size;
            sift_up(dists, idxs, size - 1);
        } else if (dist < dists[0]) {
            dists[0] = dist;
            idxs[0]  = idx;
            sift_down(dists, idxs, size, 0);
        }
    }
};


/* ======================================================================
 * Kernel
 * ====================================================================== */

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
    int* h_topk_local_idx,
    float* h_topk_dist
) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    /* ---------- Step 1. cluster  list of query_ids ---------- */
    std::vector<int> cluster_query_count(n_total_clusters, 0);
    for (int qi = 0; qi < n_query; ++qi) {
        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid >= 0 && cid < n_total_clusters) {
                cluster_query_count[cid]++;
            }
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

    /* ---------- Step 2. Per-thread heap buffers ---------- */
    int T = num_threads;
    std::vector<float> th_dists((size_t)T * n_query * topk);
    std::vector<int>   th_idxs((size_t)T * n_query * topk);
    std::vector<int>   th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);

    /* ---------- Step 3. Parallel over clusters ---------- */
    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        float*   my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*     my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*     my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopKU8 h;
            h.dists = my_dists + (size_t)qid * topk;
            h.idxs  = my_idxs  + (size_t)qid * topk;
            if (!my_touched[qid]) {
                my_touched[qid] = 1;
                my_sizes[qid] = 0;
            }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        #pragma omp for schedule(dynamic, 1)
        for (int cid = 0; cid < n_total_clusters; ++cid) {
            int qlist_start = cluster_query_offset[cid];
            int qlist_end   = cluster_query_offset[cid + 1];
            int qlist_size  = qlist_end - qlist_start;
            if (qlist_size == 0) continue;

            long long base_off = h_cluster_offsets[cid];
            int count = h_cluster_counts[cid];
            if (count <= 0) continue;

            const int* qlist = cluster_query_list.data() + qlist_start;

            int qi = 0;
            for (; qi + 4 <= qlist_size; qi += 4) {
                int q0 = qlist[qi + 0];
                int q1 = qlist[qi + 1];
                int q2 = qlist[qi + 2];
                int q3 = qlist[qi + 3];
                const uint8_t* pq0 = h_query_u8 + (size_t)q0 * dim;
                const uint8_t* pq1 = h_query_u8 + (size_t)q1 * dim;
                const uint8_t* pq2 = h_query_u8 + (size_t)q2 * dim;
                const uint8_t* pq3 = h_query_u8 + (size_t)q3 * dim;

                for (int vi = 0; vi < count; ++vi) {
                    const uint8_t* v = h_base_u8 + ((size_t)base_off + vi) * dim;
                    int32_t d0, d1, d2, d3;
                    l2_sq_v3u8_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    int gidx = (int)(base_off + vi);
                    push_local(q0, (float)d0, gidx);
                    push_local(q1, (float)d1, gidx);
                    push_local(q2, (float)d2, gidx);
                    push_local(q3, (float)d3, gidx);
                }
                total_fma += (long long)count * dim * 4;
            }
            for (; qi < qlist_size; ++qi) {
                int qid = qlist[qi];
                const uint8_t* q = h_query_u8 + (size_t)qid * dim;
                for (int vi = 0; vi < count; ++vi) {
                    const uint8_t* v = h_base_u8 + ((size_t)base_off + vi) * dim;
                    int32_t d = l2_sq_v3u8_q1(q, v, dim);
                    push_local(qid, (float)d, (int)(base_off + vi));
                }
                total_fma += (long long)count * dim;
            }
        }
    }

    /* ---------- Step 4. Merge per-thread heaps  global topk ---------- */
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

    bool numa_aware = ivftensor::numa::schedule_enabled();
    bool numa_bind = ivftensor::numa::bind_enabled();
    int numa_nodes = (numa_aware || numa_bind) ? ivftensor::numa::node_count() : 1;
    if (numa_nodes <= 1) {
        numa_aware = false;
        numa_bind = false;
        numa_nodes = 1;
    }

    std::vector<int> numa_cluster_bounds;
    std::vector<std::vector<int>> order_by_node;
    if (numa_aware) {
        numa_cluster_bounds =
            ivftensor::numa::split_clusters_by_count(h_cluster_counts, n_total_clusters, numa_nodes);
        order_by_node.assign((size_t)numa_nodes, {});
        for (int pos : order) {
            int cid = touched_clusters[(size_t)pos];
            int node = ivftensor::numa::node_for_cluster(cid, numa_cluster_bounds);
            order_by_node[(size_t)node].push_back(pos);
        }
    }

    const bool replicate_meta =
        numa_aware && ivftensor::numa::env_flag("IVFT_NUMA_REPLICATE_META");
    const bool strict_numa =
        numa_aware && ivftensor::numa::env_flag("IVFT_NUMA_STRICT");
    std::vector<std::vector<long long>> numa_offsets;
    std::vector<std::vector<int>> numa_counts;
    if (replicate_meta) {
        numa_offsets.resize((size_t)numa_nodes);
        numa_counts.resize((size_t)numa_nodes);
        std::vector<std::thread> meta_threads;
        meta_threads.reserve((size_t)numa_nodes);
        for (int node = 0; node < numa_nodes; ++node) {
            meta_threads.emplace_back([&, node]() {
                ivftensor::numa::bind_current_thread_to_node(node);
                numa_offsets[(size_t)node].resize((size_t)n_total_clusters);
                numa_counts[(size_t)node].resize((size_t)n_total_clusters);
                std::memcpy(numa_offsets[(size_t)node].data(),
                            h_cluster_offsets,
                            (size_t)n_total_clusters * sizeof(long long));
                std::memcpy(numa_counts[(size_t)node].data(),
                            h_cluster_counts,
                            (size_t)n_total_clusters * sizeof(int));
            });
        }
        for (auto& th : meta_threads) th.join();
    }

    int T = num_threads;
    std::vector<float> th_dists((size_t)T * n_query * topk);
    std::vector<int>   th_idxs((size_t)T * n_query * topk);
    std::vector<int>   th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);
    std::vector<int> numa_next((size_t)numa_nodes, 0);

    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        int my_node = 0;
        const long long* local_offsets = h_cluster_offsets;
        const int* local_counts = h_cluster_counts;
        if (numa_bind) {
            my_node = ivftensor::numa::node_for_thread(tid, T, numa_nodes);
            ivftensor::numa::bind_current_thread_to_node(my_node);
        }
        if (numa_aware) {
            if (replicate_meta) {
                local_offsets = numa_offsets[(size_t)my_node].data();
                local_counts = numa_counts[(size_t)my_node].data();
            }
        }

        float*   my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*     my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*     my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopKU8 h;
            h.dists = my_dists + (size_t)qid * topk;
            h.idxs  = my_idxs  + (size_t)qid * topk;
            if (!my_touched[qid]) {
                my_touched[qid] = 1;
                my_sizes[qid] = 0;
            }
            h.size = my_sizes[qid];
            h.push(d, gidx, topk);
            my_sizes[qid] = h.size;
        };

        auto process_pos = [&](int pos) {
            int cid = touched_clusters[(size_t)pos];
            int qlist_size = (int)cluster_query_lists[(size_t)pos].size();
            if (qlist_size == 0) return;

            long long base_off = local_offsets[cid];
            int count = local_counts[cid];
            if (count <= 0) return;

            const int* qlist = cluster_query_lists[(size_t)pos].data();
            int qi = 0;
            for (; qi + 4 <= qlist_size; qi += 4) {
                int q0 = qlist[qi + 0];
                int q1 = qlist[qi + 1];
                int q2 = qlist[qi + 2];
                int q3 = qlist[qi + 3];
                const uint8_t* pq0 = h_query_u8 + (size_t)q0 * dim;
                const uint8_t* pq1 = h_query_u8 + (size_t)q1 * dim;
                const uint8_t* pq2 = h_query_u8 + (size_t)q2 * dim;
                const uint8_t* pq3 = h_query_u8 + (size_t)q3 * dim;

                for (int vi = 0; vi < count; ++vi) {
                    const uint8_t* v = h_base_u8 + ((size_t)base_off + vi) * dim;
                    int32_t d0, d1, d2, d3;
                    l2_sq_v3u8_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    int gidx = (int)(base_off + vi);
                    push_local(q0, (float)d0, gidx);
                    push_local(q1, (float)d1, gidx);
                    push_local(q2, (float)d2, gidx);
                    push_local(q3, (float)d3, gidx);
                }
                total_fma += (long long)count * dim * 4;
            }
            for (; qi < qlist_size; ++qi) {
                int qid = qlist[qi];
                const uint8_t* q = h_query_u8 + (size_t)qid * dim;
                for (int vi = 0; vi < count; ++vi) {
                    const uint8_t* v = h_base_u8 + ((size_t)base_off + vi) * dim;
                    int32_t d = l2_sq_v3u8_q1(q, v, dim);
                    push_local(qid, (float)d, (int)(base_off + vi));
                }
                total_fma += (long long)count * dim;
            }
        };

        if (numa_aware) {
            auto try_process_node = [&](int node) -> bool {
                int local_oi;
                #pragma omp atomic capture
                local_oi = numa_next[(size_t)node]++;
                if (local_oi >= (int)order_by_node[(size_t)node].size()) return false;
                int pos = order_by_node[(size_t)node][(size_t)local_oi];
                process_pos(pos);
                return true;
            };

            while (true) {
                if (try_process_node(my_node)) continue;
                if (strict_numa) break;
                bool stolen = false;
                for (int off = 1; off < numa_nodes; ++off) {
                    int node = (my_node + off) % numa_nodes;
                    if (try_process_node(node)) {
                        stolen = true;
                        break;
                    }
                }
                if (!stolen) break;
            }
        } else {
            #pragma omp for schedule(dynamic, 1)
            for (int oi = 0; oi < (int)order.size(); ++oi) {
                process_pos(order[(size_t)oi]);
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

}  // namespace cpu_fine
}  // namespace ivftensor
