/**
 * V3V2 + Register-level Query Tiling
 *
 * cputo_do_list.md 3.2
 *   "Load one base vector, compute against Q queries simultaneously,
 *    keeping queries in AVX registers to amortize memory load costs."
 *
 *
 *   1. cluster   probing  cluster  (query_id, probe_pos)
 *   2. #pragma omp parallel for schedule(dynamic) over clusters
 *   3.  cluster  probing queries  Q=4  tile
 *   4.  tile cluster  base vectors base
 *       4  query  L2 4  FMA
 *   5.  per-thread  topk heap  topk
 *
 *  race
 *   -  thread_local heapthread_dists / thread_idxs
 *   - Merge  parallel over queries query
 *
 * -O3 -march=native -mavx512f -mfma -fopenmp
 */

#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_common.h"
#include "numa_utils.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__) && defined(__FMA__)
#include <immintrin.h>
#define IVFTENSOR_V3_HAS_AVX512 1
#else
#define IVFTENSOR_V3_HAS_AVX512 0
#endif

namespace ivftensor {
namespace cpu_fine {

/* ======================================================================
 * Q=4 Query Tiling Kernel 4  query vs 1  base vector  L2^2
 * ====================================================================== */

#if IVFTENSOR_V3_HAS_AVX512

static inline void l2_sq_v3_q4(
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

    /*  dim  16  tail SIFT/BERT  */
    for (; i < dim; ++i) {
        float a0 = q0[i] - v[i]; out0 += a0 * a0;
        float a1 = q1[i] - v[i]; out1 += a1 * a1;
        float a2 = q2[i] - v[i]; out2 += a2 * a2;
        float a3 = q3[i] - v[i]; out3 += a3 * a3;
    }
}

static inline float l2_sq_v3_q1(
    const float* __restrict__ q, const float* __restrict__ v, int dim
) {
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

#else

static inline void l2_sq_v3_q4(
    const float* q0, const float* q1, const float* q2, const float* q3,
    const float* v, int dim,
    float& o0, float& o1, float& o2, float& o3
) {
    o0 = o1 = o2 = o3 = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float a0 = q0[i] - v[i]; o0 += a0 * a0;
        float a1 = q1[i] - v[i]; o1 += a1 * a1;
        float a2 = q2[i] - v[i]; o2 += a2 * a2;
        float a3 = q3[i] - v[i]; o3 += a3 * a3;
    }
}

static inline float l2_sq_v3_q1(const float* q, const float* v, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = q[i] - v[i]; s += d * d; }
    return s;
}

#endif

/* ======================================================================
 * Per-thread  TopKHeap heap
 * ====================================================================== */

struct LocalTopK {
    int size;
    float* dists;  /* capacity == topk */
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

long long cpu_fine_kernel_v3(
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
    (void)distance_mode;  /*  L2cos  V0/V1 */
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    /* ---------- Step 1. cluster   of query_ids ---------- */
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

    /* ---------- Step 2.  per-thread  TopKHeaplazy init ---------- */
    int T = num_threads;
    /* thread_dists[t][q][k] laid out contiguously */
    std::vector<float> th_dists((size_t)T * n_query * topk);
    std::vector<int>   th_idxs((size_t)T * n_query * topk);
    std::vector<int>   th_sizes((size_t)T * n_query, 0);
    std::vector<uint8_t> th_touched((size_t)T * n_query, 0);

    /* ---------- Step 3. Parallel over clusters ---------- */
    long long total_fma = 0;
    #pragma omp parallel num_threads(T) reduction(+:total_fma)
    {
        int tid = omp_get_thread_num();
        float* my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*   my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*   my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopK h;
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

            /* Q=4 query-tiled  */
            int qi = 0;
            for (; qi + 4 <= qlist_size; qi += 4) {
                int q0 = qlist[qi + 0];
                int q1 = qlist[qi + 1];
                int q2 = qlist[qi + 2];
                int q3 = qlist[qi + 3];
                const float* pq0 = h_query + (size_t)q0 * dim;
                const float* pq1 = h_query + (size_t)q1 * dim;
                const float* pq2 = h_query + (size_t)q2 * dim;
                const float* pq3 = h_query + (size_t)q3 * dim;

                for (int vi = 0; vi < count; ++vi) {
                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d0, d1, d2, d3;
                    l2_sq_v3_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    int gidx = (int)(base_off + vi);
                    push_local(q0, d0, gidx);
                    push_local(q1, d1, gidx);
                    push_local(q2, d2, gidx);
                    push_local(q3, d3, gidx);
                }
                total_fma += (long long)count * dim * 4;
            }
            /*  4  query1..3  Q=1  */
            for (; qi < qlist_size; ++qi) {
                int qid = qlist[qi];
                const float* q = h_query + (size_t)qid * dim;
                for (int vi = 0; vi < count; ++vi) {
                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d = l2_sq_v3_q1(q, v, dim);
                    push_local(qid, d, (int)(base_off + vi));
                }
                total_fma += (long long)count * dim;
            }
        }
    }

    /* ---------- Step 4. Merge per-thread heaps  global topk ---------- */
    #pragma omp parallel for schedule(dynamic, 64) num_threads(T)
    for (int qi = 0; qi < n_query; ++qi) {
        /*  TopKHeap  */
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

long long cpu_fine_kernel_v3_touched(
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

        float* my_dists = th_dists.data() + (size_t)tid * n_query * topk;
        int*   my_idxs  = th_idxs.data()  + (size_t)tid * n_query * topk;
        int*   my_sizes = th_sizes.data() + (size_t)tid * n_query;
        uint8_t* my_touched = th_touched.data() + (size_t)tid * n_query;

        auto push_local = [&](int qid, float d, int gidx) {
            LocalTopK h;
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
                const float* pq0 = h_query + (size_t)q0 * dim;
                const float* pq1 = h_query + (size_t)q1 * dim;
                const float* pq2 = h_query + (size_t)q2 * dim;
                const float* pq3 = h_query + (size_t)q3 * dim;

                for (int vi = 0; vi < count; ++vi) {
                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d0, d1, d2, d3;
                    l2_sq_v3_q4(pq0, pq1, pq2, pq3, v, dim, d0, d1, d2, d3);
                    int gidx = (int)(base_off + vi);
                    push_local(q0, d0, gidx);
                    push_local(q1, d1, gidx);
                    push_local(q2, d2, gidx);
                    push_local(q3, d3, gidx);
                }
                total_fma += (long long)count * dim * 4;
            }
            for (; qi < qlist_size; ++qi) {
                int qid = qlist[qi];
                const float* q = h_query + (size_t)qid * dim;
                for (int vi = 0; vi < count; ++vi) {
                    const float* v = h_base + ((size_t)base_off + vi) * dim;
                    float d = l2_sq_v3_q1(q, v, dim);
                    push_local(qid, d, (int)(base_off + vi));
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

const char* variant_name_v3() { return "V3_avx512_query_tile_Q4"; }

}  // namespace cpu_fine
}  // namespace ivftensor
