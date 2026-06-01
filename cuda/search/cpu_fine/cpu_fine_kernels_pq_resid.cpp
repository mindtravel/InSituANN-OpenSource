/**
 * Residual PQ + rerank fine kernel.
 *
 * per query, per probed cluster
 *   1) q_resid = q_fp32 - centroids[cid]                             128  sub
 *   2) build LUT[M][K] = ||q_resid_seg_m - codebook[m][k]||           M*K  8-D L2
 *   3) scan cluster
 *        for v in cluster:
 *            d_approx = _m LUT[m][ pq_codes[v][m] ]
 *            push_stage1_topN(q, d_approx, v)
 *
 * Stage 2 rerankper query
 *   1) dedup top-N candidates  idx
 *   2) for each unique idx: d_exact = L2_u8(q_u8, base_u8[idx])
 *      or L2_fp32(q_fp32, base_fp32[idx]) when float rerank is enabled.
 *   3) push_topk(q, d_exact, idx)
 *
 * per-query  query
 *
 *   v2 (tile + cluster-major fan-out)
 *    " cluster  query "  pq_codes
 *   nlist=32768TILE_SIZE=16 tile-level  k = TILE_SIZE *
 *   nprobe / nlist = 16 * nprobe / 32768  < 1np=512  0.25
 *    amortization fan-out  fine
 *   +60% cluster-major  batch-level  +  heap
 *    v1-opt  PQ kernel
 */

#include "search/cpu_fine/cpu_fine_pq_resid.h"
#include "search/cpu_fine/cpu_fine_common.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define IVFT_PQ_HAS_AVX512BW 1
#else
#define IVFT_PQ_HAS_AVX512BW 0
#endif

namespace ivftensor {
namespace cpu_fine {

namespace {

/* ============================================================
 * high-resolution wall clock in ms
 * ============================================================ */
inline double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

/* ============================================================
 * Stage 2 rerank  u8 L2128  v3_u8 q1
 * ============================================================ */
static inline int32_t l2_sq_u8(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ v, int dim
) {
#if IVFT_PQ_HAS_AVX512BW
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
#else
    int32_t s = 0;
    for (int i = 0; i < dim; ++i) {
        int d = (int)q[i] - (int)v[i];
        s += d * d;
    }
    return s;
#endif
}

static inline float l2_sq_fp32(
    const float* __restrict__ q, const float* __restrict__ v, int dim
) {
    float s = 0.0f;
#if IVFT_PQ_HAS_AVX512BW
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 qv = _mm512_loadu_ps(q + i);
        __m512 bv = _mm512_loadu_ps(v + i);
        __m512 d = _mm512_sub_ps(qv, bv);
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    s = _mm512_reduce_add_ps(acc);
    for (; i < dim; ++i) {
        float d = q[i] - v[i];
        s += d * d;
    }
#else
    for (int i = 0; i < dim; ++i) {
        float d = q[i] - v[i];
        s += d * d;
    }
#endif
    return s;
}

/* ============================================================
 * LUT
 *    q_resid (1  dim float), codebook (M  K  d_sub float)
 *    LUT (M  K float)  (K   256  pq_codes  uint8)
 *
 *  mLUT[m][k] = _{i=0..d_sub-1} (q_resid[m*d_sub+i] - codebook[m][k][i])
 *
 * SIMD AVX-512d_sub=8
 *    2  code 8 float = 32 bytes __m256  subtract
 *    horizontal add  2  scalar
 *    d_sub=4/8 +
 * ============================================================ */
static inline void build_lut(
    const float* __restrict__ q_resid,   /* [dim] */
    const float* __restrict__ codebook,  /* [M, K, d_sub] */
    int M, int K, int d_sub,
    float* __restrict__ LUT              /* [M, K] */
) {
#if IVFT_PQ_HAS_AVX512BW
    if (d_sub == 8) {
        /* M*d_sub == dim, d_sub=8 __m256  8 float  */
        for (int m = 0; m < M; ++m) {
            __m256 q_seg = _mm256_loadu_ps(q_resid + m * 8);
            const float* cb_m = codebook + (size_t)m * K * 8;
            float* lut_m = LUT + (size_t)m * K;
            for (int k = 0; k < K; ++k) {
                __m256 cb = _mm256_loadu_ps(cb_m + k * 8);
                __m256 d = _mm256_sub_ps(q_seg, cb);
                __m256 sq = _mm256_mul_ps(d, d);
                /* horizontal sum of 8 floats */
                __m128 lo = _mm256_castps256_ps128(sq);
                __m128 hi = _mm256_extractf128_ps(sq, 1);
                __m128 sum = _mm_add_ps(lo, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                lut_m[k] = _mm_cvtss_f32(sum);
            }
        }
        return;
    }
#endif
    /*  fallback */
    for (int m = 0; m < M; ++m) {
        const float* q_seg = q_resid + m * d_sub;
        const float* cb_m = codebook + (size_t)m * K * d_sub;
        float* lut_m = LUT + (size_t)m * K;
        for (int k = 0; k < K; ++k) {
            const float* cb = cb_m + k * d_sub;
            float s = 0.0f;
            for (int i = 0; i < d_sub; ++i) {
                float d = q_seg[i] - cb[i];
                s += d * d;
            }
            lut_m[k] = s;
        }
    }
}

/* ============================================================
 * PQ
 *   dist = _{m=0..M-1} LUT[m][ pq[m] ]
 *
 *  M=16K25616  + 15 LUT 16 KB  L1
 * SIMD  vpgatherdd  16  float gather  CPU
 *
 * ============================================================ */
static inline float pq_distance(
    const float* __restrict__ LUT,  /* [M, K] */
    const uint8_t* __restrict__ pq, /* [M] */
    int M, int K
) {
    /*  unroll/schedule gather  */
    float s = 0.0f;
    const float* lut_m = LUT;
    for (int m = 0; m < M; ++m) {
        s += lut_m[pq[m]];
        lut_m += K;
    }
    return s;
}

/* ============================================================
 * M=16, K=256SIFT
 *  16  load +  reduction
 *  16  4 5-10%
 * ============================================================ */
static inline float pq_distance_M16_K256(
    const float* __restrict__ LUT,   /* 16 * 256 floats */
    const uint8_t* __restrict__ pq   /* 16 bytes */
) {
    float s0 = LUT[ 0 * 256 + pq[ 0]] + LUT[ 1 * 256 + pq[ 1]];
    float s1 = LUT[ 2 * 256 + pq[ 2]] + LUT[ 3 * 256 + pq[ 3]];
    float s2 = LUT[ 4 * 256 + pq[ 4]] + LUT[ 5 * 256 + pq[ 5]];
    float s3 = LUT[ 6 * 256 + pq[ 6]] + LUT[ 7 * 256 + pq[ 7]];
    float s4 = LUT[ 8 * 256 + pq[ 8]] + LUT[ 9 * 256 + pq[ 9]];
    float s5 = LUT[10 * 256 + pq[10]] + LUT[11 * 256 + pq[11]];
    float s6 = LUT[12 * 256 + pq[12]] + LUT[13 * 256 + pq[13]];
    float s7 = LUT[14 * 256 + pq[14]] + LUT[15 * 256 + pq[15]];
    float t0 = (s0 + s1) + (s2 + s3);
    float t1 = (s4 + s5) + (s6 + s7);
    return t0 + t1;
}

/* ============================================================
 * M=8, K=256BW-pq_codes  8 B/vec
 *  BW-bound  bs +  nprobe pq_M=8  Stage 1 BW
 *  rerank_n  recall
 *  38  4  2  1
 * ============================================================ */
static inline float pq_distance_M8_K256(
    const float* __restrict__ LUT,   /* 8 * 256 floats */
    const uint8_t* __restrict__ pq   /* 8 bytes */
) {
    float s0 = LUT[0 * 256 + pq[0]] + LUT[1 * 256 + pq[1]];
    float s1 = LUT[2 * 256 + pq[2]] + LUT[3 * 256 + pq[3]];
    float s2 = LUT[4 * 256 + pq[4]] + LUT[5 * 256 + pq[5]];
    float s3 = LUT[6 * 256 + pq[6]] + LUT[7 * 256 + pq[7]];
    return (s0 + s1) + (s2 + s3);
}

}  // namespace

/* ============================================================
 *  kernel
 * ============================================================ */
long long cpu_fine_kernel_pq_resid(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_pq_codes,
    const float* h_codebook,
    const float* h_centroids,
    const uint8_t* h_query_u8,
    const float* h_query_fp32,
    const float* h_base_fp32,
    int use_float_rerank,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int rerank_n,
    int pq_M,
    int pq_K,
    int num_threads,
    int* h_topk_local_idx,
    float* h_topk_dist,
    double* out_stage1_ms,
    double* out_stage2_ms
) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (rerank_n < topk) rerank_n = topk;
    if (pq_K > 256) pq_K = 256;  /* pq_codes  u8 K>256 */
    const int d_sub = dim / pq_M;

    /* ---------- Stage 1  ---------- */
    const double s1_t0 = now_ms();

    /*  query  stage1 top-N heap N  */
    std::vector<float> s1_dists((size_t)n_query * (size_t)rerank_n);
    std::vector<int>   s1_idxs ((size_t)n_query * (size_t)rerank_n);
    std::vector<int>   s1_sizes((size_t)n_query, 0);

    long long total_fma = 0;

    #pragma omp parallel num_threads(num_threads) reduction(+:total_fma)
    {
        /* per-thread scratch */
        std::vector<float> LUT((size_t)pq_M * (size_t)pq_K);
        std::vector<float> q_resid((size_t)dim);

        #pragma omp for schedule(dynamic, 4)
        for (int qi = 0; qi < n_query; ++qi) {
            float* s1d = s1_dists.data() + (size_t)qi * rerank_n;
            int*   s1i = s1_idxs .data() + (size_t)qi * rerank_n;
            int&   s1n = s1_sizes[qi];
            s1n = 0;

            /* heap helpers: max-heap keeps the N smallest distances.
             * root = worst-of-top-N. New x replaces root if x < root. */
            auto sift_up = [&](int pos){
                while (pos > 0) {
                    int parent = (pos - 1) >> 1;
                    if (s1d[parent] >= s1d[pos]) break;
                    std::swap(s1d[parent], s1d[pos]);
                    std::swap(s1i[parent], s1i[pos]);
                    pos = parent;
                }
            };
            auto sift_down = [&](int n, int pos){
                while (true) {
                    int l = 2*pos+1, r = 2*pos+2, largest = pos;
                    if (l < n && s1d[l] > s1d[largest]) largest = l;
                    if (r < n && s1d[r] > s1d[largest]) largest = r;
                    if (largest == pos) break;
                    std::swap(s1d[pos], s1d[largest]);
                    std::swap(s1i[pos], s1i[largest]);
                    pos = largest;
                }
            };
            auto push_s1 = [&](float dist, int idx){
                if (s1n < rerank_n) {
                    s1d[s1n] = dist;
                    s1i[s1n] = idx;
                    ++s1n;
                    sift_up(s1n - 1);
                } else if (dist < s1d[0]) {
                    s1d[0] = dist;
                    s1i[0] = idx;
                    sift_down(s1n, 0);
                }
            };

            const float* q_fp32 = h_query_fp32 + (size_t)qi * dim;
            const int*   cids   = h_coarse_cluster_ids + (size_t)qi * n_probes;

            /*  probed cluster */
            for (int pi = 0; pi < n_probes; ++pi) {
                int cid = cids[pi];
                if (cid < 0 || cid >= n_total_clusters) continue;
                int count = h_cluster_counts[cid];
                if (count <= 0) continue;
                long long base_off = h_cluster_offsets[cid];
                const float* c_center = h_centroids + (size_t)cid * dim;

                /* q_resid = q_fp32 - centroid[cid] */
#if IVFT_PQ_HAS_AVX512BW
                {
                    int i = 0;
                    for (; i + 16 <= dim; i += 16) {
                        __m512 a = _mm512_loadu_ps(q_fp32 + i);
                        __m512 b = _mm512_loadu_ps(c_center + i);
                        _mm512_storeu_ps(q_resid.data() + i, _mm512_sub_ps(a, b));
                    }
                    for (; i < dim; ++i) q_resid[i] = q_fp32[i] - c_center[i];
                }
#else
                for (int i = 0; i < dim; ++i) q_resid[i] = q_fp32[i] - c_center[i];
#endif

                /*  LUT */
                build_lut(q_resid.data(), h_codebook, pq_M, pq_K, d_sub, LUT.data());

                /*  cluster
                 *
                 *   (A) pq_M=16/8, pq_K=256  pq_distance
                 *   (B) threshold short-circuit s1d[0]  top_thresh
                 *       d_approx >= top_thresh  push_s1  sift
                 *        n_probes
                 */
                const uint8_t* pq_base = h_pq_codes + (size_t)base_off * pq_M;
                constexpr float FLOAT_INF = std::numeric_limits<float>::max();
                float top_thresh = (s1n >= rerank_n) ? s1d[0] : FLOAT_INF;

                if (pq_M == 16 && pq_K == 256) {
                    /* fast path M=16 */
                    for (int vi = 0; vi < count; ++vi) {
                        const uint8_t* pq = pq_base + (size_t)vi * 16;
                        float d_approx = pq_distance_M16_K256(LUT.data(), pq);
                        if (d_approx >= top_thresh) continue;
                        int gidx = (int)(base_off + vi);
                        push_s1(d_approx, gidx);
                        top_thresh = (s1n >= rerank_n) ? s1d[0] : FLOAT_INF;
                    }
                } else if (pq_M == 8 && pq_K == 256) {
                    /* fast path M=8BW-friendlypq_codes 8 B/vecStage 1 BW  */
                    for (int vi = 0; vi < count; ++vi) {
                        const uint8_t* pq = pq_base + (size_t)vi * 8;
                        float d_approx = pq_distance_M8_K256(LUT.data(), pq);
                        if (d_approx >= top_thresh) continue;
                        int gidx = (int)(base_off + vi);
                        push_s1(d_approx, gidx);
                        top_thresh = (s1n >= rerank_n) ? s1d[0] : FLOAT_INF;
                    }
                } else {
                    /* generic fallback */
                    for (int vi = 0; vi < count; ++vi) {
                        const uint8_t* pq = pq_base + (size_t)vi * pq_M;
                        float d_approx = pq_distance(LUT.data(), pq, pq_M, pq_K);
                        if (d_approx >= top_thresh) continue;
                        int gidx = (int)(base_off + vi);
                        push_s1(d_approx, gidx);
                        top_thresh = (s1n >= rerank_n) ? s1d[0] : FLOAT_INF;
                    }
                }
                total_fma += (long long)count * pq_M;  /*  FMA  */
            }
        }
    }
    const double s1_t1 = now_ms();
    if (out_stage1_ms) *out_stage1_ms = s1_t1 - s1_t0;

    /* ---------- Stage 2 rerank  ---------- */
    const double s2_t0 = now_ms();

    /*  query dedup  exact L2  top-k */
    #pragma omp parallel num_threads(num_threads)
    {
        /* thread-local topk */
        std::vector<float> tkd((size_t)topk);
        std::vector<int>   tki((size_t)topk);
        /* thread-local dedup buffer idx typical rerank_n  500 */
        std::vector<int> sorted_ids;
        sorted_ids.reserve(rerank_n);

        #pragma omp for schedule(dynamic, 8)
        for (int qi = 0; qi < n_query; ++qi) {
            int s1n = s1_sizes[qi];
            const int* s1i = s1_idxs.data() + (size_t)qi * rerank_n;

            /*  sorted_ids */
            sorted_ids.clear();
            sorted_ids.insert(sorted_ids.end(), s1i, s1i + s1n);
            std::sort(sorted_ids.begin(), sorted_ids.end());
            auto last = std::unique(sorted_ids.begin(), sorted_ids.end());
            sorted_ids.erase(last, sorted_ids.end());

            /*  top-k heap */
            TopKHeap h;
            h.init(topk, tkd.data(), tki.data());

            const uint8_t* qv_u8 = h_query_u8 ? (h_query_u8 + (size_t)qi * dim) : nullptr;
            const float* qv_fp32 = h_query_fp32 + (size_t)qi * dim;

            /* Rerank  cache-miss  seek  base  128B
             *  PREFETCH_DIST  L3/DRAM  l2_sq_u8
             * PREFETCH_DIST=8  64-thread  L1/L2
             * 128B  2  cacheline prefetch 2
             */
            constexpr int PREFETCH_DIST = 8;
            const int ns = (int)sorted_ids.size();
#if IVFT_PQ_HAS_AVX512BW
            /*  PREFETCH_DIST  */
            for (int k = 0; k < PREFETCH_DIST && k < ns; ++k) {
                int idx0 = sorted_ids[k];
                if (idx0 >= 0) {
                    if (use_float_rerank && h_base_fp32) {
                        const float* p = h_base_fp32 + (size_t)idx0 * dim;
                        _mm_prefetch((const char*)p,         _MM_HINT_T0);
                        _mm_prefetch((const char*)(p + 16),  _MM_HINT_T0);
                    } else {
                        const uint8_t* p = h_base_u8 + (size_t)idx0 * dim;
                        _mm_prefetch((const char*)p,        _MM_HINT_T0);
                        _mm_prefetch((const char*)(p + 64), _MM_HINT_T0);
                    }
                }
            }
#endif
            for (int k = 0; k < ns; ++k) {
#if IVFT_PQ_HAS_AVX512BW
                if (k + PREFETCH_DIST < ns) {
                    int nxt_idx = sorted_ids[k + PREFETCH_DIST];
                    if (nxt_idx >= 0) {
                        if (use_float_rerank && h_base_fp32) {
                            const float* nxt = h_base_fp32 + (size_t)nxt_idx * dim;
                            _mm_prefetch((const char*)nxt,        _MM_HINT_T0);
                            _mm_prefetch((const char*)(nxt + 16), _MM_HINT_T0);
                        } else {
                            const uint8_t* nxt = h_base_u8 + (size_t)nxt_idx * dim;
                            _mm_prefetch((const char*)nxt,        _MM_HINT_T0);
                            _mm_prefetch((const char*)(nxt + 64), _MM_HINT_T0);
                        }
                    }
                }
#endif
                int idx = sorted_ids[k];
                if (idx < 0) continue;
                float d_exact = 0.0f;
                if (use_float_rerank && h_base_fp32) {
                    const float* bv = h_base_fp32 + (size_t)idx * dim;
                    d_exact = l2_sq_fp32(qv_fp32, bv, dim);
                } else {
                    const uint8_t* bv = h_base_u8 + (size_t)idx * dim;
                    int32_t d_exact_i = l2_sq_u8(qv_u8, bv, dim);
                    d_exact = (float)d_exact_i;
                }
                h.push(d_exact, idx);
            }

            finalize_topk(h,
                          h_topk_dist + (size_t)qi * topk,
                          h_topk_local_idx + (size_t)qi * topk,
                          topk);
        }
    }

    const double s2_t1 = now_ms();
    if (out_stage2_ms) *out_stage2_ms = s2_t1 - s2_t0;

    return total_fma;
}

}  // namespace cpu_fine
}  // namespace ivftensor
