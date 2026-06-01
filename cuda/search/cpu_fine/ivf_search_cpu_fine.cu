/**
 * IVF GPU  + CPU
 *
 *  API
 *   (1) ivf_search_cpu_fine()
 *   (2) coarse_handle_init / _release       GPU  +  coarse API
 *       coarse_search()
 *   (3) fine_search_cpu()                    CPU fine batch
 *
 *  batch-loop / cross-batch accumulation  (2)+(3)
 *   - (2)  GPU  centers
 *   - (3)  CPU  batch  query
 *          V3  "per-cluster qlist >= 4"  batch
 *
 *  ivf_search_no_lookup.cu  cuBLAS + fused warpsort
 */

#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
#include "pch.h"
#include "search/cpu_fine/cpu_fine.h"
#include "search/coarse/fusion_dist_topk.cuh"
#include "l2norm/l2norm.cuh"
#include "utils.cuh"
#include <cub/cub.cuh>

#include <algorithm>
#include <vector>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

using namespace ivftensor;

inline double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static thread_local CoarseTimingBreakdown g_last_coarse_timing;

static double elapsed_cuda_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return (double)ms;
}

/**  GPU  query  [0, 1, 2, ..., n_total_clusters-1]
 *  warpsort kernel  [n_query, n_total_clusters] per-query  candidate index */
__global__ void gen_per_query_seq_idx_kernel(int* d_out, int n_query, int n_total_clusters) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_query * n_total_clusters;
    if (tid < total) d_out[tid] = (int)(tid % n_total_clusters);
}

constexpr int kTopkTile = 4096;
constexpr int kCubTopkTile = 4096;
constexpr int kCubTopkThreads = 256;
constexpr int kCubTopkItems = 16;
constexpr int kSmallTopkTile = 4096;
constexpr int kSmallTopkThreads = 256;
constexpr int kSmallTopkItems = 16;

__device__ __forceinline__ bool topk_pair_greater(float da, int ia, float db, int ib) {
    return (da > db) || (da == db && ia > ib);
}

__device__ __forceinline__ bool topk_pair_less(float da, int ia, float db, int ib) {
    return (da < db) || (da == db && ia < ib);
}

__device__ void bitonic_sort_tile(float* s_dist, int* s_idx) {
    for (int size = 2; size <= kTopkTile; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < kTopkTile; i += blockDim.x) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    float di = s_dist[i], dj = s_dist[j];
                    int ii = s_idx[i], ij = s_idx[j];
                    bool swap = ascending
                        ? topk_pair_greater(di, ii, dj, ij)
                        : topk_pair_less(di, ii, dj, ij);
                    if (swap) {
                        s_dist[i] = dj; s_idx[i] = ij;
                        s_dist[j] = di; s_idx[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void twostage_l2_topk_initial_kernel(
    const float* __restrict__ d_inner_product,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    __shared__ float s_dist[kTopkTile];
    __shared__ int s_idx[kTopkTile];
    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kTopkTile;
    float qn = d_query_norm[q];
    float qn2 = qn * qn;
    for (int i = threadIdx.x; i < kTopkTile; i += blockDim.x) {
        int c = base + i;
        if (c < n_centers) {
            float cn = d_centers_norm[c];
            float ip = d_inner_product[(size_t)q * n_centers + c];
            s_dist[i] = qn2 + cn * cn - 2.0f * ip;
            s_idx[i] = c;
        } else {
            s_dist[i] = FLT_MAX;
            s_idx[i] = INT_MAX;
        }
    }
    __syncthreads();
    bitonic_sort_tile(s_dist, s_idx);
    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        d_out_dist[out_base + i] = s_dist[i];
        d_out_idx[out_base + i] = s_idx[i] == INT_MAX ? -1 : s_idx[i];
    }
}

__global__ void twostage_topk_merge_kernel(
    const float* __restrict__ d_in_dist,
    const int* __restrict__ d_in_idx,
    int n_query,
    int in_count,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    __shared__ float s_dist[kTopkTile];
    __shared__ int s_idx[kTopkTile];
    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kTopkTile;
    for (int i = threadIdx.x; i < kTopkTile; i += blockDim.x) {
        int pos = base + i;
        if (pos < in_count) {
            size_t in_off = (size_t)q * in_count + pos;
            s_dist[i] = d_in_dist[in_off];
            int idx = d_in_idx[in_off];
            s_idx[i] = idx < 0 ? INT_MAX : idx;
        } else {
            s_dist[i] = FLT_MAX;
            s_idx[i] = INT_MAX;
        }
    }
    __syncthreads();
    bitonic_sort_tile(s_dist, s_idx);
    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        d_out_dist[out_base + i] = s_dist[i];
        d_out_idx[out_base + i] = s_idx[i] == INT_MAX ? -1 : s_idx[i];
    }
}

bool coarse_topk_twostage_l2(
    const float* d_inner_product,
    const float* d_query_norm,
    const float* d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    float* d_top_dist,
    int* d_top_idx)
{
    if (k <= 0 || k > 512 || n_centers <= 0) return false;
    int n_tiles = (n_centers + kTopkTile - 1) / kTopkTile;
    size_t cand_count = (size_t)n_query * n_tiles * k;
    float* d_buf1_dist = nullptr; int* d_buf1_idx = nullptr;
    float* d_buf2_dist = nullptr; int* d_buf2_idx = nullptr;
    int in_count = n_tiles * k;
    float* cur_dist = nullptr; int* cur_idx = nullptr;
    float* other_dist = nullptr; int* other_idx = nullptr;
    cudaMalloc(&d_buf1_dist, cand_count * sizeof(float));
    cudaMalloc(&d_buf1_idx,  cand_count * sizeof(int));
    if (!d_buf1_dist || !d_buf1_idx) goto fail;
    {
        dim3 grid(n_tiles, n_query);
        twostage_l2_topk_initial_kernel<<<grid, 256>>>(
            d_inner_product, d_query_norm, d_centers_norm,
            n_query, n_centers, k, n_tiles, d_buf1_dist, d_buf1_idx);
    }
    cur_dist = d_buf1_dist; cur_idx = d_buf1_idx;
    while (in_count > k) {
        int merge_tiles = (in_count + kTopkTile - 1) / kTopkTile;
        size_t out_count = (size_t)n_query * merge_tiles * k;
        if (!d_buf2_dist) {
            cudaMalloc(&d_buf2_dist, out_count * sizeof(float));
            cudaMalloc(&d_buf2_idx,  out_count * sizeof(int));
            if (!d_buf2_dist || !d_buf2_idx) goto fail;
        }
        other_dist = (cur_dist == d_buf1_dist) ? d_buf2_dist : d_buf1_dist;
        other_idx  = (cur_idx  == d_buf1_idx)  ? d_buf2_idx  : d_buf1_idx;
        dim3 grid(merge_tiles, n_query);
        twostage_topk_merge_kernel<<<grid, 256>>>(
            cur_dist, cur_idx, n_query, in_count, k, merge_tiles, other_dist, other_idx);
        cur_dist = other_dist; cur_idx = other_idx;
        in_count = merge_tiles * k;
    }
    cudaMemcpy(d_top_dist, cur_dist, (size_t)n_query * k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_top_idx,  cur_idx,  (size_t)n_query * k * sizeof(int),   cudaMemcpyDeviceToDevice);
    cudaFree(d_buf1_dist); cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return true;
fail:
    if (d_buf1_dist) cudaFree(d_buf1_dist);
    if (d_buf1_idx) cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return false;
}

__global__ void cub_l2_topk_initial_kernel(
    const float* __restrict__ d_inner_product,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kCubTopkTile;
    float qn = d_query_norm[q];
    float qn2 = qn * qn;

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        int local = threadIdx.x * kCubTopkItems + item;
        int c = base + local;
        if (c < n_centers) {
            float cn = d_centers_norm[c];
            float ip = d_inner_product[(size_t)q * n_centers + c];
            keys[item] = qn2 + cn * cn - 2.0f * ip;
            vals[item] = c;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            d_out_dist[out_base + rank] = keys[item];
            d_out_idx[out_base + rank] = vals[item] == INT_MAX ? -1 : vals[item];
        }
    }
}

__global__ void cub_topk_merge_kernel(
    const float* __restrict__ d_in_dist,
    const int* __restrict__ d_in_idx,
    int n_query,
    int in_count,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    using BlockSort = cub::BlockRadixSort<float, kCubTopkThreads, kCubTopkItems, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float keys[kCubTopkItems];
    int vals[kCubTopkItems];

    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kCubTopkTile;

    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        int local = threadIdx.x * kCubTopkItems + item;
        int pos = base + local;
        if (pos < in_count) {
            size_t in_off = (size_t)q * in_count + pos;
            keys[item] = d_in_dist[in_off];
            int idx = d_in_idx[in_off];
            vals[item] = idx < 0 ? INT_MAX : idx;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    BlockSort(sort_storage).Sort(keys, vals);

    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    #pragma unroll
    for (int item = 0; item < kCubTopkItems; ++item) {
        int rank = threadIdx.x * kCubTopkItems + item;
        if (rank < k) {
            d_out_dist[out_base + rank] = keys[item];
            d_out_idx[out_base + rank] = vals[item] == INT_MAX ? -1 : vals[item];
        }
    }
}

bool coarse_topk_cub_l2(
    const float* d_inner_product,
    const float* d_query_norm,
    const float* d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    float* d_top_dist,
    int* d_top_idx)
{
    if (k <= 0 || k > 512 || n_centers <= 0) return false;
    int n_tiles = (n_centers + kCubTopkTile - 1) / kCubTopkTile;
    size_t cand_count = (size_t)n_query * n_tiles * k;
    float* d_buf1_dist = nullptr; int* d_buf1_idx = nullptr;
    float* d_buf2_dist = nullptr; int* d_buf2_idx = nullptr;
    int in_count = n_tiles * k;
    float* cur_dist = nullptr; int* cur_idx = nullptr;
    float* other_dist = nullptr; int* other_idx = nullptr;
    cudaMalloc(&d_buf1_dist, cand_count * sizeof(float));
    cudaMalloc(&d_buf1_idx,  cand_count * sizeof(int));
    if (!d_buf1_dist || !d_buf1_idx) goto fail;
    {
        dim3 grid(n_tiles, n_query);
        cub_l2_topk_initial_kernel<<<grid, kCubTopkThreads>>>(
            d_inner_product, d_query_norm, d_centers_norm,
            n_query, n_centers, k, n_tiles, d_buf1_dist, d_buf1_idx);
    }
    cur_dist = d_buf1_dist; cur_idx = d_buf1_idx;
    while (in_count > k) {
        int merge_tiles = (in_count + kCubTopkTile - 1) / kCubTopkTile;
        size_t out_count = (size_t)n_query * merge_tiles * k;
        if (!d_buf2_dist) {
            cudaMalloc(&d_buf2_dist, out_count * sizeof(float));
            cudaMalloc(&d_buf2_idx,  out_count * sizeof(int));
            if (!d_buf2_dist || !d_buf2_idx) goto fail;
        }
        other_dist = (cur_dist == d_buf1_dist) ? d_buf2_dist : d_buf1_dist;
        other_idx  = (cur_idx  == d_buf1_idx)  ? d_buf2_idx  : d_buf1_idx;
        dim3 grid(merge_tiles, n_query);
        cub_topk_merge_kernel<<<grid, kCubTopkThreads>>>(
            cur_dist, cur_idx, n_query, in_count, k, merge_tiles, other_dist, other_idx);
        cur_dist = other_dist; cur_idx = other_idx;
        in_count = merge_tiles * k;
    }
    cudaMemcpy(d_top_dist, cur_dist, (size_t)n_query * k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_top_idx,  cur_idx,  (size_t)n_query * k * sizeof(int),   cudaMemcpyDeviceToDevice);
    cudaFree(d_buf1_dist); cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return true;
fail:
    if (d_buf1_dist) cudaFree(d_buf1_dist);
    if (d_buf1_idx) cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return false;
}

struct SmallTopkPair {
    float dist;
    int idx;
};

struct SmallTopkMin {
    __device__ __forceinline__ SmallTopkPair operator()(const SmallTopkPair& a, const SmallTopkPair& b) const {
        return topk_pair_less(b.dist, b.idx, a.dist, a.idx) ? b : a;
    }
};

__global__ void small_l2_topk_initial_kernel(
    const float* __restrict__ d_inner_product,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    using BlockReduce = cub::BlockReduce<SmallTopkPair, kSmallTopkThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ SmallTopkPair selected;
    float keys[kSmallTopkItems];
    int vals[kSmallTopkItems];

    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kSmallTopkTile;
    float qn = d_query_norm[q];
    float qn2 = qn * qn;

    #pragma unroll
    for (int item = 0; item < kSmallTopkItems; ++item) {
        int local = threadIdx.x * kSmallTopkItems + item;
        int c = base + local;
        if (c < n_centers) {
            float cn = d_centers_norm[c];
            float ip = d_inner_product[(size_t)q * n_centers + c];
            keys[item] = qn2 + cn * cn - 2.0f * ip;
            vals[item] = c;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    for (int rank = 0; rank < k; ++rank) {
        SmallTopkPair local_best{FLT_MAX, INT_MAX};
        #pragma unroll
        for (int item = 0; item < kSmallTopkItems; ++item) {
            if (topk_pair_less(keys[item], vals[item], local_best.dist, local_best.idx)) {
                local_best.dist = keys[item];
                local_best.idx = vals[item];
            }
        }
        SmallTopkPair block_best = BlockReduce(reduce_storage).Reduce(local_best, SmallTopkMin());
        if (threadIdx.x == 0) {
            selected = block_best;
            d_out_dist[out_base + rank] = block_best.dist;
            d_out_idx[out_base + rank] = block_best.idx == INT_MAX ? -1 : block_best.idx;
        }
        __syncthreads();
        #pragma unroll
        for (int item = 0; item < kSmallTopkItems; ++item) {
            if (vals[item] == selected.idx) {
                keys[item] = FLT_MAX;
                vals[item] = INT_MAX;
            }
        }
        __syncthreads();
    }
}

__global__ void small_topk_merge_kernel(
    const float* __restrict__ d_in_dist,
    const int* __restrict__ d_in_idx,
    int n_query,
    int in_count,
    int k,
    int n_tiles,
    float* __restrict__ d_out_dist,
    int* __restrict__ d_out_idx)
{
    using BlockReduce = cub::BlockReduce<SmallTopkPair, kSmallTopkThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ SmallTopkPair selected;
    float keys[kSmallTopkItems];
    int vals[kSmallTopkItems];

    int tile = blockIdx.x;
    int q = blockIdx.y;
    int base = tile * kSmallTopkTile;

    #pragma unroll
    for (int item = 0; item < kSmallTopkItems; ++item) {
        int local = threadIdx.x * kSmallTopkItems + item;
        int pos = base + local;
        if (pos < in_count) {
            size_t in_off = (size_t)q * in_count + pos;
            keys[item] = d_in_dist[in_off];
            int idx = d_in_idx[in_off];
            vals[item] = idx < 0 ? INT_MAX : idx;
        } else {
            keys[item] = FLT_MAX;
            vals[item] = INT_MAX;
        }
    }

    size_t out_base = ((size_t)q * n_tiles + tile) * k;
    for (int rank = 0; rank < k; ++rank) {
        SmallTopkPair local_best{FLT_MAX, INT_MAX};
        #pragma unroll
        for (int item = 0; item < kSmallTopkItems; ++item) {
            if (topk_pair_less(keys[item], vals[item], local_best.dist, local_best.idx)) {
                local_best.dist = keys[item];
                local_best.idx = vals[item];
            }
        }
        SmallTopkPair block_best = BlockReduce(reduce_storage).Reduce(local_best, SmallTopkMin());
        if (threadIdx.x == 0) {
            selected = block_best;
            d_out_dist[out_base + rank] = block_best.dist;
            d_out_idx[out_base + rank] = block_best.idx == INT_MAX ? -1 : block_best.idx;
        }
        __syncthreads();
        #pragma unroll
        for (int item = 0; item < kSmallTopkItems; ++item) {
            if (vals[item] == selected.idx) {
                keys[item] = FLT_MAX;
                vals[item] = INT_MAX;
            }
        }
        __syncthreads();
    }
}

bool coarse_topk_small_l2(
    const float* d_inner_product,
    const float* d_query_norm,
    const float* d_centers_norm,
    int n_query,
    int n_centers,
    int k,
    float* d_top_dist,
    int* d_top_idx)
{
    if (k <= 0 || k > 64 || n_centers <= 0) return false;
    int n_tiles = (n_centers + kSmallTopkTile - 1) / kSmallTopkTile;
    size_t cand_count = (size_t)n_query * n_tiles * k;
    float* d_buf1_dist = nullptr; int* d_buf1_idx = nullptr;
    float* d_buf2_dist = nullptr; int* d_buf2_idx = nullptr;
    int in_count = n_tiles * k;
    float* cur_dist = nullptr; int* cur_idx = nullptr;
    float* other_dist = nullptr; int* other_idx = nullptr;
    cudaMalloc(&d_buf1_dist, cand_count * sizeof(float));
    cudaMalloc(&d_buf1_idx,  cand_count * sizeof(int));
    if (!d_buf1_dist || !d_buf1_idx) goto fail;
    {
        dim3 grid(n_tiles, n_query);
        small_l2_topk_initial_kernel<<<grid, kSmallTopkThreads>>>(
            d_inner_product, d_query_norm, d_centers_norm,
            n_query, n_centers, k, n_tiles, d_buf1_dist, d_buf1_idx);
    }
    cur_dist = d_buf1_dist; cur_idx = d_buf1_idx;
    while (in_count > k) {
        int merge_tiles = (in_count + kSmallTopkTile - 1) / kSmallTopkTile;
        size_t out_count = (size_t)n_query * merge_tiles * k;
        if (!d_buf2_dist) {
            cudaMalloc(&d_buf2_dist, out_count * sizeof(float));
            cudaMalloc(&d_buf2_idx,  out_count * sizeof(int));
            if (!d_buf2_dist || !d_buf2_idx) goto fail;
        }
        other_dist = (cur_dist == d_buf1_dist) ? d_buf2_dist : d_buf1_dist;
        other_idx  = (cur_idx  == d_buf1_idx)  ? d_buf2_idx  : d_buf1_idx;
        dim3 grid(merge_tiles, n_query);
        small_topk_merge_kernel<<<grid, kSmallTopkThreads>>>(
            cur_dist, cur_idx, n_query, in_count, k, merge_tiles, other_dist, other_idx);
        cur_dist = other_dist; cur_idx = other_idx;
        in_count = merge_tiles * k;
    }
    cudaMemcpy(d_top_dist, cur_dist, (size_t)n_query * k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_top_idx,  cur_idx,  (size_t)n_query * k * sizeof(int),   cudaMemcpyDeviceToDevice);
    cudaFree(d_buf1_dist); cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return true;
fail:
    if (d_buf1_dist) cudaFree(d_buf1_dist);
    if (d_buf1_idx) cudaFree(d_buf1_idx);
    if (d_buf2_dist) cudaFree(d_buf2_dist);
    if (d_buf2_idx) cudaFree(d_buf2_idx);
    return false;
}

__global__ void gen_per_query_tile_idx_kernel(int* d_out, int n_query, int tile_len, int tile_offset) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_query * tile_len;
    if (tid < total) d_out[tid] = tile_offset + (int)(tid % tile_len);
}

__global__ void init_topk_kernel(float* d_dist, int* d_idx, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        d_dist[tid] = FLT_MAX;
        d_idx[tid] = -1;
    }
}

__global__ void merge_topk_kernel(
    const float* __restrict__ cur_dist,
    const int* __restrict__ cur_idx,
    const float* __restrict__ tile_dist,
    const int* __restrict__ tile_idx,
    int n_query,
    int k,
    int tile_k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx
) {
    int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;
    const float* cd = cur_dist + (size_t)q * k;
    const int* ci = cur_idx + (size_t)q * k;
    const float* td = tile_dist + (size_t)q * tile_k;
    const int* ti = tile_idx + (size_t)q * tile_k;
    float* od = out_dist + (size_t)q * k;
    int* oi = out_idx + (size_t)q * k;

    int a = 0, b = 0;
    for (int o = 0; o < k; ++o) {
        float av = (a < k) ? cd[a] : FLT_MAX;
        float bv = (b < tile_k) ? td[b] : FLT_MAX;
        if (av <= bv) {
            od[o] = av;
            oi[o] = (a < k) ? ci[a] : -1;
            ++a;
        } else {
            od[o] = bv;
            oi[o] = (b < tile_k) ? ti[b] : -1;
            ++b;
        }
    }
}

__global__ void group_select_topk_kernel(
    const float* __restrict__ group_dist,
    const int* __restrict__ group_idx,
    int n_query,
    int group_tiles,
    int tile_k,
    int k,
    float* __restrict__ out_dist,
    int* __restrict__ out_idx
) {
    int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;
    const int cand = group_tiles * tile_k;
    const float* gd = group_dist + (size_t)q * cand;
    const int* gi = group_idx + (size_t)q * cand;
    float* od = out_dist + (size_t)q * k;
    int* oi = out_idx + (size_t)q * k;

    for (int o = 0; o < k; ++o) {
        float best = FLT_MAX;
        int best_pos = -1;
        for (int c = 0; c < cand; ++c) {
            float v = gd[c];
            if (v < best) {
                best = v;
                best_pos = c;
            }
        }
        od[o] = best;
        oi[o] = (best_pos >= 0) ? gi[best_pos] : -1;
        if (best_pos >= 0) {
            const_cast<float*>(gd)[best_pos] = FLT_MAX;
        }
    }
}

__global__ void copy_tile_topk_to_group_kernel(
    const float* __restrict__ tile_dist,
    const int* __restrict__ tile_idx,
    int n_query,
    int tile_slot,
    int group_tiles,
    int tile_k,
    float* __restrict__ group_dist,
    int* __restrict__ group_idx
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_query * tile_k;
    if (tid >= total) return;
    int q = (int)(tid / tile_k);
    int kpos = (int)(tid % tile_k);
    size_t src = (size_t)q * tile_k + kpos;
    size_t dst = ((size_t)q * group_tiles + tile_slot) * tile_k + kpos;
    group_dist[dst] = tile_dist[src];
    group_idx[dst] = tile_idx[src];
}

static int get_env_int(const char* name, int default_value) {
    const char* s = std::getenv(name);
    if (!s || !*s) return default_value;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || v <= 0) return default_value;
    return (int)v;
}

static bool coarse_tiled_enabled(int n_query_batch, int n_total_clusters) {
    const char* mode = std::getenv("IVFT_COARSE_TILED");
    if (mode && (*mode == '1' || std::strcmp(mode, "true") == 0 || std::strcmp(mode, "on") == 0)) {
        return true;
    }
    if (mode && (*mode == '0' || std::strcmp(mode, "false") == 0 || std::strcmp(mode, "off") == 0)) {
        return false;
    }
    long long elems = (long long)n_query_batch * (long long)n_total_clusters;
    return elems >= (long long)get_env_int("IVFT_COARSE_TILED_AUTO_ELEMS", 512 * 1024 * 1024);
}

}  // namespace


/* =========================================================================
 * (2) CoarseHandle coarse
 * ========================================================================= */

void coarse_handle_init(
    CoarseHandle* h,
    const float* d_centers,
    int n_total_clusters,
    int dim
) {
    if (!h) return;
    h->d_centers = d_centers;
    h->n_total_clusters = n_total_clusters;
    h->dim = dim;
    h->distance_mode_cached = -1;

    /*  centers L2 normL2 / cosine  */
    cudaMalloc(&h->d_centers_norm, (size_t)n_total_clusters * sizeof(float));
    compute_l2_norm_gpu(d_centers, h->d_centers_norm, n_total_clusters, dim);

    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    h->cublas_handle = (void*)cublas_handle;
}

void coarse_handle_release(CoarseHandle* h) {
    if (!h) return;
    if (h->cublas_handle) {
        cublasDestroy((cublasHandle_t)h->cublas_handle);
        h->cublas_handle = nullptr;
    }
    if (h->d_centers_norm) {
        cudaFree(h->d_centers_norm);
        h->d_centers_norm = nullptr;
    }
    h->d_centers = nullptr;
    h->n_total_clusters = 0;
    h->dim = 0;
}

const CoarseTimingBreakdown* coarse_get_last_timing() {
    return &g_last_coarse_timing;
}

void coarse_search(
    const CoarseHandle* h,
    const float* h_query,
    int n_query_batch,
    int n_probes,
    int distance_mode,
    int* h_cluster_ids_out,
    double* out_coarse_ms,
    double* out_h2d_ms,
    double* out_d2h_ms
) {
    if (!h || !h->d_centers || !h->d_centers_norm) {
        std::fprintf(stderr, "[coarse_search] invalid handle\n"); std::abort();
    }
    const int n_total_clusters = h->n_total_clusters;
    const int dim = h->dim;
    if (n_query_batch <= 0 || n_probes <= 0 || n_probes > n_total_clusters) {
        std::fprintf(stderr, "[coarse_search] bad args n_q=%d nprobe=%d nlist=%d\n",
                     n_query_batch, n_probes, n_total_clusters);
        std::abort();
    }

    if (out_coarse_ms) *out_coarse_ms = 0.0;
    if (out_h2d_ms)    *out_h2d_ms    = 0.0;
    if (out_d2h_ms)    *out_d2h_ms    = 0.0;
    g_last_coarse_timing = CoarseTimingBreakdown{};

    const size_t q_bytes  = (size_t)n_query_batch * dim * sizeof(float);
    const size_t ip_bytes = (size_t)n_query_batch * n_total_clusters * sizeof(float);
    const size_t pi_bytes = (size_t)n_query_batch * n_probes * sizeof(int);
    const size_t pf_bytes = (size_t)n_query_batch * n_probes * sizeof(float);
    const size_t seq_bytes = (size_t)n_query_batch * n_total_clusters * sizeof(int);

    if (coarse_tiled_enabled(n_query_batch, n_total_clusters)) {
        const int tile_n_default = 8192;
        int tile_n = get_env_int("IVFT_COARSE_TILE_N", tile_n_default);
        if (tile_n > n_total_clusters) tile_n = n_total_clusters;
        if (tile_n < n_probes) {
            int pow2 = 1;
            while (pow2 < n_probes) pow2 <<= 1;
            tile_n = std::min(n_total_clusters, pow2);
        }

        float* d_query = nullptr;
        float* d_query_norm = nullptr;
        float* d_tile_ip = nullptr;
        int* d_tile_index = nullptr;
        float* d_tile_dist = nullptr;
        int* d_tile_top_index = nullptr;
        float* d_top_dist_a = nullptr;
        int* d_top_index_a = nullptr;
        float* d_top_dist_b = nullptr;
        int* d_top_index_b = nullptr;
        const int tile_topk = std::min(n_probes, 512);
        int tile_group = get_env_int("IVFT_COARSE_TILE_GROUP", 1);
        int max_tiles = (n_total_clusters + tile_n - 1) / tile_n;
        tile_group = std::max(1, std::min(tile_group, max_tiles));
        float* d_group_dist = nullptr;
        int* d_group_index = nullptr;
        float* d_group_top_dist = nullptr;
        int* d_group_top_index = nullptr;

        cudaMalloc(&d_query, q_bytes);
        cudaMalloc(&d_query_norm, (size_t)n_query_batch * sizeof(float));
        cudaMalloc(&d_tile_ip, (size_t)n_query_batch * tile_n * sizeof(float));
        cudaMalloc(&d_tile_index, (size_t)n_query_batch * tile_n * sizeof(int));
        cudaMalloc(&d_tile_dist, (size_t)n_query_batch * tile_topk * sizeof(float));
        cudaMalloc(&d_tile_top_index, (size_t)n_query_batch * tile_topk * sizeof(int));
        cudaMalloc(&d_top_dist_a, pf_bytes);
        cudaMalloc(&d_top_index_a, pi_bytes);
        cudaMalloc(&d_top_dist_b, pf_bytes);
        cudaMalloc(&d_top_index_b, pi_bytes);
        if (tile_group > 1) {
            cudaMalloc(&d_group_dist, (size_t)n_query_batch * tile_group * tile_topk * sizeof(float));
            cudaMalloc(&d_group_index, (size_t)n_query_batch * tile_group * tile_topk * sizeof(int));
            cudaMalloc(&d_group_top_dist, pf_bytes);
            cudaMalloc(&d_group_top_index, pi_bytes);
        }
        CHECK_CUDA_ERRORS;

        const double h2d_t0 = now_ms();
        cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;
        g_last_coarse_timing.query_h2d_ms = now_ms() - h2d_t0;
        if (out_h2d_ms) *out_h2d_ms = g_last_coarse_timing.query_h2d_ms;

        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEvent_t seg_s, seg_e;
        cudaEventCreate(&seg_s); cudaEventCreate(&seg_e);
        cudaEventRecord(s);

        cudaEventRecord(seg_s);
        compute_l2_norm_gpu(d_query, d_query_norm, n_query_batch, dim);
        cudaEventRecord(seg_e);
        cudaEventSynchronize(seg_e);
        CHECK_CUDA_ERRORS;
        g_last_coarse_timing.query_norm_ms = elapsed_cuda_ms(seg_s, seg_e);

        int init_total = n_query_batch * n_probes;
        init_topk_kernel<<<(init_total + 255) / 256, 256>>>(d_top_dist_a, d_top_index_a, init_total);
        CHECK_CUDA_ERRORS;

        double gemm_ms_sum = 0.0;
        double topk_ms_sum = 0.0;
        cublasHandle_t handle = (cublasHandle_t)h->cublas_handle;
        float alpha = 1.0f, beta = 0.0f;

        int group_fill = 0;
        auto flush_group = [&]() {
            if (group_fill <= 0) return;
            if (group_fill == 1) {
                merge_topk_kernel<<<n_query_batch, 1>>>(
                    d_top_dist_a, d_top_index_a,
                    d_group_dist, d_group_index,
                    n_query_batch, n_probes, tile_topk,
                    d_top_dist_b, d_top_index_b);
            } else {
                group_select_topk_kernel<<<n_query_batch, 1>>>(
                    d_group_dist, d_group_index,
                    n_query_batch, group_fill, tile_topk, n_probes,
                    d_group_top_dist, d_group_top_index);
                merge_topk_kernel<<<n_query_batch, 1>>>(
                    d_top_dist_a, d_top_index_a,
                    d_group_top_dist, d_group_top_index,
                    n_query_batch, n_probes, n_probes,
                    d_top_dist_b, d_top_index_b);
            }
            CHECK_CUDA_ERRORS;
            std::swap(d_top_dist_a, d_top_dist_b);
            std::swap(d_top_index_a, d_top_index_b);
            group_fill = 0;
        };

        for (int tile0 = 0; tile0 < n_total_clusters; tile0 += tile_n) {
            int cur_tile = std::min(tile_n, n_total_clusters - tile0);
            long long tile_total = (long long)n_query_batch * cur_tile;

            cudaEventRecord(seg_s);
            cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        cur_tile, n_query_batch, dim,
                        &alpha,
                        h->d_centers + (size_t)tile0 * dim, dim,
                        d_query, dim,
                        &beta,
                        d_tile_ip, cur_tile);
            cudaEventRecord(seg_e);
            cudaEventSynchronize(seg_e);
            float seg_ms = 0.0f;
            cudaEventElapsedTime(&seg_ms, seg_s, seg_e);
            gemm_ms_sum += seg_ms;
            CHECK_CUDA_ERRORS;

            cudaEventRecord(seg_s);
            gen_per_query_tile_idx_kernel<<<(int)((tile_total + 255) / 256), 256>>>(
                d_tile_index, n_query_batch, cur_tile, tile0);
            CHECK_CUDA_ERRORS;
            if (distance_mode == COSINE_DISTANCE) {
                INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                    d_query_norm, h->d_centers_norm + tile0, d_tile_ip, d_tile_index,
                    n_query_batch, cur_tile, tile_topk,
                    d_tile_dist, d_tile_top_index,
                    /*select_min=*/true, /*stream=*/0);
            } else {
                INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                    d_query_norm, h->d_centers_norm + tile0, d_tile_ip, d_tile_index,
                    n_query_batch, cur_tile, tile_topk,
                    d_tile_dist, d_tile_top_index,
                    /*select_min=*/true, /*stream=*/0);
            }
            if (tile_group > 1) {
                copy_tile_topk_to_group_kernel<<<(int)(((long long)n_query_batch * tile_topk + 255) / 256), 256>>>(
                    d_tile_dist, d_tile_top_index,
                    n_query_batch, group_fill, tile_group, tile_topk,
                    d_group_dist, d_group_index);
                CHECK_CUDA_ERRORS;
                ++group_fill;
                if (group_fill == tile_group) flush_group();
            } else {
                merge_topk_kernel<<<n_query_batch, 1>>>(
                    d_top_dist_a, d_top_index_a,
                    d_tile_dist, d_tile_top_index,
                    n_query_batch, n_probes, tile_topk,
                    d_top_dist_b, d_top_index_b);
                CHECK_CUDA_ERRORS;
                std::swap(d_top_dist_a, d_top_dist_b);
                std::swap(d_top_index_a, d_top_index_b);
            }
            cudaEventRecord(seg_e);
            cudaEventSynchronize(seg_e);
            cudaEventElapsedTime(&seg_ms, seg_s, seg_e);
            topk_ms_sum += seg_ms;
        }
        if (tile_group > 1) {
            cudaEventRecord(seg_s);
            flush_group();
            cudaEventRecord(seg_e);
            cudaEventSynchronize(seg_e);
            float seg_ms = 0.0f;
            cudaEventElapsedTime(&seg_ms, seg_s, seg_e);
            topk_ms_sum += seg_ms;
        }

        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms_coarse = 0.0f;
        cudaEventElapsedTime(&ms_coarse, s, e);
        g_last_coarse_timing.gemm_ms = gemm_ms_sum;
        g_last_coarse_timing.topk_ms = topk_ms_sum;
        g_last_coarse_timing.coarse_compute_ms = (double)ms_coarse;
        if (out_coarse_ms) *out_coarse_ms = (double)ms_coarse;
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaEventDestroy(seg_s); cudaEventDestroy(seg_e);

        const double d2h_t0 = now_ms();
        cudaMemcpy(h_cluster_ids_out, d_top_index_a, pi_bytes, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        g_last_coarse_timing.cluster_d2h_ms = now_ms() - d2h_t0;
        if (out_d2h_ms) *out_d2h_ms = g_last_coarse_timing.cluster_d2h_ms;
        g_last_coarse_timing.coarse_total_ms =
            g_last_coarse_timing.query_h2d_ms +
            g_last_coarse_timing.query_norm_ms +
            g_last_coarse_timing.gemm_ms +
            g_last_coarse_timing.topk_ms +
            g_last_coarse_timing.cluster_d2h_ms;

        cudaFree(d_query);
        cudaFree(d_query_norm);
        cudaFree(d_tile_ip);
        cudaFree(d_tile_index);
        cudaFree(d_tile_dist);
        cudaFree(d_tile_top_index);
        cudaFree(d_top_dist_a);
        cudaFree(d_top_index_a);
        cudaFree(d_top_dist_b);
        cudaFree(d_top_index_b);
        if (d_group_dist) cudaFree(d_group_dist);
        if (d_group_index) cudaFree(d_group_index);
        if (d_group_top_dist) cudaFree(d_group_top_dist);
        if (d_group_top_index) cudaFree(d_group_top_index);
        CHECK_CUDA_ERRORS;
        return;
    }

    /* Legacy full-matrix path guard: avoid cuBLAS int32 overflow and OOM when
     * callers explicitly disable tiled coarse. */
    {
        constexpr long long kGiB = 1024LL * 1024 * 1024;
        const int kAlign = 32;
        const long long kMaxGemmElems = 1900000000LL;
        const long long gemm_elems = (long long)n_query_batch * n_total_clusters;

        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);

        const long long headroom = std::max(4LL * kGiB, (long long)gpu_total / 8);
        long long usable = (long long)gpu_free - headroom;
        if (usable < (long long)gpu_free / 5) usable = (long long)gpu_free / 5;
        usable = usable * 9 / 10;

        long long bytes_per_q = 2LL * n_total_clusters * sizeof(float)
                              + (long long)n_probes * (sizeof(float) + sizeof(int))
                              + (long long)dim * sizeof(float) + 4096;
        int max_sub_mem = (int)std::min((long long)n_query_batch,
                                        usable / std::max(bytes_per_q, 1LL));
        int max_sub_int = (int)(kMaxGemmElems / std::max(n_total_clusters, 1));
        int max_sub = std::min(max_sub_mem, max_sub_int);
        if (max_sub >= kAlign) max_sub = (max_sub / kAlign) * kAlign;
        if (max_sub < 1) max_sub = 1;

        if (gemm_elems > kMaxGemmElems || n_query_batch > max_sub) {
            double total_coarse = 0.0, total_h2d = 0.0, total_d2h = 0.0;
            CoarseTimingBreakdown acc{};
            for (int off = 0; off < n_query_batch; off += max_sub) {
                int n_this = std::min(max_sub, n_query_batch - off);
                double co = 0.0, h2 = 0.0, d2 = 0.0;
                coarse_search(
                    h,
                    h_query + (size_t)off * dim,
                    n_this, n_probes, distance_mode,
                    h_cluster_ids_out + (size_t)off * n_probes,
                    &co, &h2, &d2);
                total_coarse += co;
                total_h2d += h2;
                total_d2h += d2;
                const CoarseTimingBreakdown* bd = coarse_get_last_timing();
                if (bd) {
                    acc.query_h2d_ms += bd->query_h2d_ms;
                    acc.seq_init_ms += bd->seq_init_ms;
                    acc.query_norm_ms += bd->query_norm_ms;
                    acc.gemm_ms += bd->gemm_ms;
                    acc.topk_ms += bd->topk_ms;
                    acc.cluster_d2h_ms += bd->cluster_d2h_ms;
                    acc.coarse_compute_ms += bd->coarse_compute_ms;
                    acc.coarse_total_ms += bd->coarse_total_ms;
                }
            }
            g_last_coarse_timing = acc;
            if (out_coarse_ms) *out_coarse_ms = total_coarse;
            if (out_h2d_ms) *out_h2d_ms = total_h2d;
            if (out_d2h_ms) *out_d2h_ms = total_d2h;
            return;
        }
    }

    float* d_query = nullptr;
    float* d_query_norm = nullptr;
    float* d_inner_product = nullptr;
    int*   d_seq_index = nullptr;
    int*   d_top_index = nullptr;
    float* d_top_dist  = nullptr;

    cudaMalloc(&d_query, q_bytes);
    cudaMalloc(&d_query_norm, (size_t)n_query_batch * sizeof(float));
    cudaMalloc(&d_inner_product, ip_bytes);
    cudaMalloc(&d_seq_index, seq_bytes);
    cudaMalloc(&d_top_index, pi_bytes);
    cudaMalloc(&d_top_dist,  pf_bytes);
    CHECK_CUDA_ERRORS;

    const double h2d_t0 = now_ms();
    cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    g_last_coarse_timing.query_h2d_ms = now_ms() - h2d_t0;
    if (out_h2d_ms) *out_h2d_ms = g_last_coarse_timing.query_h2d_ms;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEvent_t seg_s, seg_e;
    cudaEventCreate(&seg_s); cudaEventCreate(&seg_e);
    {
        int bs = 256;
        long long total = (long long)n_query_batch * n_total_clusters;
        int gs = (int)((total + bs - 1) / bs);
        cudaEventRecord(seg_s);
        gen_per_query_seq_idx_kernel<<<gs, bs>>>(d_seq_index, n_query_batch, n_total_clusters);
        cudaEventRecord(seg_e);
        cudaEventSynchronize(seg_e);
        CHECK_CUDA_ERRORS;
        g_last_coarse_timing.seq_init_ms = elapsed_cuda_ms(seg_s, seg_e);
    }

    cudaEventRecord(s);

    cudaEventRecord(seg_s);
    compute_l2_norm_gpu(d_query, d_query_norm, n_query_batch, dim);
    cudaEventRecord(seg_e);
    cudaEventSynchronize(seg_e);
    CHECK_CUDA_ERRORS;
    g_last_coarse_timing.query_norm_ms = elapsed_cuda_ms(seg_s, seg_e);

    /* GEMMquery  centers^T  [n_query_batch, n_total_clusters] col-major */
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = (cublasHandle_t)h->cublas_handle;
        cudaEventRecord(seg_s);
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n_total_clusters, n_query_batch, dim,
                    &alpha,
                    h->d_centers, dim,
                    d_query, dim,
                    &beta,
                    d_inner_product, n_total_clusters);
        cudaEventRecord(seg_e);
        cudaEventSynchronize(seg_e);
        CHECK_CUDA_ERRORS;
        g_last_coarse_timing.gemm_ms = elapsed_cuda_ms(seg_s, seg_e);
    }

    /* Multi-pass coarse top-k: split into chunks of <= 512, merge on GPU */
    const int MAX_K_PER_PASS = 512;

    const char* topk_impl_env = std::getenv("IVFT_COARSE_TOPK_IMPL");
    const bool force_legacy_topk = topk_impl_env && std::strcmp(topk_impl_env, "legacy") == 0;
    const bool force_twostage_topk = topk_impl_env && std::strcmp(topk_impl_env, "twostage") == 0;
    const bool force_cub_topk = topk_impl_env && std::strcmp(topk_impl_env, "cub") == 0;
    const bool force_small_topk = topk_impl_env && std::strcmp(topk_impl_env, "small") == 0;
    const bool auto_topk = !topk_impl_env || std::strcmp(topk_impl_env, "auto") == 0;
    const bool can_optimized_l2_topk = !force_legacy_topk && distance_mode != COSINE_DISTANCE &&
        n_probes <= MAX_K_PER_PASS &&
        n_total_clusters >= 131072;

    cudaEventRecord(seg_s);
    bool topk_done = false;
    const bool auto_small_topk = auto_topk && n_probes <= 32;
    if ((force_small_topk || auto_small_topk) && can_optimized_l2_topk && n_probes <= 64) {
        topk_done = coarse_topk_small_l2(
            d_inner_product, d_query_norm, h->d_centers_norm,
            n_query_batch, n_total_clusters, n_probes,
            d_top_dist, d_top_index);
        CHECK_CUDA_ERRORS;
        if (!topk_done) {
            std::fprintf(stderr, "[coarse_search] small top-k failed; falling back to legacy\n");
        }
    }

    if (!topk_done && (force_cub_topk || auto_topk || force_small_topk) && can_optimized_l2_topk) {
        topk_done = coarse_topk_cub_l2(
            d_inner_product, d_query_norm, h->d_centers_norm,
            n_query_batch, n_total_clusters, n_probes,
            d_top_dist, d_top_index);
        CHECK_CUDA_ERRORS;
        if (!topk_done) {
            std::fprintf(stderr, "[coarse_search] cub top-k failed; falling back to legacy\n");
        }
    } else if (force_twostage_topk && can_optimized_l2_topk) {
        topk_done = coarse_topk_twostage_l2(
            d_inner_product, d_query_norm, h->d_centers_norm,
            n_query_batch, n_total_clusters, n_probes,
            d_top_dist, d_top_index);
        CHECK_CUDA_ERRORS;
        if (!topk_done && force_twostage_topk) {
            std::fprintf(stderr, "[coarse_search] twostage top-k failed; falling back to legacy\n");
        }
    }

    if (!topk_done && n_probes <= MAX_K_PER_PASS) {
        {
            dim3 blk(256);
            dim3 grd(((long long)n_query_batch * n_probes + blk.x - 1) / blk.x);
            fill_kernel<<<grd, blk>>>(d_top_dist, FLT_MAX, n_query_batch * n_probes);
            CHECK_CUDA_ERRORS;
        }
        if (distance_mode == COSINE_DISTANCE) {
            INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                d_query_norm, h->d_centers_norm, d_inner_product, d_seq_index,
                n_query_batch, n_total_clusters, n_probes,
                d_top_dist, d_top_index,
                /*select_min=*/true, /*stream=*/0);
        } else {
            INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                d_query_norm, h->d_centers_norm, d_inner_product, d_seq_index,
                n_query_batch, n_total_clusters, n_probes,
                d_top_dist, d_top_index,
                /*select_min=*/true, /*stream=*/0);
        }
    } else if (!topk_done) {
        /* Two-pass: first get top-MAX_K_PER_PASS from full distance matrix,
         * then get top-MAX_K_PER_PASS from the remaining distances (excluding first-pass winners),
         * finally merge and re-select top-n_probes via standalone warpsort. */
        const int k1 = MAX_K_PER_PASS;
        const size_t pass_pi = (size_t)n_query_batch * k1;

        float* d_pass1_dist = nullptr;
        int*   d_pass1_idx  = nullptr;
        cudaMalloc(&d_pass1_dist, pass_pi * sizeof(float));
        cudaMalloc(&d_pass1_idx,  pass_pi * sizeof(int));
        {
            dim3 blk(256);
            dim3 grd((pass_pi + blk.x - 1) / blk.x);
            fill_kernel<<<grd, blk>>>(d_pass1_dist, FLT_MAX, (int)pass_pi);
        }

        if (distance_mode == COSINE_DISTANCE) {
            INSITUANN::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
                d_query_norm, h->d_centers_norm, d_inner_product, d_seq_index,
                n_query_batch, n_total_clusters, k1,
                d_pass1_dist, d_pass1_idx,
                /*select_min=*/true, /*stream=*/0);
        } else {
            INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                d_query_norm, h->d_centers_norm, d_inner_product, d_seq_index,
                n_query_batch, n_total_clusters, k1,
                d_pass1_dist, d_pass1_idx,
                /*select_min=*/true, /*stream=*/0);
        }

        /* Pass 2: invalidate pass-1 winners in the distance matrix, re-select top-k1 */
        float* d_pass2_dist = nullptr;
        int*   d_pass2_idx  = nullptr;
        cudaMalloc(&d_pass2_dist, pass_pi * sizeof(float));
        cudaMalloc(&d_pass2_idx,  pass_pi * sizeof(int));

        /* Zero out pass-1 winners by setting their distances to FLT_MAX */
        {
            dim3 blk(256);
            dim3 grd((pass_pi + blk.x - 1) / blk.x);
            /* Inline kernel: for each pass-1 result, set the corresponding distance to FLT_MAX */
            /* We do this on CPU since it's simpler and the copy is fast */
        }
        /* Simpler approach: copy pass-1 indices to host, mask on GPU is complex.
         * Instead, use a single standalone warpsort on the FULL distance matrix
         * to get top-(n_probes) directly. The standalone warpsort supports up to
         * k=512 (kMaxCapacity=1024). For n_probes > 512, we concatenate two
         * top-512 passes and re-select. */

        /* Actually, the simplest correct approach:
         * 1. Compute L2 distances for ALL nlist centroids (already done in d_inner_product)
         * 2. Copy the full distance row to host for each query
         * 3. Do CPU top-n_probes selection (nth_element)
         * This avoids GPU kernel limitations entirely. */
        {
            /* Allocate host buffer for distances */
            const size_t row_bytes = (size_t)n_total_clusters * sizeof(float);
            /* Compute actual L2 distances in-place in d_inner_product */
            /* d_inner_product currently holds inner products; need to convert to L2 */
            /* L2 = ||q||^2 + ||c||^2 - 2*<q,c> */
            /* This is done per-element; use a simple kernel */
            dim3 blk(256);
            long long total_elem = (long long)n_query_batch * n_total_clusters;
            dim3 grd((int)((total_elem + 255) / 256));
            /* We need a kernel to convert IP to L2. Let's just copy to host and do it there. */
        }

        /* Fallback: copy all distances to host, CPU top-k */
        std::vector<float> h_all_dist((size_t)n_query_batch * n_total_clusters);
        cudaMemcpy(h_all_dist.data(), d_inner_product,
                   (size_t)n_query_batch * n_total_clusters * sizeof(float),
                   cudaMemcpyDeviceToHost);

        std::vector<float> h_qnorm(n_query_batch);
        cudaMemcpy(h_qnorm.data(), d_query_norm, n_query_batch * sizeof(float),
                   cudaMemcpyDeviceToHost);
        std::vector<float> h_cnorm(n_total_clusters);
        cudaMemcpy(h_cnorm.data(), h->d_centers_norm, n_total_clusters * sizeof(float),
                   cudaMemcpyDeviceToHost);

        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < n_query_batch; ++q) {
            float* row = h_all_dist.data() + (size_t)q * n_total_clusters;
            float qn = h_qnorm[q];
            for (int c = 0; c < n_total_clusters; ++c) {
                row[c] = qn * qn + h_cnorm[c] * h_cnorm[c] - 2.0f * row[c];
            }
            /* Partial sort to find top-n_probes smallest */
            std::vector<std::pair<float, int>> dists(n_total_clusters);
            for (int c = 0; c < n_total_clusters; ++c) dists[c] = {row[c], c};
            std::partial_sort(dists.begin(), dists.begin() + n_probes, dists.end());
            for (int j = 0; j < n_probes; ++j) {
                h_cluster_ids_out[(size_t)q * n_probes + j] = dists[j].second;
            }
        }

        cudaFree(d_pass1_dist); cudaFree(d_pass1_idx);
        cudaFree(d_pass2_dist); cudaFree(d_pass2_idx);

        /* Skip the normal D2H copy since we already filled h_cluster_ids_out */
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms_coarse = 0.0f;
        cudaEventElapsedTime(&ms_coarse, s, e);
        g_last_coarse_timing.coarse_compute_ms = (double)ms_coarse;
        g_last_coarse_timing.topk_ms =
            (double)ms_coarse - g_last_coarse_timing.query_norm_ms - g_last_coarse_timing.gemm_ms;
        if (g_last_coarse_timing.topk_ms < 0.0) g_last_coarse_timing.topk_ms = 0.0;
        g_last_coarse_timing.coarse_total_ms =
            g_last_coarse_timing.query_h2d_ms +
            g_last_coarse_timing.seq_init_ms +
            g_last_coarse_timing.coarse_compute_ms +
            g_last_coarse_timing.cluster_d2h_ms;
        if (out_coarse_ms) *out_coarse_ms = (double)ms_coarse;
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaEventDestroy(seg_s); cudaEventDestroy(seg_e);

        cudaFree(d_query); cudaFree(d_query_norm);
        cudaFree(d_inner_product); cudaFree(d_seq_index);
        cudaFree(d_top_index); cudaFree(d_top_dist);
        CHECK_CUDA_ERRORS;
        return;  /* early return - already filled output */
    }

    cudaEventRecord(seg_e);
    cudaEventSynchronize(seg_e);
    CHECK_CUDA_ERRORS;
    g_last_coarse_timing.topk_ms = elapsed_cuda_ms(seg_s, seg_e);

    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms_coarse = 0.0f;
    cudaEventElapsedTime(&ms_coarse, s, e);
    g_last_coarse_timing.coarse_compute_ms = (double)ms_coarse;
    if (out_coarse_ms) *out_coarse_ms = (double)ms_coarse;
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaEventDestroy(seg_s); cudaEventDestroy(seg_e);

    const double t0 = now_ms();
    cudaMemcpy(h_cluster_ids_out, d_top_index, pi_bytes, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    g_last_coarse_timing.cluster_d2h_ms = now_ms() - t0;
    if (out_d2h_ms) *out_d2h_ms = g_last_coarse_timing.cluster_d2h_ms;
    g_last_coarse_timing.coarse_total_ms =
        g_last_coarse_timing.query_h2d_ms +
        g_last_coarse_timing.seq_init_ms +
        g_last_coarse_timing.query_norm_ms +
        g_last_coarse_timing.gemm_ms +
        g_last_coarse_timing.topk_ms +
        g_last_coarse_timing.cluster_d2h_ms;

    cudaFree(d_query);
    cudaFree(d_query_norm);
    cudaFree(d_inner_product);
    cudaFree(d_seq_index);
    cudaFree(d_top_index);
    cudaFree(d_top_dist);
    CHECK_CUDA_ERRORS;
}


/* =========================================================================
 * (3) fine_search_cpu CPU fine
 * ========================================================================= */

void fine_search_cpu(
    CpuFineVariant variant,
    const float* h_base_rowmajor,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_reordered_indices,
    int n_total_clusters,
    int dim,
    const float* h_query,
    const int* h_coarse_cluster_ids,
    int n_query_accum,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_index,
    float* h_topk_dist,
    double* out_fine_ms,
    long long* out_n_fma
) {
    if (n_query_accum <= 0 || n_probes <= 0 || topk <= 0) {
        std::fprintf(stderr, "[fine_search_cpu] bad args\n"); std::abort();
    }
    ivftensor::cpu_fine::CpuFineKernelFn kfn = ivftensor::cpu_fine::dispatch(variant);
    if (!kfn) { std::fprintf(stderr, "[fine_search_cpu] unknown variant %d\n", (int)variant); std::abort(); }

    std::vector<int> h_topk_local_idx((size_t)n_query_accum * (size_t)topk);

    const double t0 = now_ms();
    long long nfma = kfn(
        h_base_rowmajor,
        h_base_aosoa,
        h_aosoa_offsets,
        h_cluster_offsets,
        h_cluster_counts,
        h_query,
        h_coarse_cluster_ids,
        n_query_accum, dim, n_total_clusters, n_probes, topk,
        distance_mode == COSINE_DISTANCE ? 1 : 0,
        num_threads,
        h_topk_local_idx.data(),
        h_topk_dist
    );
    if (out_fine_ms) *out_fine_ms = now_ms() - t0;
    if (out_n_fma)   *out_n_fma = nfma;

    const size_t n_out = (size_t)n_query_accum * (size_t)topk;
    if (h_reordered_indices) {
        for (size_t i = 0; i < n_out; ++i) {
            int pos = h_topk_local_idx[i];
            h_topk_index[i] = (pos >= 0) ? h_reordered_indices[pos] : -1;
        }
    } else {
        std::memcpy(h_topk_index, h_topk_local_idx.data(), n_out * sizeof(int));
    }
}


/* =========================================================================
 * (1)  ivf_search_cpu_fine (2)+(3)
 * ========================================================================= */

void ivf_search_cpu_fine(
    const float* d_centers,
    const float* /*d_centers_norm_in*/,   /* handle  */
    int n_total_clusters,
    int dim,
    const float* h_base_rowmajor,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_reordered_indices,
    int /*n_total_vectors*/,
    const float* h_query,
    int n_query,
    int n_probes,
    int topk,
    int distance_mode,
    CpuFineVariant variant,
    int num_threads,
    int* h_topk_index,
    float* h_topk_dist,
    CpuFineStats* stats
) {
    if (n_query <= 0 || dim <= 0 || n_total_clusters <= 0 ||
        n_probes <= 0 || n_probes > n_total_clusters || topk <= 0) {
        std::fprintf(stderr, "[ivf_search_cpu_fine] invalid args\n"); std::abort();
    }

    const double t_begin = now_ms();

    CoarseHandle chandle;
    coarse_handle_init(&chandle, d_centers, n_total_clusters, dim);

    std::vector<int> h_coarse_ids((size_t)n_query * (size_t)n_probes);
    double t_coarse = 0.0, t_h2d = 0.0, t_d2h = 0.0;
    coarse_search(&chandle, h_query, n_query, n_probes, distance_mode,
                  h_coarse_ids.data(), &t_coarse, &t_h2d, &t_d2h);

    double t_fine = 0.0;
    long long nfma = 0;
    fine_search_cpu(variant,
                    h_base_rowmajor, h_base_aosoa, h_aosoa_offsets,
                    h_cluster_offsets, h_cluster_counts, h_reordered_indices,
                    n_total_clusters, dim,
                    h_query, h_coarse_ids.data(),
                    n_query, n_probes, topk, distance_mode, num_threads,
                    h_topk_index, h_topk_dist,
                    &t_fine, &nfma);

    coarse_handle_release(&chandle);

    if (stats) {
        stats->coarse_ms = t_coarse;
        stats->h2d_ms    = t_h2d;
        stats->d2h_ms    = t_d2h;
        stats->fine_ms   = t_fine;
        stats->total_ms  = now_ms() - t_begin;
        stats->n_fma     = nfma;
    }
}
