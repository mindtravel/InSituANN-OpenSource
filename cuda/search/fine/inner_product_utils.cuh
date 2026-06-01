#ifndef INNER_PRODUCT_UTILS_CUH
#define INNER_PRODUCT_UTILS_CUH

#include "pch.h"
#include <stdint.h>

/**
 *
 *
 *
 * - float4
 * -
 * -
 */

/**
 * tilefloat4
 *
 * @tparam Tile tile
 * @param lhs_vec4 float4
 * @param rhs_vec4 float4
 * @param base_idx
 * @param vec4_count float4
 * @param lhs_tile tile
 * @param rhs_tile tile
 */
template<int Tile>
__device__ __forceinline__ void load_tile_vec4(const float4* lhs_vec4,
                                               const float4* rhs_vec4,
                                               int base_idx,
                                               int vec4_count,
                                               float4 (&lhs_tile)[Tile],
                                               float4 (&rhs_tile)[Tile]) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        int idx = base_idx + t;
        if (idx < vec4_count) {
            lhs_tile[t] = lhs_vec4[idx];
            rhs_tile[t] = rhs_vec4[idx];
        } else {
            lhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
            rhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

/**
 * tilefloat4
 *
 * @tparam Tile tile
 * @param lhs_tile tile
 * @param rhs_tile tile
 * @param valid_count
 * @param sum
 * @return
 */
template<int Tile>
__device__ __forceinline__ float accumulate_tile(const float4 (&lhs_tile)[Tile],
                                                 const float4 (&rhs_tile)[Tile],
                                                 int valid_count,
                                                 float sum) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if (t < valid_count) {
            const float4& l = lhs_tile[t];
            const float4& r = rhs_tile[t];
            sum = fmaf(l.x, r.x, sum);
            sum = fmaf(l.y, r.y, sum);
            sum = fmaf(l.z, r.z, sum);
            sum = fmaf(l.w, r.w, sum);
        }
    }
    return sum;
}

/**
 * tiled
 *
 * @tparam Dim
 * @param lhs
 * @param rhs
 * @return
 */
template<int Dim>
__device__ __forceinline__ float dot_product_tiled(const float* __restrict__ lhs,
                                                   const float* __restrict__ rhs) {
    constexpr int kVec4Count = Dim / 4;
    constexpr int kTile = 4;
    if constexpr (kVec4Count == 0) {
        return 0.0f;
    } else {
        constexpr int tile_count = (kVec4Count + kTile - 1) / kTile;
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);

        float4 cur_lhs[kTile];
        float4 cur_rhs[kTile];
        load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, 0, kVec4Count, cur_lhs, cur_rhs);

        float sum = 0.0f;

        if constexpr (tile_count == 1) {
            sum = accumulate_tile<kTile>(cur_lhs, cur_rhs, kVec4Count, sum);
        } else {
            float4 next_lhs[kTile];
            float4 next_rhs[kTile];

            #pragma unroll
            for (int tile = 0; tile < tile_count; ++tile) {
                int next_base = (tile + 1) * kTile;
                if (tile + 1 < tile_count) {
                    load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, next_base, kVec4Count, next_lhs, next_rhs);
                }

                int valid = kVec4Count - tile * kTile;
                valid = valid > kTile ? kTile : valid;
                sum   = accumulate_tile<kTile>(cur_lhs, cur_rhs, valid, sum);

                if (tile + 1 < tile_count) {
                    #pragma unroll
                    for (int t = 0; t < kTile; ++t) {
                        cur_lhs[t] = next_lhs[t];
                        cur_rhs[t] = next_rhs[t];
                    }
                }
            }
        }

        return sum;
    }
}

/**
 * float4
 *
 * @param lhs 16
 * @param rhs 16
 * @param length 4
 * @return
 */
__device__ __forceinline__ float dot_product_vec4_aligned(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    const int vec4_elems = length >> 2;
    const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
    const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);

    #pragma unroll
    for (int v = 0; v < vec4_elems; ++v) {
        const float4 lhs_val = lhs_vec4[v];
        const float4 rhs_val = rhs_vec4[v];
        sum += lhs_val.x * rhs_val.x +
               lhs_val.y * rhs_val.y +
               lhs_val.z * rhs_val.z +
               lhs_val.w * rhs_val.w;
    }
    return sum;
}

/**
 *
 *
 *
 * 1.
 * 2. float4
 * 3.
 *
 * @param lhs
 * @param rhs
 * @param length
 * @return
 */
__device__ __forceinline__ float dot_product_accumulate(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    int i = 0;

    while (i < length &&
           ((reinterpret_cast<uintptr_t>(lhs + i) |
             reinterpret_cast<uintptr_t>(rhs + i)) & (sizeof(float4) - 1))) {
        sum += lhs[i] * rhs[i];
        ++i;
    }

    const int remaining = length - i;
    const int vec4_elems = remaining >> 2;

    if (vec4_elems > 0) {
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs + i);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs + i);

        #pragma unroll
        for (int v = 0; v < vec4_elems; ++v) {
            const float4 lhs_val = lhs_vec4[v];
            const float4 rhs_val = rhs_vec4[v];
            sum += lhs_val.x * rhs_val.x +
                   lhs_val.y * rhs_val.y +
                   lhs_val.z * rhs_val.z +
                   lhs_val.w * rhs_val.w;
        }
        i += vec4_elems << 2;
    }

    for (; i < length; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#endif // INNER_PRODUCT_UTILS_CUH
