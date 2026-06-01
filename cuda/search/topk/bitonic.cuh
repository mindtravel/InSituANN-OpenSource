/**
 * Bitonic Sort Implementation for INSITUANN
 *
 * Based on RAFT (RAPIDS AI) bitonic sort implementation:
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 *
 * This implementation provides warp-level bitonic sorting operations
 * for GPU-accelerated top-k selection algorithms.
 *
 * Copyright (c) 2024, INSITUANN
 * Adapted from RAFT (Apache 2.0 License)
 */

#ifndef BITONIC_CUH
#define BITONIC_CUH

#include "warpsort_utils.cuh"

namespace INSITUANN {
namespace bitonic {

/*  warpsort_utils  */
using namespace warpsort_utils;

// ============================================================================
// Bitonic Sort Implementation
// ============================================================================

/**
 * Bitonic merge and sort operations for warp-level data.
 *
 * Data is distributed across threads in a warp, with each thread holding
 * 'Size' elements. The total sorted data size is 'Size * warp_width'.
 *
 * @tparam Size Number of elements per thread (must be power of 2)
 */
template<int Size = 1>
class Bitonic {
    static_assert(isPowerOf2(Size), "Size must be power of 2");

public:
    __device__ __forceinline__ Bitonic(bool ascending, int warp_width = kWarpSize)
        : ascending_(ascending), warp_width_(warp_width)
    {
    }

    /**
     * Bitonic Sort
     *
     */
    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void merge(KeyT* keys, IdxT* indices) const
    {
        merge_impl(ascending_, warp_width_, keys, indices);
    }

    /**
     *
     */
    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void sort(KeyT* keys, IdxT* indices) const
    {
        sort_impl(ascending_, warp_width_, keys, indices);
    }

private:
    const bool ascending_;
    const int warp_width_;

    /*  Size  Bitonic  */
    template<int AnotherSize>
    friend class Bitonic;

    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void merge_impl(
        bool ascending, int warp_width, KeyT* keys, IdxT* indices)
    {
        const int lane = laneId();

        /*  */
        #pragma unroll
        for (int size = Size; size > 1; size >>= 1) {
            const int stride = size >> 1;

            #pragma unroll
            for (int offset = 0; offset < Size; offset += size) {
                #pragma unroll
                for (int i = 0; i < stride; i++) {
                    compare_and_swap(
                        ascending,
                        keys[offset + i],
                        keys[offset + i + stride],
                        indices[offset + i],
                        indices[offset + i + stride]
                    );
                }
            }
        }

        // Cross-thread comparisons using shuffle
        #pragma unroll
        for (int stride = warp_width >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int offset = 0; offset < Size; offset++) {
                KeyT other_key = shfl(keys[offset], lane ^ stride, warp_width);
                IdxT other_idx = shfl(indices[offset], lane ^ stride, warp_width);

                /*
                 * is_second:
                 *  RAFT
                 * - (ascending XOR is_second)  true  key > other
                 * - (ascending XOR is_second)  false  key < other
                 */
                const bool is_second = bool(lane & stride);
                const bool do_swap = (ascending != is_second)
                    ? (keys[offset] > other_key)
                    : (keys[offset] < other_key);

                if (do_swap) {
                    keys[offset] = other_key;
                    indices[offset] = other_idx;
                }
            }
        }
    }

    /**
     *
     *
     *
     * -  constexpr if
     * -
     * -
     */
    template<int S, typename KeyT, typename IdxT>
    static __device__ __forceinline__ void sort_recursive(
        bool ascending, int warp_width, KeyT* keys, IdxT* indices)
    {
        if constexpr (S == 1) {
            /*
             * Size == 1
             *  warp  bitonic
             */
            const int lane = laneId();
            #pragma unroll
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, keys, indices);
            }
        } else {
            constexpr int HalfSize = S / 2;

            /*  */
            sort_recursive<HalfSize>(false, warp_width, keys, indices);

            /*  */
            sort_recursive<HalfSize>(true, warp_width, keys + HalfSize, indices + HalfSize);
        }

        /*  merge */
        Bitonic<S>::merge_impl(ascending, warp_width, keys, indices);
    }

    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void sort_impl(
        bool ascending, int warp_width, KeyT* keys, IdxT* indices)
    {
        sort_recursive<Size>(ascending, warp_width, keys, indices);
    }

    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void compare_and_swap(
        bool ascending, KeyT& a, KeyT& b, IdxT& ia, IdxT& ib)
    {
        bool cond = ascending ? (a > b) : (a < b);
        if (cond) {
            swap(a, b);
            swap(ia, ib);
        }
    }
};

} // namespace bitonic
} // namespace INSITUANN

#endif // BITONIC_CUH
