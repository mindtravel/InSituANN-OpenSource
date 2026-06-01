/**
 * Warp-Sort Top-K Implementation for INSITUANN
 *
 * Based on RAFT (RAPIDS AI) warp-sort implementation:
 * - raft/cpp/include/raft/matrix/detail/select_warpsort.cuh
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 *
 * This implementation provides GPU-accelerated top-k selection using
 * warp-level primitives and bitonic sorting networks.
 *
 * Key features:
 * - Support for k up to 256 (kMaxCapacity)
 * - Warp-level parallelism using shuffle operations
 * - Register-based storage for low memory overhead
 * - Bitonic merge network for efficient sorting
 *
 * Copyright (c) 2024, INSITUANN
 * Adapted from RAFT (Apache 2.0 License)
 */

#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "l2norm/l2norm.cuh"
#include "pch.h"
#include "warpsort_utils.cuh"
#include "bitonic.cuh"


namespace INSITUANN {
namespace warpsort {

using namespace warpsort_utils;
using namespace bitonic;

// ============================================================================
// Warp Sort Base Class
// ============================================================================

/**
 * warptop-k
 *
 * warpk
 *  Capacity/kWarpWidth
 *
 *  store
 *  top-k  WarpSortFiltered
 *
 * @tparam Capacity 2kMaxCapacity
 * @tparam Ascending truekfalsek
 * @tparam T float
 * @tparam IdxT intuint32_t
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSort {
    static_assert(isPowerOf2(Capacity), "Capacity must be power of 2");
    static_assert(Capacity/2 <= kMaxCapacity, "Capacity exceeds maximum");

public:
    static constexpr int kWarpWidth = (Capacity < kWarpSize) ? Capacity : kWarpSize;
    static constexpr int kMaxArrLen = Capacity / kWarpWidth;

    /* dummy value */
    static __device__ __forceinline__ T kDummy()
    {
        return Ascending ? upper_bound<T>() : lower_bound<T>();
    }

    const int k;  /*  */

    __device__ WarpSort(int k_val) : k(k_val)
    {
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            val_arr_[i] = kDummy();
            idx_arr_[i] = IdxT{};
        }
    }

    /**
     *  queue
     *
     *  k queue
     */
    __device__ void store(T* out_val, IdxT* out_idx) const
    {
        int idx = Pow2<kWarpWidth>::mod(laneId());

        #pragma unroll
        for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
            out_val[idx] = val_arr_[i];
            out_idx[idx] = idx_arr_[i];
        }
    }

protected:
    T val_arr_[kMaxArrLen];
    IdxT idx_arr_[kMaxArrLen];
};

// ============================================================================
// Warp Sort Filtered (Optimized for Large Inputs)
// ============================================================================

/**
 * WarpSortFilteredwarp-sort
 *
 * - val_arr_  queue top-k
 * - val_arr_  buffer
 * - buffer  val_arr_
 *
 *
 * 1. k buffer
 * 2. buffer queue + buffer
 * 3.
 * 4. k
 *
 * @tparam Capacity  >= 64 kMaxArrLen >= 2
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSortFiltered : public WarpSort<Capacity, Ascending, T, IdxT> {
    using Base = WarpSort<Capacity, Ascending, T, IdxT>;

public:
    using Base::kWarpWidth;
    using Base::kMaxArrLen;
    using Base::k;

    /*
     *
     * - val_arr_[0 .. k_arr_len_-1]: queue  top-k
     * - val_arr_[k_arr_len_ .. kMaxArrLen-1]: buffer
     *
     *
     * - queue  k k_arr_len_ = ceil(k / kWarpWidth)
     * - buffer  = kMaxArrLen - k_arr_len_
     * -  queue  k
     */
    static_assert(kMaxArrLen >= 2, "Capacity must be >= 64 for WarpSortFiltered");

    __device__ WarpSortFiltered(int k_val, T limit = Base::kDummy())
        : Base(k_val), buf_len_(0), k_th_(limit)
    {
        /*  k  */
        k_arr_len_ = (k + kWarpWidth - 1) / kWarpWidth;

        /*  */
        // if (k_arr_len_ >= kMaxArrLen) {
        //     k_arr_len_ = kMaxArrLen - 1;  /* 1buffer */
        // }

        /* val_arr_  kDummy */
    }

    /**
     *
     *  buffer
     *
     * merge
     *
     */
    __device__ void add(T val, IdxT idx)
    {
        /*  buffer  */
        int buf_max_len = kMaxArrLen - k_arr_len_;

        /*
         *  merge  do_add
         *  merge
         */
        if (any(buf_len_ >= buf_max_len)) {
            merge_buf_();
        }

        /*  filter  */
        bool do_add = is_ordered<Ascending>(val, k_th_);

        if (do_add) {
            /*  buffer  */
            add_to_buf_(val, idx);
        }
    }

    /**
     *  buffer
     */
    __device__ void done()
    {
        if (any(buf_len_ != 0)) {
            merge_buf_();
        }
    }

private:
    using Base::val_arr_;
    using Base::idx_arr_;

    int buf_len_;    /*  buffer  */
    int k_arr_len_;  /* queue  k */
    T k_th_;         /*  k */

    /**
     *  buffer val_arr_
     *
     * buffer  val_arr_[k_arr_len_]
     */
    __device__ __forceinline__ void add_to_buf_(T val, IdxT idx)
    {
        /*
         *
         *  if
         *
         *  k_arr_len_
         */
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            /* buffer  k_arr_len_  */
            if (i == k_arr_len_ + buf_len_) {
                val_arr_[i] = val;
                idx_arr_[i] = idx;
            }
        }
        buf_len_++;
    }

    /**
     *  buffer  queue
     *
     *
     * 1.  val_arr_[0..kMaxArrLen-1]  bitonic sort
     * 2.
     * 3.  buffer  k_arr_len_
     * 4.  k_th_
     */
    __device__ __forceinline__ void merge_buf_()
    {
        /* queue + buffer */
        Bitonic<kMaxArrLen>(Ascending, kWarpWidth).sort(val_arr_, idx_arr_);

        /*  buffer  k_arr_len_  */
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            if (i >= k_arr_len_) {
                val_arr_[i] = Base::kDummy();
                idx_arr_[i] = IdxT{};
            }
        }

        /*  buffer  */
        buf_len_ = 0;

        /*  k  */
        set_k_th_();
    }

    /**
     *  k
     *
     *  k  warp
     *  shuffle
     */
    __device__ __forceinline__ void set_k_th_()
    {
        /*
         *  k
         * -  k=16, kWarpWidth=32
         * - 16thread_id = (16-1) % 32 = 15, arr_idx = (16-1) / 32 = 0
         * - thread 15  val_arr_[0]
         */
        int k_thread = (k - 1) % kWarpWidth;  /*  k  */
        int k_idx = (k - 1) / kWarpWidth;      /*  k  */

        /*  k  */
        k_th_ = shfl(val_arr_[k_idx], k_thread, kWarpWidth);
    }
};

} // namespace warpsort
} // namespace INSITUANN
