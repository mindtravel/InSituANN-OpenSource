#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "l2norm/l2norm.cuh"
#include "pch.h"
#include "warpsort_utils.cuh"
#include "warpsort.cuh"
#include "bitonic.cuh"

namespace INSITUANN {
namespace warpsort_topk {

using namespace warpsort_utils;
using namespace warpsort;

// ============================================================================
// Public API: Warpsort Top-K Selection Kernel
// ============================================================================

/**
 *  top-k
 *
 *  CUDA block  block  warp  WarpSortFiltered
 *
 * @param[in] input         [batch_size, len]
 * @param[in] batch_size
 * @param[in] len
 * @param[in] k
 * @param[out] output_vals  top-k  [batch_size, k]
 * @param[out] output_idx   top-k  [batch_size, k]
 * @param[in] select_min    true  k  k
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void select_k_kernel(
    const T* __restrict__ input,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize; /* WarpSort1 */

    /* warpk */
    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k);

    /*  */
    __syncwarp();

    /*
     *
     *  WarpSortFiltered  any()  __any_sync()
     *
     *  `for (int i = ...; i < len; i += ...)`
     *  queue.add()  __any_sync()
     *
     *
     *  queue.add()
     */

    /* ceil(len / (n_warps * kWarpSize)) */
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);

    /*  laneId  */
    const T* row_input = input + row * len;

    for (int iter = 0; iter < max_iter; iter++) {
        /*  */
        __syncwarp();

        /*  */
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;

        /*  dummy  */
        if (i < len) {
        queue.add(row_input[i], static_cast<IdxT>(i));
        } else {
            /*  WarpSort dummy */
            using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
            const T dummy_val = BaseWarpSort::kDummy();
            queue.add(dummy_val, static_cast<IdxT>(-1));
        }
    }

    /*  buffer  queue  */
    queue.done();

    /*  warp done()  store */
    __syncwarp();

    /*  queue */
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
}

/**
 * Host function to launch top-k selection.
 * Automatically chooses appropriate capacity based on k.
 */
template<typename T, typename IdxT>
cudaError_t select_k(
    const T* input,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0)
{
    //
    if (k <= 0 || k > kMaxCapacity) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (len <= 0) {
        return cudaErrorInvalidValue;
    }
    if (input == nullptr || output_vals == nullptr || output_idx == nullptr) {
        return cudaErrorInvalidValue;
    }

    // CUDA grid
    if (batch_size > 2147483647) {
        return cudaErrorInvalidValue;
    }

    /*
     *  Capacity
     *
     * WarpSortFiltered  buffer
     * - Capacity  > k k
     * -  64 kMaxArrLen >= 2
     * -  Capacity > k  2
     */
    int capacity = 32;  /*  32 */
    while (capacity <= k) capacity <<= 1;  /* Capacity  > k */

    dim3 block(32);  /* 32warp*/
    dim3 grid(batch_size);

#define LAUNCH_SEL(CAP, ASC) \
    select_k_kernel<CAP, ASC, T, IdxT><<<grid, block, 0, stream>>>( \
        input, batch_size, len, k, output_vals, output_idx)

    if (select_min) {
        if      (capacity <= 64)  { LAUNCH_SEL( 64, true); }
        else if (capacity <= 128) { LAUNCH_SEL(128, true); }
        else if (capacity <= 256) { LAUNCH_SEL(256, true); }
        else if (capacity <= 512)  { LAUNCH_SEL(512, true); }
        else                       { LAUNCH_SEL(1024, true); }
    } else {
        if      (capacity <= 64)   { LAUNCH_SEL( 64, false); }
        else if (capacity <= 128)  { LAUNCH_SEL(128, false); }
        else if (capacity <= 256)  { LAUNCH_SEL(256, false); }
        else if (capacity <= 512)  { LAUNCH_SEL(512, false); }
        else                       { LAUNCH_SEL(1024, false); }
    }
#undef LAUNCH_SEL

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t select_k<float, int>(
    const float*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t select_k<float, uint32_t>(
    const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort_topk
} // namespace INSITUANN
