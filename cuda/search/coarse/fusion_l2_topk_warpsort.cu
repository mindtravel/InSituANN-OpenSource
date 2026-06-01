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
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "fusion_dist_topk.cuh"
#include "pch.h"
#include "search/topk/warpsort_utils.cuh"
#include "search/topk/warpsort.cuh"

#define ENABLE_CUDA_TIMING 0

namespace INSITUANN {
namespace fusion_dist_topk_warpsort {

using namespace warpsort_utils;
using namespace warpsort;

// ============================================================================
// Public API: Top-K Selection Kernel
// ============================================================================

/**
 *  top-k
 *
 *  CUDA block  block  warp  WarpSortFiltered
 *
 * @param[in] d_query_norm        query L2 [n_query]
 * @param[in] d_data_norm        data L2 [n_batch]
 * @param[in] d_inner_product         [n_query, n_batch]
 * @param[in] d_index         [n_query, n_batch]
 * @param[in] batch_size
 * @param[in] len
 * @param[in] k
 * @param[out] output_vals  top-k  [n_query, k]
 * @param[out] output_idx   top-k  [n_query, k]
 * @param[in] select_min    true  k  k
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_l2_topk_warpsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    const IdxT* __restrict__ d_index,
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

    float query_norm = d_query_norm[row];

    /* dummy*/
    /*  WarpSort dummy */
    using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
    const T dummy_val = BaseWarpSort::kDummy();

    /*  laneId  */
    const T* row_inner_product = d_inner_product + (long long)row * len;
    const IdxT* row_index = d_index + (long long)row * len;

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

    /*  */
    __syncwarp();

    /* ceil(len / (n_warps * kWarpSize)) */
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);

    for (int iter = 0; iter < max_iter; iter++) {
        /*  */
        __syncwarp();

        /*  */
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);

        if (has_data) {
            float data_norm = d_data_norm[i];
            float inner_product = row_inner_product[i];
            IdxT index = row_index[i];

            /*  */
            /*  */
            bool is_valid_index = true;
            if (std::is_signed<IdxT>::value) {
                is_valid_index = (index >= 0);
            }

            if (data_norm >= 1e-6f && is_valid_index) {
                float l2_distance =  - 2*inner_product + query_norm * query_norm + data_norm * data_norm;
                // float cos_distance = 1.0f - cos_similarity;
                queue.add(l2_distance, index);
            } else {
                /* dummy */
                queue.add(dummy_val, IdxT{});
            }
        } else {
            /* dummy */
            queue.add(dummy_val, IdxT{});
        }
    }

    /*  */
    __syncwarp();

    /*  buffer  queue  */
    queue.done();

    /*  queue */
    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
}

/**
 * Host function to launch top-k selection.
 * Automatically chooses appropriate capacity based on k.
 */
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream
)
{
    if (k > kMaxCapacity) {
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
    int capacity = 64;  /* WarpSortFiltered needs queue storage plus buffer storage. */
    const int lanes_needed = ((k + 31) / 32 + 1) * 32;
    while (capacity < lanes_needed) capacity <<= 1;

    dim3 block(32);  /* 32warp*/
    dim3 grid(batch_size);

    /*  */
#define LAUNCH_L2(CAP, ASC) \
    fusion_l2_topk_warpsort_kernel<CAP, ASC, T, IdxT><<<grid, block, 0, stream>>>( \
        d_query_norm, d_data_norm, d_inner_product, d_index, \
        batch_size, len, k, output_vals, output_idx)

    if (select_min) {
        if      (capacity <= 64)  { LAUNCH_L2( 64, true); }
        else if (capacity <= 128) { LAUNCH_L2(128, true); }
        else if (capacity <= 256) { LAUNCH_L2(256, true); }
        else if (capacity <= 512)  { LAUNCH_L2(512, true); }
        else                       { LAUNCH_L2(1024, true); }
    } else {
        if      (capacity <= 64)   { LAUNCH_L2( 64, false); }
        else if (capacity <= 128)  { LAUNCH_L2(128, false); }
        else if (capacity <= 256)  { LAUNCH_L2(256, false); }
        else if (capacity <= 512)  { LAUNCH_L2(512, false); }
        else                       { LAUNCH_L2(1024, false); }
    }
#undef LAUNCH_L2

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fusion_l2_topk_warpsort<float, int>(
    const float*, const float*, const float*, const int*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t fusion_l2_topk_warpsort<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort
} // namespace INSITUANN


void cuda_l2_topk_warpsort(
    float** h_query_vectors, float** h_data_vectors,
    int** h_index, int** h_topk_index, float** h_topk_l2_dist,
    int n_query, int n_batch, int n_dim,
    int k /**/
){
    /**
    * batchtopk [batch, k]
    **/
//    table_2D("h_topk_index", h_topk_index, n_query, k);

    float alpha = 1.0f;
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda
    bool query_copied = false; // query

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

    cudaStream_t streams[NUM_STREAMS];

    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_dist = n_query * n_batch * sizeof(float);
    size_t size_index = n_query * n_batch * sizeof(int);
    size_t size_topk_dist = n_query * k * sizeof(float);
    size_t size_topk_idx = n_query * k * sizeof(int);

    // cuBLAS
    cublasHandle_t handle;
    // cublasSetStream(handle, streams[0]);

    //
    float *d_query_vectors, *d_data_vectors, *d_inner_product, *d_topk_l2_dist,
        *d_query_norm, *d_data_norm;
    int *d_index, *d_topk_index;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING);

        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_inner_product, size_dist);/*querydata*/
        cudaMalloc(&d_index, size_index);/*querydata*/
        cudaMalloc(&d_topk_l2_dist, size_topk_dist);/*topk*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*topk*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*queryl2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*datal2 Norm*/

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        cublasCreate(&handle);
    }

    //
    {
        // COUT_ENDL("begin data transfer");

        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        //
        if(query_copied == false){
            cudaMemcpy2D(
                d_query_vectors,
                n_dim * sizeof(float),
                h_query_vectors[0],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_query,
                cudaMemcpyHostToDevice
            );
            query_copied = true;
        }

        /* data */
        cudaMemcpy2D(
            d_data_vectors,
            n_dim * sizeof(float),
            h_data_vectors[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_data_vectors, h_data_vectors, size_data, cudaMemcpyHostToDevice);

        /*  */
        cudaMemcpy2D(
            d_index,
            n_batch * sizeof(int),
            h_index[0],
            n_batch * sizeof(int),
            n_batch * sizeof(int),
            n_query,
            cudaMemcpyHostToDevice
        );
        // CHECK_CUDA_ERRORS;

        /* -1 */
        thrust::fill(
            thrust::device_pointer_cast(d_topk_l2_dist),/*pointer_cast*/
            thrust::device_pointer_cast(d_topk_l2_dist) + (n_query * k),  /*  */
            FLT_MAX
        );
        // cudaMemset((void*)d_topk_l2_dist, (int)0xEF, n_query * k * sizeof(float)) /*memset*/
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);


        // COUT_ENDL("finish data transfer");


    }

    // print_cuda_2D("index matrix", d_index, n_query, n_batch);
    // print_cuda_2D("l2 distance matrix", d_inner_product, n_query, n_batch);
    // print_2D("query vector", h_query_vectors, n_query, n_dim);
    // print_2D("data vector", h_data_vectors, n_batch, n_dim);
    // print_cuda_2D("query vector", d_query_vectors, n_query, n_dim);
    // print_cuda_2D("data vector", d_data_vectors, n_batch, n_dim);
    // print_cuda_2D("topk index matrix", d_topk_index, n_query, k);
    // print_cuda_2D("topk l2 distance matrix", d_topk_l2_dist, n_query, k);

    /*  */
    {
        // COUT_ENDL("begin_kernel");

        CUDATimer timer_compute("Kernel Execution: l2 Norm + matrix multiply", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm,
            n_query, n_dim
        );

        l2_norm_kernel<<<dataDim, vectorDim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm,
            n_batch, n_dim
        );
        // COUT_ENDL("finish l2 norm");

        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);

        // table_cuda_1D("query_norm", d_query_norm, n_query);
        // table_cuda_1D("data_norm", d_data_norm, n_batch);
        // table_cuda_2D("data vectors", d_data_vectors, n_batch, n_dim);

        /**
        * cuBLAS
        * cuBLASleading dimension
        * */
       cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_batch, n_query, n_dim,
            &alpha,
            d_data_vectors, n_dim,
            d_query_vectors, n_dim,
            &beta,
            d_inner_product, n_batch
        );

        cudaDeviceSynchronize();
        // COUT_ENDL("finish matrix multiply");

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);
    }

    {
        CUDATimer timer_compute("Kernel Execution: l2 + topk", ENABLE_CUDA_TIMING);

        INSITUANN::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            n_query, n_batch, k,
            d_topk_l2_dist, d_topk_index,
            true /* select min */
        );

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);

        cudaDeviceSynchronize();

        // CUDARAFT


    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(h_topk_index[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_topk_l2_dist[0], d_topk_l2_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_inner_product);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        cudaFree(d_index);
        cudaFree(d_topk_l2_dist);
        cudaFree(d_topk_index);
        // CUDA
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}
