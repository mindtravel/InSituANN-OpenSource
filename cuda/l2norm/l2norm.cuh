#ifndef L2NORM_H
#define L2NORM_H

#include <cuda_runtime.h>

/**
 * L2kernel
 *
 */
enum L2NormVersion {
    L2NORM_AUTO = 0,           /**<  */
    L2NORM_BASIC,              /**<  */
    L2NORM_OPTIMIZED,          /**< 1dim */
    L2NORM_OPTIMIZED_V2,       /**< 2 */
    L2NORM_OPTIMIZED_V3        /**< 3float4dim4 */
};

__global__ void l2_norm_kernel_basic(
    float *vector_data,
    float *vector_square_sum,
    int n_batch,
    int n_dim
);

__global__ void l2_norm_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim
);

__global__ void l2_norm_kernel_optimized_v2(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim
);

__global__ void l2_norm_kernel_optimized_v3(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_norm,
    int n_batch,
    int n_dim
);

/**
 * L2host
 *
 * kernel
 *
 * @param vectors device memory
 * @param vector_l2_squared L2device memory
 * @param n_batch
 * @param n_dim
 * @param version L2NORM_AUTO
 * @param stream CUDANULL
 *
 *
 * - n_dim <= 32: warp shuffle
 * - 32 < n_dim <= 128: warp shuffle + shared memory
 * - n_dim > 128: shared memory
 * - dim4: float4
 */
void compute_l2_norm_gpu(
    const float* vectors,
    float* vector_l2_squared,
    int n_batch,
    int n_dim,
    L2NormVersion version = L2NORM_AUTO,
    cudaStream_t stream = nullptr
);

// CUDA
__global__ void l2_squared_kernel_basic(
    float *vector_data,
    float *vector_square_sum,
    int n_batch,
    int n_dim
);

__global__ void l2_squared_kernel(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_squared,
    int n_batch,
    int n_dim
);

__global__ void l2_squared_kernel_optimized_v2(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_squared,
    int n_batch,
    int n_dim
);

__global__ void l2_squared_kernel_optimized_v3(
    const float* __restrict__ vectors,
    float* __restrict__ vector_l2_squared,
    int n_batch,
    int n_dim
);

/**
 * L2host
 *
 * kernel
 *
 * @param vectors device memory
 * @param vector_l2_squared L2device memory
 * @param n_batch
 * @param n_dim
 * @param version L2NORM_AUTO
 * @param stream CUDANULL
 *
 *
 * - n_dim <= 32: warp shuffle
 * - 32 < n_dim <= 128: warp shuffle + shared memory
 * - n_dim > 128: shared memory
 * - dim4: float4
 */
void compute_l2_squared_gpu(
    const float* vectors,
    float* vector_l2_squared,
    int n_batch,
    int n_dim,
    L2NormVersion version = L2NORM_AUTO,
    cudaStream_t stream = nullptr
);

#endif