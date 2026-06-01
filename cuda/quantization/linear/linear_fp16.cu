/**
 * LinearQuantizerFP16fp32 <-> fp16 cast +  scale/zero_point
 */
#include "quantization/linear/linear_fp16.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

namespace ivf {

namespace {

__global__ void encode_fp32_to_fp16_kernel(
    const float* __restrict__ d_data,
    int n, int dim,
    float scale, float zero_point,
    half* __restrict__ d_codes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * dim;
    if (i >= total) return;
    float v = (d_data[i] - zero_point) * scale;
    d_codes[i] = __float2half_rn(v);
}

__global__ void decode_fp16_to_fp32_kernel(
    const half* __restrict__ d_codes,
    int n, int dim,
    float scale, float zero_point,
    float* __restrict__ d_data
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * dim;
    if (i >= total) return;
    float v = __half2float(d_codes[i]);
    d_data[i] = v / scale + zero_point;
}

} // namespace

LinearQuantizerFP16GPU::LinearQuantizerFP16GPU(float scale, float zero_point)
    : scale_(scale), zero_point_(zero_point) {}

void LinearQuantizerFP16GPU::train(const float* /*h_data*/, int /*n*/, int /*dim*/) {
    // fp16  cast  per-tensor scale  min/max  scale/zero_point
}

void LinearQuantizerFP16GPU::encode(const float* d_data, int n, int dim, void* d_codes) {
    int total = n * dim;
    if (total <= 0) return;
    dim3 blk(256);
    dim3 grid((total + 255) / 256);
    encode_fp32_to_fp16_kernel<<<grid, blk>>>(d_data, n, dim, scale_, zero_point_, (half*)d_codes);
}

void LinearQuantizerFP16GPU::decode(const void* d_codes, int n, int dim, float* d_data) {
    int total = n * dim;
    if (total <= 0) return;
    dim3 blk(256);
    dim3 grid((total + 255) / 256);
    decode_fp16_to_fp32_kernel<<<grid, blk>>>((const half*)d_codes, n, dim, scale_, zero_point_, d_data);
}

int LinearQuantizerFP16GPU::code_size_bytes(int n, int dim) const {
    return n * dim * (int)sizeof(half);
}

} // namespace ivf
