/**
 * LinearQuantizerFP16CPUhost  fp32 <-> fp16
 *  GPU half  binary16  GPU
 */
#include "quantization/linear/linear_fp16.cuh"
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace ivf {

namespace {

static float uint32_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static uint32_t float_to_uint32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    return u;
}

//  float -> half (round to nearest) GPU half
inline uint16_t float_to_half_rn(float f) {
    uint32_t fint = float_to_uint32(f);
    uint32_t sign = (fint >> 16) & 0x8000u;
    fint &= 0x7fffffffu;
    if (fint >= 0x7f800000u) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (fint < 0x38800000u) {
        return (uint16_t)sign;
    }
    uint32_t exp = (fint >> 23) - 127 + 15;
    uint32_t mant = fint & 0x7fffffu;
    if (exp >= 31u) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

//  half -> float
inline float half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp_mant = (uint32_t)(h & 0x7fffu);
    if (exp_mant == 0) {
        return uint32_to_float(sign);
    }
    uint32_t exp = (exp_mant >> 10) & 0x1fu;
    uint32_t mant = exp_mant & 0x3ffu;
    if (exp == 31u) {
        uint32_t u = sign | 0x7f800000u | (mant << 13);
        return uint32_to_float(u);
    }
    if (exp == 0u) {
        // half denormal: value = mant * 2^-24
        uint32_t u = sign | (103u << 23) | (mant << 13);
        return uint32_to_float(u);
    }
    uint32_t u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return uint32_to_float(u);
}

} // namespace

LinearQuantizerFP16CPU::LinearQuantizerFP16CPU(float scale, float zero_point)
    : scale_(scale), zero_point_(zero_point) {}

void LinearQuantizerFP16CPU::train(const float* /*h_data*/, int /*n*/, int /*dim*/) {}

void LinearQuantizerFP16CPU::encode(const float* h_data, int n, int dim, void* h_codes) {
    const int total = n * dim;
    uint16_t* codes = static_cast<uint16_t*>(h_codes);
    for (int i = 0; i < total; ++i) {
        float v = (h_data[i] - zero_point_) * scale_;
        codes[i] = float_to_half_rn(v);
    }
}

void LinearQuantizerFP16CPU::decode(const void* h_codes, int n, int dim, float* h_data) {
    const int total = n * dim;
    const uint16_t* codes = static_cast<const uint16_t*>(h_codes);
    for (int i = 0; i < total; ++i) {
        float v = half_to_float(codes[i]);
        h_data[i] = v / scale_ + zero_point_;
    }
}

int LinearQuantizerFP16CPU::code_size_bytes(int n, int dim) const {
    return n * dim * (int)sizeof(uint16_t);
}

} // namespace ivf
