#ifndef IVF_QUANTIZER_CUH
#define IVF_QUANTIZER_CUH

#include "pch.h"
#include <cstddef>

namespace ivf {

enum class QuantizeType {
    NONE,
    LINEAR_INT8,
    LINEAR_FP16,
    PQ
};

struct QuantizerConfig {
    QuantizeType type = QuantizeType::NONE;
    int n_subspace = 0;   // PQ
    int n_codes = 0;      // PQ
    // linear:  per-tensor scale (fp16 )
    float scale = 1.0f;
    float zero_point = 0.0f;
};

/**
 * GPU device train -> encode -> decode
 */
class QuantizerInterface {
public:
    virtual void train(const float* h_data, int n, int dim) = 0;
    virtual void encode(const float* d_data, int n, int dim, void* d_codes) = 0;
    virtual void decode(const void* d_codes, int n, int dim, float* d_data) = 0;
    virtual int code_size_bytes(int n, int dim) const = 0;
    virtual ~QuantizerInterface() = default;
};

/**
 * CPU host
 */
class QuantizerInterfaceCPU {
public:
    virtual void train(const float* h_data, int n, int dim) = 0;
    virtual void encode(const float* h_data, int n, int dim, void* h_codes) = 0;
    virtual void decode(const void* h_codes, int n, int dim, float* h_data) = 0;
    virtual int code_size_bytes(int n, int dim) const = 0;
    virtual ~QuantizerInterfaceCPU() = default;
};

} // namespace ivf

#endif
