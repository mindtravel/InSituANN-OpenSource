#ifndef IVF_LINEAR_FP16_CUH
#define IVF_LINEAR_FP16_CUH

#include "quantization/quantizer.cuh"
#include <cstddef>

namespace ivf {

/**
 *  fp16 GPUfp32 -> fp16device  GPU
 */
class LinearQuantizerFP16GPU : public QuantizerInterface {
public:
    explicit LinearQuantizerFP16GPU(float scale = 1.0f, float zero_point = 0.0f);
    void train(const float* h_data, int n, int dim) override;
    void encode(const float* d_data, int n, int dim, void* d_codes) override;
    void decode(const void* d_codes, int n, int dim, float* d_data) override;
    int code_size_bytes(int n, int dim) const override;

    float get_scale() const { return scale_; }
    float get_zero_point() const { return zero_point_; }

private:
    float scale_;
    float zero_point_;
};

/**
 *  fp16 CPUfp32 -> fp16host
 */
class LinearQuantizerFP16CPU : public QuantizerInterfaceCPU {
public:
    explicit LinearQuantizerFP16CPU(float scale = 1.0f, float zero_point = 0.0f);
    void train(const float* h_data, int n, int dim) override;
    void encode(const float* h_data, int n, int dim, void* h_codes) override;
    void decode(const void* h_codes, int n, int dim, float* h_data) override;
    int code_size_bytes(int n, int dim) const override;

    float get_scale() const { return scale_; }
    float get_zero_point() const { return zero_point_; }

private:
    float scale_;
    float zero_point_;
};

} // namespace ivf

#endif
