//=== cudatimer.h ===//
#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <cuda_runtime.h>

class CUDATimer {
private:
    std::string name_;
    bool enable_;
    bool use_cuda_event_;
    cudaEvent_t start_event_, stop_event_;
    std::chrono::high_resolution_clock::time_point start_cpu_, stop_cpu_;
    float gpu_time_ms_;

public:
    // CUDAtrue
    CUDATimer(const std::string& name, bool enable = true, bool use_cuda_event = true)
        : name_(name), enable_(enable), use_cuda_event_(use_cuda_event), gpu_time_ms_(0.0f) {
        if (!enable_) return;

        if (use_cuda_event_) {
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
            cudaEventRecord(start_event_); //
            cudaEventSynchronize(start_event_); //
        } else {
            start_cpu_ = std::chrono::high_resolution_clock::now();
        }
    }

    //
    ~CUDATimer() {
        if (!enable_) return;

        if (use_cuda_event_) {
            cudaEventRecord(stop_event_);
            cudaEventSynchronize(stop_event_); // GPU
            cudaEventElapsedTime(&gpu_time_ms_, start_event_, stop_event_);
            std::cout << "[CUDA Event Timer] " << name_ << " took: " << gpu_time_ms_ << " ms" << std::endl;
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
        } else {
            stop_cpu_ = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu_ - start_cpu_);
            std::cout << "[CPU Timer] " << name_ << " took: " << duration.count() << " ms" << std::endl;
        }
    }

    //
    float getElapsedMilliseconds() const {
        if (!enable_) return 0.0f;
        if (use_cuda_event_) {
            return gpu_time_ms_;
        } else {
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - start_cpu_);
            return static_cast<float>(duration.count());
        }
    }

    //
    static bool global_enable;
};

//
// bool CUDATimer::global_enable = true;
#endif