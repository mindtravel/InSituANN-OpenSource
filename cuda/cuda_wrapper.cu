/**
 * cuda_wrapper.cu
 */
#include "pch.h"
#include "cuda_wrapper.h"

#include <cuda_runtime.h>

/**
 * Ccuda
 **/
extern "C" {
    /*
    * CUDA
    */
    bool cuda_is_available() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            printf("DEBUG: cudaGetDeviceCount : %s\n", cudaGetErrorString(err));
            return false;
        }
        if (device_count <= 0) {
            printf("DEBUG:  CUDA \n");
            return false;
        }
        return true;
    }

    /**
     *  Pinned Memory
     */
    void* cuda_alloc_pinned(size_t size)
    {
        void* ptr = NULL;
        cudaError_t cuda_err = cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined);

        if (cuda_err != cudaSuccess) {
            printf("CUDA Pinned Alloc failed: %s", cudaGetErrorString(cuda_err));
            return NULL;
        }

        return ptr;
    }

    /**
     *  Pinned Memory
     */
    void cuda_free_pinned(void* ptr)
    {
        cudaError_t cuda_err = cudaFreeHost(ptr);
        if (cuda_err != cudaSuccess){
            printf("CUDA Pinned Free failed: %s\n", cudaGetErrorString(cuda_err));
        }
        return;
    }

    /**
     *  GPU
     */
    void** cuda_malloc(void** d_ptr, size_t size)
    {
        cudaError_t err = cudaMalloc((void**)d_ptr, size);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        }
        return d_ptr;
    }

    void cuda_free(void* d_ptr)
    {
        cudaError_t err = cudaFree(d_ptr);
        if (err != cudaSuccess) {
            printf("CUDA free failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     *  Host to Device
     */
    void cuda_memcpy_h2d(void* d_dst, const void* h_src, size_t size)
    {
        cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     *  Device to Host
     */
    void cuda_memcpy_d2h(void* h_dst, const void* d_src, size_t size)
    {

        cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA memcpy D2H failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     *  Host to Device
     */
    void cuda_memcpy_async_h2d(void* d_dst, const void* h_src, size_t size)
    {
        cudaError_t err = cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy async H2D failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     *  Device to Host
     */
    void cuda_memcpy_async_d2h(void* h_dst, const void* d_src, size_t size)
    {
        cudaError_t err = cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA memcpy async D2H failed: %s\n", cudaGetErrorString(err));
        }
    }

    /* ========================================================================= */
    /*                              Pipeline                              */
    /* ========================================================================= */

    void cuda_pipeline_init(GpuPipelineContext* ctx, int dim, size_t total_vectors,
                float* d_vec_ptr)
    {
        if (!ctx) return;

        ctx->dimensions = dim;
        ctx->chunk_capacity = PIPELINE_CHUNK_SIZE;
        ctx->total_uploaded = 0;
        ctx->active_buf_idx = 0;
        ctx->d_vectors_base = d_vec_ptr;

        /*  */
        for (int i = 0; i < 2; i++) {
            size_t bytes = ctx->chunk_capacity * dim * sizeof(float);

            /* 1.  Pinned Memory */
            cudaError_t err = cudaHostAlloc((void**)&ctx->h_vec_buffers[i], bytes, cudaHostAllocWriteCombined);
            if (err != cudaSuccess) {
                printf("CUDA Pipeline Init: Alloc failed for buf %d: %s\n", i, cudaGetErrorString(err));
                ctx->h_vec_buffers[i] = NULL; /*  */
            }

            ctx->current_counts[i] = 0;

            /* 2.  CUDA Events () */
            cudaEvent_t evt;
            cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
            ctx->events[i] = (void*)evt;

            /* 3.  Event Wait  */
            cudaEventRecord(evt, 0);
        }
    }

    /*
     *  flush
     *  cuda_pipeline_flush_vectors_only
     */
    void cuda_pipeline_flush(GpuPipelineContext* ctx)
    {
        cuda_pipeline_flush_vectors_only(ctx);
    }

    /*
     *
     * 1.  Buffer
     * 2.  Buffer
     * 3.  Buffer
     * 4.  Buffer  GPU CPU
     */
    void cuda_pipeline_flush_vectors_only(GpuPipelineContext* ctx)
    {
        if (!ctx) return;

        int idx = ctx->active_buf_idx;

        /*  Buffer  flush */
        if (ctx->current_counts[idx] == 0) return;

        size_t bytes_vec = ctx->current_counts[idx] * ctx->dimensions * sizeof(float);

        /*  GPU  */
        float* d_dest = ctx->d_vectors_base + (ctx->total_uploaded * ctx->dimensions);

        /* ---  1:  DMA  (H2D) --- */
        /*  (0) */
        cudaMemcpyAsync(d_dest, ctx->h_vec_buffers[idx], bytes_vec, cudaMemcpyHostToDevice, 0);

        /* ---  2:  Event --- */
        /*  GPU: " Memcpy  Event  Signaled" */
        cudaEventRecord((cudaEvent_t)ctx->events[idx], 0);

        /*  */
        ctx->total_uploaded += ctx->current_counts[idx];
        ctx->current_counts[idx] = 0; /*  */

        /* ---  3:  (Ping-Pong) --- */
        int next_idx = 1 - idx;

        /* ---  4:  Buffer  --- */
        /* CPU  GPU  next_idx Buffer  */
        /*  GPU  GPU CPU  */
        cudaEventSynchronize((cudaEvent_t)ctx->events[next_idx]);

        ctx->active_buf_idx = next_idx;
    }

    void cuda_pipeline_free(GpuPipelineContext* ctx)
    {
        if (!ctx) return;

        /*  GPU  */
        cudaDeviceSynchronize();

        for (int i = 0; i < 2; i++) {
            if (ctx->h_vec_buffers[i]) {
                cudaFreeHost(ctx->h_vec_buffers[i]);
                ctx->h_vec_buffers[i] = NULL;
            }
            if (ctx->events[i]) {
                cudaEventDestroy((cudaEvent_t)ctx->events[i]);
                ctx->events[i] = NULL;
            }
        }
    }

    /*  */
    void
    cuda_cleanup_memory(float* d_query_batch, int* d_cluster_size, float* d_cluster_vectors,
                    float* d_cluster_centers, int* d_initial_indices, float* d_topk_dist, int* d_topk_index)
    {
        if (d_query_batch) cudaFree(d_query_batch);
        if (d_cluster_size) cudaFree(d_cluster_size);
        if (d_cluster_vectors) cudaFree(d_cluster_vectors);
        if (d_cluster_centers) cudaFree(d_cluster_centers);
        if (d_initial_indices) cudaFree(d_initial_indices);
        if (d_topk_dist) cudaFree(d_topk_dist);
        if (d_topk_index) cudaFree(d_topk_index);
    }

    /* CUDA  */
    void cuda_device_synchronize(void)
    {
        cudaDeviceSynchronize();
    }

    /*  CUDA  */
    const char* cuda_get_last_error_string(void)
    {
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return "no error";
        }
        return cudaGetErrorString(err);
    }

    /*  CUDA  true */
    bool cuda_check_last_error(void)
    {
        cudaError_t err = cudaGetLastError();
        return (err != cudaSuccess);
    }
}