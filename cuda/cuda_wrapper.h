#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/*
 *
 * 64K  * 1024  * 4 bytes = 256MB per chunk
 *  CPU
 */
#define PIPELINE_CHUNK_SIZE 65536

/*
 *  C  C++
 *  CUDA  cudaEvent_t void*
 */
typedef struct GpuPipelineContext {
    /* ---  --- */
    int         dimensions;
    size_t      chunk_capacity;

    /* ---  --- */
    size_t      total_uploaded;

    /* ---  (Double Buffering) --- */
    /* CPU Pinned Memory Buffers (Host) */
    float*      h_vec_buffers[2];

    /* Buffer  */
    size_t      current_counts[2];

    /*  Buffer  (0  1) - CPU  */
    int         active_buf_idx;

    /* CUDA Events ( void*) -  */
    void*       events[2];

    /* --- GPU  --- */
    float*      d_vectors_base;

} GpuPipelineContext;

/*
* cuda
*/
extern bool cuda_is_available(void);

/* GPU  */
extern void** cuda_malloc(void** d_ptr, size_t size);
extern void* cuda_alloc_pinned(size_t size);
extern void cuda_free_pinned(void* ptr);
extern void cuda_free(void* d_ptr);
extern void cuda_memcpy_h2d(void* d_dst, const void* h_src, size_t size);
extern void cuda_memcpy_d2h(void* h_dst, const void* d_src, size_t size);
extern void cuda_memcpy_async_h2d(void* d_dst, const void* h_src, size_t size);
extern void cuda_memcpy_async_d2h(void* h_dst, const void* d_src, size_t size);

/*  */
extern void cuda_cleanup_memory(float* d_query_batch, int* d_cluster_size, float* d_cluster_vectors,
                                float* d_cluster_centers, int* d_initial_indices, float* d_topk_dist, int* d_topk_index);

/* */
extern void cuda_pipeline_init(GpuPipelineContext* ctx, int dim, size_t total_vectors,
    float* d_vec_ptr);
extern void cuda_pipeline_flush(GpuPipelineContext* ctx);
extern void cuda_pipeline_flush_vectors_only(GpuPipelineContext* ctx);
extern void cuda_pipeline_free(GpuPipelineContext* ctx);

/* CUDA  */
extern void cuda_device_synchronize(void);
extern const char* cuda_get_last_error_string(void);
extern bool cuda_check_last_error(void);


#ifdef __cplusplus
}
#endif

#endif /* CUDA_WRAPPER_H */