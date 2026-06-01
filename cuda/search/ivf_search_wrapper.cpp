#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "search/ivf_search.cuh"

extern "C" {
/**
 * CCivf_search_pipeline device
 * C++
 */
int ivf_search_pipeline_wrapper(float* d_query_batch,
                                  int* d_cluster_size,
                                  float* d_cluster_vectors,
                                  float* d_cluster_centers,
                                  int* d_initial_indices,
                                  float* d_topk_dist,
                                  int* d_topk_index,
                                  int n_query, int n_dim, int n_total_cluster,
                                  int n_total_vectors, int n_probes, int k, int distance_mode)
{
    /* PostgreSQL */
    fprintf(stderr, "ivf_search_pipeline_wrapper: ==========  ==========\n");
    fprintf(stderr, "ivf_search_pipeline_wrapper: n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d\n",
            n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k);

    /* GPU */
    fprintf(stderr, "ivf_search_pipeline_wrapper: GPU:\n");
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_query_batch=%p (expected size: %zu bytes)\n",
            (void*)d_query_batch, (size_t)n_query * n_dim * sizeof(float));
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_cluster_size=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_size, (size_t)n_total_cluster * sizeof(int));
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_cluster_vectors=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_vectors, (size_t)n_total_vectors * n_dim * sizeof(float));
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_cluster_centers=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_centers, (size_t)n_total_cluster * n_dim * sizeof(float));
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_initial_indices=%p (expected size: %zu bytes)\n",
            (void*)d_initial_indices, d_initial_indices ? (size_t)n_query * n_total_cluster * sizeof(int) : 0);
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_topk_dist=%p (expected size: %zu bytes)\n",
            (void*)d_topk_dist, (size_t)n_query * k * sizeof(float));
    fprintf(stderr, "ivf_search_pipeline_wrapper:   d_topk_index=%p (expected size: %zu bytes)\n",
            (void*)d_topk_index, (size_t)n_query * k * sizeof(int));

    /*  */
    if (!d_query_batch || !d_cluster_size || !d_cluster_vectors ||
        !d_cluster_centers || !d_topk_dist || !d_topk_index) {
        fprintf(stderr, "ivf_search_pipeline_wrapper: : GPUNULL\n");
        return -1;
    }

    if (n_query <= 0 || n_dim <= 0 || n_total_cluster <= 0 ||
        n_total_vectors <= 0 || n_probes <= 0 || k <= 0) {
        fprintf(stderr, "ivf_search_pipeline_wrapper: : \n");
        return -1;
    }

    if (n_probes > n_total_cluster) {
        fprintf(stderr, "ivf_search_pipeline_wrapper: : n_probes(%d) > n_total_cluster(%d)\n",
                n_probes, n_total_cluster);
    }

    fprintf(stderr, "ivf_search_pipeline_wrapper: ==========  ivf_search_pipeline ==========\n");

    try {
        ivf_search_pipeline_wrapper(d_query_batch, d_cluster_size, d_cluster_vectors, d_cluster_centers,
                             d_initial_indices,  // nullptr
                             d_topk_dist, d_topk_index,
                             n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k, distance_mode);
        fprintf(stderr, "ivf_search_pipeline_wrapper: \n");
        return 0;
    } catch (const std::exception& e) {
        /* PostgreSQL */
        fprintf(stderr, "ivf_search_pipeline_wrapper:  - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "ivf_search_pipeline_wrapper: \n");
        return -1;
    }
}

/*
 *  C wrapper
 *  C  C++
 */

void* ivf_create_index_context_wrapper(void) {
    try {
        return ivf_create_index_context();
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_create_index_context_wrapper:  - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "ivf_create_index_context_wrapper: \n");
        return nullptr;
    }
}

void ivf_destroy_index_context_wrapper(void* ctx_ptr) {
    try {
        ivf_destroy_index_context(ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_destroy_index_context_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_destroy_index_context_wrapper: \n");
    }
}

int ivf_load_dataset_wrapper(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    try {
        return ivf_load_dataset(idx_ctx_ptr, d_cluster_size, d_cluster_vectors, d_cluster_centers,
                               n_total_clusters, n_total_vectors, n_dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_load_dataset_wrapper:  - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "ivf_load_dataset_wrapper: \n");
        return -1;
    }
}

void* ivf_create_batch_context_wrapper(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters) {
    try {
        return ivf_create_batch_context(max_n_query, n_dim, max_n_probes, max_k, n_total_clusters);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_create_batch_context_wrapper:  - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "ivf_create_batch_context_wrapper: \n");
        return nullptr;
    }
}

void ivf_destroy_batch_context_wrapper(void* ctx_ptr) {
    try {
        ivf_destroy_batch_context(ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_destroy_batch_context_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_destroy_batch_context_wrapper: \n");
    }
}

void ivf_pipeline_stage1_prepare_wrapper(
    void* batch_ctx_ptr,
    float* query_batch_host,
    int n_query
) {
    try {
        ivf_pipeline_stage1_prepare(batch_ctx_ptr, query_batch_host, n_query);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_stage1_prepare_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_stage1_prepare_wrapper: \n");
    }
}

/*  */
int ivf_check_index_initialized_wrapper(void* idx_ctx_ptr) {
    if (!idx_ctx_ptr) {
        fprintf(stderr, "ivf_check_index_initialized_wrapper: idx_ctx_ptr is NULL\n");
        return 0;
    }
    try {
        /*  IVFIndexContext  C++  */
        /*  */
        /*  ivf_pipeline_stage2_compute  */
        return 1; /* 1 compute  */
    } catch (...) {
        fprintf(stderr, "ivf_check_index_initialized_wrapper: \n");
        return 0;
    }
}

void ivf_pipeline_stage2_compute_wrapper(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
) {
    /*  */
    if (!batch_ctx_ptr) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: ERROR - batch_ctx_ptr is NULL\n");
        fflush(stderr);
        return;
    }
    if (!idx_ctx_ptr) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: ERROR - idx_ctx_ptr is NULL\n");
        fflush(stderr);
        return;
    }
    if (n_query <= 0 || n_probes <= 0 || k <= 0) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: ERROR -  - n_query=%d, n_probes=%d, k=%d\n",
                n_query, n_probes, k);
        fflush(stderr);
        return;
    }

    fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper:  - batch_ctx_ptr=%p, idx_ctx_ptr=%p, n_query=%d, n_probes=%d, k=%d\n",
            batch_ctx_ptr, idx_ctx_ptr, n_query, n_probes, k);
    fflush(stderr);

    try {
        ivf_pipeline_stage2_compute(batch_ctx_ptr, idx_ctx_ptr, n_query, n_probes, k, distance_mode);
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: \n");
        fflush(stderr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: EXCEPTION - %s\n", e.what());
        fflush(stderr);
        /*  C  C++  */
        /*  stderrPostgreSQL  */
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: UNKNOWN EXCEPTION\n");
        fflush(stderr);
    }
}

void ivf_pipeline_get_results_wrapper(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
) {
    /*  */
    if (!batch_ctx_ptr) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: ERROR - batch_ctx_ptr is NULL\n");
        fflush(stderr);
        return;
    }
    if (!topk_dist || !topk_index) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: ERROR - topk_dist or topk_index is NULL\n");
        fflush(stderr);
        return;
    }
    if (n_query <= 0 || k <= 0) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: ERROR -  - n_query=%d, k=%d\n",
                n_query, k);
        fflush(stderr);
        return;
    }

    fprintf(stderr, "ivf_pipeline_get_results_wrapper:  - batch_ctx_ptr=%p, n_query=%d, k=%d\n",
            batch_ctx_ptr, n_query, k);
    fflush(stderr);

    try {
        ivf_pipeline_get_results(batch_ctx_ptr, topk_dist, topk_index, n_query, k);
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: \n");
        fflush(stderr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: EXCEPTION - %s\n", e.what());
        fflush(stderr);
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: UNKNOWN EXCEPTION\n");
        fflush(stderr);
    }
}

void ivf_pipeline_sync_batch_wrapper(void* batch_ctx_ptr) {
    try {
        ivf_pipeline_sync_batch(batch_ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_sync_batch_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_sync_batch_wrapper: \n");
    }
}

/*  C wrapper */
void ivf_init_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    try {
        ivf_init_streaming_upload(idx_ctx_ptr, n_total_clusters, n_total_vectors, n_dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_init_streaming_upload_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_init_streaming_upload_wrapper: \n");
    }
}

void ivf_append_cluster_data_wrapper(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
) {
    try {
        ivf_append_cluster_data(idx_ctx_ptr, cluster_id, host_vector_data, count, start_offset_idx);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_append_cluster_data_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_append_cluster_data_wrapper: \n");
    }
}

void ivf_finalize_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
) {
    try {
        ivf_finalize_streaming_upload(idx_ctx_ptr, center_data_flat, total_vectors_check);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_finalize_streaming_upload_wrapper:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_finalize_streaming_upload_wrapper: \n");
    }
}

/* ========================================================================= */
/*                                                              */
/* ========================================================================= */

/* Session */
#define MAX_GPU_INDICES 64

typedef struct {
    unsigned int index_oid;  /* Oid  C++  unsigned int */
    void* handle;
    bool active;
} GpuIndexEntry;

static GpuIndexEntry g_gpu_indices[MAX_GPU_INDICES] = {0};
static int g_gpu_indices_count = 0;

/**
 *
 *
 * @param index_oid  OID
 * @param gpu_handle GPU
 */
void ivf_register_index_instance(unsigned int index_oid, void* gpu_handle) {
    if (gpu_handle == NULL) {
        fprintf(stderr, "ivf_register_index_instance: gpu_handle  NULL\n");
        return;
    }

    /*  OID  */
    for (int i = 0; i < g_gpu_indices_count; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            /*  */
            if (g_gpu_indices[i].handle != NULL) {
                ivf_destroy_index_context_wrapper(g_gpu_indices[i].handle);
            }
            g_gpu_indices[i].handle = gpu_handle;
            fprintf(stderr, "ivf_register_index_instance:  OID %u \n", index_oid);
            return;
        }
    }

    /*  */
    if (g_gpu_indices_count < MAX_GPU_INDICES) {
        g_gpu_indices[g_gpu_indices_count].index_oid = index_oid;
        g_gpu_indices[g_gpu_indices_count].handle = gpu_handle;
        g_gpu_indices[g_gpu_indices_count].active = true;
        g_gpu_indices_count++;
        fprintf(stderr, "ivf_register_index_instance:  OID %u: %d\n", index_oid, g_gpu_indices_count);
    } else {
        /*  */
        for (int i = 0; i < MAX_GPU_INDICES; i++) {
            if (!g_gpu_indices[i].active) {
                g_gpu_indices[i].index_oid = index_oid;
                g_gpu_indices[i].handle = gpu_handle;
                g_gpu_indices[i].active = true;
                fprintf(stderr, "ivf_register_index_instance:  %d  OID %u\n", i, index_oid);
                return;
            }
        }
        fprintf(stderr, "ivf_register_index_instance:  -  %d \n", MAX_GPU_INDICES);
    }
}

/**
 *  OID  GPU
 *
 * @param index_oid  OID
 * @return GPU  NULL
 */
void* ivf_get_index_instance(unsigned int index_oid) {
    for (int i = 0; i < g_gpu_indices_count; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            return g_gpu_indices[i].handle;
        }
    }

    /*  count  */
    for (int i = 0; i < MAX_GPU_INDICES; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            return g_gpu_indices[i].handle;
        }
    }

    return NULL;
}

/**
 *
 *
 * @param index_oid  OID
 */
void ivf_unregister_index_instance(unsigned int index_oid) {
    for (int i = 0; i < MAX_GPU_INDICES; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            if (g_gpu_indices[i].handle != NULL) {
                ivf_destroy_index_context_wrapper(g_gpu_indices[i].handle);
            }
            g_gpu_indices[i].active = false;
            g_gpu_indices[i].handle = NULL;
            fprintf(stderr, "ivf_unregister_index_instance:  OID %u\n", index_oid);
            return;
        }
    }
}
}
