#ifndef IVFTENSOR_CPU_FINE_H
#define IVFTENSOR_CPU_FINE_H

/**
 * CPU Fine Search
 *
 * "GPU  + CPU " CPU
 *  CPU  cputo_do_list.md
 *  6  V0..V5
 *
 *   V0   FP32 +  (-O0)              Reference scalar path
 *   V1  V0  + -O3 -march=native -ffast-math
 *   V2   AVX-512 + FMA intrinsics
 *   V3  V2 + Register-level Query Tiling           base  vs  query
 *   V4  V3 + _mm_prefetch                   DRAM
 *   V5  V4 + AoSoA<kTile>                   gather
 *
 *  orchestrationGPU  (GEMM + fused warpsort)
 *  [n_query, n_probes]  cluster idcudaMemcpy  host
 *  OpenMP  CPU  + top-k
 *
 *
 *   - V0..V4row-major [n_total_vectors, dim] cluster
 *   - V5   AoSoA<kTile> cpu_fine_layout.h
 *
 * V2..V5  dim % 16 == 0AVX-512
 * SIFT (dim=128)BERT-like (dim=768)
 */

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**  kernels_v{0..5}.cpp  */
typedef enum {
    CPU_FINE_V0 = 0,  /*  FP32-O0 */
    CPU_FINE_V1 = 1,  /*  FP32-O3 -march=native */
    CPU_FINE_V2 = 2,  /*  AVX-512 FMA */
    CPU_FINE_V3 = 3,  /* V2 + Query Tiling */
    CPU_FINE_V4 = 4,  /* V3 + Prefetch */
    CPU_FINE_V5 = 5,  /* V4 + AoSoA per-query reference for AoSoA */
    CPU_FINE_V6 = 6,  /* V3  + V5 AoSoA best-of-both */
    CPU_FINE_V3_U8 = 7,  /* B1V3  + uint8 AVX-512 madd_epi16*/
    CPU_FINE_PQ_RESID = 8,  /* Residual PQ + rerankLUT  + u8  */
    CPU_FINE_V3_TOUCHED = 9,  /* V3 + iterate only clusters touched by current batch */
    CPU_FINE_V3_U8_TOUCHED = 10,  /* V3_U8 + iterate only clusters touched by current batch */
    CPU_FINE_V1_TOUCHED = 11,  /* V1 + iterate only clusters touched by current batch */
    CPU_FINE_V2_TOUCHED = 12,  /* V2 + iterate only clusters touched by current batch */
    CPU_FINE_N_VARIANTS = 13
} CpuFineVariant;

/** Runtime statistics. */
typedef struct {
    double coarse_ms;     /* GPU GEMM + warpsortwall time */
    double h2d_ms;        /* query H2D  */
    double d2h_ms;        /* cluster id +  D2H  */
    double fine_ms;       /* CPU  + top-k  wall time */
    double total_ms;      /*  wall time */
    long long n_fma;      /*  FMA */
} CpuFineStats;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace ivftensor {
namespace cpu_fine {

/** AoSoA tile AVX-512  16AVX2  8NEON  4 */
#ifndef IVFTENSOR_CPU_FINE_TILE
#define IVFTENSOR_CPU_FINE_TILE 16
#endif
constexpr int kTile = IVFTENSOR_CPU_FINE_TILE;

/**
 * CPU  kernel
 * cpu_fine_kernel_vN
 *
 *
 *   h_base          row-major [n_total_vectors, dim]V5  h_base_aosoa
 *   h_base_aosoa    AoSoA  V5  nullptr
 *   h_aosoa_offsets [n_total_clusters+1] V5  cluster  AoSoA
 *   h_query         [n_query, dim]
 *   h_cluster_offsets [n_total_clusters+1] row-major  cluster V0..V4
 *   h_cluster_counts  [n_total_clusters]    AoSoA
 *   h_coarse_cluster_ids [n_query, n_probes]
 *
 * " topk "
 *   h_topk_index    [n_query, topk]
 *   h_topk_dist     [n_query, topk]
 *
 *  FMA  Roofline /
 */
typedef long long (*CpuFineKernelFn)(
    const float* h_base,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const float* h_query,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int distance_mode,    /* 0=L2, 1=COSINE */
    int num_threads,
    int* h_topk_local_idx,     /* [n_query, topk] cluster-- */
    float* h_topk_dist         /* [n_query, topk] */
);

long long cpu_fine_kernel_v0(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v1(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v1_touched(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v2(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v2_touched(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v3(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v3_touched(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v4(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v5(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);
long long cpu_fine_kernel_v6(
    const float*, const float*, const long long*,
    const long long*, const int*,
    const float*, const int*, int, int, int, int, int, int, int, int*, float*);

/** kernel  */
CpuFineKernelFn dispatch(CpuFineVariant v);

/**  */
const char* variant_name(CpuFineVariant v);

}  // namespace cpu_fine
}  // namespace ivftensor


/**
 *  GPU-coarse + CPU-fine
 *
 *  KMeans  GPU
 *
 *   1.  h_query  GPUd_query
 *   2.  cuBLAS GEMM + fused_l2/cos_topk_warpsort  cluster id
 *   3. cudaMemcpy  host
 *   4.  CPU  kernel
 *   5.  h_reordered_indices cluster-
 *
 * @param d_centers       [n_total_clusters, dim]
 * @param d_centers_norm  [n_total_clusters]       L2 norm
 *                         nullptr
 *                        SIFT-1B
 * @param h_base_rowmajor row-major baseV0..V4
 * @param h_base_aosoa    AoSoA baseV5  nullptr
 * @param h_aosoa_offsets [n_total_clusters+1]V5  nullptr
 */
void ivf_search_cpu_fine(
    /* GPU-resident index */
    const float* d_centers,
    const float* d_centers_norm,     /*  nullptr */
    int n_total_clusters,
    int dim,
    /* Host-resident base */
    const float* h_base_rowmajor,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_reordered_indices,  /* [n_total_vectors]  nullptr */
    int n_total_vectors,
    /* Query */
    const float* h_query,
    int n_query,
    int n_probes,
    int topk,
    int distance_mode,               /* 0=L2, 1=COSINE */
    /*  */
    CpuFineVariant variant,
    int num_threads,                 /* 0 = OpenMP  */
    /*  */
    int* h_topk_index,               /* [n_query, topk]  h_reordered_indices  */
    float* h_topk_dist,              /* [n_query, topk] */
    CpuFineStats* stats              /*  nullptr */
);


/* =========================================================================
 *  coarse handle batch-loop  cudaMalloc / memcpy centers
 *
 * offline batch
 *   CoarseHandle h;
 *   coarse_handle_init(&h, d_centers, nlist, dim);
 *   for (batch b: ...) {
 *       coarse_search(&h, h_query_batch, n_batch, nprobe, L2,
 *                     h_cluster_ids_batch, &t_coarse, &t_d2h);
 *   }
 *   coarse_handle_release(&h);
 *
 * d_centers  handle  freed_centers_norm  handle
 *  init
 * ========================================================================= */
struct CoarseHandle {
    const float* d_centers;          /*  owned */
    float*       d_centers_norm;     /* handle owned */
    void*        cublas_handle;      /* cublasHandle_tvoid*  */
    int          n_total_clusters;
    int          dim;
    int          distance_mode_cached; /*  norm  mode L2  norm */
};

struct CoarseTimingBreakdown {
    double query_h2d_ms = 0.0;
    double seq_init_ms = 0.0;
    double query_norm_ms = 0.0;
    double gemm_ms = 0.0;
    double topk_ms = 0.0;
    double cluster_d2h_ms = 0.0;
    double coarse_compute_ms = 0.0;
    double coarse_total_ms = 0.0;
};

void coarse_handle_init(
    CoarseHandle* h,
    const float* d_centers,
    int n_total_clusters,
    int dim
);

void coarse_handle_release(CoarseHandle* h);

/**
 *  GPU  cluster id  host
 *  cudaMalloc/cudaFree  per-batch  scratchquery / inner-product / seq
 *  centers / norm handle
 */
void coarse_search(
    const CoarseHandle* h,
    const float* h_query,            /* [n_query_batch, dim] */
    int n_query_batch,
    int n_probes,
    int distance_mode,               /* 0=L2, 1=COSINE */
    int* h_cluster_ids_out,          /* [n_query_batch, n_probes] */
    double* out_coarse_ms,           /*  nullptr */
    double* out_h2d_ms,              /*  nullptr */
    double* out_d2h_ms               /*  nullptr */
);

const CoarseTimingBreakdown* coarse_get_last_timing();

/**
 *  CPU fine coarse coarse_search  cross-batch
 * h_coarse_cluster_ids  [n_query_accum, n_probes]
 */
void fine_search_cpu(
    CpuFineVariant variant,
    const float* h_base_rowmajor,
    const float* h_base_aosoa,
    const long long* h_aosoa_offsets,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const int* h_reordered_indices,  /*  nullptr */
    int n_total_clusters,
    int dim,
    const float* h_query,
    const int* h_coarse_cluster_ids,
    int n_query_accum,
    int n_probes,
    int topk,
    int distance_mode,
    int num_threads,
    int* h_topk_index,               /* [n_query_accum, topk] */
    float* h_topk_dist,              /* [n_query_accum, topk] */
    double* out_fine_ms,             /*  nullptr */
    long long* out_n_fma             /*  nullptr */
);

#endif /* __cplusplus */

#endif  /* IVFTENSOR_CPU_FINE_H */
