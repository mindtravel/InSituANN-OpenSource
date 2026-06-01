#ifndef INSITUANN_CUDA_KMEANS_CUH
#define INSITUANN_CUDA_KMEANS_CUH

#include "pch.h"
#include "clustering/clustering.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// Config
// ============================================================
enum DataType {
    USE_FP32 = 0,
    USE_FP16 = 1,
};

struct KMeansCase {
    int n;          // number of vectors
    int dim;        // vector dimension
    int k;          // number of clusters
    int iters;      // Lloyd iterations
    int minibatch_iters;      // Minibatch iters
    int seed;       // random seed
    DistanceType dist;  //  pch.h  DistanceType
    DataType dtype;
};

// ============================================================
// StreamEnv:  buffer
// ============================================================
struct StreamEnv {
    float* d_xnorm2;      // [B]
    float* d_best_dist2;  // [B]
    int*   d_best_idx;    // [B]
    float* d_dot;         // [B * Ktile]

    StreamEnv() : d_xnorm2(nullptr), d_best_dist2(nullptr), d_best_idx(nullptr), d_dot(nullptr) {}

    void allocate(int B, int Ktile);
    void free();
};

// ============================================================
// GPU Kernels Declaration
// ============================================================

/**
 * Kernel:
 *
 * @param centroids  [k, dim]
 * @param accum  [k, dim]
 * @param counts  [k]
 * @param k
 * @param dim
 */
__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
);

/**
 * Kernel: Minibatch
 *
 * @param centroids  [k, dim]
 * @param accum minibatch [k, dim]
 * @param counts minibatch [k]
 * @param total_counts  [k]
 * @param k
 * @param dim
 */
__global__ void kernel_update_centroids_minibatch(
    float* __restrict__ centroids,      // [k, dim] (in/out)
    const float* __restrict__ accum,    // [k, dim]
    const int* __restrict__ counts,     // [k]
    int* __restrict__ total_counts,      // [k] (in/out)
    int k, int dim
);

/**
 * Kernel: best_dist2 = INF, best_idx = 0
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    int* __restrict__ best_idx,
    int n
);

/**
 * Kernel:  GEMM col-major dotT
 */
__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] centroid
    int curB,
    int curK,
    int cbase,                            // centroid
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
);

/**
 * Kernel:  GEMM
 */
__global__ void kernel_update_best_from_dotT_warp_point(
    const float* __restrict__ dotT,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
);

__global__ void kernel_update_best_from_dotT_warp_point_no_xnorm(
    const float* __restrict__ dotT,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
);

__global__ void kernel_update_best_from_dotT_warp_point_no_xnorm_half(
    const __half* __restrict__ dotT,
    const float* __restrict__ cnorm2_global,
    int curB,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
);

__global__ void kernel_assign_wmma_argmin_tile(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
);

__global__ void kernel_assign_fused_argmin64_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
);

__global__ void kernel_assign_fused_argmin64x128_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
);

__global__ void kernel_assign_fused_argmin64x128s_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
);

__global__ void kernel_assign_fused_argmin64x256_fp16_tile(
    const __half* __restrict__ data,
    const __half* __restrict__ centroids,
    const float* __restrict__ xnorm2,
    const float* __restrict__ cnorm2_global,
    int curB,
    int dim,
    int curK,
    int cbase,
    int cblocks64,
    float* __restrict__ partial_dist2,
    int* __restrict__ partial_idx
);

__global__ void kernel_reduce_fused_argmin_partials(
    const float* __restrict__ partial_dist2,
    const int* __restrict__ partial_idx,
    int curB,
    int cblocks64,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist2
);

__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const int* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
);

/**
 * Kernel: block reduce
 */
__global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  //  float
    int n
);

__global__ void kernel_u8_to_f32_batch(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int n,
    int dim
);

__global__ void kernel_u8_to_f16_batch(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int n,
    int dim
);

__global__ void kernel_f32_to_f16_batch(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n,
    int dim
);

__global__ void kernel_u8_to_bf16_batch(
    const uint8_t* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n,
    int dim
);

__global__ void kernel_f32_to_bf16_batch(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n,
    int dim
);

// ============================================================
// GPU KMeans Runner
// ============================================================

/**
 *  assignment
 *  Final Assignment Pass d_assign  d_centroids
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major
 * @param d_assign  [n]
 * @param d_centroids  [k, dim]
 * @param d_cnorm2 centroid  [k]
 * @param env StreamEnv buffer
 * @param d_data_buffer  [B, dim]/
 * @param stream CUDA
 * @param handle cuBLAS handle
 * @param B batch
 * @param Ktile centroid
 * @param n
 * @param dim
 * @param k
 */
void perform_assignment_only(
    const KMeansCase& cfg,
    const float* h_data,
    int* d_assign,
    const float* d_centroids,
    float* d_cnorm2,
    StreamEnv& env,
    float* d_data_buffer,
    cudaStream_t stream,
    cublasHandle_t handle,
    int B,
    int Ktile,
    int n,
    int dim,
    int k
);

/**
 * GPU KMeans Lloyd GEMM
 *
 *  GEMM  K
 *  l2norm  cublas GEMM
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major (float)
 * @param d_assign  [n]
 * @param d_centroids  [k, dim] float
 * @param h_objective
 */
void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major
    int* d_assign,                 // [n]
    float* d_centroids,             // [k, dim] float (in/out)
    float* h_objective               // scalar (sum dist2)
);

/**
 * Multi-GPU KMeans Lloyd: distribute data across all available GPUs.
 * Same interface as gpu_kmeans_lloyd but uses N GPUs for ~Nx speedup.
 * d_assign and d_centroids live on device 0.
 */
void gpu_kmeans_lloyd_multigpu(
    const KMeansCase& cfg,
    const float* h_data,
    int* d_assign,
    float* d_centroids,
    float* h_objective
);

/**
 * GPU KMeans Minibatch GEMM
 *
 * minibatch
 *
 *
 * @param cfg KMeans
 * @param h_data  [n, dim] row-major (float)
 * @param d_assign  [n] (minibatchassign)
 * @param d_centroids  [k, dim] float
 * @param h_objective
 */
void gpu_kmeans_minibatch(
    const KMeansCase& cfg,
    const float* h_data,            // [n, dim] row-major
    int* d_assign,                 // [n] ()
    float* d_centroids,             // [k, dim] float (in/out)
    float* h_objective               // scalar (sum dist2) ()
);

// ============================================================
// CPU Initialization Functions
// ============================================================

/**
 *  k
 *
 *  Fisher-Yates k
 *  CPU  GPU
 *
 * @param cfg KMeans  seed
 * @param data  [n, dim] row-major
 * @param out_centroids  [k, dim] row-major
 */
__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
);

// ============================================================
// Vector Reordering After Clustering
// (ClusterInfo  clustering/clustering.cuh)
// ============================================================

/**
 * GPU Kernel: cluster
 */
__global__ void kernel_compute_cluster_indices(
    const int* __restrict__ assign,      // [n]
    int* __restrict__ cluster_indices,   // [n] cluster
    int* __restrict__ cluster_counts,    // [k] cluster
    int n, int k
);

/**
 * GPU Kernel: Exclusive scan ()
 *  offsets[i] = sum(counts[0..i-1])
 *
 * @param counts cluster [k]
 * @param offsets cluster [k]
 * @param k cluster
 */
__global__ void kernel_exclusive_scan(
    const int* __restrict__ counts,   // [k]
    int* __restrict__ offsets,        // [k]
    int k
);

/**
 * GPU Kernel: assign
 *
 * @param data_in  [n, dim] row-major
 * @param assign  [n]assign[i] icluster
 * @param cluster_offsets cluster [k]
 * @param cluster_indices cluster [n]
 * @param data_out  [n, dim] row-major
 * @param n
 * @param dim
 */
__global__ void kernel_reorder_vectors_by_cluster(
    const float* __restrict__ data_in,   // [n, dim]
    const int* __restrict__ assign,      // [n]
    const int* __restrict__ cluster_offsets,  // [k]
    const int* __restrict__ cluster_indices,   // [n]
    float* __restrict__ data_out,        // [n, dim]
    int n, int dim
);

/**
 * GPU
 *
 * cluster
 * GPU
 *
 * @param cfg KMeans
 * @param h_data_in  [n, dim] row-majorpageablepinned memory
 * @param h_assign  [n]
 * @param h_data_out  [n, dim] row-major
 * @param h_cluster_info clusteroffsetscountsnullptr
 * @param device_id GPUID
 * @param B batch 1<<20 = 1M
 * @param stream CUDA0
 */
void gpu_reorder_vectors_by_cluster(
    const KMeansCase& cfg,
    const float* h_data_in,    // [n,dim] CPU
    const int*   h_assign,     // [n] CPU
    float*       h_data_out,   // [n,dim] CPU
    ClusterInfo* h_cluster_info, // optional host output
    int device_id = 0,
    int B = (1 << 20),         // batch size (e.g. 1<<20)
    cudaStream_t stream = 0
);

/**
 * GPUpermutation
 *
 * permutationperm[0..n-1]cluster 0cluster 1
 * cluster
 *
 * assign
 *  out[p] = in[perm[p]]
 *
 *  assign  H2D
 *
 * @param cfg KMeans
 * @param d_assign  [n]
 * @param h_perm_out permutation [n]nullptrcluster_info
 * @param h_cluster_info clusteroffsetscountsnullptr
 * @param device_id GPUID
 * @param B batch 1<<20 = 1M
 * @param stream CUDA0
 */
void gpu_build_permutation_by_cluster(
    const KMeansCase& cfg,
    const int* d_assign,             // [n] device
    int* h_perm_out,                 // [n] host (optional, can be nullptr)
    ClusterInfo* h_cluster_info,      // optional host output
    int device_id = 0,
    int B = (1 << 20),               // e.g. 1<<20
    cudaStream_t stream = 0
);

/**
 * CPUpermutation
 *
 *
 * pageable memorypinned memory
 *
 * @param h_data_in  [n, dim] row-major
 * @param h_perm permutation [n]perm[p] p
 * @param h_data_out  [n, dim] row-major
 * @param n
 * @param dim
 */
void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,    // [n, dim] CPU
    const int* h_perm,         // [n] CPU permutation array
    float* h_data_out,         // [n, dim] CPU output
    int n, int dim
);

/**
 * IVF K-meansK-means
 *
 *
 * 1. K-meansLloydMinibatch
 * 2. permutationcluster
 * 3.
 * 4.
 *
 * @param cfg KMeans
 * @param h_data_in  [n, dim] row-major (pageable memory)
 * @param h_data_out  [n, dim] row-major (pageable memory)
 * @param d_centroids  [k, dim] ()
 * @param h_cluster_info clusteroffsetscountsnullptr
 * @param use_minibatch Minibatchtrue=Minibatch, false=Lloyd
 * @param device_id GPUID
 * @param batch_size permutationbatch 1<<20
 * @param h_objective nullptr
 * @param h_indices_in  [n] [0, 1, 2, ..., n-1]nullptr
 * @param h_indices_out  [n]nullptr
 * @return truefalse
 */
bool ivf_kmeans(
    const KMeansCase& cfg,
    const float* h_data_in,        // [n, dim] CPU input
    float* h_data_out,             // [n, dim] CPU output (must be pre-allocated)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch = false,    // true for minibatch, false for Lloyd
    int device_id = 0,
    int batch_size = (1 << 20),   // batch size for permutation building
    float* h_objective = nullptr, // optional output
    const int* h_indices_in = nullptr,  // [n] CPU
    int* h_indices_out = nullptr        // [n] CPU
);

/**
 * ClusterInfo
 */
void free_cluster_info(ClusterInfo* info, bool is_device);

#endif // INSITUANN_CUDA_KMEANS_CUH
