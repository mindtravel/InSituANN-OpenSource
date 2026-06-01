#ifndef KMEANS_REORDER_UTILS_CUH
#define KMEANS_REORDER_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

// ============================================================
// Utility Functions
// ============================================================

/**
 * CUDA
 */
static inline void cuda_check_reorder(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

/**
 * CUDA kernel
 */
static inline void cuda_check_last_reorder(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

// ============================================================
// GPU Kernels
// ============================================================

/**
 * Pass1: cluster
 *
 * @param assign cluster [n]
 * @param counts cluster [k]
 * @param n
 * @param k cluster
 * @param invalid_counter
 */
__global__ void kernel_count_clusters(const int* __restrict__ assign,
                                      int* __restrict__ counts,
                                      int n, int k,
                                      unsigned int* __restrict__ invalid_counter);

/**
 * Pass2: uint64
 *
 * @param assign cluster [n]
 * @param write_ptr cluster [k]
 * @param out_pos  [n]batch-localglobal
 * @param n
 * @param k cluster
 * @param invalid_counter
 */
__global__ void kernel_compute_positions_u64(const int* __restrict__ assign,
                                            unsigned long long* __restrict__ write_ptr, // [k]
                                            unsigned long long* __restrict__ out_pos,   // [n] (batch-local or global)
                                            int n, int k,
                                            unsigned int* __restrict__ invalid_counter);

/**
 * Pass3: scatterpermutation
 * perm[p] = global_index
 *
 * @param pos  [curB]
 * @param perm permutation [n]
 * @param base batch
 * @param curB batch
 * @param n
 * @param oob_counter
 */
__global__ void kernel_scatter_perm(const unsigned long long* __restrict__ pos, // [curB]
                                   int* __restrict__ perm,                     // [n]
                                   int base, int curB, unsigned long long n,
                                   unsigned int* __restrict__ oob_counter);

// ============================================================
// Host-side Utility Functions
// ============================================================

/**
 * hostoffsets
 *
 * @param counts cluster
 * @param offsets cluster
 * @param total
 */
static inline void build_offsets_host(const std::vector<int>& counts,
                                      std::vector<unsigned long long>& offsets,
                                      unsigned long long& total) {
    const int k = (int)counts.size();
    offsets.resize(k);
    total = 0ULL;
    for (int c = 0; c < k; ++c) {
        offsets[c] = total;
        total += (unsigned long long)counts[c];
    }
}

/**
 * permutationCPU
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
 * permutationCPU
 *
 * out[p] = in[perm[p]]
 *  perm[0]=5, perm[1]=2 out[0]=in[5], out[1]=in[2]
 *
 * @param h_indices_in  [n] [0, 1, 2, ..., n-1]
 * @param h_perm permutation [n]perm[p] p
 * @param h_indices_out  [n]
 * @param n
 */
void cpu_reorder_indices_by_permutation(
    const int* h_indices_in,   // [n] CPU
    const int* h_perm,         // [n] CPU permutation array
    int* h_indices_out,        // [n] CPU
    int n
);

#endif // KMEANS_REORDER_UTILS_CUH
