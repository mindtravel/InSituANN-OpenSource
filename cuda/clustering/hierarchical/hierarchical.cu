#include "hierarchical.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>

#include <cublas_v2.h>

namespace {

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

void free_result_buffers(
    float* reordered_data,
    int* reordered_indices,
    float* h_centroids,
    ClusterInfo* info
) {
    if (reordered_data) std::free(reordered_data);
    if (reordered_indices) std::free(reordered_indices);
    if (h_centroids) std::free(h_centroids);
    if (info) {
        if (info->offsets) std::free(info->offsets);
        if (info->counts) std::free(info->counts);
        info->offsets = nullptr;
        info->counts = nullptr;
        info->k = 0;
    }
}

float compute_pairwise_distance_sq(const float* a, const float* b, int dim, DistanceType mode) {
    if (mode == L2_DISTANCE) {
        float d2 = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = a[j] - b[j];
            d2 += diff * diff;
        }
        return d2;
    } else {
        // COSINE_DISTANCE: 1 - cos(a,b)
        float dot = 0.0f;
        float na = 0.0f;
        float nb = 0.0f;
        for (int j = 0; j < dim; ++j) {
            dot += a[j] * b[j];
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        float denom = sqrtf(na * nb);
        if (denom <= 1e-12f) {
            return 1.0f;
        }
        float cosine = dot / denom;
        return 1.0f - cosine;
    }
}

} // namespace

BalancedHierarchicalClustering::BalancedHierarchicalClustering(float alpha)
    : alpha_(alpha), fitted_(false), h_centroids_(nullptr), h_reordered_data_(nullptr), h_reordered_indices_(nullptr) {
}

BalancedHierarchicalClustering::~BalancedHierarchicalClustering() {
    release();
}

void BalancedHierarchicalClustering::release() {
    if (!fitted_) return;

    free_result_buffers(
        h_reordered_data_,
        h_reordered_indices_,
        h_centroids_,
        &cluster_info_
    );

    result_ = ClusteringResult{};
    cluster_info_ = ClusterInfo{};
    h_centroids_ = nullptr;
    h_reordered_data_ = nullptr;
    h_reordered_indices_ = nullptr;
    fitted_ = false;
}

void BalancedHierarchicalClustering::fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) {
    if (fitted_) {
        release();
    }

    if (!h_data || n <= 0 || dim <= 0 || cfg.n_clusters <= 0) {
        throw std::invalid_argument("BalancedHierarchicalClustering::fit invalid arguments");
    }

    //
    int n_train = std::max(1, (int)std::round(alpha_ * (double)n));
    if (n_train > n) n_train = n;

    std::vector<float> h_train_data;
    const float* h_train_ptr;
    std::vector<int> train_indices;
    if (n_train < n) {
        train_indices.resize(n);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::mt19937 rng(cfg.seed);
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        train_indices.resize(n_train);

        h_train_data.resize((size_t)n_train * dim);
        for (int i = 0; i < n_train; ++i) {
            const float* src = h_data + (size_t)train_indices[i] * dim;
            float* dst = h_train_data.data() + (size_t)i * dim;
            std::memcpy(dst, src, sizeof(float) * dim);
        }

        h_train_ptr = h_train_data.data();
    } else {
        h_train_ptr = h_data;
    }

    int M = cfg.n_clusters;
    int K1 = std::max(2, (int)std::ceil(std::sqrt((double)M)));
    int s = std::max(1, (int)std::ceil((double)n_train / M));

    //
    KMeansCase cfg1;
    cfg1.n = n_train;
    cfg1.dim = dim;
    cfg1.k = K1;
    cfg1.iters = cfg.max_iters;
    cfg1.minibatch_iters = cfg.max_iters * 4;
    cfg1.seed = cfg.seed;
    cfg1.dist = cfg.distance_mode;
    cfg1.dtype = USE_FP32;

    std::vector<float> h_centroids1((size_t)K1 * dim);
    init_centroids_by_sampling(cfg1, h_train_ptr, h_centroids1.data());

    float* d_centroids1 = nullptr;
    int* d_assign1 = nullptr;
    cudaMalloc(&d_centroids1, sizeof(float) * (size_t)K1 * dim);
    cudaMalloc(&d_assign1, sizeof(int) * (size_t)n_train);
    cudaMemcpy(d_centroids1, h_centroids1.data(), sizeof(float) * (size_t)K1 * dim, cudaMemcpyHostToDevice);

    float obj1 = 0.0f;
    gpu_kmeans_lloyd(cfg1, h_train_ptr, d_assign1, d_centroids1, &obj1);

    std::vector<int> h_assign1(n_train);
    cudaMemcpy(h_assign1.data(), d_assign1, sizeof(int) * (size_t)n_train, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids1.data(), d_centroids1, sizeof(float) * (size_t)K1 * dim, cudaMemcpyDeviceToHost);

    cudaFree(d_assign1);
    cudaFree(d_centroids1);

    // n'_i
    std::vector<int> n_prime(K1, 0);
    std::vector<std::vector<int>> cluster_index_in_train(K1);
    for (int i = 0; i < n_train; ++i) {
        int cid = h_assign1[i];
        if (cid < 0 || cid >= K1) {
            throw std::runtime_error("BalancedHierarchicalClustering: invalid cluster id");
        }
        n_prime[cid]++;
        cluster_index_in_train[cid].push_back(i);
    }

    //  k_i
    std::vector<float> final_centroids;
    final_centroids.reserve((size_t)M * dim * 2);

    for (int cid = 0; cid < K1; ++cid) {
        if (n_prime[cid] == 0) {
            continue;
        }

        const float* centroid_base = h_centroids1.data() + (size_t)cid * dim;
        if (n_prime[cid] <= s) {
            final_centroids.insert(final_centroids.end(), centroid_base, centroid_base + dim);
            continue;
        }

        int k_i = (int)std::ceil((double)n_prime[cid] * M / (double)n_train);
        k_i = std::max(1, k_i);
        if (k_i <= 1) {
            final_centroids.insert(final_centroids.end(), centroid_base, centroid_base + dim);
            continue;
        }

        //  cid
        std::vector<float> sub_data((size_t)n_prime[cid] * dim);
        for (int idx = 0; idx < n_prime[cid]; ++idx) {
            int ti = cluster_index_in_train[cid][idx];
            std::memcpy(sub_data.data() + (size_t)idx * dim,
                        h_train_ptr + (size_t)ti * dim,
                        sizeof(float) * dim);
        }

        //  KMeans
        KMeansCase cfg2;
        cfg2.n = n_prime[cid];
        cfg2.dim = dim;
        cfg2.k = k_i;
        cfg2.iters = cfg.max_iters;
        cfg2.minibatch_iters = cfg.max_iters * 4;
        cfg2.seed = cfg.seed + cid;
        cfg2.dist = cfg.distance_mode;
        cfg2.dtype = USE_FP32;

        std::vector<float> h_centroids2((size_t)k_i * dim);
        init_centroids_by_sampling(cfg2, sub_data.data(), h_centroids2.data());

        float* d_centroids2 = nullptr;
        int* d_assign2 = nullptr;
        cudaMalloc(&d_centroids2, sizeof(float) * (size_t)k_i * dim);
        cudaMalloc(&d_assign2, sizeof(int) * (size_t)cfg2.n);
        cudaMemcpy(d_centroids2, h_centroids2.data(), sizeof(float) * (size_t)k_i * dim, cudaMemcpyHostToDevice);

        gpu_kmeans_lloyd(cfg2, sub_data.data(), d_assign2, d_centroids2, nullptr);

        cudaMemcpy(h_centroids2.data(), d_centroids2, sizeof(float) * (size_t)k_i * dim, cudaMemcpyDeviceToHost);
        cudaFree(d_assign2);
        cudaFree(d_centroids2);

        final_centroids.insert(final_centroids.end(), h_centroids2.begin(), h_centroids2.end());
    }

    int final_k = (int)(final_centroids.size() / dim);
    if (final_k <= 0) {
        throw std::runtime_error("BalancedHierarchicalClustering: no final clusters generated");
    }

    //  n final_k
    float* d_centroids_final = nullptr;
    int* d_assign_global = nullptr;
    float* d_cnorm2 = nullptr;
    float* d_data_buffer = nullptr;
    StreamEnv stream_env;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;

    cudaMalloc(&d_centroids_final, sizeof(float) * (size_t)final_k * dim);
    cudaMalloc(&d_assign_global, sizeof(int) * (size_t)n);
    cudaMalloc(&d_cnorm2, sizeof(float) * (size_t)final_k);

    cudaMemcpy(d_centroids_final, final_centroids.data(), sizeof(float) * (size_t)final_k * dim, cudaMemcpyHostToDevice);

    int B = 1 << 15;
    int Ktile = 4096;
    stream_env.allocate(B, Ktile);
    cudaMalloc(&d_data_buffer, sizeof(float) * (size_t)B * dim);
    cudaStreamCreate(&stream);

    cublas_check(cublasCreate(&cublas_handle), "cublasCreate");
    cublas_check(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    KMeansCase cfg3;
    cfg3.n = n;
    cfg3.dim = dim;
    cfg3.k = final_k;
    cfg3.iters = 1;
    cfg3.minibatch_iters = 1;
    cfg3.seed = cfg.seed;
    cfg3.dist = cfg.distance_mode;
    cfg3.dtype = USE_FP32;

    perform_assignment_only(
        cfg3,
        h_data,
        d_assign_global,
        d_centroids_final,
        d_cnorm2,
        stream_env,
        d_data_buffer,
        stream,
        cublas_handle,
        B,
        Ktile,
        n,
        dim,
        final_k
    );

    cudaStreamSynchronize(stream);

    std::vector<int> h_assign_global(n);
    cudaMemcpy(h_assign_global.data(), d_assign_global, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);

    //  GPU
    stream_env.free();
    cudaFree(d_data_buffer);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas_handle);
    cudaFree(d_centroids_final);
    cudaFree(d_assign_global);
    cudaFree(d_cnorm2);

    //  cluster counts + offsets
    std::vector<int> final_counts(final_k, 0);
    for (int i = 0; i < n; ++i) {
        int cid = h_assign_global[i];
        if (cid < 0 || cid >= final_k) {
            throw std::runtime_error("BalancedHierarchicalClustering: invalid final assignment id");
        }
        final_counts[cid]++;
    }

    std::vector<long long> offsets(final_k, 0);
    long long accum = 0;
    for (int i = 0; i < final_k; ++i) {
        offsets[i] = accum;
        accum += final_counts[i];
    }

    //
    size_t data_size = (size_t)n * dim;
    h_reordered_data_ = (float*)std::malloc(data_size * sizeof(float));
    h_reordered_indices_ = (int*)std::malloc((size_t)n * sizeof(int));
    if (!h_reordered_data_ || !h_reordered_indices_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }

    std::vector<int> write_ptr(final_k);
    for (int i = 0; i < final_k; ++i) {
        write_ptr[i] = (int)offsets[i];
    }

    for (int i = 0; i < n; ++i) {
        int cid = h_assign_global[i];
        int dst_pos = write_ptr[cid]++;
        float* dst = h_reordered_data_ + (size_t)dst_pos * dim;
        const float* src = h_data + (size_t)i * dim;
        std::memcpy(dst, src, sizeof(float) * dim);
        h_reordered_indices_[dst_pos] = i;
    }

    //
    h_centroids_ = (float*)std::malloc((size_t)final_k * dim * sizeof(float));
    if (!h_centroids_) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, nullptr, nullptr);
        throw std::bad_alloc();
    }
    std::memcpy(h_centroids_, final_centroids.data(), sizeof(float) * (size_t)final_k * dim);

    cluster_info_.k = final_k;
    cluster_info_.offsets = (long long*)std::malloc((size_t)final_k * sizeof(long long));
    cluster_info_.counts = (int*)std::malloc((size_t)final_k * sizeof(int));
    if (!cluster_info_.offsets || !cluster_info_.counts) {
        free_result_buffers(h_reordered_data_, h_reordered_indices_, h_centroids_, &cluster_info_);
        throw std::bad_alloc();
    }
    std::memcpy(cluster_info_.offsets, offsets.data(), sizeof(long long) * (size_t)final_k);
    std::memcpy(cluster_info_.counts, final_counts.data(), sizeof(int) * (size_t)final_k);

    result_.centroids = h_centroids_;
    result_.assignments = nullptr;
    result_.cluster_sizes = cluster_info_.counts;
    result_.reordered_data = h_reordered_data_;
    result_.reordered_indices = h_reordered_indices_;
    result_.cluster_info = &cluster_info_;
    result_.n_vectors = n;
    result_.dim = dim;

    fitted_ = true;
}

ClusteringResult BalancedHierarchicalClustering::get_result() {
    return result_;
}
