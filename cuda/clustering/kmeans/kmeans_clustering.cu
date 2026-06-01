/**
 * KMeansClustering ClusteringInterface ivf_kmeans
 */
#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

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

} // namespace

void KMeansClustering::fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) {
    if (fitted_) {
        free_result_buffers(
            result_.reordered_data,
            result_.reordered_indices,
            h_centroids_,
            &cluster_info_
        );
        result_ = ClusteringResult{};
        cluster_info_ = ClusterInfo{};
        h_centroids_ = nullptr;
        fitted_ = false;
    }

    const int k = cfg.n_clusters;
    const size_t data_size = (size_t)n * dim;
    const size_t centroid_size = (size_t)k * dim;

    float* reordered_data = (float*)std::malloc(data_size * sizeof(float));
    int* reordered_indices = (int*)std::malloc((size_t)n * sizeof(int));
    float* h_centroids = (float*)std::malloc(centroid_size * sizeof(float));
    if (!reordered_data || !reordered_indices || !h_centroids) {
        free_result_buffers(reordered_data, reordered_indices, h_centroids, nullptr);
        throw std::bad_alloc();
    }

    int* original_indices = (int*)std::malloc((size_t)n * sizeof(int));
    if (!original_indices) {
        free_result_buffers(reordered_data, reordered_indices, h_centroids, nullptr);
        throw std::bad_alloc();
    }
    for (int i = 0; i < n; ++i) original_indices[i] = i;

    KMeansCase kmeans_cfg;
    kmeans_cfg.n = n;
    kmeans_cfg.dim = dim;
    kmeans_cfg.k = k;
    kmeans_cfg.iters = cfg.max_iters;
    kmeans_cfg.minibatch_iters = cfg.max_iters * 4;
    kmeans_cfg.seed = cfg.seed;
    kmeans_cfg.dist = cfg.distance_mode;
    kmeans_cfg.dtype = USE_FP32;

    init_centroids_by_sampling(kmeans_cfg, h_data, h_centroids);

    float* d_centroids = nullptr;
    cudaError_t err = cudaMalloc(&d_centroids, centroid_size * sizeof(float));
    if (err != cudaSuccess) {
        std::free(original_indices);
        free_result_buffers(reordered_data, reordered_indices, h_centroids, nullptr);
        throw std::runtime_error("cudaMalloc centroids failed");
    }
    cudaMemcpy(d_centroids, h_centroids, centroid_size * sizeof(float), cudaMemcpyHostToDevice);

    cluster_info_.offsets = nullptr;
    cluster_info_.counts = nullptr;
    cluster_info_.k = 0;

    bool ok = ivf_kmeans(
        kmeans_cfg,
        h_data,
        reordered_data,
        d_centroids,
        &cluster_info_,
        cfg.use_minibatch,
        cfg.device_id,
        cfg.batch_size,
        nullptr,
        original_indices,
        reordered_indices
    );
    std::free(original_indices);
    if (!ok) {
        cudaFree(d_centroids);
        free_result_buffers(reordered_data, reordered_indices, h_centroids, &cluster_info_);
        throw std::runtime_error("ivf_kmeans failed");
    }

    cudaMemcpy(h_centroids, d_centroids, centroid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_centroids);

    h_centroids_ = h_centroids;

    result_.centroids = h_centroids_;
    result_.assignments = nullptr;
    result_.cluster_sizes = cluster_info_.counts;
    result_.reordered_data = reordered_data;
    result_.reordered_indices = reordered_indices;
    result_.cluster_info = &cluster_info_;
    result_.n_vectors = n;
    result_.dim = dim;
    fitted_ = true;
}

ClusteringResult KMeansClustering::get_result() {
    return result_;
}

KMeansClustering::~KMeansClustering() {
    if (!fitted_) return;
    free_result_buffers(
        result_.reordered_data,
        result_.reordered_indices,
        h_centroids_,
        &cluster_info_
    );
}
