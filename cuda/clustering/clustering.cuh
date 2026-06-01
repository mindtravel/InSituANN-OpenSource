#ifndef IVF_CLUSTERING_CUH
#define IVF_CLUSTERING_CUH

#include "pch.h"
#include <cstddef>

/**
 *
 *  KMeans  HNSW/
 */

// ============================================================
//  IVF  ClusterInfo
// ============================================================
struct ClusterInfo {
    long long* offsets;   // [k]  cluster
    int* counts;          // [k]  cluster
    int k;                // cluster
};

// ============================================================
//
// ============================================================
struct ClusteringConfig {
    int n_clusters;
    int max_iters;
    DistanceType distance_mode;
    unsigned int seed;
    bool use_minibatch;
    int batch_size;
    int device_id;
};

// ============================================================
// fit  get_result()
//
// ============================================================
struct ClusteringResult {
    float* centroids;           // [k, dim]
    int* assignments;            // [n]  cluster
    int* cluster_sizes;          // [k] ClusterInfo.counts
    float* reordered_data;       // [n, dim]  cluster IVF
    int* reordered_indices;      // [n] IVF
    ClusterInfo* cluster_info;   // offsets / counts / k
    int n_vectors;
    int dim;
};

// ============================================================
//
// ============================================================
class ClusteringInterface {
public:
    virtual void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) = 0;
    virtual ClusteringResult get_result() = 0;
    virtual ~ClusteringInterface() = default;
};

/**
 * KMeans  ivf_kmeans ClusteringResult
 */
class KMeansClustering : public ClusteringInterface {
public:
    KMeansClustering() = default;
    void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;
    ~KMeansClustering() override;

private:
    ClusteringResult result_{};
    ClusterInfo cluster_info_{};
    float* h_centroids_ = nullptr;
    bool fitted_ = false;
};

#endif // IVF_CLUSTERING_CUH
