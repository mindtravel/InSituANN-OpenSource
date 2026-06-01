#ifndef IVF_HIERARCHICAL_CUH
#define IVF_HIERARCHICAL_CUH

#include "clustering/clustering.cuh"
#include "clustering/kmeans/kmeans.cuh"

/**
 * Balanced Hierarchical Clustering ()
 *
 *  design.md
 * 1.  n_train = alpha * n
 * 2.  KMeansK1 = sqrt(M)
 * 3.  n'_i > s
 * 4.  M'
 */
class BalancedHierarchicalClustering : public ClusteringInterface {
public:
    explicit BalancedHierarchicalClustering(float alpha = 0.1f);
    ~BalancedHierarchicalClustering() override;

    void fit(const float* h_data, int n, int dim, const ClusteringConfig& cfg) override;
    ClusteringResult get_result() override;

private:
    void release();

    float alpha_;
    bool fitted_;

    ClusteringResult result_;
    ClusterInfo cluster_info_;

    float* h_centroids_;
    float* h_reordered_data_;
    int* h_reordered_indices_;
};

#endif // IVF_HIERARCHICAL_CUH
