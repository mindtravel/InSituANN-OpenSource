#ifndef IVFTENSOR_IVF_BCS_CUH
#define IVFTENSOR_IVF_BCS_CUH

/**
 * BCS BCS
 *  lookup  lookup  block
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * BCS
 */
int ivf_compute_bcs(
    const int* h_cluster_sizes,
    int n_cluster,
    float std_var_ratio,
    int* out_n_balanced
);

/**
 *  BCS  block  clusterblock
 *  block  lookup
 *
 * @param h_cluster_vectors      [n_total_vectors, n_dim] cluster
 * @param h_cluster_sizes        cluster  [n_cluster]
 * @param h_cluster_offsets      cluster  h_cluster_vectors  [n_cluster+1] nullptr sizes
 * @param h_centroids            [n_cluster, n_dim]
 * @param h_reordered_indices   reordered  [n_total_vectors] nullptr
 * @param n_cluster             cluster
 * @param n_dim
 * @param bcs
 * @param out_balanced_vectors   (n_total_vectors * n_dim) float
 * @param out_balanced_sizes     block  [n_balanced]
 * @param out_balanced_centers   block  cluster [n_balanced, n_dim]
 * @param out_reordered_indices  reordered  [n_total_vectors] nullptr
 * @param out_cluster_to_block_offset clusterblock  [n_cluster+1]block_ids for cluster c = [offset[c], offset[c+1])
 */
void ivf_rebalance_clusters(
    const float* h_cluster_vectors,
    const int* h_cluster_sizes,
    const long long* h_cluster_offsets,
    const float* h_centroids,
    const int* h_reordered_indices,
    int n_cluster,
    int n_dim,
    int bcs,
    float* out_balanced_vectors,
    int* out_balanced_sizes,
    float* out_balanced_centers,
    int* out_reordered_indices,
    int* out_cluster_to_block_offset
);

/**
 *  block
 *  ivf_rebalance_clusters  lookup  base_offset
 *
 * @return 0  0  stderr
 */
int ivf_validate_block_partitioning(
    const int* balanced_sizes,
    const int* cluster_to_block_offset,
    const int* cluster_sizes,
    int n_cluster,
    int n_balanced,
    int n_total_vectors,
    int bcs
);

#ifdef __cplusplus
}
#endif

#endif /* IVFTENSOR_IVF_BCS_CUH */
