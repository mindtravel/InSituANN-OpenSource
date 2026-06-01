#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "pch.h"
#include "dataset/dataset.cuh"

extern "C" {
/**
 * C ClusterDataset
 *
 * @return  ClusterDataset
 */
ClusterDataset* cluster_dataset_create(void) {
    try {
        ClusterDataset* dataset = new ClusterDataset();
        return dataset;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_create:  - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_create: \n");
        return nullptr;
    }
}

/**
 * C ClusterDataset
 *
 * @param dataset ClusterDataset
 */
void cluster_dataset_destroy(ClusterDataset* dataset) {
    if (!dataset) {
        return;
    }
    try {
        dataset->release();
        delete dataset;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_destroy:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "cluster_dataset_destroy: \n");
    }
}

/**
 * C K-means  ClusterDataset
 *
 * @param dataset_ptr ClusterDataset
 * @param h_data  [n_total_vectors, vector_dim]host
 * @param n_total_vectors
 * @param vector_dim
 * @param n_clusters cluster
 * @param h_objective K-meansnullptr
 * @param kmeans_iters K-means
 * @param use_minibatch minibatch
 * @param distance_mode 0=L2, 1=COSINE
 * @param seed
 * @param batch_size
 * @param device_id GPUID
 * @return 0-1
 */
int cluster_dataset_init_with_kmeans(
    ClusterDataset* dataset,
    float* h_data,
    int n_total_vectors,
    int vector_dim,
    int n_clusters,
    float* h_objective,
    int kmeans_iters,
    int use_minibatch,
    int distance_mode,
    unsigned int seed,
    int batch_size,
    int device_id
) {
    if (!dataset) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: dataset_ptr  NULL\n");
        return -1;
    }

    if (!h_data) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: h_data  NULL\n");
        return -1;
    }

    try {
        DistanceType dist_type = (distance_mode == 0) ? L2_DISTANCE : COSINE_DISTANCE;

        //  init_with_kmeans
        dataset->init_with_kmeans(
            n_total_vectors,
            vector_dim,
            n_clusters,
            h_objective,
            h_data,  //
            kmeans_iters,
            use_minibatch != 0,
            dist_type,
            seed,
            batch_size,
            device_id
        );

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans:  - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_init_with_kmeans: \n");
        return -1;
    }
}

/**
 * C ClusterDataset
 *
 * @param dataset ClusterDataset
 */
void cluster_dataset_release(ClusterDataset* dataset) {
    if (!dataset) {
        return;
    }
    try {
        dataset->release();
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_release:  - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "cluster_dataset_release: \n");
    }
}

/**
 * C ClusterDataset
 *
 * @param dataset ClusterDataset
 * @return 10
 */
int cluster_dataset_is_valid(ClusterDataset* dataset) {
    if (!dataset) {
        return 0;
    }
    try {
        return dataset->is_valid() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

/**
 * C ClusterDataset
 *
 * @param dataset ClusterDataset
 * @param reordered_data_out
 * @param reordered_indices_out
 * @param centroids_out
 * @param cluster_info_out cluster
 * @param n_total_vectors_out
 * @param vector_dim_out
 * @return 0-1
 */
int cluster_dataset_get_data(
    ClusterDataset* dataset,
    float** reordered_data_out,
    int** reordered_indices_out,
    float** centroids_out,
    long long** cluster_offsets_out,  //  long long*  ClusterInfo.offsets
    int** cluster_counts_out,
    int* n_clusters_out,
    int* n_total_vectors_out,
    int* vector_dim_out
) {
    if (!dataset) {
        fprintf(stderr, "cluster_dataset_get_data: dataset_ptr  NULL\n");
        return -1;
    }

    try {
        if (!dataset->is_valid()) {
            fprintf(stderr, "cluster_dataset_get_data: dataset \n");
            return -1;
        }

        if (reordered_data_out) {
            *reordered_data_out = dataset->reordered_data;
        }
        if (reordered_indices_out) {
            *reordered_indices_out = dataset->reordered_indices;
        }
        if (centroids_out) {
            *centroids_out = dataset->centroids;
        }
        if (cluster_offsets_out) {
            *cluster_offsets_out = dataset->cluster_info.offsets;
        }
        if (cluster_counts_out) {
            *cluster_counts_out = dataset->cluster_info.counts;
        }
        if (n_clusters_out) {
            *n_clusters_out = dataset->cluster_info.k;
        }
        if (n_total_vectors_out) {
            *n_total_vectors_out = dataset->n_total_vectors;
        }
        if (vector_dim_out) {
            *vector_dim_out = dataset->vector_dim;
        }

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "cluster_dataset_get_data:  - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "cluster_dataset_get_data: \n");
        return -1;
    }
}
}
