/**
 * BCS BCS  CPU CUDA
 */

#include "layout/ivf_bcs.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

float std_dev(const std::vector<int>& vec) {
    const size_t n = vec.size();
    if (n <= 1u) return 0.f;
    float sum = static_cast<float>(std::accumulate(vec.begin(), vec.end(), 0));
    float mean = sum / static_cast<float>(n);
    float accum = 0.f;
    for (int v : vec) {
        float d = static_cast<float>(v) - mean;
        accum += d * d;
    }
    return std::sqrt(accum / static_cast<float>(n - 1));
}

}  // namespace

extern "C" {

int ivf_compute_bcs(
    const int* h_cluster_sizes,
    int n_cluster,
    float std_var_ratio,
    int* out_n_balanced
) {
    if (!h_cluster_sizes || n_cluster <= 0) return 256;

    long long total = 0;
    for (int i = 0; i < n_cluster; ++i)
        total += h_cluster_sizes[i];
    int ave = static_cast<int>(total / n_cluster);

    int basenum = (total > 100000000) ? (256 * 64) : 256;
    int sta = (ave / basenum) * basenum;
    if (sta < basenum) sta = basenum;

    std::vector<int> tmp;
    bool re = false;

    while (sta >= basenum) {
        tmp.clear();
        for (int i = 0; i < n_cluster; ++i) {
            int cnum = h_cluster_sizes[i];
            while (cnum >= sta) {
                tmp.push_back(sta);
                cnum -= sta;
            }
            if (cnum > 0) tmp.push_back(cnum);
        }
        float sd = std_dev(tmp);
        float ratio = (sta > 0) ? (sd / static_cast<float>(sta)) : 0.f;
        if (ratio <= std_var_ratio || static_cast<int>(tmp.size()) > n_cluster * 11) {
            re = true;
            break;
        }
        sta -= basenum;
    }

    int bcs = re ? sta : (sta + basenum);

    if (out_n_balanced) {
        int n_bal = 0;
        for (int i = 0; i < n_cluster; ++i) {
            int cnum = h_cluster_sizes[i];
            n_bal += (cnum + bcs - 1) / bcs;
        }
        *out_n_balanced = n_bal;
    }

    return bcs;
}

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
) {
    if (!h_cluster_vectors || !h_cluster_sizes || !h_centroids || !out_balanced_vectors ||
        !out_balanced_sizes || !out_balanced_centers || !out_cluster_to_block_offset) {
        return;
    }

    std::vector<long long> offsets(n_cluster + 1);
    if (h_cluster_offsets) {
        for (int i = 0; i <= n_cluster; ++i) offsets[i] = h_cluster_offsets[i];
    } else {
        offsets[0] = 0;
        for (int i = 0; i < n_cluster; ++i)
            offsets[i + 1] = offsets[i] + h_cluster_sizes[i];
    }

    int block_id = 0;
    out_cluster_to_block_offset[0] = 0;
    long long vec_offset = 0;

    for (int c = 0; c < n_cluster; ++c) {
        int remaining = h_cluster_sizes[c];
        long long src_start = offsets[c];
        const float* centroid = h_centroids + (size_t)c * n_dim;

        while (remaining > 0) {
            int take = std::min(bcs, remaining);
            out_balanced_sizes[block_id] = take;
            std::memcpy(out_balanced_centers + (size_t)block_id * n_dim, centroid, (size_t)n_dim * sizeof(float));

            const float* src = h_cluster_vectors + src_start * n_dim;
            float* dst = out_balanced_vectors + vec_offset * n_dim;
            std::memcpy(dst, src, (size_t)take * n_dim * sizeof(float));

            if (out_reordered_indices) {
                if (h_reordered_indices) {
                    for (int i = 0; i < take; ++i)
                        out_reordered_indices[vec_offset + i] = h_reordered_indices[src_start + i];
                } else {
                    for (int i = 0; i < take; ++i)
                        out_reordered_indices[vec_offset + i] = static_cast<int>(vec_offset + i);
                }
            }

            vec_offset += take;
            src_start += take;
            remaining -= take;
            block_id++;
        }
        out_cluster_to_block_offset[c + 1] = block_id;
    }
}

int ivf_validate_block_partitioning(
    const int* balanced_sizes,
    const int* cluster_to_block_offset,
    const int* cluster_sizes,
    int n_cluster,
    int n_balanced,
    int n_total_vectors,
    int bcs
) {
    if (!balanced_sizes || !cluster_to_block_offset) {
        fprintf(stderr, "[IVF BCS validate] null pointer\n");
        return -1;
    }
    if (n_cluster <= 0 || n_balanced <= 0 || n_total_vectors <= 0 || bcs <= 0) {
        fprintf(stderr, "[IVF BCS validate] invalid dimensions n_cluster=%d n_balanced=%d n_total=%d bcs=%d\n",
                n_cluster, n_balanced, n_total_vectors, bcs);
        return -2;
    }

    /* 1. cluster_to_block_offset  */
    if (cluster_to_block_offset[0] != 0) {
        fprintf(stderr, "[IVF BCS validate] cluster_to_block_offset[0]=%d != 0\n", cluster_to_block_offset[0]);
        return 1;
    }
    if (cluster_to_block_offset[n_cluster] != n_balanced) {
        fprintf(stderr, "[IVF BCS validate] cluster_to_block_offset[n_cluster]=%d != n_balanced=%d\n",
                cluster_to_block_offset[n_cluster], n_balanced);
        return 2;
    }

    /* 2. cluster_to_block_offset  */
    for (int c = 0; c < n_cluster; ++c) {
        if (cluster_to_block_offset[c + 1] < cluster_to_block_offset[c]) {
            fprintf(stderr, "[IVF BCS validate] cluster_to_block_offset not monotonic at c=%d\n", c);
            return 3;
        }
    }

    /* 3.  block  > 0  <= bcs */
    for (int b = 0; b < n_balanced; ++b) {
        if (balanced_sizes[b] <= 0) {
            fprintf(stderr, "[IVF BCS validate] block %d size=%d <= 0\n", b, balanced_sizes[b]);
            return 4;
        }
        if (balanced_sizes[b] > bcs) {
            fprintf(stderr, "[IVF BCS validate] block %d size=%d > bcs=%d\n", b, balanced_sizes[b], bcs);
            return 5;
        }
    }

    /* 4. sum(balanced_sizes) == n_total_vectors */
    long long sum_sizes = 0;
    for (int b = 0; b < n_balanced; ++b)
        sum_sizes += balanced_sizes[b];
    if (sum_sizes != n_total_vectors) {
        fprintf(stderr, "[IVF BCS validate] sum(balanced_sizes)=%lld != n_total_vectors=%d\n",
                (long long)sum_sizes, n_total_vectors);
        return 6;
    }

    /* 5.  cluster  block  == cluster_sizes[c] */
    if (cluster_sizes) {
        for (int c = 0; c < n_cluster; ++c) {
            int b_start = cluster_to_block_offset[c];
            int b_end = cluster_to_block_offset[c + 1];
            int cluster_sum = 0;
            for (int b = b_start; b < b_end; ++b)
                cluster_sum += balanced_sizes[b];
            if (cluster_sum != cluster_sizes[c]) {
                fprintf(stderr, "[IVF BCS validate] cluster %d: sum(block_sizes)=%d != cluster_sizes=%d\n",
                        c, cluster_sum, cluster_sizes[c]);
                return 7;
            }
        }
    }

    /* 6. h_offsets  reordered index  block */
    std::vector<long long> h_offsets(static_cast<size_t>(n_balanced) + 1);
    h_offsets[0] = 0;
    for (int b = 0; b < n_balanced; ++b)
        h_offsets[b + 1] = h_offsets[b] + balanced_sizes[b];
    if (h_offsets[n_balanced] != n_total_vectors) {
        fprintf(stderr, "[IVF BCS validate] h_offsets[n_balanced]=%lld != n_total_vectors=%d\n",
                (long long)h_offsets[n_balanced], n_total_vectors);
        return 8;
    }

    /* 7.  reordered index  block */
    std::vector<int> block_of(static_cast<size_t>(n_total_vectors), -1);
    for (int b = 0; b < n_balanced; ++b) {
        long long start = h_offsets[b];
        long long end = start + balanced_sizes[b];
        for (long long r = start; r < end; ++r) {
            if (block_of[r] >= 0) {
                fprintf(stderr, "[IVF BCS validate] reordered index %lld: duplicate in block %d and %d\n",
                        (long long)r, block_of[r], b);
                return 9;
            }
            block_of[r] = b;
        }
    }
    for (int r = 0; r < n_total_vectors; ++r) {
        if (block_of[r] < 0) {
            fprintf(stderr, "[IVF BCS validate] reordered index %d: not assigned to any block\n", r);
            return 10;
        }
    }

    return 0;
}

}  // extern "C"
