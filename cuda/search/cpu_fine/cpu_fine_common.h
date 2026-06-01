#ifndef IVFTENSOR_CPU_FINE_COMMON_H
#define IVFTENSOR_CPU_FINE_COMMON_H

/**
 * CPU fine search
 *   - TopK /
 *   - OpenMP + top-k
 *
 * kernel  distance kernel / SIMD / tiled / AoSoA
 *  run_cpu_fine_host<>
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "cpu_fine.h"

namespace ivftensor {
namespace cpu_fine {

/** Max-heap-of-topk K  (dist, idx)
 *  push  O(log K) K=10
 */
struct TopKHeap {
    int k;
    int size = 0;
    float* dists;   /* heap root =  K  */
    int*   idxs;

    void init(int k_, float* dist_buf, int* idx_buf) {
        k = k_;
        size = 0;
        dists = dist_buf;
        idxs = idx_buf;
    }

    static inline void sift_down(float* d, int* i, int n, int pos) {
        while (true) {
            int l = 2 * pos + 1;
            int r = 2 * pos + 2;
            int largest = pos;
            if (l < n && d[l] > d[largest]) largest = l;
            if (r < n && d[r] > d[largest]) largest = r;
            if (largest == pos) break;
            std::swap(d[pos], d[largest]);
            std::swap(i[pos], i[largest]);
            pos = largest;
        }
    }

    static inline void sift_up(float* d, int* i, int pos) {
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (d[parent] >= d[pos]) break;
            std::swap(d[pos], d[parent]);
            std::swap(i[pos], i[parent]);
            pos = parent;
        }
    }

    inline void push(float dist, int idx) {
        if (size < k) {
            dists[size] = dist;
            idxs[size] = idx;
            ++size;
            sift_up(dists, idxs, size - 1);
        } else if (dist < dists[0]) {
            dists[0] = dist;
            idxs[0] = idx;
            sift_down(dists, idxs, size, 0);
        }
    }

    inline float worst() const {
        return size > 0 ? dists[0] : std::numeric_limits<float>::infinity();
    }

    /**  dist heap
     *   topk   std::sort  */
    inline void sort_ascending() {
        for (int i = 1; i < size; ++i) {
            float d = dists[i];
            int   x = idxs[i];
            int j = i - 1;
            while (j >= 0 && dists[j] > d) {
                dists[j + 1] = dists[j];
                idxs[j + 1]  = idxs[j];
                --j;
            }
            dists[j + 1] = d;
            idxs[j + 1]  = x;
        }
    }
};

/**  heap  k  (+inf, -1)  */
inline void finalize_topk(TopKHeap& h, float* out_dist, int* out_idx, int k) {
    h.sort_ascending();
    for (int j = h.size; j < k; ++j) {
        h.dists[j] = std::numeric_limits<float>::infinity();
        h.idxs[j] = -1;
    }
    if (out_dist != h.dists) std::memcpy(out_dist, h.dists, k * sizeof(float));
    if (out_idx  != h.idxs)  std::memcpy(out_idx,  h.idxs,  k * sizeof(int));
}

}  // namespace cpu_fine
}  // namespace ivftensor

#endif  /* IVFTENSOR_CPU_FINE_COMMON_H */
