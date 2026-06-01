#ifndef FVECS_IO_H
#define FVECS_IO_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

/**
 * TEXMEX (.fvecs / .ivecs / .bvecs)
 *
 *
 *   int32  dim
 *   T[dim] data         (fvecs: float, ivecs: int32, bvecs: uint8)
 *
 *  dim
 */

namespace fvecs_io {

/**
 *  fvecs  float
 *
 * @param path
 * @param out_n
 * @param out_dim
 * @return  free()  float* n*dim  floatrow-major
 */
inline float* read_fvecs(const std::string& path, int* out_n, int* out_dim) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fvecs_io: cannot open " + path);

    std::fseek(f, 0, SEEK_END);
    long file_sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    int32_t dim = 0;
    if (std::fread(&dim, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: empty file " + path);
    }
    if (dim <= 0 || dim > 4096) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: invalid dim " + std::to_string(dim));
    }

    long record_sz = (long)sizeof(int32_t) + (long)dim * (long)sizeof(float);
    if (file_sz % record_sz != 0) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: file size not multiple of record size");
    }
    long n = file_sz / record_sz;

    float* data = (float*)std::malloc((size_t)n * (size_t)dim * sizeof(float));
    if (!data) {
        std::fclose(f);
        throw std::bad_alloc();
    }

    std::fseek(f, 0, SEEK_SET);
    for (long i = 0; i < n; ++i) {
        int32_t d = 0;
        if (std::fread(&d, sizeof(int32_t), 1, f) != 1 || d != dim) {
            std::free(data);
            std::fclose(f);
            throw std::runtime_error("fvecs_io: dim mismatch at row " + std::to_string(i));
        }
        if (std::fread(data + (size_t)i * (size_t)dim, sizeof(float), (size_t)dim, f)
            != (size_t)dim) {
            std::free(data);
            std::fclose(f);
            throw std::runtime_error("fvecs_io: unexpected EOF at row " + std::to_string(i));
        }
    }
    std::fclose(f);

    *out_n = (int)n;
    *out_dim = (int)dim;
    return data;
}

/**
 *  ivecsint32 fvecs
 */
inline int32_t* read_ivecs(const std::string& path, int* out_n, int* out_dim) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fvecs_io: cannot open " + path);

    std::fseek(f, 0, SEEK_END);
    long file_sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    int32_t dim = 0;
    if (std::fread(&dim, sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: empty file " + path);
    }
    if (dim <= 0 || dim > 4096) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: invalid dim " + std::to_string(dim));
    }

    long record_sz = (long)sizeof(int32_t) + (long)dim * (long)sizeof(int32_t);
    if (file_sz % record_sz != 0) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: file size not multiple of record size");
    }
    long n = file_sz / record_sz;

    int32_t* data = (int32_t*)std::malloc((size_t)n * (size_t)dim * sizeof(int32_t));
    if (!data) {
        std::fclose(f);
        throw std::bad_alloc();
    }

    std::fseek(f, 0, SEEK_SET);
    for (long i = 0; i < n; ++i) {
        int32_t d = 0;
        if (std::fread(&d, sizeof(int32_t), 1, f) != 1 || d != dim) {
            std::free(data);
            std::fclose(f);
            throw std::runtime_error("fvecs_io: dim mismatch at row " + std::to_string(i));
        }
        if (std::fread(data + (size_t)i * (size_t)dim, sizeof(int32_t), (size_t)dim, f)
            != (size_t)dim) {
            std::free(data);
            std::fclose(f);
            throw std::runtime_error("fvecs_io: unexpected EOF at row " + std::to_string(i));
        }
    }
    std::fclose(f);

    *out_n = (int)n;
    *out_dim = (int)dim;
    return data;
}

/**
 * Recall@k querygpu_idx  k  gt  k
 *
 * @param gpu_idx    [n_query * k_gpu_row] GPU  top-k
 * @param gt_idx     [n_query * k_gt_row]  groundtruth
 * @param n_query
 * @param k           top-k k  k_gpu_row  k  k_gt_row
 * @param k_gpu_row  gpu_idx  topk
 * @param k_gt_row   gt_idx  SIFT-1M  gt  100
 * @return recall@k (0.0 ~ 1.0)
 */
inline double recall_at_k(const int* gpu_idx, const int32_t* gt_idx,
                          int n_query, int k, int k_gpu_row, int k_gt_row) {
    if (k > k_gpu_row) k = k_gpu_row;
    if (k > k_gt_row)  k = k_gt_row;
    long long hit = 0;
    long long total = (long long)n_query * (long long)k;
    for (int q = 0; q < n_query; ++q) {
        const int*     gpu_row = gpu_idx + (size_t)q * (size_t)k_gpu_row;
        const int32_t* gt_row  = gt_idx  + (size_t)q * (size_t)k_gt_row;
        for (int i = 0; i < k; ++i) {
            int cand = gpu_row[i];
            for (int j = 0; j < k; ++j) {
                if (cand == (int)gt_row[j]) { ++hit; break; }
            }
        }
    }
    return total > 0 ? (double)hit / (double)total : 0.0;
}

/**
 *  DiskANN  uint8 .u8bin / .bin fp32
 *
 * DiskANN  header
 *   int32  npts
 *   int32  dim
 *   uint8  data[npts * dim]
 *
 * HuggingFace Nanvivi/SIFT1B-DiskANN  base.bin  header  1B
 *  HTTP Range 10M / 100M
 *   -  max_n > 0  max_n < header_npts max_n
 *   -  header_npts  npts
 */
inline float* read_diskann_u8bin(const std::string& path, int* out_n, int* out_dim,
                                 long long max_n = -1) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fvecs_io: cannot open " + path);

    std::fseek(f, 0, SEEK_END);
    long long file_sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    int32_t hdr_npts = 0, hdr_dim = 0;
    if (std::fread(&hdr_npts, sizeof(int32_t), 1, f) != 1 ||
        std::fread(&hdr_dim,  sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann header read failed");
    }
    if (hdr_dim <= 0 || hdr_dim > 4096) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: invalid diskann dim " + std::to_string(hdr_dim));
    }

    long long avail_bytes = file_sz - (long long)sizeof(int32_t) * 2;
    long long avail_npts  = avail_bytes / (long long)hdr_dim; /* uint8 per elem */
    long long npts        = (long long)hdr_npts;
    if (npts > avail_npts) npts = avail_npts;
    if (max_n > 0 && npts > max_n) npts = max_n;

    size_t data_cnt = (size_t)npts * (size_t)hdr_dim;
    float* data = (float*)std::malloc(data_cnt * sizeof(float));
    if (!data) {
        std::fclose(f);
        throw std::bad_alloc();
    }

    /*  1 MB  fp32 */
    const size_t CHUNK_BYTES = 1 << 20;
    uint8_t* buf = (uint8_t*)std::malloc(CHUNK_BYTES);
    if (!buf) {
        std::free(data);
        std::fclose(f);
        throw std::bad_alloc();
    }

    size_t elem_done = 0;
    size_t elem_total = data_cnt;
    while (elem_done < elem_total) {
        size_t need = elem_total - elem_done;
        if (need > CHUNK_BYTES) need = CHUNK_BYTES;
        size_t got = std::fread(buf, sizeof(uint8_t), need, f);
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) data[elem_done + i] = (float)buf[i];
        elem_done += got;
    }
    std::free(buf);
    std::fclose(f);

    if (elem_done != elem_total) {
        std::free(data);
        throw std::runtime_error("fvecs_io: diskann u8bin short read");
    }

    *out_n = (int)npts;
    *out_dim = (int)hdr_dim;
    return data;
}

/**
 *  DiskANN fbin[int32 npts][int32 dim][float32 data]
 * DEEP  fp32  u8
 */
inline float* read_diskann_fbin(const std::string& path, int* out_n, int* out_dim,
                                long long max_n = -1) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fvecs_io: cannot open " + path);

    std::fseek(f, 0, SEEK_END);
    long long file_sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    int32_t hdr_npts = 0, hdr_dim = 0;
    if (std::fread(&hdr_npts, sizeof(int32_t), 1, f) != 1 ||
        std::fread(&hdr_dim,  sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: fbin header read failed");
    }
    if (hdr_dim <= 0 || hdr_dim > 4096) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: invalid fbin dim " + std::to_string(hdr_dim));
    }

    long long avail_bytes = file_sz - (long long)sizeof(int32_t) * 2;
    long long avail_npts  = avail_bytes / ((long long)hdr_dim * (long long)sizeof(float));
    long long npts        = (long long)hdr_npts;
    if (npts > avail_npts) npts = avail_npts;
    if (max_n > 0 && npts > max_n) npts = max_n;

    size_t data_cnt = (size_t)npts * (size_t)hdr_dim;
    float* data = (float*)std::malloc(data_cnt * sizeof(float));
    if (!data) {
        std::fclose(f);
        throw std::bad_alloc();
    }
    if (std::fread(data, sizeof(float), data_cnt, f) != data_cnt) {
        std::free(data);
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann fbin short read");
    }
    std::fclose(f);

    *out_n = (int)npts;
    *out_dim = (int)hdr_dim;
    return data;
}

/**
 *  DiskANN  groundtruthHuggingFace Nanvivi  gt.bin
 *   int32 npts
 *   int32 k
 *   int32 idx[npts * k]
 *   float dist[npts * k]
 *  idx malloc  [npts * k] int32
 */
inline int32_t* read_diskann_gt(const std::string& path, int* out_n, int* out_k) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("fvecs_io: cannot open " + path);

    std::fseek(f, 0, SEEK_END);
    long long file_sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);

    int32_t npts = 0, k = 0;
    if (std::fread(&npts, sizeof(int32_t), 1, f) != 1 ||
        std::fread(&k,    sizeof(int32_t), 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann gt header read failed");
    }
    if (npts <= 0 || k <= 0 || npts > 10000000 || k > 10000) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann gt bad shape " +
                                 std::to_string(npts) + "x" + std::to_string(k));
    }

    long long header_bytes = (long long)sizeof(int32_t) * 2;
    long long idx_bytes = (long long)npts * k * (long long)sizeof(int32_t);
    if (file_sz < header_bytes + idx_bytes) {
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann gt file too small");
    }

    int32_t* idx = (int32_t*)std::malloc((size_t)npts * k * sizeof(int32_t));
    if (!idx) {
        std::fclose(f);
        throw std::bad_alloc();
    }
    if (std::fread(idx, sizeof(int32_t), (size_t)npts * k, f) != (size_t)npts * k) {
        std::free(idx);
        std::fclose(f);
        throw std::runtime_error("fvecs_io: diskann gt short read");
    }
    std::fclose(f);
    *out_n = npts;
    *out_k = k;
    return idx;
}

} // namespace fvecs_io

#endif // FVECS_IO_H
