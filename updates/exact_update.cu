#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>

#include "dataset/dataset.cuh"
#include "l2norm/l2norm.cuh"
#include "search/coarse/fusion_dist_topk.cuh"
#include "search/cpu_fine/cpu_fine.h"
#include "search/cpu_fine/cpu_fine_u8.h"
#include "io.h"

namespace ivftensor {
namespace cpu_fine {
long long cpu_fine_kernel_v3_u8_touched_masked(
    const uint8_t* h_base_u8,
    const long long* h_cluster_offsets,
    const int* h_cluster_counts,
    const uint8_t* h_query_u8,
    const int* h_coarse_cluster_ids,
    int n_query,
    int dim,
    int n_total_clusters,
    int n_probes,
    int topk,
    int num_threads,
    const uint8_t* h_deleted_local,
    int* h_topk_local_idx,
    float* h_topk_dist) {
    if (num_threads <= 0) num_threads = omp_get_max_threads();

    long long total_fma = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads) reduction(+:total_fma)
    for (int qi = 0; qi < n_query; ++qi) {
        std::vector<float> best_d((size_t)topk, std::numeric_limits<float>::infinity());
        std::vector<int> best_i((size_t)topk, -1);

        auto push = [&](float dist, int local_idx) {
            int worst = 0;
            float worst_d = best_d[0];
            for (int k = 1; k < topk; ++k) {
                if (best_d[(size_t)k] > worst_d) {
                    worst_d = best_d[(size_t)k];
                    worst = k;
                }
            }
            if (dist < worst_d ||
                (dist == worst_d && (best_i[(size_t)worst] < 0 || local_idx < best_i[(size_t)worst]))) {
                best_d[(size_t)worst] = dist;
                best_i[(size_t)worst] = local_idx;
            }
        };

        const uint8_t* q = h_query_u8 + (size_t)qi * dim;
        for (int pi = 0; pi < n_probes; ++pi) {
            int cid = h_coarse_cluster_ids[(size_t)qi * n_probes + pi];
            if (cid < 0 || cid >= n_total_clusters) continue;
            long long off = h_cluster_offsets[cid];
            int count = h_cluster_counts[cid];
            for (int vi = 0; vi < count; ++vi) {
                long long local_ll = off + vi;
                if (h_deleted_local && h_deleted_local[(size_t)local_ll]) continue;
                const uint8_t* v = h_base_u8 + (size_t)local_ll * dim;
                int acc = 0;
                #pragma omp simd reduction(+:acc)
                for (int d = 0; d < dim; ++d) {
                    int diff = (int)q[d] - (int)v[d];
                    acc += diff * diff;
                }
                push((float)acc, (int)local_ll);
            }
            total_fma += (long long)count * dim;
        }

        std::vector<int> order((size_t)topk);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            float da = best_d[(size_t)a], db = best_d[(size_t)b];
            int ia = best_i[(size_t)a], ib = best_i[(size_t)b];
            return da < db || (da == db && ia < ib);
        });
        for (int k = 0; k < topk; ++k) {
            int src = order[(size_t)k];
            h_topk_dist[(size_t)qi * topk + k] = best_d[(size_t)src];
            h_topk_local_idx[(size_t)qi * topk + k] = best_i[(size_t)src];
        }
    }
    return total_fma;
}
}
}

namespace {

constexpr int L2_DISTANCE_MODE = 0;

double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

bool cpu_supports_avx512f() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

bool cpu_supports_avx512bw() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

bool cpu_supports_avx2() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

struct Args {
    std::string dataset = "SIFT1B";
    std::string data_dir = "/workspace/sift1b";
    std::string scale = "1b";
    std::string centroids;
    std::string assign;
    std::string reorder_cache_dir;
    std::string out = "update_exact_real_runner.csv";
    int nlist = 524288;
    int nprobe = 128;
    int topk = 10;
    int main_overfetch = 10;
    int delta_topk = 16;
    int batch_size = 8;
    int repeats = 2;
    int threads = 64;
    int nq_limit = 0;
    int delta_n = 0;
    int cuda_device = 0;
    double delete_ratio = 0.0;
    bool use_fbin = false;
    bool delta_sample_main = false;
    bool include_mixed_update = false;
    bool mixed_workload = false;
    bool gt_safe_updates = false;
    bool delete_gt_safe = false;
    bool insert_aware_recall = false;
    std::set<std::pair<int,int>> update_pairs;
    bool gt_only = false;
    bool pipeline = true;
    int update_steps = 4;
    int query_rounds = 1;
    uint64_t seed = 20260527ULL;
    std::string delta_mode = "far";
    std::string delta_search_mode = "flat-gpu-fp16";
    std::vector<int> batch_sizes;
    std::vector<int> delta_ns;
    std::vector<double> delete_ratios;
};

std::string need_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    return out;
}

std::set<std::pair<int,int>> parse_pair_count_list(const std::string& s) {
    std::set<std::pair<int,int>> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        size_t sep = item.find(':');
        if (sep == std::string::npos) sep = item.find('-');
        if (sep == std::string::npos) throw std::runtime_error(item);
        int a = std::stoi(item.substr(0, sep));
        int b = std::stoi(item.substr(sep + 1));
        out.insert({a, b});
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--dataset") a.dataset = need_value(i, argc, argv);
        else if (k == "--data-dir") a.data_dir = need_value(i, argc, argv);
        else if (k == "--scale") a.scale = need_value(i, argc, argv);
        else if (k == "--centroids") a.centroids = need_value(i, argc, argv);
        else if (k == "--assign") a.assign = need_value(i, argc, argv);
        else if (k == "--reorder-cache-dir") a.reorder_cache_dir = need_value(i, argc, argv);
        else if (k == "--out") a.out = need_value(i, argc, argv);
        else if (k == "--nlist") a.nlist = std::stoi(need_value(i, argc, argv));
        else if (k == "--nprobe") a.nprobe = std::stoi(need_value(i, argc, argv));
        else if (k == "--topk") a.topk = std::stoi(need_value(i, argc, argv));
        else if (k == "--delta-topk") a.delta_topk = std::stoi(need_value(i, argc, argv));
        else if (k == "--main-overfetch") a.main_overfetch = std::stoi(need_value(i, argc, argv));
        else if (k == "--batch-size") a.batch_size = std::stoi(need_value(i, argc, argv));
        else if (k == "--batch-sizes") a.batch_sizes = parse_int_list(need_value(i, argc, argv));
        else if (k == "--repeats") a.repeats = std::stoi(need_value(i, argc, argv));
        else if (k == "--threads") a.threads = std::stoi(need_value(i, argc, argv));
        else if (k == "--nq-limit") a.nq_limit = std::stoi(need_value(i, argc, argv));
        else if (k == "--cuda-device") a.cuda_device = std::stoi(need_value(i, argc, argv));
        else if (k == "--delta-n") a.delta_n = std::stoi(need_value(i, argc, argv));
        else if (k == "--delta-ns") a.delta_ns = parse_int_list(need_value(i, argc, argv));
        else if (k == "--delete-ratio") a.delete_ratio = std::stod(need_value(i, argc, argv));
        else if (k == "--delete-ratios") a.delete_ratios = parse_double_list(need_value(i, argc, argv));
        else if (k == "--delta-mode") a.delta_mode = need_value(i, argc, argv);
        else if (k == "--delta-search-mode") a.delta_search_mode = need_value(i, argc, argv);
        else if (k == "--use-fbin") a.use_fbin = std::stoi(need_value(i, argc, argv)) != 0;
        else if (k == "--include-mixed-update") a.include_mixed_update = true;
        else if (k == "--mixed-workload") a.mixed_workload = true;
        else if (k == "--gt-safe-updates") a.gt_safe_updates = true;
        else if (k == "--delete-gt-safe") a.delete_gt_safe = true;
        else if (k == "--update-pairs") a.update_pairs = parse_pair_count_list(need_value(i, argc, argv));
        else if (k == "--insert-aware-recall") a.insert_aware_recall = true;
        else if (k == "--gt-only") a.gt_only = true;
        else if (k == "--pipeline") a.pipeline = std::stoi(need_value(i, argc, argv)) != 0;
        else if (k == "--no-pipeline") a.pipeline = false;
        else if (k == "--update-steps") a.update_steps = std::stoi(need_value(i, argc, argv));
        else if (k == "--query-rounds") a.query_rounds = std::stoi(need_value(i, argc, argv));
        else if (k == "--seed") a.seed = (uint64_t)std::stoull(need_value(i, argc, argv));
        else if (k == "--delta-sample-main") {
            a.delta_sample_main = true;
            a.delta_mode = "main-sample";
        }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.batch_sizes.empty()) a.batch_sizes.push_back(a.batch_size);
    if (a.delta_ns.empty()) a.delta_ns.push_back(a.delta_n);
    if (a.delete_ratios.empty()) a.delete_ratios.push_back(a.delete_ratio);
    if (a.centroids.empty() || a.assign.empty()) {
        throw std::runtime_error("--centroids and --assign are required");
    }
    if (a.topk <= 0 || a.delta_topk <= 0 || a.nprobe <= 0 || a.nlist <= 0) {
        throw std::runtime_error("invalid topk/delta_topk/batch_size/nprobe/nlist");
    }
    if (a.delta_search_mode != "flat-gpu-fp16" &&
        a.delta_search_mode != "ivf-cpu" &&
        a.delta_search_mode != "ivf-gpu") {
        throw std::runtime_error("invalid --delta-search-mode");
    }
    for (int bs : a.batch_sizes) {
        if (bs <= 0) throw std::runtime_error("invalid batch size");
    }
    for (int dn : a.delta_ns) {
        if (dn < 0) throw std::runtime_error("invalid delta_n");
    }
    for (double dr : a.delete_ratios) {
        if (dr < 0.0 || dr >= 1.0) throw std::runtime_error("invalid delete_ratio");
    }
    if (a.update_steps <= 0) throw std::runtime_error("invalid update_steps");
    if (a.query_rounds <= 0) throw std::runtime_error("invalid query_rounds");
    if (a.delta_mode != "far" && a.delta_mode != "main-sample" &&
        a.delta_mode != "random-main-sample" &&
        a.delta_mode != "random-u8" &&
        a.delta_mode != "query-copy") {
        throw std::runtime_error("invalid --delta-mode; expected far, main-sample, random-main-sample, random-u8, or query-copy");
    }
    if (a.main_overfetch < a.topk) a.main_overfetch = a.topk;
    return a;
}

std::string strip_dataset_prefix(const std::string& s) {
    if (s.rfind("deep", 0) == 0) return s.substr(4);
    if (s.rfind("spacev", 0) == 0) return s.substr(6);
    return s;
}

long long parse_scale_n(const std::string& sx) {
    if (sx.empty()) return -1;
    char unit = sx.back();
    double v = std::atof(sx.substr(0, sx.size() - 1).c_str());
    if (unit == 'm' || unit == 'M') return (long long)(v * 1000000.0);
    if (unit == 'b' || unit == 'B') return (long long)(v * 1000000000.0);
    return -1;
}

void read_bin_header(const std::string& path, int* n, int* dim) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t h[2] = {0, 0};
    if (std::fread(h, sizeof(int32_t), 2, f) != 2) {
        std::fclose(f);
        throw std::runtime_error("short header read: " + path);
    }
    std::fclose(f);
    *n = h[0];
    *dim = h[1];
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

bool is_deleted_id(int id, double ratio) {
    if (ratio <= 0.0 || id < 0) return false;
    if (ratio >= 1.0) return true;
    const long double threshold =
        ratio * (long double)std::numeric_limits<uint64_t>::max();
    return (long double)splitmix64((uint64_t)id) < threshold;
}

bool is_protected_id(const uint8_t* protected_ids, int n, int id) {
    return protected_ids && id >= 0 && id < n && protected_ids[(size_t)id] != 0;
}

bool is_deleted_id_runtime(int id,
                           double ratio,
                           const uint8_t* protected_ids,
                           int n) {
    if (is_protected_id(protected_ids, n, id)) return false;
    return is_deleted_id(id, ratio);
}

void build_gt_protect_mask(const int32_t* gt,
                           int nq,
                           int gt_k,
                           int n,
                           std::vector<uint8_t>* protected_ids) {
    protected_ids->clear();
    if (!gt || nq <= 0 || gt_k <= 0 || n <= 0) return;
    protected_ids->assign((size_t)n, 0);
    uint8_t* out = protected_ids->data();
    for (int q = 0; q < nq; ++q) {
        const int32_t* row = gt + (size_t)q * gt_k;
        for (int j = 0; j < gt_k; ++j) {
            int id = (int)row[j];
            if (id >= 0 && id < n) out[(size_t)id] = 1;
        }
    }
}

void build_reordered_delete_mask(const int* reordered_indices,
                                 int n,
                                 double ratio,
                                 const uint8_t* protected_ids,
                                 int threads,
                                 std::vector<uint8_t>* mask) {
    mask->clear();
    if (ratio <= 0.0) return;
    mask->assign((size_t)n, 0);
    uint8_t* out = mask->data();
    if (threads <= 0) threads = omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int loc = 0; loc < n; ++loc) {
        int id = reordered_indices ? reordered_indices[loc] : loc;
        out[(size_t)loc] = is_deleted_id_runtime(id, ratio, protected_ids, n) ? 1 : 0;
    }
}

void make_query_u8(const float* q, int nq, int dim, std::vector<uint8_t>* out) {
    out->resize((size_t)nq * dim);
    for (size_t i = 0; i < out->size(); ++i) {
        float x = q[i];
        if (x < 0.0f) x = 0.0f;
        if (x > 255.0f) x = 255.0f;
        (*out)[i] = (uint8_t)std::lround(x);
    }
}

struct Candidate {
    float dist;
    int id;
};

struct ScopedEnv {
    std::string name;
    std::string old_value;
    bool had_old = false;

    ScopedEnv(const char* key, const char* value) : name(key) {
        const char* old = std::getenv(key);
        if (old) {
            had_old = true;
            old_value = old;
        }
#ifdef _WIN32
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
    }

    ~ScopedEnv() {
#ifdef _WIN32
        if (had_old) {
            _putenv_s(name.c_str(), old_value.c_str());
        } else {
            _putenv_s(name.c_str(), "");
        }
#else
        if (had_old) {
            setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }
};

void push_candidate(std::vector<Candidate>& v, float dist, int id) {
    if (id < 0 || !std::isfinite(dist)) return;
    v.push_back({dist, id});
}

void finalize_topk(std::vector<Candidate>& cands, int topk, int* out_idx, float* out_dist) {
    auto cmp = [](const Candidate& a, const Candidate& b) {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.id < b.id;
    };
    if ((int)cands.size() > topk) {
        std::nth_element(cands.begin(), cands.begin() + topk, cands.end(), cmp);
        cands.resize((size_t)topk);
    }
    std::sort(cands.begin(), cands.end(), cmp);
    for (int k = 0; k < topk; ++k) {
        if (k < (int)cands.size()) {
            out_idx[k] = cands[(size_t)k].id;
            out_dist[k] = cands[(size_t)k].dist;
        } else {
            out_idx[k] = -1;
            out_dist[k] = std::numeric_limits<float>::infinity();
        }
    }
}

float l2_float(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

float l2_u8_scalar(const uint8_t* a, const uint8_t* b, int dim) {
    int s = 0;
    for (int d = 0; d < dim; ++d) {
        int diff = (int)a[d] - (int)b[d];
        s += diff * diff;
    }
    return (float)s;
}

void build_static_gt_topk(const ClusterDataset& dataset,
                          const std::vector<uint8_t>& reordered_u8,
                          const float* queries,
                          const std::vector<uint8_t>& queries_u8,
                          bool use_fbin,
                          const int32_t* gt,
                          int nq,
                          int gt_k,
                          int n,
                          int dim,
                          int topk,
                          int threads,
                          std::vector<int>* out_ids,
                          std::vector<float>* out_dists) {
    out_ids->assign((size_t)nq * topk, -1);
    out_dists->assign((size_t)nq * topk, std::numeric_limits<float>::infinity());
    if (!gt || nq <= 0 || gt_k <= 0 || topk <= 0) return;
    if (threads <= 0) threads = omp_get_max_threads();

    double t0 = now_ms();
    std::vector<int> original_to_local((size_t)n, -1);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int loc = 0; loc < n; ++loc) {
        int id = dataset.reordered_indices ? dataset.reordered_indices[loc] : loc;
        if (id >= 0 && id < n) original_to_local[(size_t)id] = loc;
    }

    const int eval_k = std::min(topk, gt_k);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int q = 0; q < nq; ++q) {
        for (int j = 0; j < eval_k; ++j) {
            int id = (int)gt[(size_t)q * gt_k + j];
            int loc = (id >= 0 && id < n) ? original_to_local[(size_t)id] : -1;
            float dist = std::numeric_limits<float>::infinity();
            if (loc >= 0) {
                if (use_fbin) {
                    dist = l2_float(queries + (size_t)q * dim,
                                    dataset.reordered_data + (size_t)loc * dim,
                                    dim);
                } else {
                    dist = l2_u8_scalar(queries_u8.data() + (size_t)q * dim,
                                        reordered_u8.data() + (size_t)loc * dim,
                                        dim);
                }
            }
            (*out_ids)[(size_t)q * topk + j] = id;
            (*out_dists)[(size_t)q * topk + j] = dist;
        }
    }
    std::printf("[INSERT-GT] built static old-GT@%d exact distances in %.2f ms\n",
                topk, now_ms() - t0);
}

int build_filtered_base_gt_topk_from_pool(const std::vector<int>& pool_ids,
                                          const std::vector<float>& pool_dists,
                                          int nq,
                                          int pool_k,
                                          int topk,
                                          double delete_ratio,
                                          int n,
                                          std::vector<int>* out_ids,
                                          std::vector<float>* out_dists) {
    out_ids->assign((size_t)nq * topk, -1);
    out_dists->assign((size_t)nq * topk, std::numeric_limits<float>::infinity());
    if (nq <= 0 || pool_k <= 0 || topk <= 0) return 0;
    int min_kept = pool_k;
    int bad_q = -1;
    for (int q = 0; q < nq; ++q) {
        int kept_total = 0;
        int emitted = 0;
        for (int j = 0; j < pool_k; ++j) {
            int id = pool_ids[(size_t)q * pool_k + j];
            if (id < 0 || id >= n) continue;
            if (is_deleted_id_runtime(id, delete_ratio, nullptr, n)) continue;
            ++kept_total;
            if (emitted < topk) {
                (*out_ids)[(size_t)q * topk + emitted] = id;
                (*out_dists)[(size_t)q * topk + emitted] = pool_dists[(size_t)q * pool_k + j];
                ++emitted;
            }
        }
        min_kept = std::min(min_kept, kept_total);
        if (kept_total < topk && bad_q < 0) bad_q = q;
    }
    if (bad_q >= 0) {
        throw std::runtime_error("filtered GT@1000 has fewer than topk survivors for at least one query");
    }
    std::printf("[DELETE-GT] delete_ratio=%.4f filtered GT@%d min_survivors=%d -> base GT@%d\\n",
                delete_ratio, pool_k, min_kept, topk);
    return min_kept;
}

void fine_search_scalar_touched_float(const float* reordered,
                                      const long long* offsets,
                                      const int* counts,
                                      const int* reordered_indices,
                                      int nlist,
                                      int dim,
                                      const float* queries,
                                      const int* cids,
                                      int nq,
                                      int nprobe,
                                      int topk,
                                      int* out_idx,
                                      float* out_dist) {
    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < nq; ++q) {
        std::vector<Candidate> cands;
        for (int p = 0; p < nprobe; ++p) {
            int cid = cids[(size_t)q * nprobe + p];
            if (cid < 0 || cid >= nlist) continue;
            long long off = offsets[cid];
            int cnt = counts[cid];
            for (int j = 0; j < cnt; ++j) {
                int loc = (int)(off + j);
                int id = reordered_indices ? reordered_indices[loc] : loc;
                float dist = l2_float(queries + (size_t)q * dim,
                                      reordered + (size_t)loc * dim,
                                      dim);
                push_candidate(cands, dist, id);
            }
        }
        finalize_topk(cands, topk,
                      out_idx + (size_t)q * topk,
                      out_dist + (size_t)q * topk);
    }
}

void fine_search_scalar_touched_u8(const uint8_t* reordered_u8,
                                   const long long* offsets,
                                   const int* counts,
                                   const int* reordered_indices,
                                   int nlist,
                                   int dim,
                                   const uint8_t* queries_u8,
                                   const int* cids,
                                   int nq,
                                   int nprobe,
                                   int topk,
                                   int* out_idx,
                                   float* out_dist) {
    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < nq; ++q) {
        std::vector<Candidate> cands;
        for (int p = 0; p < nprobe; ++p) {
            int cid = cids[(size_t)q * nprobe + p];
            if (cid < 0 || cid >= nlist) continue;
            long long off = offsets[cid];
            int cnt = counts[cid];
            for (int j = 0; j < cnt; ++j) {
                int loc = (int)(off + j);
                int id = reordered_indices ? reordered_indices[loc] : loc;
                float dist = l2_u8_scalar(queries_u8 + (size_t)q * dim,
                                          reordered_u8 + (size_t)loc * dim,
                                          dim);
                push_candidate(cands, dist, id);
            }
        }
        finalize_topk(cands, topk,
                      out_idx + (size_t)q * topk,
                      out_dist + (size_t)q * topk);
    }
}

void cuda_check(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
    }
}

void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cublas error " + std::to_string((int)st));
    }
}

__global__ void update_init_topk_kernel(float* dist, int* idx, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dist[i] = INFINITY;
    idx[i] = -1;
}

__global__ void update_gen_tile_index_kernel(int* idx, int n_query, int tile_n, int tile0) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_query * tile_n;
    if (t >= total) return;
    int c = (int)(t % tile_n);
    idx[t] = tile0 + c;
}

__global__ void float_to_half_kernel(const float* __restrict__ in,
                                     __half* __restrict__ out,
                                     size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}


__global__ void update_merge_topk_kernel(const float* a_dist,
                                         const int* a_idx,
                                         int a_k,
                                         const float* b_dist,
                                         const int* b_idx,
                                         int b_k,
                                         int n_query,
                                         int out_k,
                                         float* out_dist,
                                         int* out_idx) {
    int q = blockIdx.x;
    if (q >= n_query || threadIdx.x != 0) return;

    const float* ad = a_dist + (size_t)q * a_k;
    const int* ai = a_idx + (size_t)q * a_k;
    const float* bd = b_dist + (size_t)q * b_k;
    const int* bi = b_idx + (size_t)q * b_k;
    float* od = out_dist + (size_t)q * out_k;
    int* oi = out_idx + (size_t)q * out_k;

    for (int out = 0; out < out_k; ++out) {
        float best_d = INFINITY;
        int best_i = -1;
        for (int k = 0; k < a_k; ++k) {
            float d = ad[k];
            int id = ai[k];
            bool emitted = false;
            for (int prev = 0; prev < out; ++prev) emitted = emitted || (oi[prev] == id);
            if (!emitted && id >= 0 && (d < best_d || (d == best_d && id < best_i))) {
                best_d = d;
                best_i = id;
            }
        }
        for (int k = 0; k < b_k; ++k) {
            float d = bd[k];
            int id = bi[k];
            bool emitted = false;
            for (int prev = 0; prev < out; ++prev) emitted = emitted || (oi[prev] == id);
            if (!emitted && id >= 0 && (d < best_d || (d == best_d && id < best_i))) {
                best_d = d;
                best_i = id;
            }
        }
        od[out] = best_d;
        oi[out] = best_i;
    }
}

struct PersistentTiledCoarseWorkspace {
    int max_batch = 0;
    int dim = 0;
    int max_k = 0;
    int tile_n = 0;
    float* d_query = nullptr;
    float* d_query_norm = nullptr;
    float* d_tile_ip = nullptr;
    int* d_tile_index = nullptr;
    float* d_tile_dist = nullptr;
    int* d_tile_top_index = nullptr;
    float* d_top_dist_a = nullptr;
    int* d_top_index_a = nullptr;
    float* d_top_dist_b = nullptr;
    int* d_top_index_b = nullptr;

    void allocate(int max_batch_, int dim_, int max_k_, int tile_n_) {
        max_batch = max_batch_;
        dim = dim_;
        max_k = std::min(max_k_, 512);
        tile_n = tile_n_;
        cuda_check(cudaMalloc(&d_query, (size_t)max_batch * dim * sizeof(float)), "cudaMalloc fast d_query");
        cuda_check(cudaMalloc(&d_query_norm, (size_t)max_batch * sizeof(float)), "cudaMalloc fast d_query_norm");
        cuda_check(cudaMalloc(&d_tile_ip, (size_t)max_batch * tile_n * sizeof(float)), "cudaMalloc fast d_tile_ip");
        cuda_check(cudaMalloc(&d_tile_index, (size_t)max_batch * tile_n * sizeof(int)), "cudaMalloc fast d_tile_index");
        cuda_check(cudaMalloc(&d_tile_dist, (size_t)max_batch * max_k * sizeof(float)), "cudaMalloc fast d_tile_dist");
        cuda_check(cudaMalloc(&d_tile_top_index, (size_t)max_batch * max_k * sizeof(int)), "cudaMalloc fast d_tile_top_index");
        cuda_check(cudaMalloc(&d_top_dist_a, (size_t)max_batch * max_k * sizeof(float)), "cudaMalloc fast d_top_dist_a");
        cuda_check(cudaMalloc(&d_top_index_a, (size_t)max_batch * max_k * sizeof(int)), "cudaMalloc fast d_top_index_a");
        cuda_check(cudaMalloc(&d_top_dist_b, (size_t)max_batch * max_k * sizeof(float)), "cudaMalloc fast d_top_dist_b");
        cuda_check(cudaMalloc(&d_top_index_b, (size_t)max_batch * max_k * sizeof(int)), "cudaMalloc fast d_top_index_b");
    }

    void release() {
        if (d_query) cudaFree(d_query);
        if (d_query_norm) cudaFree(d_query_norm);
        if (d_tile_ip) cudaFree(d_tile_ip);
        if (d_tile_index) cudaFree(d_tile_index);
        if (d_tile_dist) cudaFree(d_tile_dist);
        if (d_tile_top_index) cudaFree(d_tile_top_index);
        if (d_top_dist_a) cudaFree(d_top_dist_a);
        if (d_top_index_a) cudaFree(d_top_index_a);
        if (d_top_dist_b) cudaFree(d_top_dist_b);
        if (d_top_index_b) cudaFree(d_top_index_b);
        *this = PersistentTiledCoarseWorkspace{};
    }
};

void persistent_tiled_l2_search(const CoarseHandle* h,
                                PersistentTiledCoarseWorkspace* ws,
                                const float* h_query,
                                int n_query,
                                int topk,
                                int active_n,
                                int* h_idx_out,
                                float* h_dist_out,
                                double* out_gpu_ms,
                                double* out_h2d_ms,
                                double* out_d2h_ms) {
    if (out_gpu_ms) *out_gpu_ms = 0.0;
    if (out_h2d_ms) *out_h2d_ms = 0.0;
    if (out_d2h_ms) *out_d2h_ms = 0.0;
    if (!h || !h->d_centers || !h->d_centers_norm || !ws || active_n <= 0) return;
    if (n_query > ws->max_batch || topk > ws->max_k || h->dim != ws->dim) {
        throw std::runtime_error("persistent_tiled_l2_search workspace too small");
    }
    topk = std::min(topk, active_n);

    const size_t q_bytes = (size_t)n_query * h->dim * sizeof(float);
    const size_t idx_bytes = (size_t)n_query * topk * sizeof(int);
    const size_t dist_bytes = (size_t)n_query * topk * sizeof(float);

    double h2d0 = now_ms();
    cuda_check(cudaMemcpy(ws->d_query, h_query, q_bytes, cudaMemcpyHostToDevice),
               "fast coarse query H2D");
    double h2d_ms = now_ms() - h2d0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    compute_l2_norm_gpu(ws->d_query, ws->d_query_norm, n_query, h->dim);
    int init_total = n_query * topk;
    update_init_topk_kernel<<<(init_total + 255) / 256, 256>>>(
        ws->d_top_dist_a, ws->d_top_index_a, init_total);
    cuda_check(cudaGetLastError(), "fast init topk");

    cublasHandle_t handle = (cublasHandle_t)h->cublas_handle;
    float alpha = 1.0f, beta = 0.0f;
    float* top_dist = ws->d_top_dist_a;
    int* top_idx = ws->d_top_index_a;
    float* tmp_dist = ws->d_top_dist_b;
    int* tmp_idx = ws->d_top_index_b;

    for (int tile0 = 0; tile0 < active_n; tile0 += ws->tile_n) {
        int cur_tile = std::min(ws->tile_n, active_n - tile0);
        int tile_topk = std::min(topk, cur_tile);
        cublas_check(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 cur_tile, n_query, h->dim,
                                 &alpha,
                                 h->d_centers + (size_t)tile0 * h->dim, h->dim,
                                 ws->d_query, h->dim,
                                 &beta,
                                 ws->d_tile_ip, cur_tile),
                     "fast tiled cublasSgemm");
        long long tile_total = (long long)n_query * cur_tile;
        update_gen_tile_index_kernel<<<(int)((tile_total + 255) / 256), 256>>>(
            ws->d_tile_index, n_query, cur_tile, tile0);
        cuda_check(cudaGetLastError(), "fast tile index");
        cuda_check(pgvector::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                       ws->d_query_norm, h->d_centers_norm + tile0,
                       ws->d_tile_ip, ws->d_tile_index,
                       n_query, cur_tile, tile_topk,
                       ws->d_tile_dist, ws->d_tile_top_index,
                       true, 0),
                   "fast tile l2 topk");
        update_merge_topk_kernel<<<n_query, 1>>>(
            top_dist, top_idx, topk,
            ws->d_tile_dist, ws->d_tile_top_index, tile_topk,
            n_query, topk,
            tmp_dist, tmp_idx);
        cuda_check(cudaGetLastError(), "fast merge topk");
        std::swap(top_dist, tmp_dist);
        std::swap(top_idx, tmp_idx);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    double d2h0 = now_ms();
    cuda_check(cudaMemcpy(h_idx_out, top_idx, idx_bytes, cudaMemcpyDeviceToHost),
               "fast coarse idx D2H");
    if (h_dist_out) {
        cuda_check(cudaMemcpy(h_dist_out, top_dist, dist_bytes, cudaMemcpyDeviceToHost),
                   "fast coarse dist D2H");
    }
    double d2h_ms = now_ms() - d2h0;

    if (out_gpu_ms) *out_gpu_ms += gpu_ms;
    if (out_h2d_ms) *out_h2d_ms += h2d_ms;
    if (out_d2h_ms) *out_d2h_ms += d2h_ms;
}

struct DeltaFp16GemmWorkspace {
    int max_batch = 0;
    int dim = 0;
    int max_topk = 0;
    int max_delta_n = 0;
    int tile_n = 0;
    float* d_query_f = nullptr;
    __half* d_query_h = nullptr;
    __half* d_delta_h = nullptr;
    float* d_delta_norm = nullptr;
    float* d_query_norm = nullptr;
    float* d_tile_ip = nullptr;
    int* d_tile_index = nullptr;
    float* d_tile_dist = nullptr;
    int* d_tile_top_index = nullptr;
    float* d_top_dist_a = nullptr;
    int* d_top_index_a = nullptr;
    float* d_top_dist_b = nullptr;
    int* d_top_index_b = nullptr;
    cublasHandle_t handle = nullptr;

    void allocate(int max_batch_, int dim_, int max_topk_, int max_delta_n_, int tile_n_) {
        max_batch = max_batch_;
        dim = dim_;
        max_topk = max_topk_;
        max_delta_n = max_delta_n_;
        tile_n = tile_n_;
        cuda_check(cudaMalloc(&d_query_f, (size_t)max_batch * dim * sizeof(float)), "cudaMalloc delta fp16 d_query_f");
        cuda_check(cudaMalloc(&d_query_h, (size_t)max_batch * dim * sizeof(__half)), "cudaMalloc delta fp16 d_query_h");
        cuda_check(cudaMalloc(&d_delta_h, (size_t)max_delta_n * dim * sizeof(__half)), "cudaMalloc delta fp16 d_delta_h");
        cuda_check(cudaMalloc(&d_delta_norm, (size_t)max_delta_n * sizeof(float)), "cudaMalloc delta fp16 d_delta_norm");
        cuda_check(cudaMalloc(&d_query_norm, (size_t)max_batch * sizeof(float)), "cudaMalloc delta fp16 d_query_norm");
        cuda_check(cudaMalloc(&d_tile_ip, (size_t)max_batch * tile_n * sizeof(float)), "cudaMalloc delta fp16 d_tile_ip");
        cuda_check(cudaMalloc(&d_tile_index, (size_t)max_batch * tile_n * sizeof(int)), "cudaMalloc delta fp16 d_tile_index");
        cuda_check(cudaMalloc(&d_tile_dist, (size_t)max_batch * max_topk * sizeof(float)), "cudaMalloc delta fp16 d_tile_dist");
        cuda_check(cudaMalloc(&d_tile_top_index, (size_t)max_batch * max_topk * sizeof(int)), "cudaMalloc delta fp16 d_tile_top_index");
        cuda_check(cudaMalloc(&d_top_dist_a, (size_t)max_batch * max_topk * sizeof(float)), "cudaMalloc delta fp16 d_top_dist_a");
        cuda_check(cudaMalloc(&d_top_index_a, (size_t)max_batch * max_topk * sizeof(int)), "cudaMalloc delta fp16 d_top_index_a");
        cuda_check(cudaMalloc(&d_top_dist_b, (size_t)max_batch * max_topk * sizeof(float)), "cudaMalloc delta fp16 d_top_dist_b");
        cuda_check(cudaMalloc(&d_top_index_b, (size_t)max_batch * max_topk * sizeof(int)), "cudaMalloc delta fp16 d_top_index_b");
        cublas_check(cublasCreate(&handle), "delta fp16 cublasCreate");
        cublas_check(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "delta fp16 cublasSetMathMode");
    }

    void build_resident(const float* d_delta_f) {
        if (!d_delta_f || max_delta_n <= 0) return;
        compute_l2_norm_gpu(d_delta_f, d_delta_norm, max_delta_n, dim);
        size_t total = (size_t)max_delta_n * dim;
        float_to_half_kernel<<<(int)((total + 255) / 256), 256>>>(d_delta_f, d_delta_h, total);
        cuda_check(cudaGetLastError(), "delta fp16 convert resident");
        cuda_check(cudaDeviceSynchronize(), "delta fp16 resident sync");
    }

    void release() {
        if (handle) cublasDestroy(handle);
        if (d_query_f) cudaFree(d_query_f);
        if (d_query_h) cudaFree(d_query_h);
        if (d_delta_h) cudaFree(d_delta_h);
        if (d_delta_norm) cudaFree(d_delta_norm);
        if (d_query_norm) cudaFree(d_query_norm);
        if (d_tile_ip) cudaFree(d_tile_ip);
        if (d_tile_index) cudaFree(d_tile_index);
        if (d_tile_dist) cudaFree(d_tile_dist);
        if (d_tile_top_index) cudaFree(d_tile_top_index);
        if (d_top_dist_a) cudaFree(d_top_dist_a);
        if (d_top_index_a) cudaFree(d_top_index_a);
        if (d_top_dist_b) cudaFree(d_top_dist_b);
        if (d_top_index_b) cudaFree(d_top_index_b);
        *this = DeltaFp16GemmWorkspace{};
    }
};

void delta_fp16_gemm_topk_search(DeltaFp16GemmWorkspace* ws,
                                 const float* h_delta,
                                 const float* h_query,
                                 int n_query,
                                 int dim,
                                 int active_delta_n,
                                 int base_id,
                                 int topk,
                                 std::vector<int>* out_idx,
                                 std::vector<float>* out_dist,
                                 double* out_gpu_ms,
                                 double* out_h2d_ms,
                                 double* out_d2h_ms) {
    out_idx->assign((size_t)n_query * topk, -1);
    out_dist->assign((size_t)n_query * topk, std::numeric_limits<float>::infinity());
    if (out_gpu_ms) *out_gpu_ms = 0.0;
    if (out_h2d_ms) *out_h2d_ms = 0.0;
    if (out_d2h_ms) *out_d2h_ms = 0.0;
    if (!ws || !ws->d_delta_h || active_delta_n <= 0) return;
    if (n_query > ws->max_batch || dim != ws->dim || topk > ws->max_topk || active_delta_n > ws->max_delta_n) {
        throw std::runtime_error("delta_fp16_gemm_topk_search workspace too small");
    }
    topk = std::min(topk, active_delta_n);
    const size_t q_bytes_f = (size_t)n_query * dim * sizeof(float);
    const size_t q_elems = (size_t)n_query * dim;
    const size_t idx_bytes = (size_t)n_query * topk * sizeof(int);
    const size_t dist_bytes = (size_t)n_query * topk * sizeof(float);

    double h2d0 = now_ms();
    cuda_check(cudaMemcpy(ws->d_query_f, h_query, q_bytes_f, cudaMemcpyHostToDevice), "delta fp16 query H2D float");
    float_to_half_kernel<<<(int)((q_elems + 255) / 256), 256>>>(ws->d_query_f, ws->d_query_h, q_elems);
    cuda_check(cudaGetLastError(), "delta fp16 query convert");
    double h2d_ms = now_ms() - h2d0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    compute_l2_norm_gpu(ws->d_query_f, ws->d_query_norm, n_query, dim);
    update_init_topk_kernel<<<(n_query * topk + 255) / 256, 256>>>(ws->d_top_dist_a, ws->d_top_index_a, n_query * topk);
    cuda_check(cudaGetLastError(), "delta fp16 init topk");

    float alpha = 1.0f;
    float beta = 0.0f;
    float* top_dist = ws->d_top_dist_a;
    int* top_idx = ws->d_top_index_a;
    float* tmp_dist = ws->d_top_dist_b;
    int* tmp_idx = ws->d_top_index_b;

    for (int tile0 = 0; tile0 < active_delta_n; tile0 += ws->tile_n) {
        int cur_tile = std::min(ws->tile_n, active_delta_n - tile0);
        int tile_topk = std::min(topk, cur_tile);
        cublas_check(cublasGemmEx(ws->handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  cur_tile, n_query, dim,
                                  &alpha,
                                  ws->d_delta_h + (size_t)tile0 * dim, CUDA_R_16F, dim,
                                  ws->d_query_h, CUDA_R_16F, dim,
                                  &beta,
                                  ws->d_tile_ip, CUDA_R_32F, cur_tile,
                                  CUBLAS_COMPUTE_32F_FAST_16F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "delta fp16 tiled cublasGemmEx");
        long long tile_total = (long long)n_query * cur_tile;
        update_gen_tile_index_kernel<<<(int)((tile_total + 255) / 256), 256>>>(
            ws->d_tile_index, n_query, cur_tile, tile0);
        cuda_check(cudaGetLastError(), "delta fp16 tile index");
        cuda_check(pgvector::fusion_dist_topk_warpsort::fusion_l2_topk_warpsort<float, int>(
                       ws->d_query_norm, ws->d_delta_norm + tile0,
                       ws->d_tile_ip, ws->d_tile_index,
                       n_query, cur_tile, tile_topk,
                       ws->d_tile_dist, ws->d_tile_top_index,
                       true, 0),
                   "delta fp16 tile l2 topk");
        update_merge_topk_kernel<<<n_query, 1>>>(
            top_dist, top_idx, topk,
            ws->d_tile_dist, ws->d_tile_top_index, tile_topk,
            n_query, topk,
            tmp_dist, tmp_idx);
        cuda_check(cudaGetLastError(), "delta fp16 merge topk");
        std::swap(top_dist, tmp_dist);
        std::swap(top_idx, tmp_idx);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    std::vector<int> approx_idx((size_t)n_query * topk, -1);
    std::vector<float> approx_dist((size_t)n_query * topk, std::numeric_limits<float>::infinity());
    double d2h0 = now_ms();
    cuda_check(cudaMemcpy(approx_idx.data(), top_idx, idx_bytes, cudaMemcpyDeviceToHost), "delta fp16 idx D2H");
    cuda_check(cudaMemcpy(approx_dist.data(), top_dist, dist_bytes, cudaMemcpyDeviceToHost), "delta fp16 dist D2H");
    double d2h_ms = now_ms() - d2h0;

    for (int q = 0; q < n_query; ++q) {
        std::vector<Candidate> cands;
        cands.reserve((size_t)topk);
        for (int k = 0; k < topk; ++k) {
            int local = approx_idx[(size_t)q * topk + k];
            if (local < 0 || local >= active_delta_n) continue;
            float exact = approx_dist[(size_t)q * topk + k];
            if (h_delta) {
                exact = l2_float(h_query + (size_t)q * dim,
                                 h_delta + (size_t)local * dim,
                                 dim);
            }
            push_candidate(cands, exact, base_id + local);
        }
        finalize_topk(cands, topk,
                      out_idx->data() + (size_t)q * topk,
                      out_dist->data() + (size_t)q * topk);
    }

    if (out_gpu_ms) *out_gpu_ms += gpu_ms;
    if (out_h2d_ms) *out_h2d_ms += h2d_ms;
    if (out_d2h_ms) *out_d2h_ms += d2h_ms;
}

struct DeltaIvfCpuSegment {
    int active_n = 0;
    int nlist = 0;
    int dim = 0;
    bool use_u8 = false;
    std::vector<int> counts;
    std::vector<long long> offsets;
    std::vector<float> vectors;
    std::vector<uint8_t> vectors_u8;
    std::vector<int> ids;
};

int cluster_for_reordered_loc(const ClusterDataset& dataset, int nlist, int loc) {
    const long long* begin = dataset.cluster_info.offsets;
    const long long* end = dataset.cluster_info.offsets + nlist;
    int c = (int)(std::upper_bound(begin, end, (long long)loc) - begin) - 1;
    if (c < 0) c = 0;
    if (c >= nlist) c = nlist - 1;
    return c;
}

int delta_cluster_for_insert(const ClusterDataset& dataset,
                             const std::vector<int>& delta_locs,
                             int nlist,
                             int local_id) {
    int loc = (local_id < (int)delta_locs.size()) ? delta_locs[(size_t)local_id] : -1;
    if (loc >= 0) return cluster_for_reordered_loc(dataset, nlist, loc);
    return (int)(((uint64_t)local_id * 11400714819323198485ull) % (uint64_t)nlist);
}

void build_delta_ivf_cpu_segment(const ClusterDataset& dataset,
                                 const std::vector<float>& delta,
                                 const std::vector<uint8_t>& delta_u8,
                                 const std::vector<int>& delta_locs,
                                 int active_n,
                                 int base_id,
                                 int nlist,
                                 int dim,
                                 DeltaIvfCpuSegment* seg) {
    if (!seg || active_n <= 0) return;
    seg->active_n = active_n;
    seg->nlist = nlist;
    seg->dim = dim;
    seg->use_u8 = !delta_u8.empty();
    seg->counts.assign((size_t)nlist, 0);
    seg->offsets.assign((size_t)nlist, 0);

    std::vector<int> reserve_counts((size_t)nlist, 0);
    for (int i = 0; i < active_n; ++i) {
        int c = delta_cluster_for_insert(dataset, delta_locs, nlist, i);
        ++reserve_counts[(size_t)c];
    }

    double append_t0 = now_ms();
    std::vector<std::vector<int>> mutable_delta_lists((size_t)nlist);
    for (int c = 0; c < nlist; ++c) {
        if (reserve_counts[(size_t)c] > 0) {
            mutable_delta_lists[(size_t)c].reserve((size_t)reserve_counts[(size_t)c]);
        }
    }
    for (int i = 0; i < active_n; ++i) {
        int c = delta_cluster_for_insert(dataset, delta_locs, nlist, i);
        mutable_delta_lists[(size_t)c].push_back(i);
    }
    double append_ms = now_ms() - append_t0;

    double publish_t0 = now_ms();
    int max_list = 0;
    long long nonempty = 0;
    for (int c = 0; c < nlist; ++c) {
        int cnt = (int)mutable_delta_lists[(size_t)c].size();
        seg->counts[(size_t)c] = cnt;
        max_list = std::max(max_list, cnt);
        nonempty += (cnt > 0);
    }
    for (int c = 1; c < nlist; ++c) {
        seg->offsets[(size_t)c] = seg->offsets[(size_t)c - 1] + seg->counts[(size_t)c - 1];
    }
    if (seg->use_u8) {
        seg->vectors_u8.assign((size_t)active_n * dim, 0);
        seg->vectors.clear();
    } else {
        seg->vectors.assign((size_t)active_n * dim, 0.0f);
        seg->vectors_u8.clear();
    }
    seg->ids.assign((size_t)active_n, -1);
    for (int c = 0; c < nlist; ++c) {
        long long dst = seg->offsets[(size_t)c];
        const std::vector<int>& list = mutable_delta_lists[(size_t)c];
        for (int local_id : list) {
            if (seg->use_u8) {
                std::memcpy(seg->vectors_u8.data() + (size_t)dst * dim,
                            delta_u8.data() + (size_t)local_id * dim,
                            (size_t)dim * sizeof(uint8_t));
            } else {
                std::memcpy(seg->vectors.data() + (size_t)dst * dim,
                            delta.data() + (size_t)local_id * dim,
                            (size_t)dim * sizeof(float));
            }
            seg->ids[(size_t)dst] = base_id + local_id;
            ++dst;
        }
    }
    double publish_ms = now_ms() - publish_t0;
    std::printf("[DELTA-MUTABLE] active_n=%d appended_to_lists_ms=%.2f publish_flatten_ms=%.2f nonempty_lists=%lld max_list=%d\n",
                active_n, append_ms, publish_ms, nonempty, max_list);
}

const DeltaIvfCpuSegment* find_delta_ivf_segment(const std::vector<DeltaIvfCpuSegment>& segs,
                                                 int active_n) {
    for (const auto& s : segs) {
        if (s.active_n == active_n) return &s;
    }
    return nullptr;
}

void delta_ivf_cpu_search(const DeltaIvfCpuSegment* seg,
                          const float* h_query,
                          const uint8_t* h_query_u8,
                          const int* h_cids,
                          int n_query,
                          int nprobe,
                          int dim,
                          int topk,
                          int num_threads,
                          std::vector<int>* out_idx,
                          std::vector<float>* out_dist,
                          double* out_ms) {
    out_idx->assign((size_t)n_query * topk, -1);
    out_dist->assign((size_t)n_query * topk, std::numeric_limits<float>::infinity());
    if (out_ms) *out_ms = 0.0;
    if (!seg || seg->active_n <= 0) return;
    if (seg->use_u8 && !h_query_u8) {
        throw std::runtime_error("u8 delta-IVF CPU search requires u8 queries");
    }
    double t0 = now_ms();
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int q = 0; q < n_query; ++q) {
        std::vector<Candidate> cands;
        cands.reserve((size_t)nprobe * 32);
        const float* query = h_query + (size_t)q * dim;
        const uint8_t* query_u8 = h_query_u8 ? h_query_u8 + (size_t)q * dim : nullptr;
        for (int p = 0; p < nprobe; ++p) {
            int c = h_cids[(size_t)q * nprobe + p];
            if (c < 0 || c >= seg->nlist) continue;
            long long start = seg->offsets[(size_t)c];
            int len = seg->counts[(size_t)c];
            for (int j = 0; j < len; ++j) {
                long long pos = start + j;
                int id = seg->ids[(size_t)pos];
                if (id < 0) continue;
                float dist = seg->use_u8
                    ? l2_u8_scalar(query_u8, seg->vectors_u8.data() + (size_t)pos * dim, dim)
                    : l2_float(query, seg->vectors.data() + (size_t)pos * dim, dim);
                push_candidate(cands, dist, id);
            }
        }
        finalize_topk(cands, topk,
                      out_idx->data() + (size_t)q * topk,
                      out_dist->data() + (size_t)q * topk);
    }
    if (out_ms) *out_ms = now_ms() - t0;
}

__global__ void delta_ivf_gpu_topk_kernel(const float* __restrict__ queries,
                                          const uint8_t* __restrict__ queries_u8,
                                          const int* __restrict__ cids,
                                          const long long* __restrict__ offsets,
                                          const int* __restrict__ counts,
                                          const float* __restrict__ vectors,
                                          const uint8_t* __restrict__ vectors_u8,
                                          const int* __restrict__ ids,
                                          int n_query,
                                          int nprobe,
                                          int dim,
                                          int nlist,
                                          int use_u8,
                                          int topk,
                                          int* out_idx,
                                          float* out_dist) {
    constexpr int MAXK = 16;
    int q = blockIdx.x;
    if (q >= n_query) return;
    int tid = threadIdx.x;
    extern __shared__ unsigned char smem[];
    float* sh_dist = reinterpret_cast<float*>(smem);
    int* sh_idx = reinterpret_cast<int*>(sh_dist + (size_t)blockDim.x * MAXK);

    float local_dist[MAXK];
    int local_idx[MAXK];
    for (int k = 0; k < MAXK; ++k) {
        local_dist[k] = INFINITY;
        local_idx[k] = -1;
    }
    const float* query = queries ? queries + (size_t)q * dim : nullptr;
    const uint8_t* query_u8 = queries_u8 ? queries_u8 + (size_t)q * dim : nullptr;
    for (int p = 0; p < nprobe; ++p) {
        int c = cids[(size_t)q * nprobe + p];
        if (c < 0 || c >= nlist) continue;
        long long start = offsets[c];
        int len = counts[c];
        for (int j = tid; j < len; j += blockDim.x) {
            long long pos = start + j;
            float acc = 0.0f;
            if (use_u8) {
                const uint8_t* x = vectors_u8 + (size_t)pos * dim;
                for (int d = 0; d < dim; ++d) {
                    int diff = (int)query_u8[d] - (int)x[d];
                    acc += (float)(diff * diff);
                }
            } else {
                const float* x = vectors + (size_t)pos * dim;
                for (int d = 0; d < dim; ++d) {
                    float diff = query[d] - x[d];
                    acc += diff * diff;
                }
            }
            int id = ids[pos];
            if (id < 0) continue;
            for (int k = 0; k < topk && k < MAXK; ++k) {
                if (acc < local_dist[k] || (acc == local_dist[k] && id < local_idx[k])) {
                    for (int m = min(topk, MAXK) - 1; m > k; --m) {
                        local_dist[m] = local_dist[m - 1];
                        local_idx[m] = local_idx[m - 1];
                    }
                    local_dist[k] = acc;
                    local_idx[k] = id;
                    break;
                }
            }
        }
    }
    for (int k = 0; k < MAXK; ++k) {
        sh_dist[(size_t)tid * MAXK + k] = local_dist[k];
        sh_idx[(size_t)tid * MAXK + k] = local_idx[k];
    }
    __syncthreads();
    if (tid == 0) {
        float best_dist[MAXK];
        int best_idx[MAXK];
        for (int k = 0; k < MAXK; ++k) {
            best_dist[k] = INFINITY;
            best_idx[k] = -1;
        }
        int tk = min(topk, MAXK);
        for (int t = 0; t < blockDim.x; ++t) {
            for (int kk = 0; kk < tk; ++kk) {
                float d = sh_dist[(size_t)t * MAXK + kk];
                int id = sh_idx[(size_t)t * MAXK + kk];
                if (id < 0) continue;
                for (int k = 0; k < tk; ++k) {
                    if (d < best_dist[k] || (d == best_dist[k] && id < best_idx[k])) {
                        for (int m = tk - 1; m > k; --m) {
                            best_dist[m] = best_dist[m - 1];
                            best_idx[m] = best_idx[m - 1];
                        }
                        best_dist[k] = d;
                        best_idx[k] = id;
                        break;
                    }
                }
            }
        }
        for (int k = 0; k < topk; ++k) {
            out_dist[(size_t)q * topk + k] = (k < tk) ? best_dist[k] : INFINITY;
            out_idx[(size_t)q * topk + k] = (k < tk) ? best_idx[k] : -1;
        }
    }
}

struct DeltaIvfGpuSegment {
    int active_n = 0;
    int nlist = 0;
    int dim = 0;
    int max_batch = 0;
    int max_topk = 0;
    bool use_u8 = false;
    float* d_vectors = nullptr;
    uint8_t* d_vectors_u8 = nullptr;
    int* d_ids = nullptr;
    long long* d_offsets = nullptr;
    int* d_counts = nullptr;
    float* d_query = nullptr;
    uint8_t* d_query_u8 = nullptr;
    int* d_cids = nullptr;
    int* d_out_idx = nullptr;
    float* d_out_dist = nullptr;

    void allocate_from_cpu(const DeltaIvfCpuSegment& cpu, int max_batch_, int max_topk_, int nprobe) {
        active_n = cpu.active_n;
        nlist = cpu.nlist;
        dim = cpu.dim;
        max_batch = max_batch_;
        max_topk = max_topk_;
        use_u8 = cpu.use_u8;
        if (use_u8) {
            cuda_check(cudaMalloc(&d_vectors_u8, (size_t)active_n * dim * sizeof(uint8_t)), "cudaMalloc delta ivf u8 vectors");
        } else {
            cuda_check(cudaMalloc(&d_vectors, (size_t)active_n * dim * sizeof(float)), "cudaMalloc delta ivf vectors");
        }
        cuda_check(cudaMalloc(&d_ids, (size_t)active_n * sizeof(int)), "cudaMalloc delta ivf ids");
        cuda_check(cudaMalloc(&d_offsets, (size_t)nlist * sizeof(long long)), "cudaMalloc delta ivf offsets");
        cuda_check(cudaMalloc(&d_counts, (size_t)nlist * sizeof(int)), "cudaMalloc delta ivf counts");
        if (use_u8) {
            cuda_check(cudaMalloc(&d_query_u8, (size_t)max_batch * dim * sizeof(uint8_t)), "cudaMalloc delta ivf u8 query");
        } else {
            cuda_check(cudaMalloc(&d_query, (size_t)max_batch * dim * sizeof(float)), "cudaMalloc delta ivf query");
        }
        cuda_check(cudaMalloc(&d_cids, (size_t)max_batch * nprobe * sizeof(int)), "cudaMalloc delta ivf cids");
        cuda_check(cudaMalloc(&d_out_idx, (size_t)max_batch * max_topk * sizeof(int)), "cudaMalloc delta ivf out idx");
        cuda_check(cudaMalloc(&d_out_dist, (size_t)max_batch * max_topk * sizeof(float)), "cudaMalloc delta ivf out dist");
        if (use_u8) {
            cuda_check(cudaMemcpy(d_vectors_u8, cpu.vectors_u8.data(), (size_t)active_n * dim * sizeof(uint8_t), cudaMemcpyHostToDevice), "delta ivf H2D u8 vectors");
        } else {
            cuda_check(cudaMemcpy(d_vectors, cpu.vectors.data(), (size_t)active_n * dim * sizeof(float), cudaMemcpyHostToDevice), "delta ivf H2D vectors");
        }
        cuda_check(cudaMemcpy(d_ids, cpu.ids.data(), (size_t)active_n * sizeof(int), cudaMemcpyHostToDevice), "delta ivf H2D ids");
        cuda_check(cudaMemcpy(d_offsets, cpu.offsets.data(), (size_t)nlist * sizeof(long long), cudaMemcpyHostToDevice), "delta ivf H2D offsets");
        cuda_check(cudaMemcpy(d_counts, cpu.counts.data(), (size_t)nlist * sizeof(int), cudaMemcpyHostToDevice), "delta ivf H2D counts");
    }

    void release() {
        if (d_vectors) cudaFree(d_vectors);
        if (d_vectors_u8) cudaFree(d_vectors_u8);
        if (d_ids) cudaFree(d_ids);
        if (d_offsets) cudaFree(d_offsets);
        if (d_counts) cudaFree(d_counts);
        if (d_query) cudaFree(d_query);
        if (d_query_u8) cudaFree(d_query_u8);
        if (d_cids) cudaFree(d_cids);
        if (d_out_idx) cudaFree(d_out_idx);
        if (d_out_dist) cudaFree(d_out_dist);
        *this = DeltaIvfGpuSegment{};
    }
};

const DeltaIvfGpuSegment* find_delta_ivf_gpu_segment(const std::vector<DeltaIvfGpuSegment>& segs,
                                                     int active_n) {
    for (const auto& s : segs) {
        if (s.active_n == active_n) return &s;
    }
    return nullptr;
}

void delta_ivf_gpu_search(const DeltaIvfGpuSegment* seg,
                          const float* h_query,
                          const uint8_t* h_query_u8,
                          const int* h_cids,
                          int n_query,
                          int nprobe,
                          int dim,
                          int topk,
                          std::vector<int>* out_idx,
                          std::vector<float>* out_dist,
                          double* out_gpu_ms,
                          double* out_h2d_ms,
                          double* out_d2h_ms) {
    out_idx->assign((size_t)n_query * topk, -1);
    out_dist->assign((size_t)n_query * topk, std::numeric_limits<float>::infinity());
    if (out_gpu_ms) *out_gpu_ms = 0.0;
    if (out_h2d_ms) *out_h2d_ms = 0.0;
    if (out_d2h_ms) *out_d2h_ms = 0.0;
    if (!seg || seg->active_n <= 0) return;
    if (n_query > seg->max_batch || dim != seg->dim || topk > seg->max_topk) {
        throw std::runtime_error("delta_ivf_gpu_search workspace too small");
    }
    double h2d0 = now_ms();
    if (seg->use_u8) {
        if (!h_query_u8) throw std::runtime_error("u8 delta-IVF GPU search requires u8 queries");
        cuda_check(cudaMemcpy(seg->d_query_u8, h_query_u8, (size_t)n_query * dim * sizeof(uint8_t), cudaMemcpyHostToDevice), "delta ivf u8 query H2D");
    } else {
        cuda_check(cudaMemcpy(seg->d_query, h_query, (size_t)n_query * dim * sizeof(float), cudaMemcpyHostToDevice), "delta ivf query H2D");
    }
    cuda_check(cudaMemcpy(seg->d_cids, h_cids, (size_t)n_query * nprobe * sizeof(int), cudaMemcpyHostToDevice), "delta ivf cids H2D");
    double h2d_ms = now_ms() - h2d0;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    int threads = 256;
    size_t shmem = (size_t)threads * 16 * (sizeof(float) + sizeof(int));
    delta_ivf_gpu_topk_kernel<<<n_query, threads, shmem>>>(
        seg->d_query, seg->d_query_u8, seg->d_cids, seg->d_offsets, seg->d_counts,
        seg->d_vectors, seg->d_vectors_u8, seg->d_ids, n_query, nprobe, dim, seg->nlist,
        seg->use_u8 ? 1 : 0,
        topk, seg->d_out_idx, seg->d_out_dist);
    cuda_check(cudaGetLastError(), "delta ivf gpu kernel");
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    double d2h0 = now_ms();
    cuda_check(cudaMemcpy(out_idx->data(), seg->d_out_idx, (size_t)n_query * topk * sizeof(int), cudaMemcpyDeviceToHost), "delta ivf idx D2H");
    cuda_check(cudaMemcpy(out_dist->data(), seg->d_out_dist, (size_t)n_query * topk * sizeof(float), cudaMemcpyDeviceToHost), "delta ivf dist D2H");
    double d2h_ms = now_ms() - d2h0;
    if (out_gpu_ms) *out_gpu_ms += gpu_ms;
    if (out_h2d_ms) *out_h2d_ms += h2d_ms;
    if (out_d2h_ms) *out_d2h_ms += d2h_ms;
}

void delta_overlay_search(const Args& args,
                          DeltaFp16GemmWorkspace* fp16_ws,
                          const std::vector<DeltaIvfCpuSegment>& cpu_segments,
                          const std::vector<DeltaIvfGpuSegment>& gpu_segments,
                          const std::vector<float>& delta,
                          const std::vector<uint8_t>& delta_u8,
                          const float* h_query,
                          const uint8_t* h_query_u8,
                          const int* h_cids,
                          int n_query,
                          int dim,
                          int active_delta_n,
                          int base_id,
                          int topk,
                          std::vector<int>* out_idx,
                          std::vector<float>* out_dist,
                          double* out_stage_ms,
                          double* out_h2d_ms,
                          double* out_d2h_ms) {
    if (args.delta_search_mode == "flat-gpu-fp16") {
        delta_fp16_gemm_topk_search(fp16_ws,
                                    delta.data(),
                                    h_query,
                                    n_query,
                                    dim,
                                    active_delta_n,
                                    base_id,
                                    topk,
                                    out_idx,
                                    out_dist,
                                    out_stage_ms,
                                    out_h2d_ms,
                                    out_d2h_ms);
    } else if (args.delta_search_mode == "ivf-cpu") {
        if (!h_cids) throw std::runtime_error("ivf-cpu delta search requires coarse ids");
        const DeltaIvfCpuSegment* seg = find_delta_ivf_segment(cpu_segments, active_delta_n);
        if (!seg && active_delta_n > 0) {
            std::ostringstream oss;
            oss << "missing CPU delta-IVF segment active_delta_n=" << active_delta_n
                << " available=";
            for (const auto& s : cpu_segments) oss << s.active_n << ';';
            throw std::runtime_error(oss.str());
        }
        double cpu_ms = 0.0;
        delta_ivf_cpu_search(seg, h_query, h_query_u8, h_cids, n_query, args.nprobe, dim, topk,
                             std::min(args.threads, std::max(1, n_query)),
                             out_idx, out_dist, &cpu_ms);
        if (out_stage_ms) *out_stage_ms = cpu_ms;
        if (out_h2d_ms) *out_h2d_ms = 0.0;
        if (out_d2h_ms) *out_d2h_ms = 0.0;
    } else if (args.delta_search_mode == "ivf-gpu") {
        if (!h_cids) throw std::runtime_error("ivf-gpu delta search requires coarse ids");
        const DeltaIvfGpuSegment* seg = find_delta_ivf_gpu_segment(gpu_segments, active_delta_n);
        if (!seg && active_delta_n > 0) {
            std::ostringstream oss;
            oss << "missing GPU delta-IVF segment active_delta_n=" << active_delta_n
                << " available=";
            for (const auto& s : gpu_segments) oss << s.active_n << ';';
            throw std::runtime_error(oss.str());
        }
        delta_ivf_gpu_search(seg, h_query, h_query_u8, h_cids, n_query, args.nprobe, dim, topk,
                             out_idx, out_dist, out_stage_ms, out_h2d_ms, out_d2h_ms);
    } else {
        throw std::runtime_error("unknown delta_search_mode");
    }
}

__device__ __forceinline__ void fused_insert_topk(float* dist, int* idx, int topk, float d, int id) {
    if (id < 0) return;
    int pos = -1;
    for (int k = 0; k < topk; ++k) {
        if (d < dist[k] || (d == dist[k] && id < idx[k])) {
            pos = k;
            break;
        }
    }
    if (pos < 0) return;
    for (int k = topk - 1; k > pos; --k) {
        dist[k] = dist[k - 1];
        idx[k] = idx[k - 1];
    }
    dist[pos] = d;
    idx[pos] = id;
}

__global__ void fused_delta_partition_topk_kernel(const float* __restrict__ queries,
                                                  const float* __restrict__ delta,
                                                  int nq,
                                                  int delta_n,
                                                  int dim,
                                                  int topk,
                                                  int partitions,
                                                  float* __restrict__ partial_dist,
                                                  int* __restrict__ partial_idx) {
    constexpr int kMaxTopK = 10;
    int part = blockIdx.x;
    int q = blockIdx.y;
    if (q >= nq || part >= partitions || topk > kMaxTopK) return;

    int tid = threadIdx.x;
    long long start = ((long long)delta_n * part) / partitions;
    long long end = ((long long)delta_n * (part + 1)) / partitions;
    const float* qv = queries + (size_t)q * dim;

    float local_d[kMaxTopK];
    int local_i[kMaxTopK];
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        local_d[k] = INFINITY;
        local_i[k] = -1;
    }

    for (long long cand = start + tid; cand < end; cand += blockDim.x) {
        const float* dv = delta + (size_t)cand * dim;
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = qv[d] - dv[d];
            acc += diff * diff;
        }
        fused_insert_topk(local_d, local_i, topk, acc, (int)cand);
    }

    extern __shared__ unsigned char smem[];
    float* sdist = reinterpret_cast<float*>(smem);
    int* sidx = reinterpret_cast<int*>(sdist + (size_t)blockDim.x * topk);
    for (int k = 0; k < topk; ++k) {
        sdist[(size_t)tid * topk + k] = local_d[k];
        sidx[(size_t)tid * topk + k] = local_i[k];
    }
    __syncthreads();

    if (tid == 0) {
        float best_d[kMaxTopK];
        int best_i[kMaxTopK];
        #pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            best_d[k] = INFINITY;
            best_i[k] = -1;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            for (int k = 0; k < topk; ++k) {
                fused_insert_topk(best_d, best_i, topk,
                                  sdist[(size_t)t * topk + k],
                                  sidx[(size_t)t * topk + k]);
            }
        }
        size_t out = ((size_t)q * partitions + part) * topk;
        for (int k = 0; k < topk; ++k) {
            partial_dist[out + k] = best_d[k];
            partial_idx[out + k] = best_i[k];
        }
    }
}

__global__ void fused_delta_merge_partitions_kernel(const float* __restrict__ partial_dist,
                                                    const int* __restrict__ partial_idx,
                                                    int nq,
                                                    int partitions,
                                                    int topk,
                                                    float* __restrict__ out_dist,
                                                    int* __restrict__ out_idx) {
    constexpr int kMaxTopK = 10;
    int q = blockIdx.x;
    if (q >= nq || threadIdx.x != 0 || topk > kMaxTopK) return;

    float best_d[kMaxTopK];
    int best_i[kMaxTopK];
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        best_d[k] = INFINITY;
        best_i[k] = -1;
    }
    const size_t base = (size_t)q * partitions * topk;
    for (int p = 0; p < partitions; ++p) {
        for (int k = 0; k < topk; ++k) {
            size_t off = base + (size_t)p * topk + k;
            fused_insert_topk(best_d, best_i, topk, partial_dist[off], partial_idx[off]);
        }
    }
    for (int k = 0; k < topk; ++k) {
        out_dist[(size_t)q * topk + k] = best_d[k];
        out_idx[(size_t)q * topk + k] = best_i[k];
    }
}

struct FusedDeltaTopkWorkspace {
    int max_batch = 0;
    int dim = 0;
    int max_topk = 0;
    int max_partitions = 0;
    float* d_query = nullptr;
    float* d_partial_dist = nullptr;
    int* d_partial_idx = nullptr;
    float* d_out_dist = nullptr;
    int* d_out_idx = nullptr;

    void allocate(int max_batch_, int dim_, int max_topk_, int max_partitions_) {
        max_batch = max_batch_;
        dim = dim_;
        max_topk = max_topk_;
        max_partitions = max_partitions_;
        cuda_check(cudaMalloc(&d_query, (size_t)max_batch * dim * sizeof(float)), "cudaMalloc fused d_query");
        cuda_check(cudaMalloc(&d_partial_dist, (size_t)max_batch * max_partitions * max_topk * sizeof(float)),
                   "cudaMalloc fused partial dist");
        cuda_check(cudaMalloc(&d_partial_idx, (size_t)max_batch * max_partitions * max_topk * sizeof(int)),
                   "cudaMalloc fused partial idx");
        cuda_check(cudaMalloc(&d_out_dist, (size_t)max_batch * max_topk * sizeof(float)),
                   "cudaMalloc fused out dist");
        cuda_check(cudaMalloc(&d_out_idx, (size_t)max_batch * max_topk * sizeof(int)),
                   "cudaMalloc fused out idx");
    }

    void release() {
        if (d_query) cudaFree(d_query);
        if (d_partial_dist) cudaFree(d_partial_dist);
        if (d_partial_idx) cudaFree(d_partial_idx);
        if (d_out_dist) cudaFree(d_out_dist);
        if (d_out_idx) cudaFree(d_out_idx);
        *this = FusedDeltaTopkWorkspace{};
    }
};

void fused_delta_topk_search(FusedDeltaTopkWorkspace* ws,
                             const float* d_delta,
                             const float* h_query,
                             int nq,
                             int dim,
                             int delta_n,
                             int base_id,
                             int topk,
                             std::vector<int>* out_idx,
                             std::vector<float>* out_dist,
                             double* out_gpu_ms,
                             double* out_h2d_ms,
                             double* out_d2h_ms) {
    out_idx->assign((size_t)nq * topk, -1);
    out_dist->assign((size_t)nq * topk, std::numeric_limits<float>::infinity());
    if (out_gpu_ms) *out_gpu_ms = 0.0;
    if (out_h2d_ms) *out_h2d_ms = 0.0;
    if (out_d2h_ms) *out_d2h_ms = 0.0;
    if (!ws || !d_delta || delta_n <= 0) return;
    if (nq > ws->max_batch || dim != ws->dim || topk > ws->max_topk || topk > 10) {
        throw std::runtime_error("fused_delta_topk_search workspace too small");
    }

    const int partitions = std::max(1, std::min(ws->max_partitions, (delta_n + 4095) / 4096));
    const int threads = 256;
    const size_t q_bytes = (size_t)nq * dim * sizeof(float);
    const size_t out_i_bytes = (size_t)nq * topk * sizeof(int);
    const size_t out_d_bytes = (size_t)nq * topk * sizeof(float);

    double h2d0 = now_ms();
    cuda_check(cudaMemcpy(ws->d_query, h_query, q_bytes, cudaMemcpyHostToDevice),
               "fused delta query H2D");
    double h2d_ms = now_ms() - h2d0;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    dim3 grid(partitions, nq);
    size_t shmem = (size_t)threads * topk * (sizeof(float) + sizeof(int));
    fused_delta_partition_topk_kernel<<<grid, threads, shmem>>>(
        ws->d_query, d_delta, nq, delta_n, dim, topk, partitions,
        ws->d_partial_dist, ws->d_partial_idx);
    cuda_check(cudaGetLastError(), "fused delta partition kernel");
    fused_delta_merge_partitions_kernel<<<nq, 1>>>(
        ws->d_partial_dist, ws->d_partial_idx, nq, partitions, topk,
        ws->d_out_dist, ws->d_out_idx);
    cuda_check(cudaGetLastError(), "fused delta merge kernel");
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    double d2h0 = now_ms();
    cuda_check(cudaMemcpy(out_idx->data(), ws->d_out_idx, out_i_bytes, cudaMemcpyDeviceToHost),
               "fused delta idx D2H");
    cuda_check(cudaMemcpy(out_dist->data(), ws->d_out_dist, out_d_bytes, cudaMemcpyDeviceToHost),
               "fused delta dist D2H");
    double d2h_ms = now_ms() - d2h0;

    for (int& id : *out_idx) {
        if (id >= 0) id += base_id;
    }
    if (out_gpu_ms) *out_gpu_ms += gpu_ms;
    if (out_h2d_ms) *out_h2d_ms += h2d_ms;
    if (out_d2h_ms) *out_d2h_ms += d2h_ms;
}

void build_delta_vectors(const Args& args,
                         const ClusterDataset& dataset,
                         const std::vector<uint8_t>& base_u8,
                         int n,
                         const float* queries,
                         int nq,
                         int dim,
                         int max_delta_n,
                         const uint8_t* protected_ids,
                         bool build_float_delta,
                         std::vector<float>* delta,
                         std::vector<uint8_t>* delta_u8,
                         std::vector<int>* delta_locs = nullptr) {
    if (max_delta_n <= 0) return;
    const bool can_build_u8 = !args.use_fbin && delta_u8;
    if (build_float_delta) delta->resize((size_t)max_delta_n * dim);
    else delta->clear();
    if (can_build_u8) delta_u8->resize((size_t)max_delta_n * dim);
    if (delta_locs) delta_locs->assign((size_t)max_delta_n, -1);
    if (args.delta_mode == "query-copy") {
        if (!queries || nq <= 0) throw std::runtime_error("query-copy delta requires queries");
        const float far = 1000.0f;
        if (build_float_delta) std::fill(delta->begin(), delta->end(), far);
        if (can_build_u8) std::fill(delta_u8->begin(), delta_u8->end(), (uint8_t)255);
        int take = std::min(max_delta_n, nq);
        for (int i = 0; i < take; ++i) {
            const int qid = i;
            const float* src = queries + (size_t)qid * dim;
            float* dst = build_float_delta ? delta->data() + (size_t)i * dim : nullptr;
            uint8_t* dst_u8 = can_build_u8 ? delta_u8->data() + (size_t)i * dim : nullptr;
            for (int d = 0; d < dim; ++d) {
                if (dst) dst[d] = src[d];
                if (dst_u8) {
                    float v = std::max(0.0f, std::min(255.0f, src[d]));
                    dst_u8[d] = (uint8_t)std::lrintf(v);
                }
            }
        }
    } else if (args.delta_mode == "main-sample" || args.delta_mode == "random-main-sample") {
        int take = std::min(max_delta_n, n);
        int written = 0;
        long long stride = (long long)((args.seed % 1000000007ULL) | 1ULL);
        while (std::gcd(stride, (long long)n) != 1) stride += 2;
        long long start_loc = (long long)(args.seed % (uint64_t)n);
        const bool random_sample = (args.delta_mode == "random-main-sample");
        for (int step = 0; step < n && written < take; ++step) {
            int loc = random_sample ? (int)((start_loc + (long long)step * stride) % (long long)n) : step;
            int original_id = dataset.reordered_indices ? dataset.reordered_indices[loc] : loc;
            if (is_protected_id(protected_ids, n, original_id)) continue;
            if (!base_u8.empty()) {
                const uint8_t* src = base_u8.data() + (size_t)loc * dim;
                if (can_build_u8) {
                    std::memcpy(delta_u8->data() + (size_t)written * dim,
                                src,
                                (size_t)dim * sizeof(uint8_t));
                }
                if (build_float_delta) {
                    float* dst = delta->data() + (size_t)written * dim;
                    for (int d = 0; d < dim; ++d) dst[d] = (float)src[d];
                }
            } else {
                if (build_float_delta) {
                    std::memcpy(delta->data() + (size_t)written * dim,
                                dataset.reordered_data + (size_t)loc * dim,
                                (size_t)dim * sizeof(float));
                }
            }
            if (delta_locs) (*delta_locs)[(size_t)written] = loc;
            ++written;
        }
        for (int i = written; i < max_delta_n; ++i) {
            if (build_float_delta) {
                std::fill(delta->begin() + (size_t)i * dim,
                          delta->begin() + (size_t)(i + 1) * dim,
                          1000.0f);
            }
            if (can_build_u8) {
                std::fill(delta_u8->begin() + (size_t)i * dim,
                          delta_u8->begin() + (size_t)(i + 1) * dim,
                          (uint8_t)255);
            }
        }
        std::printf("[DELTA] %s copied %d vectors (requested=%d, gt_safe=%d, stride=%lld)\n",
                    args.delta_mode.c_str(), written, max_delta_n,
                    protected_ids ? 1 : 0, stride);

    } else if (args.delta_mode == "random-u8") {
        if (!can_build_u8) {
            throw std::runtime_error("random-u8 delta mode requires a uint8 dataset");
        }
        std::mt19937_64 rng(args.seed);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        for (size_t i = 0; i < delta_u8->size(); ++i) {
            (*delta_u8)[i] = (uint8_t)byte_dist(rng);
        }
        if (build_float_delta) {
            for (size_t i = 0; i < delta_u8->size(); ++i) {
                (*delta)[i] = (float)(*delta_u8)[i];
            }
        }
        std::printf("[DELTA] random-u8 generated %d vectors x %d dims seed=%llu\n",
                    max_delta_n, dim, (unsigned long long)args.seed);
    } else {
        float far = args.use_fbin ? 1000.0f : 255.0f;
        if (build_float_delta) std::fill(delta->begin(), delta->end(), far);
        if (can_build_u8) std::fill(delta_u8->begin(), delta_u8->end(), (uint8_t)255);
    }
}

void search_delta_gpu_fast(const CoarseHandle* delta_handle,
                           PersistentTiledCoarseWorkspace* delta_ws,
                           const float* q,
                           int nq,
                           int delta_n,
                           int base_id,
                           int topk,
                           std::vector<int>* out_idx,
                           std::vector<float>* out_dist,
                           double* out_gpu_ms,
                           double* out_h2d_ms,
                           double* out_d2h_ms) {
    out_idx->assign((size_t)nq * topk, -1);
    out_dist->assign((size_t)nq * topk, std::numeric_limits<float>::infinity());
    if (delta_n <= 0 || !delta_handle) return;

    int dk = std::min(topk, delta_n);
    std::vector<int> local_idx((size_t)nq * dk, -1);
    std::vector<float> local_dist((size_t)nq * dk, std::numeric_limits<float>::infinity());
    double tco = 0.0, th2d = 0.0, td2h = 0.0;
    persistent_tiled_l2_search(delta_handle, delta_ws, q, nq, dk, delta_n,
                               local_idx.data(), local_dist.data(),
                               &tco, &th2d, &td2h);
    if (out_gpu_ms) *out_gpu_ms += tco;
    if (out_h2d_ms) *out_h2d_ms += th2d;
    if (out_d2h_ms) *out_d2h_ms += td2h;

    for (int qi = 0; qi < nq; ++qi) {
        for (int k = 0; k < dk; ++k) {
            int lid = local_idx[(size_t)qi * dk + k];
            if (lid >= 0 && lid < delta_n) {
                (*out_idx)[(size_t)qi * topk + k] = base_id + lid;
                (*out_dist)[(size_t)qi * topk + k] = local_dist[(size_t)qi * dk + k];
            }
        }
    }
}

bool contains_id(const std::vector<int>& xs, int id) {
    return std::find(xs.begin(), xs.end(), id) != xs.end();
}

std::vector<int> update_targets_for_query(int q,
                                          int k,
                                          const int32_t* gt_row,
                                          int gt_k,
                                          int base_n,
                                          int nq,
                                          int delta_n,
                                          const std::string& delta_mode,
                                          double delete_ratio,
                                          bool gt_safe_updates) {
    std::vector<int> target;
    target.reserve((size_t)k);

    if (delta_mode == "query-copy" && delta_n > 0 && q < delta_n) {
        target.push_back(base_n + q);
    }

    if (gt_row) {
        for (int j = 0; j < gt_k && (int)target.size() < k; ++j) {
            int id = (int)gt_row[j];
            if (id < 0) continue;
            if (!gt_safe_updates && is_deleted_id(id, delete_ratio)) continue;
            if (!contains_id(target, id)) target.push_back(id);
        }
    }
    return target;
}

double recall_at_k_update(const int* idx,
                          const int32_t* gt,
                          int nq,
                          int k,
                          int row_k,
                          int gt_k,
                          int base_n,
                          int delta_n,
                          const std::string& delta_mode,
                          double delete_ratio,
                          bool gt_safe_updates) {
    if (!gt || nq <= 0 || k <= 0) return -1.0;
    if (k > row_k) return -1.0;
    long long hit = 0;
    long long total = 0;
    for (int q = 0; q < nq; ++q) {
        const int* row = idx + (size_t)q * row_k;
        const int32_t* grow = gt + (size_t)q * gt_k;
        std::vector<int> target = update_targets_for_query(
            q, k, grow, gt_k, base_n, nq, delta_n, delta_mode, delete_ratio,
            gt_safe_updates);
        total += (long long)target.size();
        for (int i = 0; i < k && i < row_k; ++i) {
            int cand = row[i];
            if (contains_id(target, cand)) {
                ++hit;
            }
        }
    }
    return total ? (double)hit / (double)total : 0.0;
}

double recall_at_k_insert_aware(const int* idx,
                                int nq,
                                int k,
                                int row_k,
                                const int* old_gt_ids,
                                const float* old_gt_dists,
                                int old_gt_k,
                                const int* delta_idx,
                                const float* delta_dists,
                                int delta_k) {
    if (!idx || !old_gt_ids || !old_gt_dists || nq <= 0 || k <= 0) return -1.0;
    if (k > row_k || k > old_gt_k) return -1.0;
    long long hit = 0;
    long long total = 0;
    std::vector<Candidate> target;
    std::vector<int> target_idx((size_t)k, -1);
    std::vector<float> target_dist((size_t)k, std::numeric_limits<float>::infinity());
    for (int q = 0; q < nq; ++q) {
        target.clear();
        target.reserve((size_t)old_gt_k + (size_t)delta_k);
        for (int j = 0; j < old_gt_k; ++j) {
            push_candidate(target,
                           old_gt_dists[(size_t)q * old_gt_k + j],
                           old_gt_ids[(size_t)q * old_gt_k + j]);
        }
        if (delta_idx && delta_dists && delta_k > 0) {
            for (int j = 0; j < delta_k; ++j) {
                push_candidate(target,
                               delta_dists[(size_t)q * delta_k + j],
                               delta_idx[(size_t)q * delta_k + j]);
            }
        }
        finalize_topk(target, k, target_idx.data(), target_dist.data());
        total += k;
        const int* row = idx + (size_t)q * row_k;
        for (int i = 0; i < k && i < row_k; ++i) {
            if (contains_id(target_idx, row[i])) ++hit;
        }
    }
    return total ? (double)hit / (double)total : 0.0;
}

void write_insert_gt_header_if_needed(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    bool need = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    in.close();
    if (!need) return;
    std::ofstream out(path);
    out << "dataset,n,nq,topk,delta_n,insert_ratio,batch_size,total_ms,"
        << "delta_gpu_stage_ms,delta_h2d_ms,delta_d2h_ms,"
        << "affected_queries,affected_query_ratio,insert_slots_top10,"
        << "insert_slot_ratio,best_insert_rank_min,best_insert_rank_avg\n";
}

void summarize_insert_gt(const std::string& dataset_name,
                         int n,
                         int nq,
                         int topk,
                         int delta_n,
                         int batch_size,
                         double total_ms,
                         double delta_stage_ms,
                         double delta_h2d_ms,
                         double delta_d2h_ms,
                         const std::vector<int>& static_gt_ids,
                         const std::vector<float>& static_gt_dists,
                         const std::vector<int>& delta_idx,
                         const std::vector<float>& delta_dists,
                         const std::string& out_path) {
    long long affected_queries = 0;
    long long insert_slots = 0;
    long long rank_sum = 0;
    int rank_min = topk + 1;
    std::vector<Candidate> target;
    std::vector<int> target_idx((size_t)topk, -1);
    std::vector<float> target_dist((size_t)topk, std::numeric_limits<float>::infinity());
    for (int q = 0; q < nq; ++q) {
        target.clear();
        target.reserve((size_t)topk * 2);
        for (int j = 0; j < topk; ++j) {
            push_candidate(target,
                           static_gt_dists[(size_t)q * topk + j],
                           static_gt_ids[(size_t)q * topk + j]);
        }
        if (delta_n > 0) {
            for (int j = 0; j < topk; ++j) {
                push_candidate(target,
                               delta_dists[(size_t)q * topk + j],
                               delta_idx[(size_t)q * topk + j]);
            }
        }
        finalize_topk(target, topk, target_idx.data(), target_dist.data());
        int q_insert = 0;
        for (int j = 0; j < topk; ++j) {
            if (target_idx[(size_t)j] >= n) {
                ++q_insert;
                ++insert_slots;
                rank_sum += (j + 1);
                rank_min = std::min(rank_min, j + 1);
            }
        }
        if (q_insert > 0) ++affected_queries;
    }
    double affected_ratio = nq ? (double)affected_queries / (double)nq : 0.0;
    double slot_ratio = (nq && topk) ? (double)insert_slots / (double)(nq * topk) : 0.0;
    double rank_avg = insert_slots ? (double)rank_sum / (double)insert_slots : 0.0;
    if (rank_min == topk + 1) rank_min = -1;

    write_insert_gt_header_if_needed(out_path);
    std::ofstream out(out_path, std::ios::app);
    out << dataset_name << ',' << n << ',' << nq << ',' << topk << ','
        << delta_n << ',' << ((double)delta_n / (double)n) << ','
        << batch_size << ',' << total_ms << ','
        << delta_stage_ms << ',' << delta_h2d_ms << ',' << delta_d2h_ms << ','
        << affected_queries << ',' << affected_ratio << ','
        << insert_slots << ',' << slot_ratio << ','
        << rank_min << ',' << rank_avg << '\n';
    out.close();
    std::printf("[INSERT-GT] delta_n=%d affected_queries=%lld/%d insert_slots=%lld/%lld total=%.2f ms\n",
                delta_n, affected_queries, nq, insert_slots, (long long)nq * topk, total_ms);
}

struct ReorderCacheMeta {
    char magic[8];
    int32_t version;
    int32_t use_fbin;
    int32_t n;
    int32_t dim;
    int32_t nlist;
    uint64_t assign_size;
    int64_t assign_mtime;
    uint64_t centroids_size;
    int64_t centroids_mtime;
    uint64_t base_size;
    int64_t base_mtime;
    uint64_t vec_bytes;
};

bool file_stat(const std::string& path, uint64_t* size, int64_t* mtime) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    if (size) *size = (uint64_t)st.st_size;
    if (mtime) *mtime = (int64_t)st.st_mtime;
    return true;
}

void ensure_dir(const std::string& dir) {
    if (dir.empty()) return;
    std::string cmd = "mkdir -p " + dir;
    int rc = std::system(cmd.c_str());
    if (rc != 0) throw std::runtime_error("mkdir failed for " + dir);
}

std::string cache_prefix(const Args& args, int n, int dim) {
    std::ostringstream oss;
    oss << args.reorder_cache_dir << '/' << args.dataset
        << "_n" << n << "_d" << dim
        << "_nlist" << args.nlist
        << (args.use_fbin ? "_f32" : "_u8");
    return oss.str();
}

bool read_file_exact(const std::string& path, void* dst, size_t bytes) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    size_t got = std::fread(dst, 1, bytes, fp);
    std::fclose(fp);
    return got == bytes;
}

void write_file_exact(const std::string& path, const void* src, size_t bytes) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("cannot write " + path);
    size_t wrote = std::fwrite(src, 1, bytes, fp);
    std::fclose(fp);
    if (wrote != bytes) throw std::runtime_error("short write " + path);
}

bool load_centroids_for_dataset(const std::string& path, ClusterDataset* dataset, int nlist, int dim) {
    FILE* fc = std::fopen(path.c_str(), "rb");
    if (!fc) return false;
    int32_t file_nlist = 0, file_dim = 0;
    if (std::fread(&file_nlist, 4, 1, fc) != 1 ||
        std::fread(&file_dim, 4, 1, fc) != 1 ||
        file_nlist != nlist || file_dim != dim) {
        std::fclose(fc);
        return false;
    }
    dataset->centroids = (float*)memalign(64, (size_t)nlist * dim * sizeof(float));
    if (!dataset->centroids) {
        std::fclose(fc);
        throw std::runtime_error("OOM loading cached centroids");
    }
    size_t need = (size_t)nlist * dim;
    size_t got = std::fread(dataset->centroids, sizeof(float), need, fc);
    std::fclose(fc);
    return got == need;
}

ReorderCacheMeta make_reorder_cache_meta(const Args& args,
                                         const std::string& base_path,
                                         int n,
                                         int dim) {
    ReorderCacheMeta m{};
    std::memcpy(m.magic, "IVFTRC1", 7);
    m.version = 1;
    m.use_fbin = args.use_fbin ? 1 : 0;
    m.n = n;
    m.dim = dim;
    m.nlist = args.nlist;
    file_stat(args.assign, &m.assign_size, &m.assign_mtime);
    file_stat(args.centroids, &m.centroids_size, &m.centroids_mtime);
    file_stat(base_path, &m.base_size, &m.base_mtime);
    m.vec_bytes = (uint64_t)n * (uint64_t)dim * (args.use_fbin ? sizeof(float) : sizeof(uint8_t));
    return m;
}

bool same_reorder_cache_meta(const ReorderCacheMeta& a, const ReorderCacheMeta& b) {
    return std::memcmp(a.magic, "IVFTRC1", 7) == 0 &&
           a.version == b.version &&
           a.use_fbin == b.use_fbin &&
           a.n == b.n &&
           a.dim == b.dim &&
           a.nlist == b.nlist &&
           a.assign_size == b.assign_size &&
           a.assign_mtime == b.assign_mtime &&
           a.centroids_size == b.centroids_size &&
           a.centroids_mtime == b.centroids_mtime &&
           a.base_size == b.base_size &&
           a.base_mtime == b.base_mtime &&
           a.vec_bytes == b.vec_bytes;
}

bool load_reorder_cache(const Args& args,
                        const std::string& base_path,
                        ClusterDataset* dataset,
                        std::vector<uint8_t>* h_base_u8,
                        int n,
                        int dim) {
    if (args.reorder_cache_dir.empty()) return false;
    const std::string p = cache_prefix(args, n, dim);
    const std::string meta_path = p + ".meta";
    const std::string vec_path = p + ".vec";
    const std::string ids_path = p + ".ids";
    const std::string counts_path = p + ".counts";
    const std::string offsets_path = p + ".offsets";

    ReorderCacheMeta expected = make_reorder_cache_meta(args, base_path, n, dim);
    ReorderCacheMeta found{};
    if (!read_file_exact(meta_path, &found, sizeof(found))) return false;
    if (!same_reorder_cache_meta(found, expected)) {
        std::printf("[REORDER-CACHE] stale metadata, rebuilding: %s\n", meta_path.c_str());
        return false;
    }

    dataset->reordered_data = nullptr;
    dataset->reordered_indices = nullptr;
    dataset->centroids = nullptr;
    dataset->cluster_info.k = args.nlist;
    dataset->cluster_info.counts = (int*)memalign(64, (size_t)args.nlist * sizeof(int));
    dataset->cluster_info.offsets = (long long*)memalign(64, (size_t)args.nlist * sizeof(long long));
    dataset->reordered_indices = (int*)memalign(64, (size_t)n * sizeof(int));
    dataset->n_total_vectors = n;
    dataset->vector_dim = dim;
    if (!dataset->cluster_info.counts || !dataset->cluster_info.offsets || !dataset->reordered_indices) {
        throw std::runtime_error("OOM loading reorder cache metadata");
    }
    if (!load_centroids_for_dataset(args.centroids, dataset, args.nlist, dim) ||
        !read_file_exact(counts_path, dataset->cluster_info.counts, (size_t)args.nlist * sizeof(int)) ||
        !read_file_exact(offsets_path, dataset->cluster_info.offsets, (size_t)args.nlist * sizeof(long long)) ||
        !read_file_exact(ids_path, dataset->reordered_indices, (size_t)n * sizeof(int))) {
        std::printf("[REORDER-CACHE] missing/corrupt sidecar, rebuilding: %s\n", p.c_str());
        return false;
    }
    if (args.use_fbin) {
        dataset->reordered_data = (float*)memalign(64, (size_t)n * dim * sizeof(float));
        if (!dataset->reordered_data) throw std::runtime_error("OOM loading cached reordered f32");
        if (!read_file_exact(vec_path, dataset->reordered_data, (size_t)expected.vec_bytes)) {
            std::printf("[REORDER-CACHE] missing/corrupt f32 vectors, rebuilding: %s\n", vec_path.c_str());
            return false;
        }
    } else {
        h_base_u8->resize((size_t)n * dim);
        if (!read_file_exact(vec_path, h_base_u8->data(), (size_t)expected.vec_bytes)) {
            std::printf("[REORDER-CACHE] missing/corrupt u8 vectors, rebuilding: %s\n", vec_path.c_str());
            return false;
        }
    }
    std::printf("[REORDER-CACHE] loaded %s (%.2f GB vectors)\n",
                p.c_str(), (double)expected.vec_bytes / 1e9);
    return true;
}

void save_reorder_cache(const Args& args,
                        const std::string& base_path,
                        const ClusterDataset& dataset,
                        const std::vector<uint8_t>& h_base_u8,
                        int n,
                        int dim) {
    if (args.reorder_cache_dir.empty()) return;
    ensure_dir(args.reorder_cache_dir);
    const std::string p = cache_prefix(args, n, dim);
    const std::string meta_path = p + ".meta";
    const std::string vec_path = p + ".vec";
    const std::string ids_path = p + ".ids";
    const std::string counts_path = p + ".counts";
    const std::string offsets_path = p + ".offsets";
    ReorderCacheMeta meta = make_reorder_cache_meta(args, base_path, n, dim);

    double t0 = now_ms();
    if (args.use_fbin) {
        write_file_exact(vec_path, dataset.reordered_data, (size_t)meta.vec_bytes);
    } else {
        write_file_exact(vec_path, h_base_u8.data(), (size_t)meta.vec_bytes);
    }
    write_file_exact(ids_path, dataset.reordered_indices, (size_t)n * sizeof(int));
    write_file_exact(counts_path, dataset.cluster_info.counts, (size_t)args.nlist * sizeof(int));
    write_file_exact(offsets_path, dataset.cluster_info.offsets, (size_t)args.nlist * sizeof(long long));
    write_file_exact(meta_path, &meta, sizeof(meta));
    std::printf("[REORDER-CACHE] saved %s in %.2f ms (%.2f GB vectors)\n",
                p.c_str(), now_ms() - t0, (double)meta.vec_bytes / 1e9);
}

void write_header_if_needed(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    bool need = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    in.close();
    if (!need) return;
    std::ofstream out(path);
    out << "dataset,n,nq,nlist,nprobe,batch_size,threads,topk,main_topk,"
        << "delta_n,delete_count,insert_ratio,delete_ratio,delta_mode,repeat,total_ms,"
        << "pipeline,p50_ms,p99_ms,coarse_ms,h2d_ms,d2h_ms,main_fine_ms,delta_ms,delta_h2d_ms,"
        << "delta_d2h_ms,merge_ms,qps,recall1,recall10,recall100\n";
}

void write_mixed_header_if_needed(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    bool need = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    in.close();
    if (!need) return;
    std::ofstream out(path);
    out << "dataset,n,nq,nlist,nprobe,batch_size,threads,topk,"
        << "target_delta_n,target_insert_ratio,target_delete_ratio,"
        << "update_steps,query_rounds,seed,repeat,num_query_batches,"
        << "pipeline,total_query_ms,query_qps,p50_ms,p90_ms,p99_ms,"
        << "recall1,recall10,avg_delta_n,avg_insert_ratio,avg_delete_ratio,"
        << "insert_publish_ms,delete_publish_ms,coarse_ms,h2d_ms,d2h_ms,"
        << "main_fine_ms,delta_ms,delta_h2d_ms,delta_d2h_ms,merge_ms\n";
}

double percentile(std::vector<double> xs, double p) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    double pos = (p / 100.0) * (double)(xs.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    if (lo == hi) return xs[lo];
    double t = pos - (double)lo;
    return xs[lo] * (1.0 - t) + xs[hi] * t;
}

void accumulate_recall_at_k_update(const int* idx,
                                   const int32_t* gt,
                                   int qb,
                                   int nb,
                                   int global_nq,
                                   int k,
                                   int row_k,
                                   int gt_k,
                                   int base_n,
                                   int delta_n,
                                   const std::string& delta_mode,
                                   double delete_ratio,
                                   bool gt_safe_updates,
                                   long long* hit,
                                   long long* total) {
    if (!gt || nb <= 0 || k <= 0 || k > row_k) return;
    for (int q = 0; q < nb; ++q) {
        int global_q = qb + q;
        const int* row = idx + (size_t)q * row_k;
        const int32_t* grow = gt + (size_t)global_q * gt_k;
        std::vector<int> target = update_targets_for_query(
            global_q, k, grow, gt_k, base_n, global_nq, delta_n, delta_mode,
            delete_ratio, gt_safe_updates);
        *total += (long long)target.size();
        for (int i = 0; i < k && i < row_k; ++i) {
            if (contains_id(target, row[i])) ++(*hit);
        }
    }
}

struct DeltaHandleEntry {
    int delta_n = 0;
    CoarseHandle handle{};
};

const CoarseHandle* find_delta_handle(const std::vector<DeltaHandleEntry>& entries, int delta_n) {
    for (const auto& e : entries) {
        if (e.delta_n == delta_n) return &e.handle;
    }
    return nullptr;
}

struct MainFineJob {
    int qb = 0;
    int nb = 0;
    int topk = 0;
    std::vector<int> cids;
    std::vector<int> idx;
    std::vector<float> dist;
    std::vector<int> delta_idx;
    std::vector<float> delta_dist;
    const uint8_t* deleted_local_mask = nullptr;
    double fine_ms = 0.0;
};

void run_main_fine_job(const Args& args,
                       const ClusterDataset& dataset,
                       const std::vector<uint8_t>& h_base_u8,
                       const std::vector<uint8_t>& h_query_u8,
                       const float* h_query,
                       int n,
                       int dim,
                       bool has_avx512f,
                       bool has_avx512bw,
                       MainFineJob* job) {
    if (!job || job->nb <= 0 || job->topk <= 0) return;
    job->idx.assign((size_t)job->nb * job->topk, -1);
    job->dist.assign((size_t)job->nb * job->topk, std::numeric_limits<float>::infinity());

    const double t0 = now_ms();
    if (args.use_fbin) {
        if (has_avx512f) {
            double tf = 0.0;
            long long nfma = 0;
            fine_search_cpu(
                CPU_FINE_V3_TOUCHED,
                dataset.reordered_data,
                nullptr,
                nullptr,
                dataset.cluster_info.offsets,
                dataset.cluster_info.counts,
                dataset.reordered_indices,
                args.nlist,
                dim,
                h_query + (size_t)job->qb * dim,
                job->cids.data(),
                job->nb,
                args.nprobe,
                job->topk,
                L2_DISTANCE_MODE,
                args.threads,
                job->idx.data(),
                job->dist.data(),
                &tf,
                &nfma);
        } else {
            fine_search_scalar_touched_float(
                dataset.reordered_data,
                dataset.cluster_info.offsets,
                dataset.cluster_info.counts,
                dataset.reordered_indices,
                args.nlist,
                dim,
                h_query + (size_t)job->qb * dim,
                job->cids.data(),
                job->nb,
                args.nprobe,
                job->topk,
                job->idx.data(),
                job->dist.data());
        }
    } else {
        std::vector<uint8_t> q_u8((size_t)job->nb * dim);
        std::memcpy(q_u8.data(), h_query_u8.data() + (size_t)job->qb * dim,
                    (size_t)job->nb * dim);
        if (job->deleted_local_mask) {
            ivftensor::cpu_fine::cpu_fine_kernel_v3_u8_touched_masked(
                h_base_u8.data(),
                dataset.cluster_info.offsets,
                dataset.cluster_info.counts,
                q_u8.data(),
                job->cids.data(),
                job->nb,
                dim,
                args.nlist,
                args.nprobe,
                job->topk,
                args.threads,
                job->deleted_local_mask,
                job->idx.data(),
                job->dist.data());
        } else {
            ivftensor::cpu_fine::cpu_fine_kernel_v3_u8_touched(
                h_base_u8.data(),
                dataset.cluster_info.offsets,
                dataset.cluster_info.counts,
                q_u8.data(),
                job->cids.data(),
                job->nb,
                dim,
                args.nlist,
                args.nprobe,
                job->topk,
                args.threads,
                job->idx.data(),
                job->dist.data());
        }
        for (size_t i = 0; i < job->idx.size(); ++i) {
            int loc = job->idx[i];
            job->idx[i] = (loc >= 0 && loc < n) ? dataset.reordered_indices[loc] : -1;
        }
    }
    job->fine_ms = now_ms() - t0;
}

int count_query_copy_top1_miss(const std::vector<int>& delta_idx,
                               int nb,
                               int topk,
                               int qb,
                               int nq,
                               int base_n,
                               int delta_n,
                               const std::string& delta_mode) {
    if (delta_mode != "query-copy" || delta_n <= 0) return 0;
    int miss = 0;
    for (int q = 0; q < nb; ++q) {
        int global_q = qb + q;
        int expected_local = global_q;
        if (expected_local >= delta_n) continue;
        int expected_id = base_n + expected_local;
        bool found = false;
        for (int k = 0; k < topk; ++k) {
            if (delta_idx[(size_t)q * topk + k] == expected_id) {
                found = true;
                break;
            }
        }
        if (!found) ++miss;
    }
    return miss;
}

enum class EventType { Query, Insert, Delete };

struct Event {
    EventType type;
    int qb = 0;
    int qn = 0;
    int insert_to = 0;
    double delete_to = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        Args args = parse_args(argc, argv);
        cuda_check(cudaSetDevice(args.cuda_device), "cudaSetDevice");
        std::printf("[CUDA] using device %d\n", args.cuda_device);
        omp_set_num_threads(args.threads);

        std::string sx = strip_dataset_prefix(args.scale);
        long long cap = parse_scale_n(sx);
        if (cap <= 0) throw std::runtime_error("cannot parse scale " + args.scale);

        std::string base_path = args.data_dir + "/base_" + sx + (args.use_fbin ? ".fbin" : ".bin");
        std::string query_path = args.data_dir + (args.use_fbin ? "/query.fbin" : "/query.bin");
        std::string gt_path = args.data_dir + "/groundtruth_" + args.scale + ".bin";

        int hdr_n = 0, dim = 0;
        read_bin_header(base_path, &hdr_n, &dim);
        int n = (int)std::min<long long>((long long)hdr_n, cap);
        int max_delta_n = 0;
        for (int dn : args.delta_ns) max_delta_n = std::max(max_delta_n, dn);
        std::printf("[CFG] dataset=%s base=%s n=%d dim=%d nlist=%d nprobe=%d topk=%d main_topk=%d\n",
                    args.dataset.c_str(), base_path.c_str(), n, dim,
                    args.nlist, args.nprobe, args.topk, args.main_overfetch);
        std::printf("[CFG] batch_sizes=");
        for (size_t i = 0; i < args.batch_sizes.size(); ++i) {
            std::printf("%s%d", i ? "," : "", args.batch_sizes[i]);
        }
        std::printf(" delta_ns=");
        for (size_t i = 0; i < args.delta_ns.size(); ++i) {
            std::printf("%s%d", i ? "," : "", args.delta_ns[i]);
        }
        std::printf(" delete_ratios=");
        for (size_t i = 0; i < args.delete_ratios.size(); ++i) {
            std::printf("%s%.6f", i ? "," : "", args.delete_ratios[i]);
        }
        std::printf(" delta_mode=%s delta_search=%s use_fbin=%d gt_safe_updates=%d delete_gt_safe=%d insert_aware_recall=%d\n",
                    args.delta_mode.c_str(), args.delta_search_mode.c_str(), args.use_fbin ? 1 : 0,
                    args.gt_safe_updates ? 1 : 0,
                    args.delete_gt_safe ? 1 : 0,
                    args.insert_aware_recall ? 1 : 0);

        int nq = 0, qdim = 0;
        float* h_query = args.use_fbin
            ? fvecs_io::read_diskann_fbin(query_path, &nq, &qdim, -1)
            : fvecs_io::read_diskann_u8bin(query_path, &nq, &qdim, -1);
        if (qdim != dim) throw std::runtime_error("query dim mismatch");
        if (args.nq_limit > 0 && args.nq_limit < nq) nq = args.nq_limit;
        std::vector<uint8_t> h_query_u8;
        if (!args.use_fbin) make_query_u8(h_query, nq, dim, &h_query_u8);

        int ngt = 0, kgt = 0;
        int32_t* h_gt = nullptr;
        FILE* fgt = std::fopen(gt_path.c_str(), "rb");
        if (fgt) {
            std::fclose(fgt);
            h_gt = fvecs_io::read_diskann_gt(gt_path, &ngt, &kgt);
            if (ngt < nq) throw std::runtime_error("gt has fewer queries than requested");
            std::printf("[GT] %s n=%d k=%d\n", gt_path.c_str(), ngt, kgt);
        } else {
            std::printf("[GT] missing %s; recall fields will be -1\n", gt_path.c_str());
        }
        std::vector<uint8_t> gt_protect_ids;
        if (args.gt_safe_updates || args.delete_gt_safe) {
            build_gt_protect_mask(h_gt, nq, kgt, n, &gt_protect_ids);
            size_t protected_count = 0;
            for (uint8_t v : gt_protect_ids) protected_count += (v != 0);
            std::printf("[GT-SAFE] protecting %zu original GT ids (gt_safe_updates=%d delete_gt_safe=%d)\n",
                        protected_count, args.gt_safe_updates ? 1 : 0, args.delete_gt_safe ? 1 : 0);
        }

        ClusterDataset dataset;
        std::vector<uint8_t> h_base_u8;
        double t_load0 = now_ms();
        bool cache_hit = load_reorder_cache(args, base_path, &dataset, &h_base_u8, n, dim);
        if (!cache_hit) {
            if (args.use_fbin) {
                dataset.init_from_dump_stream_fbin(
                    args.centroids.c_str(), args.assign.c_str(), base_path.c_str(),
                    n, dim, args.nlist);
            } else {
                dataset.init_from_dump_stream_u8_direct(
                    args.centroids.c_str(), args.assign.c_str(), base_path.c_str(),
                    n, dim, args.nlist, h_base_u8);
            }
            save_reorder_cache(args, base_path, dataset, h_base_u8, n, dim);
        }
        double t_load1 = now_ms();
        std::printf("[INDEX] loaded/reordered in %.2f ms cache_hit=%d\n",
                    t_load1 - t_load0, cache_hit ? 1 : 0);

        std::vector<int> static_gt_ids;
        std::vector<float> static_gt_dists;
        std::vector<int> static_gt_pool_ids;
        std::vector<float> static_gt_pool_dists;
        int static_gt_pool_k = 0;
        if (args.insert_aware_recall) {
            build_static_gt_topk(dataset,
                                 h_base_u8,
                                 h_query,
                                 h_query_u8,
                                 args.use_fbin,
                                 h_gt,
                                 nq,
                                 kgt,
                                 n,
                                 dim,
                                 args.topk,
                                 args.threads,
                                 &static_gt_ids,
                                 &static_gt_dists);
            static_gt_pool_k = std::min(kgt, 1000);
            build_static_gt_topk(dataset,
                                 h_base_u8,
                                 h_query,
                                 h_query_u8,
                                 args.use_fbin,
                                 h_gt,
                                 nq,
                                 kgt,
                                 n,
                                 dim,
                                 static_gt_pool_k,
                                 args.threads,
                                 &static_gt_pool_ids,
                                 &static_gt_pool_dists);
        }

        float* d_centers = nullptr;
        cuda_check(cudaMalloc(&d_centers, (size_t)args.nlist * dim * sizeof(float)),
                   "cudaMalloc d_centers");
        cuda_check(cudaMemcpy(d_centers, dataset.centroids,
                              (size_t)args.nlist * dim * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_centers");
        CoarseHandle ch;
        coarse_handle_init(&ch, d_centers, args.nlist, dim);

        int max_batch_size = 0;
        for (int bs : args.batch_sizes) max_batch_size = std::max(max_batch_size, bs);
        if (max_batch_size <= 0) max_batch_size = args.batch_size;
        const int delta_tile_n = 262144;
        const int delta_topk = std::max(args.topk, args.delta_topk);
        PersistentTiledCoarseWorkspace main_ws;
        std::printf("[COARSE] optimized main coarse_search path: max_bs=%d nprobe=%d\n",
                    max_batch_size, args.nprobe);

        std::vector<float> delta;
        std::vector<uint8_t> delta_u8;
        std::vector<int> delta_locs;
        const bool need_flat_delta_for_recall =
            args.insert_aware_recall || args.gt_only || args.delta_search_mode == "flat-gpu-fp16";
        const bool build_float_delta = args.use_fbin || need_flat_delta_for_recall;
        build_delta_vectors(args, dataset, h_base_u8, n, h_query, nq, dim, max_delta_n,
                            (args.gt_safe_updates && !gt_protect_ids.empty()) ? gt_protect_ids.data() : nullptr,
                            build_float_delta, &delta, &delta_u8, &delta_locs);
        float* d_delta = nullptr;
        std::vector<DeltaHandleEntry> delta_handles;
        CoarseHandle delta_ch{};
        bool delta_ch_valid = false;
        PersistentTiledCoarseWorkspace delta_ws;
        DeltaFp16GemmWorkspace delta_fp16_ws;
        std::vector<DeltaIvfCpuSegment> delta_ivf_cpu_segments;
        std::vector<DeltaIvfGpuSegment> delta_ivf_gpu_segments;
        if (!delta.empty() || !delta_u8.empty()) {
            if (!delta.empty()) {
                std::printf("[DELTA] host delta vectors fp32: max=%d x %d %.2f MB\n",
                            max_delta_n, dim,
                            (double)delta.size() * sizeof(float) / (1024.0 * 1024.0));
            }
            if (!delta_u8.empty()) {
                std::printf("[DELTA] host delta vectors u8: max=%d x %d %.2f MB\n",
                            max_delta_n, dim,
                            (double)delta_u8.size() * sizeof(uint8_t) / (1024.0 * 1024.0));
            }
            if (need_flat_delta_for_recall && max_delta_n > 0) {
                if (delta.empty()) throw std::runtime_error("flat delta path requires fp32 delta buffer");
                cuda_check(cudaMalloc(&d_delta, delta.size() * sizeof(float)),
                           "cudaMalloc d_delta");
                cuda_check(cudaMemcpy(d_delta, delta.data(), delta.size() * sizeof(float),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy d_delta");
                delta_fp16_ws.allocate(max_batch_size, dim, delta_topk, max_delta_n, delta_tile_n);
                delta_fp16_ws.build_resident(d_delta);
                cudaFree(d_delta);
                d_delta = nullptr;
                std::printf("[DELTA] resident FP16 GEMM workspace: max_bs=%d tile_n=%d return_topk=%d fp16_delta=%.2f MB\n",
                            max_batch_size, delta_tile_n, delta_topk,
                            (double)max_delta_n * dim * sizeof(__half) / (1024.0 * 1024.0));
            }
            if ((args.delta_search_mode == "ivf-cpu" || args.delta_search_mode == "ivf-gpu") &&
                max_delta_n > 0) {
                std::vector<int> unique_delta_ns;
                for (int dn : args.delta_ns) {
                    if (dn > 0 && std::find(unique_delta_ns.begin(), unique_delta_ns.end(), dn) == unique_delta_ns.end()) {
                        unique_delta_ns.push_back(dn);
                    }
                }
                std::sort(unique_delta_ns.begin(), unique_delta_ns.end());
                for (int dn : unique_delta_ns) {
                    double tb0 = now_ms();
                    DeltaIvfCpuSegment seg;
                    build_delta_ivf_cpu_segment(dataset, delta, delta_u8, delta_locs, dn, n, args.nlist, dim, &seg);
                    long long nonempty = 0;
                    int max_len = 0;
                    for (int cnt : seg.counts) {
                        nonempty += (cnt > 0);
                        max_len = std::max(max_len, cnt);
                    }
                    std::printf("[DELTA-IVF-CPU] built delta_n=%d nonempty_lists=%lld max_list=%d time=%.2f ms\n",
                                dn, nonempty, max_len, now_ms() - tb0);
                    delta_ivf_cpu_segments.push_back(std::move(seg));
                }
                if (args.delta_search_mode == "ivf-gpu") {
                    for (const auto& cpu_seg : delta_ivf_cpu_segments) {
                        double tg0 = now_ms();
                        DeltaIvfGpuSegment gseg;
                        gseg.allocate_from_cpu(cpu_seg, max_batch_size, delta_topk, args.nprobe);
                        double resident_mb = cpu_seg.use_u8
                            ? (double)cpu_seg.active_n * dim * sizeof(uint8_t) / (1024.0 * 1024.0)
                            : (double)cpu_seg.active_n * dim * sizeof(float) / (1024.0 * 1024.0);
                        std::printf("[DELTA-IVF-GPU] resident delta_n=%d %s_vectors=%.2f MB time=%.2f ms\n",
                                    cpu_seg.active_n,
                                    cpu_seg.use_u8 ? "u8" : "fp32",
                                    resident_mb,
                                    now_ms() - tg0);
                        delta_ivf_gpu_segments.push_back(std::move(gseg));
                    }
                }
            }
            std::printf("[DELTA] publish changes only update active length; no per-step handle rebuild\n");
        }

        if (args.gt_only) {
            if (!args.insert_aware_recall) {
                throw std::runtime_error("--gt-only requires --insert-aware-recall");
            }
            write_insert_gt_header_if_needed(args.out);
            const int gt_batch = max_batch_size;
            for (int delta_n_cfg : args.delta_ns) {
                if (delta_n_cfg > 0 && delta.empty()) {
                    throw std::runtime_error("gt-only requested delta_n>0 without delta buffer");
                }
                std::vector<int> all_delta_idx((size_t)nq * args.topk, -1);
                std::vector<float> all_delta_dist((size_t)nq * args.topk,
                                                  std::numeric_limits<float>::infinity());
                double total0 = now_ms();
                double delta_stage_ms = 0.0, delta_h2d_ms = 0.0, delta_d2h_ms = 0.0;
                if (delta_n_cfg > 0) {
                    const int n_batches = (nq + gt_batch - 1) / gt_batch;
                    for (int b = 0; b < n_batches; ++b) {
                        int qb = b * gt_batch;
                        int qe = std::min(qb + gt_batch, nq);
                        int nb = qe - qb;
                        std::vector<int> delta_idx;
                        std::vector<float> delta_dist;
                        double t0 = now_ms();
                        double dgpu = 0.0, dh2d = 0.0, dd2h = 0.0;
                        delta_fp16_gemm_topk_search(&delta_fp16_ws,
                                               delta.data(),
                                               h_query + (size_t)qb * dim,
                                               nb,
                                               dim,
                                               delta_n_cfg,
                                               n,
                                               delta_topk,
                                               &delta_idx,
                                               &delta_dist,
                                               &dgpu,
                                               &dh2d,
                                               &dd2h);
                        delta_stage_ms += now_ms() - t0;
                        delta_h2d_ms += dh2d;
                        delta_d2h_ms += dd2h;
                        const int gt_copy_k = std::min(args.topk, delta_topk);
                        for (int q = 0; q < nb; ++q) {
                            std::memcpy(all_delta_idx.data() + (size_t)(qb + q) * args.topk,
                                        delta_idx.data() + (size_t)q * delta_topk,
                                        (size_t)gt_copy_k * sizeof(int));
                            std::memcpy(all_delta_dist.data() + (size_t)(qb + q) * args.topk,
                                        delta_dist.data() + (size_t)q * delta_topk,
                                        (size_t)gt_copy_k * sizeof(float));
                        }
                    }
                }
                double total_ms = now_ms() - total0;
                summarize_insert_gt(args.dataset,
                                    n,
                                    nq,
                                    args.topk,
                                    delta_n_cfg,
                                    gt_batch,
                                    total_ms,
                                    delta_stage_ms,
                                    delta_h2d_ms,
                                    delta_d2h_ms,
                                    static_gt_ids,
                                    static_gt_dists,
                                    all_delta_idx,
                                    all_delta_dist,
                                    args.out);
                std::fflush(stdout);
            }
            coarse_handle_release(&ch);
            if (delta_ch_valid) coarse_handle_release(&delta_ch);
            for (auto& e : delta_handles) coarse_handle_release(&e.handle);
            main_ws.release();
            delta_ws.release();
            delta_fp16_ws.release();
            for (auto& s : delta_ivf_gpu_segments) s.release();
            cudaFree(d_centers);
            if (d_delta) cudaFree(d_delta);
            std::free(h_query);
            if (h_gt) std::free(h_gt);
            return 0;
        }

        const bool has_avx512f = cpu_supports_avx512f();
        const bool has_avx512bw = cpu_supports_avx512bw();
        const bool has_avx2 = cpu_supports_avx2();
        if (args.use_fbin && !has_avx512f) {
            std::printf("[CPU-FINE] AVX-512F unavailable; using scalar touched float fallback\n");
        }
        if (!args.use_fbin && !has_avx512bw) {
            if (!has_avx2) {
                throw std::runtime_error("SIFT/uint8 update runner requires AVX2 or AVX-512BW for the optimized main query path");
            }
            std::printf("[CPU-FINE] AVX-512BW unavailable; using local V3 touched u8 AVX2 path\n");
        }

        {
            int warm_nb = std::min(max_batch_size, nq);
            std::vector<int> warm_cids((size_t)warm_nb * args.nprobe);
            double warm_gpu = 0.0, warm_h2d = 0.0, warm_d2h = 0.0;
            coarse_search(&ch, h_query, warm_nb, args.nprobe,
                          L2_DISTANCE_MODE, warm_cids.data(),
                          &warm_gpu, &warm_h2d, &warm_d2h);
            if ((!delta.empty() || !delta_u8.empty()) && max_delta_n > 0) {
                std::vector<int> warm_delta_idx;
                std::vector<float> warm_delta_dist;
                double warm_delta_gpu = 0.0, warm_delta_h2d = 0.0, warm_delta_d2h = 0.0;
                delta_overlay_search(args,
                                     &delta_fp16_ws,
                                     delta_ivf_cpu_segments,
                                     delta_ivf_gpu_segments,
                                     delta,
                                     delta_u8,
                                     h_query,
                                     h_query_u8.empty() ? nullptr : h_query_u8.data(),
                                     warm_cids.data(),
                                     warm_nb,
                                     dim,
                                     max_delta_n,
                                     n,
                                     delta_topk,
                                     &warm_delta_idx,
                                     &warm_delta_dist,
                                     &warm_delta_gpu,
                                     &warm_delta_h2d,
                                     &warm_delta_d2h);
            }
            cuda_check(cudaDeviceSynchronize(), "warmup sync");
            std::printf("[WARMUP] done nb=%d\n", warm_nb);
        }

        if (args.mixed_workload) {
            write_mixed_header_if_needed(args.out);
            for (int bs : args.batch_sizes) {
                const int base_batches = (nq + bs - 1) / bs;
                for (int target_delta_n : args.delta_ns) {
                    for (double target_delete_ratio : args.delete_ratios) {
                        for (int rep = 0; rep < args.repeats; ++rep) {
                            std::vector<Event> query_events_vec;
                            query_events_vec.reserve((size_t)base_batches * args.query_rounds);
                            for (int round = 0; round < args.query_rounds; ++round) {
                                for (int b = 0; b < base_batches; ++b) {
                                    int qb = b * bs;
                                    int qe = std::min(qb + bs, nq);
                                    query_events_vec.push_back({EventType::Query, qb, qe - qb, 0, 0.0});
                                }
                            }
                            uint64_t cell_seed = args.seed ^
                                ((uint64_t)bs << 48) ^
                                ((uint64_t)(uint32_t)target_delta_n << 16) ^
                                (uint64_t)std::llround(target_delete_ratio * 1000000.0) ^
                                (uint64_t)rep;
                            std::mt19937_64 rng(cell_seed);
                            std::shuffle(query_events_vec.begin(), query_events_vec.end(), rng);

                            std::vector<Event> events;
                            events.reserve(query_events_vec.size() + (size_t)args.update_steps * 2);
                            size_t next_query = 0;
                            int next_insert_step = 1;
                            int next_delete_step = 1;
                            while (next_query < query_events_vec.size() ||
                                   (target_delta_n > 0 && next_insert_step <= args.update_steps) ||
                                   (target_delete_ratio > 0.0 && next_delete_step <= args.update_steps)) {
                                std::vector<EventType> choices;
                                if (next_query < query_events_vec.size()) choices.push_back(EventType::Query);
                                if (target_delta_n > 0 && next_insert_step <= args.update_steps) {
                                    choices.push_back(EventType::Insert);
                                }
                                if (target_delete_ratio > 0.0 && next_delete_step <= args.update_steps) {
                                    choices.push_back(EventType::Delete);
                                }
                                std::uniform_int_distribution<size_t> pick(0, choices.size() - 1);
                                EventType chosen = choices[pick(rng)];
                                if (chosen == EventType::Query) {
                                    events.push_back(query_events_vec[next_query++]);
                                } else if (chosen == EventType::Insert) {
                                    int to = (int)(((long long)target_delta_n * next_insert_step +
                                                    args.update_steps - 1) / args.update_steps);
                                    events.push_back({EventType::Insert, 0, 0,
                                                      std::min(to, target_delta_n), 0.0});
                                    ++next_insert_step;
                                } else {
                                    double to = target_delete_ratio *
                                                (double)next_delete_step / (double)args.update_steps;
                                    events.push_back({EventType::Delete, 0, 0, 0, to});
                                    ++next_delete_step;
                                }
                            }

                            int current_delta_n = 0;
                            double current_delete_ratio = 0.0;

                            std::vector<double> latencies;
                            latencies.reserve((size_t)base_batches * args.query_rounds);
                            double total_query_ms = 0.0;
                            double coarse_ms = 0.0, h2d_ms = 0.0, d2h_ms = 0.0;
                            double fine_ms = 0.0, delta_ms = 0.0, merge_ms = 0.0;
                            double delta_h2d_ms = 0.0, delta_d2h_ms = 0.0;
                            double insert_publish_ms = 0.0, delete_publish_ms = 0.0;
                            long long total_queries = 0;
                            long long r1_hit = 0, r1_total = 0, r10_hit = 0, r10_total = 0;
                            long long delta_top1_miss = 0;
                            double delta_state_sum = 0.0, delete_state_sum = 0.0;
                            long long query_events = 0;

                            for (const Event& ev : events) {
                                if (ev.type == EventType::Insert) {
                                    double ti0 = now_ms();
                                    current_delta_n = ev.insert_to;
                                    if (current_delta_n > 0) {
                                        cuda_check(cudaDeviceSynchronize(), "delta publish sync");
                                    }
                                    insert_publish_ms += now_ms() - ti0;
                                    continue;
                                }
                                if (ev.type == EventType::Delete) {
                                    double td0 = now_ms();
                                    current_delete_ratio = ev.delete_to;
                                    delete_publish_ms += now_ms() - td0;
                                    continue;
                                }

                                int qb = ev.qb;
                                int nb = ev.qn;
                                double batch0 = now_ms();

                                std::vector<int> cids((size_t)nb * args.nprobe);
                                double tco = 0.0, th2d = 0.0, td2h = 0.0;
                                coarse_search(&ch, h_query + (size_t)qb * dim,
                                              nb, args.nprobe, L2_DISTANCE_MODE,
                                              cids.data(), &tco, &th2d, &td2h);
                                coarse_ms += tco;
                                h2d_ms += th2d;
                                d2h_ms += td2h;

                                const int main_topk_cfg =
                                    (current_delete_ratio > 0.0 && args.main_overfetch < 128)
                                        ? 128
                                        : args.main_overfetch;
                                std::vector<int> main_idx((size_t)nb * main_topk_cfg);
                                std::vector<float> main_dist((size_t)nb * main_topk_cfg);
                                double tf0 = now_ms();
                                if (args.use_fbin) {
                                    if (has_avx512f) {
                                        double tf = 0.0;
                                        long long nfma = 0;
                                        fine_search_cpu(
                                            CPU_FINE_V3_TOUCHED,
                                            dataset.reordered_data,
                                            nullptr,
                                            nullptr,
                                            dataset.cluster_info.offsets,
                                            dataset.cluster_info.counts,
                                            dataset.reordered_indices,
                                            args.nlist,
                                            dim,
                                            h_query + (size_t)qb * dim,
                                            cids.data(),
                                            nb,
                                            args.nprobe,
                                            main_topk_cfg,
                                            L2_DISTANCE_MODE,
                                            args.threads,
                                            main_idx.data(),
                                            main_dist.data(),
                                            &tf,
                                            &nfma);
                                    } else {
                                        fine_search_scalar_touched_float(
                                            dataset.reordered_data,
                                            dataset.cluster_info.offsets,
                                            dataset.cluster_info.counts,
                                            dataset.reordered_indices,
                                            args.nlist,
                                            dim,
                                            h_query + (size_t)qb * dim,
                                            cids.data(),
                                            nb,
                                            args.nprobe,
                                            main_topk_cfg,
                                            main_idx.data(),
                                            main_dist.data());
                                    }
                                } else {
                                    std::vector<uint8_t> q_u8((size_t)nb * dim);
                                    std::memcpy(q_u8.data(), h_query_u8.data() + (size_t)qb * dim,
                                                (size_t)nb * dim);
                                    const uint8_t* mask_ptr_direct = nullptr;
                                    if (mask_ptr_direct) {
                                        ivftensor::cpu_fine::cpu_fine_kernel_v3_u8_touched_masked(
                                            h_base_u8.data(),
                                            dataset.cluster_info.offsets,
                                            dataset.cluster_info.counts,
                                            q_u8.data(),
                                            cids.data(),
                                            nb,
                                            dim,
                                            args.nlist,
                                            args.nprobe,
                                            main_topk_cfg,
                                            args.threads,
                                            mask_ptr_direct,
                                            main_idx.data(),
                                            main_dist.data());
                                    } else {
                                        ivftensor::cpu_fine::cpu_fine_kernel_v3_u8_touched(
                                            h_base_u8.data(),
                                            dataset.cluster_info.offsets,
                                            dataset.cluster_info.counts,
                                            q_u8.data(),
                                            cids.data(),
                                            nb,
                                            dim,
                                            args.nlist,
                                            args.nprobe,
                                            main_topk_cfg,
                                            args.threads,
                                            main_idx.data(),
                                            main_dist.data());
                                    }
                                    for (size_t i = 0; i < main_idx.size(); ++i) {
                                        int loc = main_idx[i];
                                        main_idx[i] = (loc >= 0 && loc < n) ? dataset.reordered_indices[loc] : -1;
                                    }
                                }
                                fine_ms += now_ms() - tf0;

                                std::vector<int> delta_idx;
                                std::vector<float> delta_dist;
                                double td0 = now_ms();
                                double delta_gpu_ms = 0.0, delta_h2d = 0.0, delta_d2h = 0.0;
                                delta_overlay_search(args,
                                                     &delta_fp16_ws,
                                                     delta_ivf_cpu_segments,
                                                     delta_ivf_gpu_segments,
                                                     delta,
                                                     delta_u8,
                                                     h_query + (size_t)qb * dim,
                                                     h_query_u8.empty() ? nullptr : h_query_u8.data() + (size_t)qb * dim,
                                                     cids.data(),
                                                     nb,
                                                     dim,
                                                     current_delta_n,
                                                     n,
                                                     delta_topk,
                                                     &delta_idx,
                                                     &delta_dist,
                                                     &delta_gpu_ms,
                                                     &delta_h2d,
                                                     &delta_d2h);
                                delta_ms += now_ms() - td0;
                                delta_h2d_ms += delta_h2d;
                                delta_d2h_ms += delta_d2h;
                                delta_top1_miss += count_query_copy_top1_miss(
                                    delta_idx, nb, delta_topk, qb, nq, n,
                                    current_delta_n, args.delta_mode);

                                std::vector<int> batch_idx((size_t)nb * args.topk);
                                std::vector<float> batch_dist((size_t)nb * args.topk);
                                double tm0 = now_ms();
                                for (int q = 0; q < nb; ++q) {
                                    std::vector<Candidate> cands;
                                    cands.reserve((size_t)main_topk_cfg + (size_t)delta_topk);
                                    for (int k = 0; k < main_topk_cfg; ++k) {
                                        int id = main_idx[(size_t)q * main_topk_cfg + k];
                                        if (is_deleted_id_runtime(id, current_delete_ratio,
                                                                  ((args.gt_safe_updates || args.delete_gt_safe) && !gt_protect_ids.empty()) ? gt_protect_ids.data() : nullptr,
                                                                  n)) continue;
                                        push_candidate(cands,
                                                       main_dist[(size_t)q * main_topk_cfg + k],
                                                       id);
                                    }
                                    for (int k = 0; k < delta_topk && current_delta_n > 0; ++k) {
                                        push_candidate(cands,
                                                       delta_dist[(size_t)q * delta_topk + k],
                                                       delta_idx[(size_t)q * delta_topk + k]);
                                    }
                                    finalize_topk(cands, args.topk,
                                                  batch_idx.data() + (size_t)q * args.topk,
                                                  batch_dist.data() + (size_t)q * args.topk);
                                }
                                merge_ms += now_ms() - tm0;

                                accumulate_recall_at_k_update(
                                    batch_idx.data(), h_gt, qb, nb, nq, 1, args.topk, kgt,
                                    n, current_delta_n, args.delta_mode, current_delete_ratio,
                                    args.gt_safe_updates,
                                    &r1_hit, &r1_total);
                                accumulate_recall_at_k_update(
                                    batch_idx.data(), h_gt, qb, nb, nq, 10, args.topk, kgt,
                                    n, current_delta_n, args.delta_mode, current_delete_ratio,
                                    args.gt_safe_updates,
                                    &r10_hit, &r10_total);

                                double batch_ms = now_ms() - batch0;
                                latencies.push_back(batch_ms);
                                total_query_ms += batch_ms;
                                total_queries += nb;
                                delta_state_sum += (double)current_delta_n;
                                delete_state_sum += current_delete_ratio;
                                ++query_events;
                            }

                            double qps = total_query_ms > 0.0
                                ? (double)total_queries / (total_query_ms / 1000.0)
                                : 0.0;
                            double r1 = r1_total ? (double)r1_hit / (double)r1_total : -1.0;
                            double r10 = r10_total ? (double)r10_hit / (double)r10_total : -1.0;
                            double avg_delta_n = query_events ? delta_state_sum / (double)query_events : 0.0;
                            double avg_delete_ratio = query_events ? delete_state_sum / (double)query_events : 0.0;
                            double avg_insert_ratio = avg_delta_n / (double)n;
                            double target_insert_ratio = (double)target_delta_n / (double)n;

                            std::ofstream out(args.out, std::ios::app);
                            out << args.dataset << ',' << n << ',' << nq << ',' << args.nlist << ','
                                << args.nprobe << ',' << bs << ',' << args.threads << ','
                                << args.topk << ',' << target_delta_n << ',' << target_insert_ratio << ','
                                << target_delete_ratio << ',' << args.update_steps << ','
                                << args.query_rounds << ',' << cell_seed << ',' << rep << ','
                                << query_events << ',' << (args.pipeline ? 1 : 0) << ','
                                << total_query_ms << ',' << qps << ','
                                << percentile(latencies, 50.0) << ',' << percentile(latencies, 90.0) << ','
                                << percentile(latencies, 99.0) << ',' << r1 << ',' << r10 << ','
                                << avg_delta_n << ',' << avg_insert_ratio << ',' << avg_delete_ratio << ','
                                << insert_publish_ms << ',' << delete_publish_ms << ','
                                << coarse_ms << ',' << h2d_ms << ',' << d2h_ms << ','
                                << fine_ms << ',' << delta_ms << ',' << delta_h2d_ms << ','
                                << delta_d2h_ms << ',' << merge_ms << '\n';
                            out.close();

                            std::printf("[MIXED] dataset=%s bs=%d target_delta=%d target_del=%.4f rep=%d qps=%.2f R10=%.4f p99=%.2f avg_delta=%.0f avg_del=%.5f\n",
                                        args.dataset.c_str(), bs, target_delta_n,
                                        target_delete_ratio, rep, qps, r10,
                                        percentile(latencies, 99.0), avg_delta_n,
                                        avg_delete_ratio);
                            if (delta_top1_miss > 0) {
                                std::printf("[WARN] mixed query-copy delta top1 missing for %lld query batches/items\n",
                                            delta_top1_miss);
                            }
                            std::fflush(stdout);
                        }
                    }
                }
            }

            coarse_handle_release(&ch);
            if (delta_ch_valid) coarse_handle_release(&delta_ch);
            for (auto& e : delta_handles) coarse_handle_release(&e.handle);
            main_ws.release();
            delta_ws.release();
            delta_fp16_ws.release();
            for (auto& s : delta_ivf_gpu_segments) s.release();
            cudaFree(d_centers);
            if (d_delta) cudaFree(d_delta);
            std::free(h_query);
            if (h_gt) std::free(h_gt);
            return 0;
        }

        write_header_if_needed(args.out);

        for (int bs : args.batch_sizes) {
            const int n_batches = (nq + bs - 1) / bs;
            for (int delta_n_cfg : args.delta_ns) {
                if (delta_n_cfg > 0 && delta.empty() && delta_u8.empty()) {
                    throw std::runtime_error("missing initialized delta buffer");
                }
                for (double delete_ratio_cfg : args.delete_ratios) {
                    int delete_count_cfg = (int)llround(delete_ratio_cfg * (double)n);
                    if (!args.update_pairs.empty() && !args.update_pairs.count({delta_n_cfg, delete_count_cfg})) {
                        continue;
                    }
                    if (!args.include_mixed_update && args.update_pairs.empty() &&
                        delta_n_cfg > 0 && delete_ratio_cfg > 0.0) {
                        continue;
                    }
                    const int main_topk_cfg = args.main_overfetch;
                    std::vector<uint8_t> delete_local_mask;
                    build_reordered_delete_mask(dataset.reordered_indices,
                                                n,
                                                delete_ratio_cfg,
                                                ((args.gt_safe_updates || args.delete_gt_safe) && !gt_protect_ids.empty()) ? gt_protect_ids.data() : nullptr,
                                                args.threads,
                                                &delete_local_mask);
                    const uint8_t* delete_mask_ptr = delete_local_mask.empty()
                        ? nullptr
                        : delete_local_mask.data();
                    std::vector<int> final_idx((size_t)nq * args.topk);
                    std::vector<float> final_dist((size_t)nq * args.topk);
                    std::vector<int> filtered_base_gt_ids;
                    std::vector<float> filtered_base_gt_dists;
                    int filtered_gt_min_survivors = -1;
                    if (args.insert_aware_recall) {
                        filtered_gt_min_survivors = build_filtered_base_gt_topk_from_pool(
                            static_gt_pool_ids,
                            static_gt_pool_dists,
                            nq,
                            static_gt_pool_k,
                            args.topk,
                            delete_ratio_cfg,
                            n,
                            &filtered_base_gt_ids,
                            &filtered_base_gt_dists);
                    }
                    std::vector<int> all_delta_idx;
                    std::vector<float> all_delta_dist;
                    if (args.insert_aware_recall && delta_n_cfg > 0) {
                        all_delta_idx.assign((size_t)nq * delta_topk, -1);
                        all_delta_dist.assign((size_t)nq * delta_topk,
                                              std::numeric_limits<float>::infinity());
                    }

                    for (int rep = 0; rep < args.repeats; ++rep) {
                        if (!all_delta_idx.empty()) {
                            std::fill(all_delta_idx.begin(), all_delta_idx.end(), -1);
                            std::fill(all_delta_dist.begin(), all_delta_dist.end(),
                                      std::numeric_limits<float>::infinity());
                        }
                        double total0 = now_ms();
                        double coarse_ms = 0.0, h2d_ms = 0.0, d2h_ms = 0.0;
                        double fine_ms = 0.0, delta_ms = 0.0, merge_ms = 0.0;
                        double delta_h2d_ms = 0.0, delta_d2h_ms = 0.0;
                        long long delta_top1_miss = 0;
                        std::vector<double> batch_latency_ms;
                        batch_latency_ms.reserve((size_t)n_batches);

                        if (!args.pipeline) {
                        for (int b = 0; b < n_batches; ++b) {
                            double batch_t0 = now_ms();
                            int qb = b * bs;
                            int qe = std::min(qb + bs, nq);
                            int nb = qe - qb;

                            std::vector<int> cids((size_t)nb * args.nprobe);
                            double tco = 0.0, th2d = 0.0, td2h = 0.0;
                            coarse_search(&ch, h_query + (size_t)qb * dim,
                                          nb, args.nprobe, L2_DISTANCE_MODE,
                                          cids.data(), &tco, &th2d, &td2h);
                            coarse_ms += tco;
                            h2d_ms += th2d;
                            d2h_ms += td2h;

                            MainFineJob main_job;
                            main_job.qb = qb;
                            main_job.nb = nb;
                            main_job.topk = main_topk_cfg;
                            main_job.deleted_local_mask = delete_mask_ptr;
                            main_job.cids = std::move(cids);
                            run_main_fine_job(args, dataset, h_base_u8, h_query_u8,
                                              h_query, n, dim, has_avx512f, has_avx512bw,
                                              &main_job);
                            fine_ms += main_job.fine_ms;
                            const std::vector<int>& main_idx = main_job.idx;
                            const std::vector<float>& main_dist = main_job.dist;

                            std::vector<int> delta_idx;
                            std::vector<float> delta_dist;
                            double td0 = now_ms();
                            double delta_gpu_ms = 0.0, delta_h2d = 0.0, delta_d2h = 0.0;
                            delta_overlay_search(args,
                                                 &delta_fp16_ws,
                                                 delta_ivf_cpu_segments,
                                                 delta_ivf_gpu_segments,
                                                 delta,
                                                 delta_u8,
                                                 h_query + (size_t)qb * dim,
                                                 h_query_u8.empty() ? nullptr : h_query_u8.data() + (size_t)qb * dim,
                                                 cids.data(),
                                                 nb,
                                                 dim,
                                                 delta_n_cfg,
                                                 n,
                                                 delta_topk,
                                                 &delta_idx,
                                                 &delta_dist,
                                                 &delta_gpu_ms,
                                                 &delta_h2d,
                                                 &delta_d2h);
                            delta_ms += now_ms() - td0;
                            delta_h2d_ms += delta_h2d;
                            delta_d2h_ms += delta_d2h;
                            delta_top1_miss += count_query_copy_top1_miss(
                                delta_idx, nb, delta_topk, qb, nq, n,
                                delta_n_cfg, args.delta_mode);
                            if (!all_delta_idx.empty()) {
                                for (int q = 0; q < nb; ++q) {
                                    std::memcpy(all_delta_idx.data() + (size_t)(qb + q) * delta_topk,
                                                delta_idx.data() + (size_t)q * delta_topk,
                                                (size_t)delta_topk * sizeof(int));
                                    std::memcpy(all_delta_dist.data() + (size_t)(qb + q) * delta_topk,
                                                delta_dist.data() + (size_t)q * delta_topk,
                                                (size_t)delta_topk * sizeof(float));
                                }
                            }

                            double tm0 = now_ms();
                            for (int q = 0; q < nb; ++q) {
                                std::vector<Candidate> cands;
                                cands.reserve((size_t)main_topk_cfg + (size_t)delta_topk);
                                for (int k = 0; k < main_topk_cfg; ++k) {
                                    int id = main_idx[(size_t)q * main_topk_cfg + k];
                                    if (is_deleted_id_runtime(id, delete_ratio_cfg,
                                                              ((args.gt_safe_updates || args.delete_gt_safe) && !gt_protect_ids.empty()) ? gt_protect_ids.data() : nullptr,
                                                              n)) continue;
                                    push_candidate(cands,
                                                   main_dist[(size_t)q * main_topk_cfg + k],
                                                   id);
                                }
                                for (int k = 0; k < delta_topk && delta_n_cfg > 0; ++k) {
                                    push_candidate(cands,
                                                   delta_dist[(size_t)q * delta_topk + k],
                                                   delta_idx[(size_t)q * delta_topk + k]);
                                }
                                finalize_topk(cands, args.topk,
                                              final_idx.data() + (size_t)(qb + q) * args.topk,
                                              final_dist.data() + (size_t)(qb + q) * args.topk);
                            }
                            merge_ms += now_ms() - tm0;
                            batch_latency_ms.push_back(now_ms() - batch_t0);
                        }
                        } else {
                            MainFineJob pipe_state;
                            std::thread pipe_worker;
                            std::mutex pipe_mu;
                            std::condition_variable pipe_cv;
                            bool pipe_has_job = false;
                            bool pipe_job_done = false;
                            bool pipe_inflight = false;
                            bool pipe_stop = false;
                            std::exception_ptr pipe_error = nullptr;
                            std::vector<double> pipe_batch_start((size_t)n_batches, 0.0);

                            auto merge_finished = [&](const MainFineJob& job) {
                                double tm0 = now_ms();
                                if (!all_delta_idx.empty()) {
                                    for (int q = 0; q < job.nb; ++q) {
                                        std::memcpy(all_delta_idx.data() + (size_t)(job.qb + q) * delta_topk,
                                                    job.delta_idx.data() + (size_t)q * delta_topk,
                                                    (size_t)delta_topk * sizeof(int));
                                        std::memcpy(all_delta_dist.data() + (size_t)(job.qb + q) * delta_topk,
                                                    job.delta_dist.data() + (size_t)q * delta_topk,
                                                    (size_t)delta_topk * sizeof(float));
                                    }
                                }
                                for (int q = 0; q < job.nb; ++q) {
                                    std::vector<Candidate> cands;
                                    cands.reserve((size_t)main_topk_cfg + (size_t)delta_topk);
                                    for (int k = 0; k < main_topk_cfg; ++k) {
                                        int id = job.idx[(size_t)q * main_topk_cfg + k];
                                        if (is_deleted_id_runtime(id, delete_ratio_cfg,
                                                                  ((args.gt_safe_updates || args.delete_gt_safe) && !gt_protect_ids.empty()) ? gt_protect_ids.data() : nullptr,
                                                                  n)) continue;
                                        push_candidate(cands,
                                                       job.dist[(size_t)q * main_topk_cfg + k],
                                                       id);
                                    }
                                    for (int k = 0; k < delta_topk && delta_n_cfg > 0; ++k) {
                                        push_candidate(cands,
                                                       job.delta_dist[(size_t)q * delta_topk + k],
                                                       job.delta_idx[(size_t)q * delta_topk + k]);
                                    }
                                    finalize_topk(cands, args.topk,
                                                  final_idx.data() + (size_t)(job.qb + q) * args.topk,
                                                  final_dist.data() + (size_t)(job.qb + q) * args.topk);
                                }
                                merge_ms += now_ms() - tm0;
                                int bi = job.qb / bs;
                                if (bi >= 0 && bi < (int)pipe_batch_start.size() && pipe_batch_start[(size_t)bi] > 0.0) {
                                    batch_latency_ms.push_back(now_ms() - pipe_batch_start[(size_t)bi]);
                                }
                            };

                            pipe_worker = std::thread([&]() {
                                while (true) {
                                    MainFineJob local_job;
                                    {
                                        std::unique_lock<std::mutex> lk(pipe_mu);
                                        pipe_cv.wait(lk, [&]() { return pipe_has_job || pipe_stop; });
                                        if (pipe_stop && !pipe_has_job) break;
                                        local_job = std::move(pipe_state);
                                        pipe_has_job = false;
                                    }
                                    try {
                                        run_main_fine_job(args, dataset, h_base_u8, h_query_u8,
                                                          h_query, n, dim, has_avx512f, has_avx512bw,
                                                          &local_job);
                                    } catch (...) {
                                        std::lock_guard<std::mutex> lk(pipe_mu);
                                        pipe_error = std::current_exception();
                                        pipe_job_done = true;
                                        pipe_cv.notify_all();
                                        continue;
                                    }
                                    {
                                        std::lock_guard<std::mutex> lk(pipe_mu);
                                        pipe_state = std::move(local_job);
                                        pipe_job_done = true;
                                    }
                                    pipe_cv.notify_all();
                                }
                            });

                            auto finish_pipeline_job = [&]() {
                                if (!pipe_inflight) return;
                                MainFineJob done;
                                {
                                    std::unique_lock<std::mutex> lk(pipe_mu);
                                    pipe_cv.wait(lk, [&]() { return pipe_job_done || pipe_error; });
                                    if (pipe_error) std::rethrow_exception(pipe_error);
                                    done = std::move(pipe_state);
                                    pipe_state = MainFineJob();
                                    pipe_job_done = false;
                                    pipe_inflight = false;
                                }
                                fine_ms += done.fine_ms;
                                merge_finished(done);
                            };

                            auto launch_pipeline_job = [&](MainFineJob&& job) {
                                finish_pipeline_job();
                                {
                                    std::lock_guard<std::mutex> lk(pipe_mu);
                                    pipe_state = std::move(job);
                                    pipe_job_done = false;
                                    pipe_has_job = true;
                                    pipe_inflight = true;
                                }
                                pipe_cv.notify_one();
                            };

                            for (int b = 0; b < n_batches; ++b) {
                                pipe_batch_start[(size_t)b] = now_ms();
                                int qb = b * bs;
                                int qe = std::min(qb + bs, nq);
                                int nb = qe - qb;

                                MainFineJob job;
                                job.qb = qb;
                                job.nb = nb;
                                job.topk = main_topk_cfg;
                                job.deleted_local_mask = delete_mask_ptr;
                                job.cids.resize((size_t)nb * args.nprobe);

                                double tco = 0.0, th2d = 0.0, td2h = 0.0;
                                coarse_search(&ch, h_query + (size_t)qb * dim,
                                              nb, args.nprobe, L2_DISTANCE_MODE,
                                              job.cids.data(), &tco, &th2d, &td2h);
                                coarse_ms += tco;
                                h2d_ms += th2d;
                                d2h_ms += td2h;

                                double td0 = now_ms();
                                double delta_gpu_ms = 0.0, delta_h2d = 0.0, delta_d2h = 0.0;
                                delta_overlay_search(args,
                                                     &delta_fp16_ws,
                                                     delta_ivf_cpu_segments,
                                                     delta_ivf_gpu_segments,
                                                     delta,
                                                     delta_u8,
                                                     h_query + (size_t)qb * dim,
                                                     h_query_u8.empty() ? nullptr : h_query_u8.data() + (size_t)qb * dim,
                                                     job.cids.data(),
                                                     nb,
                                                     dim,
                                                     delta_n_cfg,
                                                     n,
                                                     delta_topk,
                                                     &job.delta_idx,
                                                     &job.delta_dist,
                                                     &delta_gpu_ms,
                                                     &delta_h2d,
                                                     &delta_d2h);
                                delta_ms += now_ms() - td0;
                                delta_h2d_ms += delta_h2d;
                                delta_d2h_ms += delta_d2h;
                                delta_top1_miss += count_query_copy_top1_miss(
                                    job.delta_idx, nb, delta_topk, qb, nq, n,
                                    delta_n_cfg, args.delta_mode);

                                launch_pipeline_job(std::move(job));
                            }

                            finish_pipeline_job();
                            {
                                std::lock_guard<std::mutex> lk(pipe_mu);
                                pipe_stop = true;
                            }
                            pipe_cv.notify_all();
                            if (pipe_worker.joinable()) pipe_worker.join();
                        }

                        double total_ms = now_ms() - total0;
                        double p50_ms = percentile(batch_latency_ms, 50.0);
                        double p99_ms = percentile(batch_latency_ms, 99.0);
                        double qps = (double)nq / (total_ms / 1000.0);
                        double r1 = -1.0, r10 = -1.0, r100 = -1.0;
                        if (args.insert_aware_recall) {
                            const int* dyn_delta_idx = all_delta_idx.empty() ? nullptr : all_delta_idx.data();
                            const float* dyn_delta_dist = all_delta_dist.empty() ? nullptr : all_delta_dist.data();
                            const int dyn_delta_k = all_delta_idx.empty() ? 0 : delta_topk;
                            r1 = recall_at_k_insert_aware(final_idx.data(), nq, 1, args.topk,
                                                          filtered_base_gt_ids.data(), filtered_base_gt_dists.data(),
                                                          args.topk, dyn_delta_idx, dyn_delta_dist,
                                                          dyn_delta_k);
                            r10 = recall_at_k_insert_aware(final_idx.data(), nq, 10, args.topk,
                                                           filtered_base_gt_ids.data(), filtered_base_gt_dists.data(),
                                                           args.topk, dyn_delta_idx, dyn_delta_dist,
                                                           dyn_delta_k);
                            r100 = recall_at_k_insert_aware(final_idx.data(), nq, 100, args.topk,
                                                            filtered_base_gt_ids.data(), filtered_base_gt_dists.data(),
                                                            args.topk, dyn_delta_idx, dyn_delta_dist,
                                                            dyn_delta_k);
                        } else {
                            r1 = recall_at_k_update(final_idx.data(), h_gt, nq, 1,
                                                    args.topk, kgt, n, delta_n_cfg,
                                                    args.delta_mode, delete_ratio_cfg,
                                                    args.gt_safe_updates);
                            r10 = recall_at_k_update(final_idx.data(), h_gt, nq, 10,
                                                     args.topk, kgt, n, delta_n_cfg,
                                                     args.delta_mode, delete_ratio_cfg,
                                                     args.gt_safe_updates);
                            r100 = recall_at_k_update(final_idx.data(), h_gt, nq, 100,
                                                      args.topk, kgt, n, delta_n_cfg,
                                                      args.delta_mode, delete_ratio_cfg,
                                                      args.gt_safe_updates);
                        }
                        double insert_ratio = (double)delta_n_cfg / (double)n;

                        std::ofstream out(args.out, std::ios::app);
                        out << args.dataset << ',' << n << ',' << nq << ',' << args.nlist << ','
                            << args.nprobe << ',' << bs << ',' << args.threads << ','
                            << args.topk << ',' << main_topk_cfg << ',' << delta_n_cfg << ',' << delete_count_cfg << ','
                            << insert_ratio << ',' << delete_ratio_cfg << ',' << args.delta_mode << ','
                            << rep << ',' << total_ms << ',' << (args.pipeline ? 1 : 0) << ','
                            << p50_ms << ',' << p99_ms << ',' << coarse_ms << ','
                            << h2d_ms << ',' << d2h_ms << ',' << fine_ms << ','
                            << delta_ms << ',' << delta_h2d_ms << ',' << delta_d2h_ms << ','
                            << merge_ms << ',' << qps << ',' << r1 << ',' << r10 << ',' << r100 << '\n';
                        out.close();

                        std::printf("[CSV] dataset=%s bs=%d delta_n=%d del=%.4f rep=%d pipeline=%d nq=%d total=%.2f ms qps=%.2f R1=%.4f R10=%.4f coarse=%.2f fine=%.2f delta=%.2f merge=%.2f\n",
                                    args.dataset.c_str(), bs, delta_n_cfg, delete_ratio_cfg, rep,
                                    args.pipeline ? 1 : 0, nq, total_ms, qps, r1, r10,
                                    coarse_ms, fine_ms, delta_ms, merge_ms);
                        if (delta_top1_miss > 0) {
                            std::printf("[WARN] query-copy delta top1 missing for %lld / %d queries\n",
                                        delta_top1_miss, nq);
                        }
                        std::fflush(stdout);
                    }
                }
            }
        }

        coarse_handle_release(&ch);
        if (delta_ch_valid) coarse_handle_release(&delta_ch);
        for (auto& e : delta_handles) coarse_handle_release(&e.handle);
        main_ws.release();
        delta_ws.release();
        delta_fp16_ws.release();
        for (auto& s : delta_ivf_gpu_segments) s.release();
        cudaFree(d_centers);
        if (d_delta) cudaFree(d_delta);
        std::free(h_query);
        if (h_gt) std::free(h_gt);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FATAL] %s\n", e.what());
        return 2;
    }
}
