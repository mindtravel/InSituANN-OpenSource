#ifndef IVFTENSOR_NUMA_UTILS_H
#define IVFTENSOR_NUMA_UTILS_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#endif

namespace ivftensor {
namespace numa {

inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return false;
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

inline int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v || x <= 0) return fallback;
    return (int)x;
}

inline bool enabled() {
    return env_flag("IVFT_NUMA_AWARE");
}

inline bool placement_enabled() {
    return env_flag("IVFT_NUMA_PLACE") ||
           (enabled() && !env_flag("IVFT_NUMA_NO_PLACE"));
}

inline bool schedule_enabled() {
    return env_flag("IVFT_NUMA_SCHEDULE") ||
           (enabled() && !env_flag("IVFT_NUMA_NO_SCHEDULE"));
}

inline bool bind_enabled() {
    return env_flag("IVFT_NUMA_BIND") ||
           ((schedule_enabled() || placement_enabled()) &&
            !env_flag("IVFT_NUMA_NO_BIND"));
}

inline bool any_enabled() {
    return enabled() || env_flag("IVFT_NUMA_PLACE") ||
           env_flag("IVFT_NUMA_SCHEDULE") || env_flag("IVFT_NUMA_BIND");
}

inline bool verbose() {
    return env_flag("IVFT_NUMA_VERBOSE");
}

inline std::vector<int> parse_cpu_list(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::atoi(part.c_str()));
        } else {
            int lo = std::atoi(part.substr(0, dash).c_str());
            int hi = std::atoi(part.substr(dash + 1).c_str());
            if (hi < lo) std::swap(lo, hi);
            for (int c = lo; c <= hi; ++c) cpus.push_back(c);
        }
    }
    cpus.erase(std::remove_if(cpus.begin(), cpus.end(),
                              [](int c) { return c < 0; }),
               cpus.end());
    return cpus;
}

inline std::vector<std::vector<int>> detect_node_cpus(int requested_nodes) {
    std::vector<std::vector<int>> nodes;
#ifdef __linux__
    cpu_set_t allowed;
    bool have_allowed = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    for (int node = 0; node < 16; ++node) {
        std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
        std::ifstream f(path);
        if (!f.good()) continue;
        std::string line;
        std::getline(f, line);
        std::vector<int> cpus = parse_cpu_list(line);
        if (have_allowed) {
            cpus.erase(std::remove_if(cpus.begin(), cpus.end(),
                                      [&](int cpu) {
                                          return cpu < 0 || cpu >= CPU_SETSIZE ||
                                                 !CPU_ISSET(cpu, &allowed);
                                      }),
                       cpus.end());
        }
        if (!cpus.empty()) nodes.push_back(std::move(cpus));
    }
#endif
    if (!nodes.empty()) {
        if (requested_nodes > 0 && requested_nodes < (int)nodes.size()) {
            nodes.resize((size_t)requested_nodes);
        }
        return nodes;
    }

    int hw = (int)std::thread::hardware_concurrency();
    if (hw <= 0) hw = 1;
    int n = requested_nodes > 0 ? requested_nodes : 2;
    n = std::max(1, std::min(n, hw));
    nodes.assign((size_t)n, {});
    for (int cpu = 0; cpu < hw; ++cpu) {
        int node = (int)(((long long)cpu * n) / hw);
        if (node >= n) node = n - 1;
        nodes[(size_t)node].push_back(cpu);
    }
    return nodes;
}

inline const std::vector<std::vector<int>>& node_cpus() {
    static std::vector<std::vector<int>> cpus =
        detect_node_cpus(env_int("IVFT_NUMA_NODES", 2));
    return cpus;
}

inline int node_count() {
    if (!any_enabled()) return 1;
    int n = (int)node_cpus().size();
    return std::max(1, n);
}

inline int node_for_thread(int tid, int num_threads, int nodes) {
    if (nodes <= 1 || num_threads <= 1) return 0;
    int node = (int)(((long long)tid * nodes) / num_threads);
    return std::max(0, std::min(node, nodes - 1));
}

inline bool bind_current_thread_to_node(int node) {
#ifdef __linux__
    const auto& nodes = node_cpus();
    if (node < 0 || node >= (int)nodes.size()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    int nset = 0;
    for (int cpu : nodes[(size_t)node]) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &set);
            ++nset;
        }
    }
    if (nset == 0) return false;
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)node;
    return false;
#endif
}

inline std::vector<int> split_clusters_by_count(const int* counts, int nlist, int nodes) {
    nodes = std::max(1, nodes);
    std::vector<int> bounds((size_t)nodes + 1, 0);
    bounds[0] = 0;
    bounds[(size_t)nodes] = nlist;
    if (!counts || nlist <= 0 || nodes == 1) return bounds;

    long long total = 0;
    for (int c = 0; c < nlist; ++c) total += std::max(0, counts[c]);
    long long acc = 0;
    int c = 0;
    for (int node = 1; node < nodes; ++node) {
        long long target = (total * node) / nodes;
        while (c < nlist && acc + std::max(0, counts[c]) <= target) {
            acc += std::max(0, counts[c]);
            ++c;
        }
        bounds[(size_t)node] = std::max(bounds[(size_t)node - 1], std::min(c, nlist));
    }
    return bounds;
}

inline int node_for_cluster(int cid, const std::vector<int>& bounds) {
    int nodes = (int)bounds.size() - 1;
    if (nodes <= 1) return 0;
    auto it = std::upper_bound(bounds.begin() + 1, bounds.end(), cid);
    int node = (int)(it - (bounds.begin() + 1));
    return std::max(0, std::min(node, nodes - 1));
}

inline std::vector<long long> cluster_prefix(const int* counts, int nlist) {
    std::vector<long long> prefix((size_t)nlist + 1, 0);
    for (int c = 0; c < nlist; ++c) {
        prefix[(size_t)c + 1] = prefix[(size_t)c] + std::max(0, counts[c]);
    }
    return prefix;
}

#ifdef __linux__
inline void advise_discard_pages(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    uintptr_t begin = (uintptr_t)ptr;
    uintptr_t end = begin + bytes;
    uintptr_t aligned_begin = (begin + (uintptr_t)page - 1) & ~((uintptr_t)page - 1);
    uintptr_t aligned_end = end & ~((uintptr_t)page - 1);
    if (aligned_end > aligned_begin) {
        madvise((void*)aligned_begin, (size_t)(aligned_end - aligned_begin), MADV_DONTNEED);
    }
}
#endif

inline void touch_zero_pages(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
#ifdef __linux__
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
#else
    long page = 4096;
#endif
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    for (size_t off = 0; off < bytes; off += (size_t)page) {
        p[off] = 0;
    }
    p[bytes - 1] = 0;
}

inline void place_cluster_major_memory(
    void* base,
    size_t bytes_per_vector,
    const int* counts,
    int nlist,
    const char* label
) {
    if (!placement_enabled() || !base || !counts || nlist <= 0 || bytes_per_vector == 0) return;
    int nodes = node_count();
    if (nodes <= 1) return;

    std::vector<int> bounds = split_clusters_by_count(counts, nlist, nodes);
    std::vector<long long> prefix = cluster_prefix(counts, nlist);
    std::vector<std::thread> threads;
    threads.reserve((size_t)nodes);

    for (int node = 0; node < nodes; ++node) {
        int c0 = bounds[(size_t)node];
        int c1 = bounds[(size_t)node + 1];
        long long v0 = prefix[(size_t)c0];
        long long v1 = prefix[(size_t)c1];
        uint8_t* p = (uint8_t*)base + (size_t)v0 * bytes_per_vector;
        size_t bytes = (size_t)(v1 - v0) * bytes_per_vector;
        threads.emplace_back([node, p, bytes]() {
            bind_current_thread_to_node(node);
#ifdef __linux__
            advise_discard_pages(p, bytes);
#endif
            touch_zero_pages(p, bytes);
        });
    }
    for (auto& th : threads) th.join();

    if (verbose()) {
        std::fprintf(stderr, "[NUMA] placed %s across %d nodes by cluster range", label, nodes);
        for (int node = 0; node < nodes; ++node) {
            std::fprintf(stderr, " node%d=[%d,%d)", node,
                         bounds[(size_t)node], bounds[(size_t)node + 1]);
        }
        std::fprintf(stderr, "\n");
    }
}

}  // namespace numa
}  // namespace ivftensor

#endif  // IVFTENSOR_NUMA_UTILS_H
