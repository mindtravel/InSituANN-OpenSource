#ifndef CPU_ARRAY_UTILS_H
#define CPU_ARRAY_UTILS_H

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <type_traits>
/**
 * CPU
 *
 */



 /**
  *
  *
  * @tparam T  ( int, float, double )
  * @param data
  * @param size
  * @param seed
  * @param min_val
  * @param max_val
  * @param num_threads
  */
 template <typename T>
 inline void init_array_multithreaded(
     T* data,
     size_t size,
     unsigned int seed = 1234,
     T min_val = static_cast<T>(-10),
     T max_val = static_cast<T>(10),
     int num_threads = 0
 ) {
     //  T
     static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type (int, float, double, etc.)");

     // 1.
     if (num_threads <= 0) {
         unsigned int hw_concurrency = std::thread::hardware_concurrency();
         num_threads = std::max(1, (int)hw_concurrency);
     }

     // 2.  T
     //  -> uniform_real_distribution
     //    -> uniform_int_distribution
     using DistributionType = typename std::conditional<
         std::is_floating_point<T>::value,
         std::uniform_real_distribution<T>,
         std::uniform_int_distribution<T>
     >::type;

     //
     const size_t chunk_size = (size + num_threads - 1) / num_threads;
     std::vector<std::thread> workers;
     workers.reserve(num_threads);

     // 3.
     for (int tid = 0; tid < num_threads; ++tid) {
         workers.emplace_back([=]() {
             const size_t start = tid * chunk_size;
             //
             if (start >= size) return;
             const size_t end = std::min(start + chunk_size, size);

             //  RNG
             std::mt19937 rng(seed + tid);

             //  DistributionType
             DistributionType dist(min_val, max_val);

             //
             for (size_t i = start; i < end; ++i) {
                 data[i] = dist(rng);
             }
         });
     }

     // 4.
     for (auto& t : workers) {
         if (t.joinable()) {
             t.join();
         }
     }
 }

#endif // CPU_ARRAY_UTILS_H
