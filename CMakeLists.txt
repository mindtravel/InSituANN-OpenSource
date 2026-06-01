cmake_minimum_required(VERSION 3.18)

#  gcc-10 Ubuntu 24.04 + CUDA 13
if(EXISTS "/usr/bin/gcc-10" AND EXISTS "/usr/bin/g++-10")
    set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")
endif()

project(InSituANN LANGUAGES CXX CUDA)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

#  C++/CUDA CUDA 13  thrust/cub  C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(EXISTS "/usr/bin/g++-10")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++-10")
endif()


# CUDA
enable_language(CUDA)

#  CUDAToolkit CUDA::cudart / CUDAToolkit_INCLUDE_DIRS
#  .cpp  nvcc  cuda_runtime.h
find_package(CUDAToolkit REQUIRED)
include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})

#
# -O0:
# -use_fast_math:
# -lineinfo: -g
# -w:
# -Xcompiler -w:
# --disable-warnings: CUDA
# --threads 0: CPU
# --expt-relaxed-constexpr: constexpr
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo -w -Xcompiler -w --disable-warnings --expt-relaxed-constexpr")

# CUDA -DCMAKE_CUDA_ARCHITECTURES=86  RTX 30/40
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
#
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")
# nvcc--threads
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
#  ptxas  -v
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# CUDA
set(CUDA_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

# CUDA
set(CUDA_LIBRARIES
    ${CUDA_LIBRARIES}
    cudart
    curand
    cublas
    cusparse
)

#
set(COMMON_HEADERS "-include ${CMAKE_CURRENT_SOURCE_DIR}/cuda/pch.h")

# l2norm
# device
add_library(l2norm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/l2norm/l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/l2norm/l2norm_squared.cu
)
target_include_directories(l2norm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(l2norm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)

# utils
#
#   - utils.cu: kernelhost
add_library(utils_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/utils.cu
)
target_include_directories(utils_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(utils_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)

# kmeans KMeansClustering  dataset_wrapper /
add_library(kmeans_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans_minibatch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans_reorder_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/kmeans/kmeans_clustering.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/clustering/hierarchical/hierarchical.cu
)
target_include_directories(kmeans_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(kmeans_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)
target_link_libraries(kmeans_lib PUBLIC
    utils_lib
    l2norm_lib
    cublas
    curand
)

# indexed_gemm
#
#   - indexed_gemm_cos.cu: v5 entry-based
#   - indexed_gemm_l2.cu: v5 entry-based
#
#
#   - inner_product_utils.cuh
add_library(indexed_gemm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/fine/indexed_gemm_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/fine/indexed_gemm_l2.cu
)
target_include_directories(indexed_gemm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(indexed_gemm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)
#
target_link_libraries(indexed_gemm_lib
    PUBLIC
    utils_lib
)

# fusion_dist_topk_warpsortcosineL2warpsort
#
#   - ivf_search_lib (fusion_cos_topk_warpsort)
#
#
#   - fusion_cos_topk_warpsort.cu: cosinewarpsort
#   - fusion_l2_topk_warpsort.cu: L2warpsort
#   - warpsort_topk.cu: warp-level top-k
add_library(fusion_dist_topk_warpsort_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/coarse/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/coarse/fusion_l2_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/topk/warpsort_topk.cu
)
target_include_directories(fusion_dist_topk_warpsort_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(fusion_dist_topk_warpsort_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)
target_link_libraries(fusion_dist_topk_warpsort_lib
    PUBLIC
    l2norm_lib
    utils_lib
    cublas
)

# fp16 GPU + CPU
add_library(quantization_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/quantization/linear/linear_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/quantization/linear/linear_fp16_cpu.cpp
)
target_include_directories(quantization_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(quantization_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)

# ivf_searchivftensor
#
#   - ivftensorivfscanbatch.c  batch_search_pipeline
#
#
#   - ivf_search.cu: batch_search_pipeline
#
add_library(ivf_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_no_lookup.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_stages.cu
)
target_include_directories(ivf_search_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
# NVTX  Nsight Systems Stage0~Stage4 OFF  nvToolsExt
option(IVFTENSOR_USE_NVTX "Enable NVTX ranges for Nsight Systems" OFF)
if(IVFTENSOR_USE_NVTX)
    target_compile_definitions(ivf_search_lib PUBLIC USE_NVTX=1)
else()
    target_compile_definitions(ivf_search_lib PUBLIC USE_NVTX=0)
endif()
set_target_properties(ivf_search_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)
target_link_libraries(ivf_search_lib
    PUBLIC
    l2norm_lib
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
)
if(IVFTENSOR_USE_NVTX)
    target_link_libraries(ivf_search_lib PUBLIC nvToolsExt)
endif()

# ivf_search_pipeline
#
#
#
#   - ivf_search_pipeline.cu: batch_search_pipeline

add_library(ivf_search_pipeline_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_separated.cu  #
)
target_include_directories(ivf_search_pipeline_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(ivf_search_pipeline_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #
)
target_link_libraries(ivf_search_pipeline_lib
    PUBLIC
    l2norm_lib
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
)

#

# CPU Fine SearchGPU coarse + CPU fine
add_subdirectory(cuda/search/cpu_fine)

#

#  ivf_search_wrapper  C
#  STATIC  PIC
# ivf_bcs.cppBCS  CPU lookup
add_library(ivf_search_wrapper_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/dataset/dataset_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/layout/ivf_bcs.cpp
)
target_include_directories(ivf_search_wrapper_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(ivf_search_wrapper_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON  #  PIC
)
target_link_libraries(ivf_search_wrapper_lib
    PUBLIC
    kmeans_lib  #  kmeans  init_centroids_by_sampling
    ivf_search_lib
    ivf_search_pipeline_lib
    l2norm_lib
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
    curand
    cusparse
    cudart
)

# Python IVFTensor
#  Python 3.10 Python
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    #  Python
    find_package(Python COMPONENTS Interpreter Development QUIET)
    if(Python_FOUND)
        set(Python3_EXECUTABLE ${Python_EXECUTABLE})
        set(Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
        set(Python3_LIBRARIES ${Python_LIBRARIES})
    endif()
endif()

#  pybind11
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "Found pybind11, building Python bindings")
    if(Python3_EXECUTABLE)
        message(STATUS "Using Python: ${Python3_EXECUTABLE}")
        message(STATUS "Python version: ${Python3_VERSION}")
    endif()

    #  Python  Python
    pybind11_add_module(PyIVFTensor
        ${CMAKE_CURRENT_SOURCE_DIR}/python/ivf_tensor_bindings.cpp
    )

    #  Python
    if(Python3_EXECUTABLE)
        set_target_properties(PyIVFTensor PROPERTIES
            Python_EXECUTABLE ${Python3_EXECUTABLE}
        )
    endif()

    target_include_directories(PyIVFTensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${pybind11_INCLUDE_DIRS}
    )

    target_link_libraries(PyIVFTensor PRIVATE
        ivf_search_wrapper_lib
        kmeans_lib  #  kmeans
        ivf_search_lib
        ivf_search_pipeline_lib
        l2norm_lib
        utils_lib
        indexed_gemm_lib
        fusion_dist_topk_warpsort_lib
        cudart
        cublas
        curand
        cusparse
    )

    #
    set_target_properties(PyIVFTensor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/build
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    #  ctypes
    add_library(ivf_search_shared SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/search/ivf_search_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/dataset/dataset_wrapper.cpp
    )
    target_include_directories(ivf_search_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_link_libraries(ivf_search_shared PUBLIC
        kmeans_lib  #  kmeans
        ivf_search_lib
        ivf_search_pipeline_lib
        l2norm_lib
        utils_lib
        indexed_gemm_lib
        fusion_dist_topk_warpsort_lib
        cudart
        cublas
        curand
        cusparse
    )
    set_target_properties(ivf_search_shared PROPERTIES
        OUTPUT_NAME "ivf_search"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
else()
    message(WARNING "pybind11 not found, skipping Python bindings. Install with: pip install pybind11")
endif()
