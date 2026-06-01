# cuda/search/cpu_fineCPU Fine Search
#
#  .cpp  per-file COMPILE_OPTIONS
#   V0  -O0 -fno-tree-vectorize     Reference scalar path
#   V1  -O3 -march=native -ffast-math -funroll-loops
#   V2  V1 + -mavx512f -mfma
#   V3  V2 intrinsics  AVX512
#   V4  V3 + prefetch
#   V5  V3 + AoSoA
#
#  x86_64 Linuxgcc/clang V0/V1
#  Windows/ARM  V0/V1 AVX-512  V2..V5
#  #ifdef  fallback

cmake_minimum_required(VERSION 3.18)

set(CPU_FINE_SOURCES
    cpu_fine_kernels_v0.cpp
    cpu_fine_kernels_v1.cpp
    cpu_fine_kernels_v2.cpp
    cpu_fine_kernels_v3.cpp
    cpu_fine_kernels_v4.cpp
    cpu_fine_kernels_v5.cpp
    cpu_fine_kernels_v6.cpp
    cpu_fine_kernels_v3_u8.cpp
    cpu_fine_kernels_pq_resid.cpp
    cpu_fine_dispatch.cpp
    cpu_fine_layout.cpp
    ivf_search_cpu_fine.cu
)

add_library(cpu_fine_lib STATIC ${CPU_FINE_SOURCES})

target_include_directories(cpu_fine_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

set_target_properties(cpu_fine_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cpu_fine_lib PUBLIC
    l2norm_lib
    utils_lib
    fusion_dist_topk_warpsort_lib
    cublas
)

# OpenMPCPU fine
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_fine_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# -------------------------------------------------------------------
# Per-file compile options""
# -------------------------------------------------------------------
#  Linux/GCC/Clang Windows MSVC  -march=native
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # V0 + -O0 reference path
    set_source_files_properties(cpu_fine_kernels_v0.cpp PROPERTIES
        COMPILE_OPTIONS "-O0;-fno-tree-vectorize;-fno-tree-slp-vectorize;-fopenmp"
    )
    # V1..V5-O3 -march=native -ffast-math
    set(_COMMON_OPTS "-O3;-march=native;-ffast-math;-funroll-loops;-fopenmp")
    set_source_files_properties(cpu_fine_kernels_v1.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS}"
    )
    set_source_files_properties(cpu_fine_kernels_v2.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mfma"
    )
    set_source_files_properties(cpu_fine_kernels_v3.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mfma"
    )
    set_source_files_properties(cpu_fine_kernels_v4.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mfma"
    )
    set_source_files_properties(cpu_fine_kernels_v5.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mfma"
    )
    set_source_files_properties(cpu_fine_kernels_v6.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mfma"
    )
    # V3-u8 AVX-512 BW_mm512_cvtepu8_epi16 / _mm512_madd_epi16
    set_source_files_properties(cpu_fine_kernels_v3_u8.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mavx512bw;-mfma"
    )
    # PQ residual AVX-512 FMALUT AVX-512 BWrerank u8 L2
    set_source_files_properties(cpu_fine_kernels_pq_resid.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS};-mavx512f;-mavx512bw;-mfma"
    )
    # dispatch / layout
    set_source_files_properties(cpu_fine_dispatch.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS}"
    )
    set_source_files_properties(cpu_fine_layout.cpp PROPERTIES
        COMPILE_OPTIONS "${_COMMON_OPTS}"
    )
endif()
