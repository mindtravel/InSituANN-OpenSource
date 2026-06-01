cmake_minimum_required(VERSION 3.18)
project(insituann_updates LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED INSITUANN_ROOT)
    get_filename_component(INSITUANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED INSITUANN_BUILD)
    set(INSITUANN_BUILD "${INSITUANN_ROOT}/build")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(UPDATE_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${INSITUANN_ROOT}/cuda"
)

set(UPDATE_LIBS
    "${INSITUANN_BUILD}/cuda/search/cpu_fine/libcpu_fine_lib.a"
    "${INSITUANN_BUILD}/libivf_search_lib.a"
    "${INSITUANN_BUILD}/libkmeans_lib.a"
    "${INSITUANN_BUILD}/libfusion_dist_topk_warpsort_lib.a"
    "${INSITUANN_BUILD}/libl2norm_lib.a"
    "${INSITUANN_BUILD}/libutils_lib.a"
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    OpenMP::OpenMP_CXX
)

function(add_update_runner target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${UPDATE_INCLUDES})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math -Xcompiler=-fopenmp -Xcompiler=-march=native>
    )
    target_link_libraries(${target} PRIVATE ${UPDATE_LIBS})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

add_update_runner(exact_update exact_update.cu)
add_update_runner(pq_update pq_update.cu)
