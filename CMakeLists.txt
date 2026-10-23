cmake_minimum_required(VERSION 3.24)

project(gemm_foundry
    VERSION 0.1.0
    DESCRIPTION "A CUDA GEMM library with a portable SIMT baseline and architecture-specific kernels"
    LANGUAGES CXX CUDA
)

include(GNUInstallDirs)

# This project uses CUDA FP8 types introduced in CUDA Toolkit 11.8.
find_package(CUDAToolkit 11.8 REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
    message(FATAL_ERROR
        "gemm-foundry requires CUDA Toolkit 11.8 or newer; found ${CMAKE_CUDA_COMPILER_VERSION}."
    )
endif()

# Default baseline targets cover Turing, Ampere, Ada, and Hopper.
# Override this cache variable for your hardware/toolkit, e.g.:
#   -DGEMM_FOUNDRY_CUDA_ARCHITECTURES="80;90"
#   -DGEMM_FOUNDRY_CUDA_ARCHITECTURES="native"  (CMake 3.24+, GPU visible at configure time)
# Blackwell targets (e.g. 100 and 120) require a CUDA Toolkit that supports them.
set(GEMM_FOUNDRY_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING
    "Semicolon-separated CUDA architectures for gemm-foundry (or 'native')"
)

if(GEMM_FOUNDRY_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "GEMM_FOUNDRY_CUDA_ARCHITECTURES must not be empty.")
endif()

option(GEMM_FOUNDRY_BUILD_EXAMPLES "Build gemm-foundry examples" OFF)
option(GEMM_FOUNDRY_BUILD_TESTS "Build gemm-foundry correctness tests" OFF)
option(GEMM_FOUNDRY_BUILD_BENCHMARKS "Build gemm-foundry benchmarks" OFF)

# Apply consistent language standards and architecture targets to every CUDA target.
function(gemm_foundry_configure_cuda_target target_name)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
        CUDA_ARCHITECTURES "${GEMM_FOUNDRY_CUDA_ARCHITECTURES}"
    )
endfunction()

add_library(gemm_foundry
    src/gemm.cu
    include/gemm_foundry/gemm.hpp
    include/gemm_foundry/types.hpp
    include/gemm_foundry/type_traits.hpp
    src/kernels/gemm_naive.cuh
    src/kernels/gemm_tiled.cuh
    src/kernels/gemm_register_blocked.cuh
    src/kernels/gemm_precision.cuh
)

add_library(gemm_foundry::gemm_foundry ALIAS gemm_foundry)

target_include_directories(gemm_foundry
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gemm_foundry
    PUBLIC CUDA::cudart
)

target_compile_features(gemm_foundry
    PUBLIC cxx_std_17 cuda_std_17
)

gemm_foundry_configure_cuda_target(gemm_foundry)

set_target_properties(gemm_foundry PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Optional basic example. The source should exercise the public API in gemm.hpp.
if(GEMM_FOUNDRY_BUILD_EXAMPLES)
    set(_example_source "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_gemm.cu")
    if(NOT EXISTS "${_example_source}")
        message(FATAL_ERROR
            "GEMM_FOUNDRY_BUILD_EXAMPLES=ON, but examples/basic_gemm.cu was not found."
        )
    endif()

    add_executable(gemm_foundry_basic ${_example_source})
    target_link_libraries(gemm_foundry_basic PRIVATE gemm_foundry::gemm_foundry)
    gemm_foundry_configure_cuda_target(gemm_foundry_basic)
endif()

# Optional correctness-test driver. Keep disabled until this source is present and updated.
if(GEMM_FOUNDRY_BUILD_TESTS)
    set(_test_source "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemm_tests.cu")
    if(NOT EXISTS "${_test_source}")
        message(FATAL_ERROR
            "GEMM_FOUNDRY_BUILD_TESTS=ON, but tests/gemm_tests.cu was not found."
        )
    endif()

    enable_testing()
    add_executable(gemm_foundry_tests ${_test_source})
    target_link_libraries(gemm_foundry_tests PRIVATE gemm_foundry::gemm_foundry)
    gemm_foundry_configure_cuda_target(gemm_foundry_tests)
    add_test(NAME gemm_foundry_tests COMMAND gemm_foundry_tests)
endif()

# Optional benchmark driver. The benchmark source must use the current public API.
if(GEMM_FOUNDRY_BUILD_BENCHMARKS)
    set(_benchmark_source "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/gemm_compare.cu")
    if(NOT EXISTS "${_benchmark_source}")
        message(FATAL_ERROR
            "GEMM_FOUNDRY_BUILD_BENCHMARKS=ON, but benchmarks/gemm_compare.cu was not found."
        )
    endif()

    add_executable(gemm_foundry_benchmark ${_benchmark_source})
    target_link_libraries(gemm_foundry_benchmark PRIVATE gemm_foundry::gemm_foundry)
    gemm_foundry_configure_cuda_target(gemm_foundry_benchmark)
endif()

# Install the compiled library and public headers.
install(TARGETS gemm_foundry
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/gemm_foundry
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

message(STATUS "gemm-foundry version: ${PROJECT_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "CUDA architectures: ${GEMM_FOUNDRY_CUDA_ARCHITECTURES}")
message(STATUS "Build examples: ${GEMM_FOUNDRY_BUILD_EXAMPLES}")
message(STATUS "Build tests: ${GEMM_FOUNDRY_BUILD_TESTS}")
message(STATUS "Build benchmarks: ${GEMM_FOUNDRY_BUILD_BENCHMARKS}")
