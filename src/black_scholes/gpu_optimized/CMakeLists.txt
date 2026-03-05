cmake_minimum_required(VERSION 3.20)

add_executable(black_scholes_gpu_optimized
    src/bs_gpu_optimized.cu
    src/main.cpp
)

target_include_directories(black_scholes_gpu_optimized PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(black_scholes_gpu_optimized core)

set_target_properties(black_scholes_gpu_optimized PROPERTIES CUDA_ARCHITECTURES "61;75;86")
