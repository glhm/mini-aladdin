cmake_minimum_required(VERSION 3.20)

add_executable(monte_carlo_gpu_optimized
    src/mc_gpu_optimized.cu
    src/main.cpp
)

target_include_directories(monte_carlo_gpu_optimized PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(monte_carlo_gpu_optimized core)

set_target_properties(monte_carlo_gpu_optimized PROPERTIES CUDA_ARCHITECTURES "61;75;86")
