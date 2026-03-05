cmake_minimum_required(VERSION 3.20)

add_executable(monte_carlo_gpu_naive
    src/mc_gpu_naive.cu
    src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/monte_carlo/cpu/src/mc_cpu_pricer.cpp
)

target_include_directories(monte_carlo_gpu_naive PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/monte_carlo/cpu/include
)

target_link_libraries(monte_carlo_gpu_naive core)

target_link_libraries(monte_carlo_gpu_naive curand)

set_target_properties(monte_carlo_gpu_naive PROPERTIES CUDA_ARCHITECTURES "61;75;86")
