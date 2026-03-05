cmake_minimum_required(VERSION 3.20)

add_executable(black_scholes_gpu_naive
    src/bs_gpu_naive.cu
    src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/black_scholes/cpu/src/bs_cpu_pricer.cpp
)

target_include_directories(black_scholes_gpu_naive PRIVATE
    include
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/black_scholes/cpu/include
)

target_link_libraries(black_scholes_gpu_naive core)

set_target_properties(black_scholes_gpu_naive PROPERTIES CUDA_ARCHITECTURES "61;75;86")
