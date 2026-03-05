cmake_minimum_required(VERSION 3.20)
project(MiniAladdin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Core library (header-only)
add_library(core INTERFACE)
target_include_directories(core INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/include
)

# CPU optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /fp:fast /std:c++latest")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")
endif()

# Black-Scholes CPU
add_executable(black_scholes_cpu 
    src/black_scholes/cpu/src/bs_cpu_pricer.cpp
    src/black_scholes/cpu/src/main.cpp
)
target_include_directories(black_scholes_cpu PRIVATE 
    src/black_scholes/cpu/include
    src/core/include
)
target_link_libraries(black_scholes_cpu core)

# Monte Carlo CPU
add_executable(monte_carlo_cpu 
    src/monte_carlo/cpu/src/mc_cpu_pricer.cpp
    src/monte_carlo/cpu/src/main.cpp
)
target_include_directories(monte_carlo_cpu PRIVATE 
    src/monte_carlo/cpu/include
    src/core/include
)
target_link_libraries(monte_carlo_cpu core)

# GPU Projects
add_subdirectory(src/black_scholes/gpu_naive)
add_subdirectory(src/black_scholes/gpu_optimized)
add_subdirectory(src/monte_carlo/gpu_naive)
add_subdirectory(src/monte_carlo/gpu_optimized)
