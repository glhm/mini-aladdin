#!/bin/bash
# Profiles naive and optimized GPU versions with Nsight Systems and Nsight Compute.
# Usage: ./scripts/profile_nsight.sh
# Requires: Nsight Systems (nsys) and Nsight Compute (ncu) installed

set -e
mkdir -p results/nsight

echo "=== Nsight Systems: macro timeline profiling ==="

# Profile all 4 GPU executables
nsys profile --output results/nsight/bs_gpu_naive \
     --trace cuda,osrt \
     ./build/black_scholes_gpu_naive

nsys profile --output results/nsight/bs_gpu_optimized \
     --trace cuda,osrt \
     ./build/black_scholes_gpu_optimized

nsys profile --output results/nsight/mc_gpu_naive \
     --trace cuda,osrt \
     ./build/monte_carlo_gpu_naive

nsys profile --output results/nsight/mc_gpu_optimized \
     --trace cuda,osrt \
     ./build/monte_carlo_gpu_optimized

echo ""
echo "=== Nsight Compute: kernel-level analysis ==="

# Detailed kernel analysis — focus on Monte Carlo (most complex kernel)
ncu --set full \
    --output results/nsight/mc_gpu_naive_kernel \
    ./build/monte_carlo_gpu_naive

ncu --set full \
    --output results/nsight/mc_gpu_optimized_kernel \
    ./build/monte_carlo_gpu_optimized

echo ""
echo "Nsight profiles saved to results/nsight/"
echo "Open .nsys-rep files in Nsight Systems GUI"
echo "Open .ncu-rep files in Nsight Compute GUI"
echo ""
echo "Key metrics to compare:"
echo "  - Global Load Efficiency (naive ~3% vs optimized ~98%)"
echo "  - SM Occupancy            (naive ~20% vs optimized ~75%)"
echo "  - H->D / D->H overlap     (naive: none vs optimized: streams overlap)"
