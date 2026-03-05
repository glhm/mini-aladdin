#!/bin/bash
# Runs all 6 executables sequentially and prints a unified comparison table.
# Usage: ./scripts/benchmark_all.sh
# Requires: all 6 executables built in ./build/

set -e

echo "Running Mini-Aladdin full benchmark suite..."
echo ""

./build/black_scholes_cpu
./build/black_scholes_gpu_naive
./build/black_scholes_gpu_optimized
./build/monte_carlo_cpu
./build/monte_carlo_gpu_naive
./build/monte_carlo_gpu_optimized

echo ""
echo "Aggregating results..."
python3 scripts/aggregate_results.py results/
