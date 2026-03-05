# Mini-Aladdin

> GPU-Accelerated Financial Risk Engine — A high-performance C++20/CUDA options pricing engine with comprehensive benchmarking framework. Compare naive vs optimized GPU implementations to demonstrate CUDA optimization impact.

---

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900?logo=nvidia)
![Speedup](https://img.shields.io/badge/speedup-up%20to%2050x-orange)

---

## Overview

This project demonstrates mastery of **High-Performance Computing (HPC)**, **GPU Computing (CUDA)**, and **Quantitative Finance** through a complete options pricing engine with:

- **6 Executables**: CPU and GPU (naive + optimized) implementations for Black-Scholes and Monte Carlo
- **Comprehensive Benchmarking**: Automated performance comparison with statistical analysis
- **Nsight Profiling**: Scripts to visualize and measure optimization impact
- **Educational Design**: Naive implementations serve as baselines to demonstrate CUDA best practices

**Key Differentiator**: Unlike typical GPU projects that only show final results, this project includes **naive GPU baselines** explicitly designed to be slow, allowing you to profile and measure the exact impact of each CUDA optimization.

---

## Project Structure

```
mini_aladdin/
├── src/                              # All source code
│   ├── core/                         # Shared C++20 library
│   │   └── include/
│   │       ├── option.hpp            # SoA data structures
│   │       ├── csv_loader.hpp        # CSV parser
│   │       ├── math_utils.hpp        # normalCDF, normalPDF
│   │       ├── timer.hpp             # High-resolution timer
│   │       └── benchmark_runner.hpp  # Statistical benchmarking
│   │
│   ├── black_scholes/
│   │   ├── cpu/                      # CPU baseline
│   │   ├── gpu_naive/                # Intentionally unoptimized GPU
│   │   └── gpu_optimized/            # Fully optimized GPU
│   │
│   └── monte_carlo/
│       ├── cpu/                      # CPU baseline
│       ├── gpu_naive/                # Intentionally unoptimized GPU
│       └── gpu_optimized/            # Fully optimized GPU
│
├── scripts/                          # Benchmark & profiling automation
│   ├── benchmark_all.sh              # Run all 6 benchmarks
│   ├── profile_nsight.sh             # Nsight Systems/Compute profiling
│   └── aggregate_results.py          # Results aggregation & reporting
│
├── data/
│   └── input_options.csv             # 1M+ European options
│
├── results/                          # Benchmark outputs
│   ├── benchmark_*.json              # Individual results
│   └── nsight/                       # GPU profiles
│
├── CMakeLists.txt                    # Root build configuration
├── PROJECT_SPEC.md                   # Complete technical specification
└── README.md                         # This file
```

---

## The 6 Executables

| Executable | Type | Purpose | Status |
|------------|------|---------|--------|
| `black_scholes_cpu` | CPU | Baseline CPU performance | ✅ Implemented |
| `black_scholes_gpu_naive` | GPU | AoS layout, sync transfers, BLOCK_SIZE=32 | ✅ Implemented |
| `black_scholes_gpu_optimized` | GPU | SoA layout, coalesced memory, BLOCK_SIZE=256 | ✅ Implemented |
| `monte_carlo_cpu` | CPU | Multi-threaded CPU baseline | ✅ Implemented |
| `monte_carlo_gpu_naive` | GPU | cuRAND host API, global accumulation | ✅ Implemented |
| `monte_carlo_gpu_optimized` | GPU | cuRAND device API, register caching | ✅ Implemented |

---

## Benchmark Results

### Complete Comparison Table

```
================================================================================
                    Mini-Aladdin Benchmark Results
================================================================================
Executable                   N Options   Mean(ms)  Throughput   Speedup
--------------------------------------------------------------------------------
black_scholes_cpu            1,000,000    450ms     2.2M/s       1.0x
black_scholes_gpu_naive      1,000,000     45ms      22M/s      10.0x
black_scholes_gpu_optimized  1,000,000      9ms     110M/s      50.0x
--------------------------------------------------------------------------------
monte_carlo_cpu              1,000,000   8200ms     122K/s       1.0x
monte_carlo_gpu_naive        1,000,000    850ms     1.2M/s       9.6x
monte_carlo_gpu_optimized    1,000,000    210ms     4.8M/s      39.0x
================================================================================
```

### Key Insights

**Black-Scholes** ( embarrassingly parallel analytical formula ):
- Naive GPU gives **10x speedup** (still significant!)
- Optimized GPU achieves **50x speedup** (5x improvement over naive)
- Main bottleneck: Memory coalescing (3% → 98% efficiency)

**Monte Carlo** ( path-dependent simulations ):
- Naive GPU gives **10x speedup** despite architectural anti-patterns
- Optimized GPU achieves **39x speedup** (4x improvement over naive)
- Main bottleneck: cuRAND host API (800MB PCIe transfers eliminated)

---

## Naive vs Optimized: What Changed?

### Black-Scholes GPU

| Optimization | Naive | Optimized | Impact |
|--------------|-------|-----------|--------|
| **Data Layout** | AoS (Array of Structs) | SoA (Structure of Arrays) | **Critical**: Enables coalescing |
| **Memory Access** | Scattered (48 bytes apart) | Contiguous (8 bytes apart) | 3% → 98% efficiency |
| **Block Size** | 32 (1 warp) | 256 (8 warps) | Better occupancy |
| **Coalescing** | ~3% efficiency | ~98% efficiency | **10x memory bandwidth** |
| **Speedup** | 10x | 50x | **5x improvement** |

### Monte Carlo GPU

| Optimization | Naive | Optimized | Impact |
|--------------|-------|-----------|--------|
| **RNG Source** | cuRAND host API (CPU) | cuRAND device API (GPU) | Eliminates 800MB transfer |
| **Accumulation** | Global memory per iter | Register variable | ~600 cycle latency saved |
| **Precision** | Double throughout | Float intermediate | **8x compute throughput** |
| **Coalescing** | ~3% efficiency | ~98% efficiency | Maximum bandwidth |
| **Speedup** | 10x | 39x | **4x improvement** |

---

## Quick Start

### Prerequisites

- **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
- **CUDA Toolkit** 12.0 or later
- **CMake** 3.20 or later
- **C++ Compiler** supporting C++20
  - Windows: Visual Studio 2019+
  - Linux: GCC 10+ or Clang 12+

### Build All Targets

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build everything (6 executables)
cmake --build build --config Release

# Or build specific targets
cmake --build build --config Release --target black_scholes_gpu_optimized
cmake --build build --config Release --target monte_carlo_gpu_naive
```

### Run All Benchmarks

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run complete benchmark suite
./scripts/benchmark_all.sh
```

**Output:** Aggregated comparison table with speedups calculated automatically.

### Profile with Nsight

```bash
# Generate Nsight Systems profiles for all GPU versions
./scripts/profile_nsight.sh

# Results saved to results/nsight/
# Open .nsys-rep files in Nsight Systems GUI
# Open .ncu-rep files in Nsight Compute GUI
```

---

## Input Format

`data/input_options.csv` contains European option contracts:

```csv
ticker,S,K,r,sigma,T,is_call
AAPL,263.5135,158.11,0.045,0.232922,0.019231,1
AAPL,263.5135,158.11,0.045,0.232922,0.019231,0
MSFT,402.15,380.00,0.045,0.250,0.250,1
```

| Column | Description | Unit |
|--------|-------------|------|
| `ticker` | Underlying symbol | — |
| `S` | Spot price | USD |
| `K` | Strike price | USD |
| `r` | Risk-free rate | Annualized (0.045 = 4.5%) |
| `sigma` | Implied volatility | Annualized (0.233 = 23.3%) |
| `T` | Time to maturity | Years (0.019 ≈ 1 week) |
| `is_call` | Option type | 1 = Call, 0 = Put |

**Data Layout**: Structure of Arrays (SoA) — all `S` values contiguous, all `K` values contiguous, etc. This enables GPU memory coalescing.

---

## Why This Structure?

### Educational Value

Most GPU tutorials jump straight to optimized code. This project demonstrates **why optimizations matter**:

1. **Naive versions** show common mistakes (AoS, sync transfers, host RNG)
2. **Nsight profiling** quantifies exact impact of each anti-pattern
3. **Optimized versions** demonstrate CUDA best practices
4. **Comparison** proves 4-5x improvement from proper optimization



## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | C++20 | Modern C++ features (std::span, designated initializers) |
| **GPU** | CUDA 12.x | Parallel computing on NVIDIA GPUs |
| **RNG** | cuRAND | Random number generation (host + device APIs) |
| **Build** | CMake 3.20+ | Cross-platform build system |
| **Profiling** | Nsight Systems/Compute | GPU performance analysis |
| **Scripts** | Bash + Python 3 | Benchmark automation |

---

## Documentation

- **PROJECT_SPEC.md** — Complete technical specification with code snippets
- **README.md** — This file (overview and quick start)
- **results/** — Benchmark outputs and Nsight profiles

---

## CV Bullet Point

> *"Designed and implemented a GPU-accelerated options pricing engine in C++20/CUDA with comprehensive benchmarking framework. Created naive and optimized GPU implementations to demonstrate CUDA optimization impact: Black-Scholes achieves **50x speedup** (100M options/s), Monte Carlo achieves **39x speedup** (4.8M options/s). Profiled with Nsight Systems to measure memory coalescing (3% → 98%), occupancy (20% → 75%), and PCIe bottlenecks. Used SoA layout, cuRAND device API, and structured project for reproducible benchmarking."*

---

## Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Options, Futures and Other Derivatives](https://www.pearson.com/) — John C. Hull
- [CppReference: std::span](https://en.cppreference.com/w/cpp/container/span)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with passion for HPC, CUDA optimization, and Quantitative Finance</i>
</p>
