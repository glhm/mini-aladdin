# Mini-Aladdin

> GPU-Accelerated Financial Risk Engine — High-performance options pricing with naive-to-optimized CUDA comparisons demonstrating 50x speedups through memory coalescing, occupancy tuning, and algorithmic optimization.

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.0-76B900?logo=nvidia)
![Speedup](https://img.shields.io/badge/speedup-up%20to%2050x-orange)

**What makes this different**: Unlike typical GPU tutorials that only show final optimized code, this project includes intentionally slow "naive" baselines with documented anti-patterns. Nsight profiling quantifies the exact impact of each optimization — turning abstract CUDA best practices into measurable speedups.

---

## Benchmark Results

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

**Key Insights**:
- **Black-Scholes**: 50x speedup driven by SoA memory layout (coalescing efficiency: 3% → 98%) and occupancy tuning (BLOCK_SIZE 32 → 256)
- **Monte Carlo**: 39x speedup driven by cuRAND device API (eliminates 800MB PCIe transfers) and register accumulation (~600 cycle latency saved per simulation)
- Even "naive" GPU implementations achieve 10x speedup — demonstrating why naive-to-optimized comparisons matter for understanding true optimization impact

---

## What This Demonstrates

- **CUDA Optimization**: Memory coalescing, SM occupancy tuning, cuRAND device vs host API, register caching, mixed-precision compute
- **C++20 Low-Latency Idioms**: `std::span`, `[[nodiscard]]`, `noexcept`, parallel algorithms, designated initializers
- **Quantitative Finance**: Black-Scholes analytical pricing, Monte Carlo path simulation, Greeks computation
- **Benchmarking Methodology**: Statistical timing (warmup + multiple runs), Nsight Systems/Compute profiling, JSON result aggregation

---

## Project Structure

```
mini_aladdin/
├── src/                              # All source code
│   ├── core/                         # Shared C++20 library (header-only)
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
├── PROJECT_SPEC.md                   # Technical documentation
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites

- **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
- **CUDA Toolkit** 12.0 or later
- **CMake** 3.20 or later
- **C++ Compiler** supporting C++20 (Visual Studio 2019+, GCC 10+, Clang 12+)

### Build

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build all 6 executables
cmake --build build --config Release
```

### Run Benchmarks

```bash
# Run complete benchmark suite
./scripts/benchmark_all.sh

# Profile with Nsight
./scripts/profile_nsight.sh
```

---

## Input Format

`data/input_options.csv` contains European option contracts:

```csv
 ticker,S,K,r,sigma,T,is_call
AAPL,263.5135,158.11,0.045,0.232922,0.019231,1
MSFT,402.15,380.00,0.045,0.250,0.250,0
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

---

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

## Troubleshooting

### Visual Studio shows red squiggles under CUDA keywords

This is normal — Visual Studio's IntelliSense doesn't always understand CUDA code, even though nvcc compiles correctly.

**Solutions:**
1. **The code compiles fine** — Ignore red squiggles, build succeeds
2. **Close and reopen VS** — Sometimes helps IntelliSense refresh
3. **Delete .vs folder** and reopen:
   ```bash
   rmdir /s /q .vs
   ```
4. **Check CMakeSettings.json** — Ensure CUDA path is correct

All `.cu` files include IntelliSense helpers that define CUDA keywords when not compiling with nvcc.

---

## Documentation

- **PROJECT_SPEC.md** — Complete technical specification with architecture decisions, optimization rationale, and developer guidelines
- **README.md** — This file (overview and quick start)

---

## Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Options, Futures and Other Derivatives](https://www.pearson.com/) — John C. Hull
- [CppReference: std::span](https://en.cppreference.com/w/cpp/container/span)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with passion for HPC, CUDA optimization, and Quantitative Finance</i>
</p>
