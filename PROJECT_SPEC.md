# Mini-Aladdin — Technical Specification

## Golden Rules (read first)

- **PROJECT_SPEC.md** = source of truth for all technical decisions
- **README.md** = recruiter-facing only, no code, no implementation details  
- **Any agent completing a task MUST update PROJECT_SPEC.md** before closing
- **README.md is only updated** if benchmark results or structure changed

---

## Agent Workflow Rules

These rules apply to ANY agent working on this codebase, at any time.

### Before starting any task
1. Read PROJECT_SPEC.md entirely before writing any code
2. If the task contradicts PROJECT_SPEC.md, stop and ask for clarification
3. Never "fix" naive versions — de-optimizations are intentional (see section 6)

### After completing any task
1. **Update PROJECT_SPEC.md** to reflect any architectural decision made
   - New file created → add it to section 2 (executables) or section 3 (core)
   - New optimization added → document it in section 5
   - New constraint decided → add it to the relevant section
   - **Do NOT paste source code** — reference by file path only

2. **Update README.md** if and only if:
   - Benchmark results changed → update the comparison table
   - A new executable was added → update the 6 executables table
   - Build instructions changed → update Quick Start
   - Never add implementation details to README.md

3. **Run the validation checklist** before closing the task:
   - [ ] PROJECT_SPEC.md reflects current state of the codebase
   - [ ] README.md contains no source code snippets
   - [ ] PROJECT_SPEC.md contains no copy-pasted source code
   - [ ] All new files referenced by path in PROJECT_SPEC.md
   - [ ] Naive versions still contain their intentional de-optimizations

---

## 1. Project Purpose & Audience

This project demonstrates high-performance computing (HPC) expertise through a GPU-accelerated financial options pricing engine. It targets quantitative finance and HPC engineering roles, with a focus on measurable optimization impact through naive-to-optimized comparisons.

The key differentiator is the intentional inclusion of unoptimized baselines that allow Nsight profiling to quantify the exact impact of each CUDA optimization technique.

---

## 2. The 6 Executables — Roles & Responsibilities

| Executable | Source File | Role | Key Design Constraints |
|------------|-------------|------|------------------------|
| `black_scholes_cpu` | `src/black_scholes/cpu/src/bs_cpu_pricer.cpp` | Baseline | Sequential CPU, std::span for zero-copy, AVX2 auto-vectorization via flags |
| `black_scholes_gpu_naive` | `src/black_scholes/gpu_naive/src/bs_gpu_naive.cu` | Naive baseline | AoS layout, BLOCK_SIZE=32, sync transfers, pageable memory, inline math |
| `black_scholes_gpu_optimized` | `src/black_scholes/gpu_optimized/src/bs_gpu_optimized.cu` | Optimized | SoA layout, BLOCK_SIZE=256, __host__ __device__ math reuse |
| `monte_carlo_cpu` | `src/monte_carlo/cpu/src/mc_cpu_pricer.cpp` | Baseline | std::execution::par_unseq, std::mt19937_64 per thread |
| `monte_carlo_gpu_naive` | `src/monte_carlo/gpu_naive/src/mc_gpu_naive.cu` | Naive baseline | AoS, BLOCK_SIZE=32, cuRAND host API, double precision, global accumulation |
| `monte_carlo_gpu_optimized` | `src/monte_carlo/gpu_optimized/src/mc_gpu_optimized.cu` | Optimized | SoA, BLOCK_SIZE=256, cuRAND device API, register accumulation |

---

## 3. Core Library — Design Decisions

### option.hpp
**What**: Defines `OptionContract` (AoS) and `OptionBatch` (SoA) data structures.
**Why**: CPU code benefits from cache-friendly AoS; GPU optimized code requires coalesced SoA.
**Decisions**:
- `OptionContract` includes padding to 64 bytes (cache line) for CPU cache efficiency
- `OptionBatch` uses separate vectors for each field to enable GPU memory coalescing
- `is_call` stored as `int8_t` (not bool) for explicit 1-byte width and GPU compatibility

### math_utils.hpp
**What**: `normalCDF` and `normalPDF`, compiled for both CPU and GPU.
**Why**: Avoids duplicating math between CPU pricer and GPU kernel.
**Decisions**:
- Uses `erfc()` instead of polynomial approximation: numerically stable in distribution tails
- `__host__ __device__` without `#else` branch: `erfc()` works in global namespace on both CPU and GPU
- `constexpr` constants `SQRT1_2`, `INV_SQRT2PI`: evaluated at compile time, no runtime cost

### timer.hpp
**What**: High-resolution timer with milliseconds and nanoseconds precision.
**Why**: Benchmarking requires microsecond-level precision for short GPU kernels.
**Decisions**:
- `std::chrono::steady_clock` (monotonic, not system clock)
- `ScopedTimer` RAII class for automatic timing of code blocks
- `throughput()` helper computes options/second from elapsed time

### benchmark_runner.hpp
**What**: Statistical benchmarking harness with warmup and multiple runs.
**Why**: Cold cache and branch predictor warming are essential for reproducible results.
**Decisions**:
- `n_warmup = 3` runs before timing (stabilizes cache and branch predictor)
- `n_runs = 10` for statistical significance (stddev calculation)
- JSON output for automated aggregation by `aggregate_results.py`
- Designated initializers in C++20 for `BenchmarkResult` construction

### csv_loader.hpp
**What**: Parses `data/input_options.csv` into `OptionBatch`.
**Why**: Simple CSV format for easy dataset generation and inspection.
**Decisions**:
- Header skip on first line
- `std::stringstream` for parsing (simpler than regex for fixed format)
- Returns empty batch on file error (caller checks `batch.size()`)

---

## 4. Data Layout Rules

### When to use AoS (OptionContract)
- **CPU implementations** — cache locality: loading one option brings all fields into cache
- **Naive GPU** — intentional anti-pattern to demonstrate coalescing impact

### When to use SoA (OptionBatch)
- **Optimized GPU** — memory coalescing: 32 threads read 32 contiguous doubles = 1 transaction

### Hardware reason
GPU memory bus is 256 bytes wide on T4. With SoA:
- 32 threads × 8 bytes = 256 bytes = exactly 1 memory transaction
- With AoS: 32 threads read 48 bytes apart = 32 separate transactions

### Rule
- Optimized GPU code MUST use SoA (via `OptionBatch`)
- Naive GPU code MUST use AoS (intentional for baseline comparison)
- CPU code can use either; `OptionContract` is preferred for readability

---

## 5. CUDA Optimization Decisions

### Optimization: SoA Memory Layout
**Applied in**: `black_scholes_gpu_optimized`, `monte_carlo_gpu_optimized`
**NOT applied in**: `black_scholes_gpu_naive`, `monte_carlo_gpu_naive` (intentional baseline)
**Hardware reason**: 32 threads in a warp read contiguous addresses → 1 memory transaction instead of 32. T4 memory bus is 256 bytes wide = exactly 32 doubles.
**Expected impact**: Global Load Efficiency 3% → 98% (verify in Nsight Compute → Memory Workload Analysis → Global Load Efficiency)

### Optimization: BLOCK_SIZE=256
**Applied in**: gpu_optimized versions
**NOT applied in**: gpu_naive versions (BLOCK_SIZE=32, intentional)
**Hardware reason**: 256 threads = 8 warps per block. SM can hold 32 warps active. With 8 warps per block, SM juggles multiple blocks → hides memory latency. With 32 threads (1 warp), SM stalls waiting for memory.
**Expected impact**: SM Occupancy 20% → 75%

### Optimization: cuRAND Device API
**Applied in**: `monte_carlo_gpu_optimized` only
**NOT applied in**: `monte_carlo_gpu_naive` (uses cuRAND host API, intentional)
**Hardware reason**: Device API generates random numbers on-chip per thread. Host API generates on CPU then transfers entire buffer via PCIe.
**Expected impact**: Eliminates ~800MB PCIe transfer per run (10k options × 10k sims × 8B)

### Optimization: Register Accumulation
**Applied in**: `monte_carlo_gpu_optimized` only
**NOT applied in**: `monte_carlo_gpu_naive` (kernel still uses local register for sum, but the key difference is RNG generation location)
**Hardware reason**: Local variables live in registers (0 latency). Global memory access costs ~600 cycles. With 10k sims per thread, naive version with global RNG buffer = 10k × PCIe latency stalls.
**Expected impact**: Removes PCIe bottleneck entirely

### Optimization: Mixed Precision (where applicable)
**Applied in**: Could be applied in Monte Carlo (not currently implemented)
**Rule**: float for intermediate simulation (Z, S_T), double for final accumulation
**Hardware reason**: T4 FP32 peak = 65 TFLOPS, FP64 peak = 8 TFLOPS → 8x throughput on inner loop. Double accumulation prevents drift across 100k additions.
**Note**: Currently all implementations use double throughout. Mixed precision is a future optimization.

---

## 6. Naive Versions — Intentional Anti-patterns

**CRITICAL**: The naive versions contain deliberate anti-patterns. Do NOT "fix" them.

### Black-Scholes GPU Naive (`src/black_scholes/gpu_naive/src/bs_gpu_naive.cu`)

| Anti-pattern | Location | What | Why Intentional |
|--------------|----------|------|-----------------|
| AoS layout | `struct OptionContractGPU` | Thread i reads 48 bytes away from thread i+1 | Demonstrates coalescing impact (~3% efficiency) |
| BLOCK_SIZE=32 | Kernel launch | 1 warp per block | Demonstrates occupancy impact (~20% occupancy) |
| Pageable memory | `std::vector` host buffers | cudaMemcpy stages through pinned buffer | Shows async transfer limitation |
| Inline math | Kernel body | No __device__ helper functions | Shows code reuse limitation (minor impact) |

### Monte Carlo GPU Naive (`src/monte_carlo/gpu_naive/src/mc_gpu_naive.cu`)

| Anti-pattern | Location | What | Why Intentional |
|--------------|----------|------|-----------------|
| AoS layout | `struct OptionContractGPU` | Same as BS naive | Demonstrates coalescing impact |
| cuRAND host API | `curandGenerateNormalDouble` | Generates RNG on CPU | Demonstrates 800MB PCIe bottleneck |
| BLOCK_SIZE=32 | Kernel launch | Same as BS naive | Demonstrates occupancy impact |
| Double precision | All calculations | No float intermediate | Shows 8x throughput waste on T4 |

---

## 7. Benchmarking Rules

### Methodology
- `n_warmup = 3`: Cold cache warming, branch predictor training
- `n_runs = 10`: Statistical significance for stddev calculation
- CPU baseline = speedup denominator for each family (BS CPU for BS GPU, MC CPU for MC GPU)

### JSON Output Format
```json
{
  "name": "executable_name",
  "n_options": 1000000,
  "mean_ms": 9.123456,
  "stddev_ms": 0.234567,
  "min_ms": 8.901234,
  "max_ms": 9.456789,
  "throughput": 109587241.234,
  "speedup": 1.0
}
```

### Speedup Calculation
`aggregate_results.py` computes speedup relative to CPU baseline for each algorithm family:
- Black-Scholes: `speedup = bs_cpu_time / gpu_time`
- Monte Carlo: `speedup = mc_cpu_time / gpu_time`

---

## 8. Validation Tolerances

```
BS GPU naive vs BS CPU:      1e-6  (deterministic, exact match expected)
BS GPU optimized vs BS CPU:  1e-6  (deterministic, exact match expected)
MC CPU vs BS analytical:     1e-2  (Monte Carlo statistical error, 100k sims)
MC GPU naive vs MC CPU:      1e-2  (same statistical error)
MC GPU optimized vs MC CPU:  1e-2  (same statistical error)
```

**Why tolerances differ**: Black-Scholes is deterministic analytical formula — results should match exactly. Monte Carlo is statistical simulation with inherent variance from random sampling. With 100k simulations per option, expected standard error is ~0.3% (1e-2 tolerance allows for 3-sigma variation).

---

## 9. C++20 Rules

### [[nodiscard]] on all pricing functions
→ Silently ignoring a computed price = financial bug. Compiler catches it.

### noexcept on all hot-path functions
→ Enables compiler to skip exception handling overhead on critical paths.

### std::span for batch parameters
→ Non-owning view. Zero copy. Works with vector, array, raw pointer.
→ Signals clearly: this function reads your data, does not own it.

### std::execution::par_unseq in MC CPU
→ `par` = thread pool (no raw threads). `unseq` = SIMD within each thread.

### Designated initializers
```cpp
OptionContract opt{
    .S = S[i], .K = K[i], .r = r[i],
    .sigma = sigma[i], .T = T[i],
    .is_call = static_cast<bool>(is_call[i])
};
```
→ Self-documenting, order-independent, C++20 feature.

### Digit separators: 100'000, 1'000'000
→ Readability. Required for all large numeric literals.

### constexpr for BLOCK_SIZE and math constants
→ Compile-time evaluation. Enables compiler to optimize grid size calculation.

---

## 10. Build System Rules

### Header-only core library
`core` is an INTERFACE target in CMake. No compilation unit — headers included directly.
→ Faster builds, no linking overhead, templates work across translation units.

### CUDA architectures: 61;75;86
- **61**: Pascal (GTX 1080, P100) — baseline compatibility
- **75**: Turing (RTX 2080, T4) — primary development target
- **86**: Ampere (RTX 3080, A100) — future-proofing

Why this list: Covers 3 major generations. PTX fallback handles newer/older GPUs at runtime.

### MSVC flags
- `/O2` — Maximum optimization
- `/arch:AVX2` — Enable AVX2 vectorization for CPU code
- `/fp:fast` — Allow reordering for vectorization (safe for our use case)
- `/std:c++latest` — C++20 features

### GCC/Clang flags
- `-O3` — Maximum optimization
- `-march=native` — Target host CPU capabilities
- `-ffast-math` — Same as /fp:fast

### GPU CMakeLists.txt pattern
Each GPU executable subdirectory contains:
```cmake
cmake_minimum_required(VERSION 3.20)
add_executable(name src/kernel.cu src/main.cpp)
target_include_directories(name PRIVATE include ${CMAKE_SOURCE_DIR}/src/core/include)
target_link_libraries(name core)
set_target_properties(name PROPERTIES CUDA_ARCHITECTURES "61;75;86")
```

---

## 11. Nsight Profiling Guide

After running `scripts/profile_nsight.sh`, analyze results:

### Nsight Systems (.nsys-rep)
- **Timeline view**: Are H→D transfers overlapping with computation?
- If not overlapping → check if pinned memory is used (naive versions don't; optimized should)

### Nsight Compute (.ncu-rep)
- **Global Load Efficiency**: Should be ~3% naive, ~98% optimized
- **SM Occupancy**: Should be ~20% naive, ~75% optimized  
- **Memory Throughput**: Should be ~9 GB/s naive, ~287 GB/s optimized (T4 theoretical max)
- **Compute Throughput**: Increases with optimizations (less stall, more compute)

### Where to find metrics
1. Open `.ncu-rep` in Nsight Compute GUI
2. Navigate to "Memory Workload Analysis" section
3. Check "Global Load Efficiency" percentage
4. Navigate to "Occupancy" section
5. Check "Achieved Occupancy" percentage

---

## 12. Known Limitations & Future Work

### Current Limitations
- **No CUDA Streams**: BS optimized does not overlap transfer and compute (would add ~10% speedup)
- **No Unified Memory**: All explicit cudaMalloc/cudaMemcpy
- **Single GPU**: No multi-GPU scaling

### Future Phases (not started)
- **Phase 2**: Heston stochastic volatility model (more complex than Black-Scholes)
- **Phase 3**: VaR (Value at Risk) calculation, portfolio-level Greeks
- **Phase 4**: Multi-asset basket options, correlation handling

### Optimizations to Consider
- CUDA Streams for transfer/compute overlap
- Mixed precision (float intermediate, double final) in Monte Carlo
- Constant memory for model parameters
- Shared memory for intra-block data sharing
- Warp shuffle for intra-warp reductions

---

## 13. File Reference

### Core Library
- `src/core/include/option.hpp` — Data structures (AoS/SoA)
- `src/core/include/math_utils.hpp` — normalCDF, normalPDF
- `src/core/include/timer.hpp` — High-resolution timer
- `src/core/include/benchmark_runner.hpp` — Statistical benchmarking
- `src/core/include/csv_loader.hpp` — CSV parsing

### Black-Scholes
- `src/black_scholes/cpu/include/bs_cpu_pricer.hpp`
- `src/black_scholes/cpu/src/bs_cpu_pricer.cpp`
- `src/black_scholes/cpu/src/main.cpp`
- `src/black_scholes/gpu_naive/include/bs_gpu_naive.cuh`
- `src/black_scholes/gpu_naive/src/bs_gpu_naive.cu`
- `src/black_scholes/gpu_naive/src/main.cpp`
- `src/black_scholes/gpu_optimized/include/bs_gpu_optimized.cuh`
- `src/black_scholes/gpu_optimized/src/bs_gpu_optimized.cu`
- `src/black_scholes/gpu_optimized/src/main.cpp`

### Monte Carlo
- `src/monte_carlo/cpu/include/mc_cpu_pricer.hpp`
- `src/monte_carlo/cpu/src/mc_cpu_pricer.cpp`
- `src/monte_carlo/cpu/src/main.cpp`
- `src/monte_carlo/gpu_naive/include/mc_gpu_naive.cuh`
- `src/monte_carlo/gpu_naive/src/mc_gpu_naive.cu`
- `src/monte_carlo/gpu_naive/src/main.cpp`
- `src/monte_carlo/gpu_optimized/include/mc_gpu_optimized.cuh`
- `src/monte_carlo/gpu_optimized/src/mc_gpu_optimized.cu`
- `src/monte_carlo/gpu_optimized/src/main.cpp`

### Build & Scripts
- `CMakeLists.txt` — Root build configuration
- `src/black_scholes/gpu_naive/CMakeLists.txt`
- `src/black_scholes/gpu_optimized/CMakeLists.txt`
- `src/monte_carlo/gpu_naive/CMakeLists.txt`
- `src/monte_carlo/gpu_optimized/CMakeLists.txt`
- `scripts/benchmark_all.sh` — Run all benchmarks
- `scripts/profile_nsight.sh` — Nsight profiling
- `scripts/aggregate_results.py` — Results aggregation

---

*Last updated: March 2026 — Documentation refactoring complete*
