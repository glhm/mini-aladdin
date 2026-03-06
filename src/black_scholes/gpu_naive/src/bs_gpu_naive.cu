#include "bs_gpu_naive.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Define M_SQRT1_2 for CUDA (not available by default on Windows)
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Help Visual Studio IntelliSense understand CUDA keywords
#ifndef __CUDACC__
#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__
#define threadIdx dim3(0,0,0)
#define blockIdx dim3(0,0,0)
#define blockDim dim3(1,1,1)
#define gridDim dim3(1,1,1)
#endif

namespace mini_aladdin::pricing {

// NAIVE: Array of Structs layout — deliberately breaks GPU memory coalescing.
// Thread i reads option[i].S. In memory, option[i].S and option[i+1].S are
// sizeof(OptionContractGPU) = 48 bytes apart.
// A warp of 32 threads issues 32 non-contiguous memory requests.
// → Global Load Efficiency ~3% (verify in Nsight: Memory Workload Analysis).
// OPTIMIZED version (BlackScholes.GPU.Optimized) uses SoA layout:
// all S[] contiguous → 32 threads read 256 bytes in 1 transaction → 98% efficiency.
struct OptionContractGPU {
    double S;
    double K;
    double r;
    double sigma;
    double T;
    int8_t is_call;
};

__global__ void bs_naive_kernel(
    const OptionContractGPU* options,  // AoS — bad coalescing
    double*                  prices,
    std::size_t              n
) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load option data — non-coalesced memory access pattern
    const double Si     = options[i].S;
    const double Ki     = options[i].K;
    const double ri     = options[i].r;
    const double sigi   = options[i].sigma;
    const double Ti     = options[i].T;
    const bool   call_i = static_cast<bool>(options[i].is_call);

    // NAIVE: Black-Scholes math inlined directly in kernel — no __device__ functions.
    // This is intentional to contrast with the OPTIMIZED version which reuses
    // normalCDF/normalPDF from math_utils.hpp via __host__ __device__ qualifiers.
    // Inline math here is not a performance issue per se, but demonstrates
    // lack of code reuse — the architectural anti-pattern.
    
    const double sqrt_T   = sqrt(Ti);
    const double inv_sigT = 1.0 / (sigi * sqrt_T);
    const double exp_rT   = exp(-ri * Ti);
    const double disc_K   = Ki * exp_rT;

    const double d1 = (log(Si / Ki) + (ri + 0.5 * sigi * sigi) * Ti) * inv_sigT;
    const double d2 = d1 - sigi * sqrt_T;

    // inline normal CDF using erfc
    const double Nd1 = 0.5 * erfc(-d1 * M_SQRT1_2);
    const double Nd2 = 0.5 * erfc(-d2 * M_SQRT1_2);

    prices[i] = call_i
        ? Si * Nd1 - disc_K * Nd2
        : disc_K * (1.0 - Nd2) - Si * (1.0 - Nd1);
}

[[nodiscard]] std::vector<double> price_batch_bs_gpu_naive(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call
) {
    const std::size_t n = S.size();
    const std::size_t bytes_options = n * sizeof(OptionContractGPU);
    const std::size_t bytes_prices  = n * sizeof(double);

    // NAIVE: std::vector uses pageable (non-pinned) host memory.
    // cudaMemcpy with pageable memory internally stages through a pinned buffer,
    // adding hidden overhead and preventing true async transfers.
    // OPTIMIZED version uses cudaMallocHost for all host buffers.
    std::vector<OptionContractGPU> h_options(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_options[i] = OptionContractGPU{
            S[i], K[i], r[i], sigma[i], T[i], is_call[i]
        };
    }

    OptionContractGPU* d_options;
    double*            d_prices;
    
    cudaMalloc(&d_options, bytes_options);
    cudaMalloc(&d_prices,  bytes_prices);

    // NAIVE: cudaMemcpy is synchronous — blocks CPU thread until transfer completes.
    // GPU sits idle waiting for data. No compute/transfer overlap possible.
    // OPTIMIZED version uses cudaMallocHost (pinned memory) + cudaMemcpyAsync
    // + CUDA Streams to overlap PCIe transfer with GPU computation.
    cudaMemcpy(d_options, h_options.data(), bytes_options, cudaMemcpyHostToDevice);

    // NAIVE: BLOCK_SIZE=32 = exactly one warp per block.
    // The SM can hold up to 32 warps concurrently. With only 1 warp per block,
    // the SM cannot hide memory latency by switching to another ready warp.
    // → Low occupancy, high idle cycles (verify in Nsight: Occupancy section).
    // OPTIMIZED version uses BLOCK_SIZE=256 (8 warps per block).
    constexpr int BLOCK_SIZE = 32;
    const int grid_size = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bs_naive_kernel<<<grid_size, BLOCK_SIZE>>>(d_options, d_prices, n);
    cudaDeviceSynchronize();

    std::vector<double> prices(n);
    cudaMemcpy(prices.data(), d_prices, bytes_prices, cudaMemcpyDeviceToHost);

    cudaFree(d_options);
    cudaFree(d_prices);

    return prices;
}

} // namespace mini_aladdin::pricing
