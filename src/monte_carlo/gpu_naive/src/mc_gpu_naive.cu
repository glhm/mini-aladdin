#include "mc_gpu_naive.cuh"
#include <curand.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

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

// NAIVE: AoS layout — same coalescing issue as BlackScholes.GPU.Naive.
// See bs_gpu_naive.cu for detailed explanation.
// OPTIMIZED version (MonteCarlo.GPU.Optimized) uses SoA.
struct OptionContractGPU {
    double S;
    double K;
    double r;
    double sigma;
    double T;
    int8_t is_call;
};

__global__ void mc_naive_kernel(
    const OptionContractGPU* options,  // AoS
    const double*            randoms,  // pre-generated on CPU, n * n_sims elements
    double*                  prices,
    std::size_t              n,
    int                      n_sims
) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load option parameters — non-coalesced access (AoS layout)
    const double Si    = options[i].S;
    const double Ki    = options[i].K;
    const double ri    = options[i].r;
    const double sigi  = options[i].sigma;
    const double Ti    = options[i].T;
    const bool   calli = static_cast<bool>(options[i].is_call);

    // Precompute constants
    const double drift     = (ri - 0.5 * sigi * sigi) * Ti;
    const double vol_sqrtT = sigi * sqrt(Ti);
    const double disc      = exp(-ri * Ti);

    // NAIVE: partial payoff sum written to global memory on every simulation step.
    // Global memory latency: ~600 clock cycles per access on T4.
    // With n_sims=10000 iterations, this causes 10000 * ~600 cycle stalls per thread.
    // Total stall cycles dominate execution time.
    // OPTIMIZED version accumulates in a local register variable (0 latency),
    // writing to global memory exactly once per thread at the end.
    double sum = 0.0;
    
    for (int k = 0; k < n_sims; ++k) {
        // NAIVE: all simulation in double precision throughout.
        // T4 peak throughput: 65 TFLOPS float vs 8 TFLOPS double.
        // Using double for intermediate simulation wastes 8x compute throughput.
        // OPTIMIZED version uses float for Z, S_T computation (intermediate),
        // double only for final payoff accumulation (prevents drift).
        const double Z   = randoms[(size_t)i * n_sims + k];  // double — slow
        const double S_T = Si * exp(drift + vol_sqrtT * Z);   // double exp — slow
        const double payoff = calli
            ? fmax(S_T - Ki, 0.0)
            : fmax(Ki - S_T, 0.0);
        sum += payoff;  // accumulating in register (this part is OK)
    }

    prices[i] = disc * sum / n_sims;
}

[[nodiscard]] std::vector<double> price_batch_mc_gpu_naive(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call,
    int n_sims
) {
    const std::size_t n = S.size();
    const std::size_t bytes_options = n * sizeof(OptionContractGPU);
    const std::size_t bytes_prices  = n * sizeof(double);
    const std::size_t bytes_randoms = (size_t)n * n_sims * sizeof(double);

    // NAIVE: std::vector uses pageable (non-pinned) host memory — same as BS naive
    std::vector<OptionContractGPU> h_options(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_options[i] = OptionContractGPU{
            S[i], K[i], r[i], sigma[i], T[i], is_call[i]
        };
    }

    OptionContractGPU* d_options;
    double*            d_prices;
    double*            d_randoms;
    
    cudaMalloc(&d_options, bytes_options);
    cudaMalloc(&d_prices,  bytes_prices);
    cudaMalloc(&d_randoms, bytes_randoms);

    // NAIVE: cuRAND host API generates ALL random numbers on CPU, then
    // transfers the full buffer to GPU before kernel launch.
    // Buffer size = n_options * n_sims * sizeof(double).
    // Example: 10k options * 10k sims * 8 bytes = 800MB PCIe transfer per run.
    // This is the single biggest bottleneck in this naive version.
    // OPTIMIZED version uses cuRAND device API: each thread generates its own
    // random numbers on-chip using its own curandState — zero transfer overhead.
    std::vector<double> h_randoms((size_t)n * n_sims);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    curandGenerateNormalDouble(gen, h_randoms.data(), (size_t)n * n_sims, 0.0, 1.0);
    curandDestroyGenerator(gen);

    // NAIVE: cudaMemcpy is synchronous — same as BS naive
    cudaMemcpy(d_options, h_options.data(), bytes_options, cudaMemcpyHostToDevice);
    cudaMemcpy(d_randoms, h_randoms.data(), bytes_randoms, cudaMemcpyHostToDevice);

    // NAIVE: BLOCK_SIZE=32 — same as BS naive
    constexpr int BLOCK_SIZE = 32;
    const int grid_size = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    mc_naive_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_options, d_randoms, d_prices, n, n_sims
    );
    cudaDeviceSynchronize();

    std::vector<double> prices(n);
    cudaMemcpy(prices.data(), d_prices, bytes_prices, cudaMemcpyDeviceToHost);

    cudaFree(d_options);
    cudaFree(d_prices);
    cudaFree(d_randoms);

    return prices;
}

} // namespace mini_aladdin::pricing
