#include "mc_gpu_optimized.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

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

__global__ void init_curand_states(
    curandState* states,
    std::size_t  n,
    unsigned long long seed
) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    curand_init(seed, i, 0, &states[i]);
}

__global__ void mc_kernel(
    std::size_t    n,
    const double*  S,
    const double*  K,
    const double*  r,
    const double*  sigma,
    const double*  T,
    const int8_t*  is_call,
    double*        prices,
    curandState*   states,
    int            n_sims
) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double Si    = S[i];
    const double Ki    = K[i];
    const double ri    = r[i];
    const double sigi  = sigma[i];
    const double Ti    = T[i];
    const bool   calli = static_cast<bool>(is_call[i]);

    const double drift     = (ri - 0.5 * sigi * sigi) * Ti;
    const double vol_sqrtT = sigi * sqrt(Ti);
    const double disc      = exp(-ri * Ti);

    curandState local_state = states[i];

    double sum = 0.0;
    for (int k = 0; k < n_sims; ++k) {
        const double Z      = curand_normal_double(&local_state);
        const double S_T    = Si * exp(drift + vol_sqrtT * Z);
        const double payoff = calli
            ? fmax(S_T - Ki, 0.0)
            : fmax(Ki - S_T, 0.0);
        sum += payoff;
    }

    states[i] = local_state;
    prices[i] = disc * sum / n_sims;
}

std::vector<double> price_batch_mc_gpu_managed(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call,
    int                        n_simulations
) {
    const std::size_t n        = S.size();
    const std::size_t bytes_d  = n * sizeof(double);
    const std::size_t bytes_i8 = n * sizeof(int8_t);

    double*  d_S;
    double*  d_K;
    double*  d_r;
    double*  d_sigma;
    double*  d_T;
    double*  d_prices;
    int8_t*  d_is_call;
    curandState* d_states;
    
    cudaMalloc(&d_S,       bytes_d);
    cudaMalloc(&d_K,       bytes_d);
    cudaMalloc(&d_r,       bytes_d);
    cudaMalloc(&d_sigma,   bytes_d);
    cudaMalloc(&d_T,       bytes_d);
    cudaMalloc(&d_is_call, bytes_i8);
    cudaMalloc(&d_prices,  bytes_d);
    cudaMalloc(&d_states,  n * sizeof(curandState));

    cudaMemcpy(d_S,       S.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       K.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,       r.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma,   sigma.data(),   bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_T,       T.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_call, is_call.data(), bytes_i8, cudaMemcpyHostToDevice);

    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_curand_states<<<grid_size, BLOCK_SIZE>>>(d_states, n, 42);
    cudaDeviceSynchronize();

    mc_kernel<<<grid_size, BLOCK_SIZE>>>(
        n, d_S, d_K, d_r, d_sigma, d_T, d_is_call, d_prices, d_states, n_simulations
    );
    cudaDeviceSynchronize();

    std::vector<double> prices(n);
    cudaMemcpy(prices.data(), d_prices, bytes_d, cudaMemcpyDeviceToHost);

    cudaFree(d_S); 
    cudaFree(d_K); 
    cudaFree(d_r);
    cudaFree(d_sigma); 
    cudaFree(d_T); 
    cudaFree(d_is_call);
    cudaFree(d_prices);
    cudaFree(d_states);

    return prices;
}

} // namespace mini_aladdin::pricing
