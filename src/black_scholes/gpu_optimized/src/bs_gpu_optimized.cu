#include "bs_gpu_optimized.cuh"
#include "math_utils.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

namespace mini_aladdin::pricing {

__global__ void bs_kernel(
    std::size_t    n,
    const double*  S,
    const double*  K,
    const double*  r,
    const double*  sigma,
    const double*  T,
    const int8_t*  is_call,
    double*        prices
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (i >= n) return;

    const double Si     = S[i];
    const double Ki     = K[i];
    const double ri     = r[i];
    const double sigi   = sigma[i];
    const double Ti     = T[i];
    const bool   call_i = static_cast<bool>(is_call[i]);

    const double sqrt_T   = sqrt(Ti);
    const double inv_sigT = 1.0 / (sigi * sqrt_T);
    const double exp_rT   = exp(-ri * Ti);
    const double disc_K   = Ki * exp_rT;

    const double d1 = (log(Si / Ki) + (ri + 0.5 * sigi * sigi) * Ti) * inv_sigT;
    const double d2 = d1 - sigi * sqrt_T;

    const double Nd1 = mini_aladdin::math::normalCDF(d1);
    const double Nd2 = mini_aladdin::math::normalCDF(d2);

    prices[i] = call_i
        ? Si * Nd1 - disc_K * Nd2
        : disc_K * (1.0 - Nd2) - Si * (1.0 - Nd1);
}

void price_batch_bs_gpu(
    std::size_t    n,
    const double*  d_S,
    const double*  d_K,
    const double*  d_r,
    const double*  d_sigma,
    const double*  d_T,
    const int8_t*  d_is_call,
    double*        d_prices
) {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bs_kernel<<<grid_size, BLOCK_SIZE>>>(
        n, d_S, d_K, d_r, d_sigma, d_T, d_is_call, d_prices
    );

    cudaDeviceSynchronize();
}

std::vector<double> price_batch_bs_gpu_managed(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call
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
    
    cudaMalloc(&d_S,       bytes_d);
    cudaMalloc(&d_K,       bytes_d);
    cudaMalloc(&d_r,       bytes_d);
    cudaMalloc(&d_sigma,   bytes_d);
    cudaMalloc(&d_T,       bytes_d);
    cudaMalloc(&d_is_call, bytes_i8);
    cudaMalloc(&d_prices,  bytes_d);

    cudaMemcpy(d_S,       S.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,       K.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,       r.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma,   sigma.data(),   bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_T,       T.data(),       bytes_d,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_call, is_call.data(), bytes_i8, cudaMemcpyHostToDevice);

    price_batch_bs_gpu(n, d_S, d_K, d_r, d_sigma, d_T, d_is_call, d_prices);

    std::vector<double> prices(n);
    cudaMemcpy(prices.data(), d_prices, bytes_d, cudaMemcpyDeviceToHost);

    cudaFree(d_S); 
    cudaFree(d_K); 
    cudaFree(d_r);
    cudaFree(d_sigma); 
    cudaFree(d_T); 
    cudaFree(d_is_call); 
    cudaFree(d_prices);

    return prices;
}

} // namespace mini_aladdin::pricing
