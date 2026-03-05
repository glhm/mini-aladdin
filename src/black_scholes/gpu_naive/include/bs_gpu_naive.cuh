#pragma once
#include <cstddef>
#include <vector>

namespace mini_aladdin::pricing {

/// Naive Black-Scholes GPU pricer.
/// Intentionally unoptimized to serve as Nsight profiling baseline.
/// Compare with BlackScholes.GPU.Optimized to measure optimization impact.
[[nodiscard]] std::vector<double> price_batch_bs_gpu_naive(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call
);

} // namespace mini_aladdin::pricing
