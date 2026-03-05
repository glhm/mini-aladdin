#pragma once
#include <cstddef>
#include <vector>

namespace mini_aladdin::pricing {

/// Naive Monte Carlo GPU pricer.
/// Intentionally unoptimized to serve as Nsight profiling baseline.
/// Compare with MonteCarlo.GPU.Optimized to measure optimization impact.
[[nodiscard]] std::vector<double> price_batch_mc_gpu_naive(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call,
    int n_sims = 10'000
);

} // namespace mini_aladdin::pricing
