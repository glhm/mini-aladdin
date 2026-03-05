#pragma once
#include <cstddef>
#include <vector>

namespace mini_aladdin::pricing {

std::vector<double> price_batch_mc_gpu_managed(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call,
    int                        n_simulations = 100000
);

} // namespace mini_aladdin::pricing
