#pragma once
#include <cstddef>
#include <vector>

namespace mini_aladdin::pricing {

void price_batch_bs_gpu(
    std::size_t    n,
    const double*  d_S,
    const double*  d_K,
    const double*  d_r,
    const double*  d_sigma,
    const double*  d_T,
    const int8_t*  d_is_call,
    double*        d_prices
);

std::vector<double> price_batch_bs_gpu_managed(
    const std::vector<double>& S,
    const std::vector<double>& K,
    const std::vector<double>& r,
    const std::vector<double>& sigma,
    const std::vector<double>& T,
    const std::vector<int8_t>& is_call
);

} // namespace mini_aladdin::pricing
