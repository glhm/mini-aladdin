#pragma once
#include "option.hpp"
#include <span>

namespace mini_aladdin::pricing {

[[nodiscard]] double price_option_mc_cpu(
    const OptionContract& opt,
    int                   n_simulations = 100000,
    uint64_t              seed = 42
) noexcept;

void price_batch_mc_cpu(
    std::span<const double> S,
    std::span<const double> K,
    std::span<const double> r,
    std::span<const double> sigma,
    std::span<const double> T,
    std::span<const int8_t> is_call,
    std::span<double>       out_prices,
    int                     n_simulations = 100000
) noexcept;

} // namespace mini_aladdin::pricing
