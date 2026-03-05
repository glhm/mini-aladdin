#pragma once
#include "option.hpp"
#include <vector>
#include <span>

namespace mini_aladdin::pricing {

struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
};

struct PricingResult {
    double price;
    Greeks greeks;
};

[[nodiscard]] PricingResult price_option_bs(const OptionContract& opt) noexcept;

void price_batch_bs_cpu(
    std::span<const double> S,
    std::span<const double> K,
    std::span<const double> r,
    std::span<const double> sigma,
    std::span<const double> T,
    std::span<const int8_t> is_call,
    std::span<double>       out_prices
) noexcept;

} // namespace mini_aladdin::pricing
