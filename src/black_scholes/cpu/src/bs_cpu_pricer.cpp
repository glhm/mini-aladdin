#include "bs_cpu_pricer.hpp"
#include "math_utils.hpp"
#include <cmath>

namespace mini_aladdin::pricing {

[[nodiscard]] PricingResult price_option_bs(const OptionContract& opt) noexcept {
    using namespace mini_aladdin::math;

    const double sqrt_T    = std::sqrt(opt.T);
    const double inv_sigT  = 1.0 / (opt.sigma * sqrt_T);
    const double exp_rT    = std::exp(-opt.r * opt.T);
    const double disc_K    = opt.K * exp_rT;

    const double d1 = (std::log(opt.S / opt.K) + (opt.r + 0.5 * opt.sigma * opt.sigma) * opt.T)
                      * inv_sigT;
    const double d2 = d1 - opt.sigma * sqrt_T;

    const double Nd1  = normalCDF(d1);
    const double Nd2  = normalCDF(d2);
    const double Nnd1 = 1.0 - Nd1;
    const double Nnd2 = 1.0 - Nd2;
    const double nd1  = normalPDF(d1);

    PricingResult result;

    if (opt.is_call) {
        result.price         = opt.S * Nd1 - disc_K * Nd2;
        result.greeks.delta  = Nd1;
        result.greeks.theta  = -(opt.S * nd1 * opt.sigma / (2.0 * sqrt_T))
                                - opt.r * disc_K * Nd2;
    } else {
        result.price         = disc_K * Nnd2 - opt.S * Nnd1;
        result.greeks.delta  = Nd1 - 1.0;
        result.greeks.theta  = -(opt.S * nd1 * opt.sigma / (2.0 * sqrt_T))
                                + opt.r * disc_K * Nnd2;
    }
    
    result.greeks.gamma  = nd1 / (opt.S * opt.sigma * sqrt_T);
    result.greeks.vega   = opt.S * nd1 * sqrt_T * 0.01;

    return result;
}

void price_batch_bs_cpu(
    std::span<const double> S,
    std::span<const double> K,
    std::span<const double> r,
    std::span<const double> sigma,
    std::span<const double> T,
    std::span<const int8_t> is_call,
    std::span<double>       out_prices
) noexcept {
    const std::size_t n = S.size();
    
    for (std::size_t i = 0; i < n; ++i) {
        OptionContract opt{
            .S = S[i], .K = K[i], .r = r[i],
            .sigma = sigma[i], .T = T[i],
            .is_call = static_cast<bool>(is_call[i])
        };
        out_prices[i] = price_option_bs(opt).price;
    }
}

} // namespace mini_aladdin::pricing
