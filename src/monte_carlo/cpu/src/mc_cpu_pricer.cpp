#include "mc_cpu_pricer.hpp"
#include "math_utils.hpp"
#include <random>
#include <execution>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mini_aladdin::pricing {

[[nodiscard]] double price_option_mc_cpu(
    const OptionContract& opt,
    int                   n_simulations,
    uint64_t              seed
) noexcept {
    std::mt19937_64 rng{seed};
    std::normal_distribution<double> dist{0.0, 1.0};

    const double drift    = (opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T;
    const double vol_sqrtT = opt.sigma * std::sqrt(opt.T);
    const double disc     = std::exp(-opt.r * opt.T);

    double sum_payoffs = 0.0;

    for (int k = 0; k < n_simulations; ++k) {
        const double Z      = dist(rng);
        const double S_T    = opt.S * std::exp(drift + vol_sqrtT * Z);
        const double payoff = opt.is_call
            ? std::max(S_T - opt.K, 0.0)
            : std::max(opt.K - S_T, 0.0);
        sum_payoffs += payoff;
    }

    return disc * sum_payoffs / n_simulations;
}

void price_batch_mc_cpu(
    std::span<const double> S,
    std::span<const double> K,
    std::span<const double> r,
    std::span<const double> sigma,
    std::span<const double> T,
    std::span<const int8_t> is_call,
    std::span<double>       out_prices,
    int                     n_simulations
) noexcept {
    const std::size_t n = S.size();
    
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), std::size_t{0});

    std::for_each(std::execution::par_unseq,
        indices.begin(), indices.end(),
        [&](std::size_t i) {
            OptionContract opt{
                .S = S[i], .K = K[i], .r = r[i],
                .sigma = sigma[i], .T = T[i],
                .is_call = static_cast<bool>(is_call[i])
            };
            out_prices[i] = price_option_mc_cpu(opt, n_simulations, 42 + i);
        }
    );
}

} // namespace mini_aladdin::pricing
