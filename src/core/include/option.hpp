#pragma once
#include <string>
#include <span>
#include <vector>
#include <cstdint>

namespace mini_aladdin {

/// Represents a single European option contract loaded from CSV.
/// Uses Plain Old Data layout for cache-friendly batch processing.
struct OptionContract {
    double S;       ///< Spot price of the underlying asset
    double K;       ///< Strike price
    double r;       ///< Risk-free interest rate (annualized)
    double sigma;   ///< Implied volatility (annualized)
    double T;       ///< Time to expiration in years
    bool   is_call; ///< true = Call option, false = Put option
    
    // Padding to align struct to 64 bytes (one cache line)
    char _pad[7];
    
    std::string ticker; ///< Underlying ticker symbol (for reporting only)
};

/// Structure of Arrays (SoA) layout for GPU-friendly batch processing.
struct OptionBatch {
    std::vector<double> S;
    std::vector<double> K;
    std::vector<double> r;
    std::vector<double> sigma;
    std::vector<double> T;
    std::vector<int8_t> is_call;
    std::vector<std::string> tickers;
    
    [[nodiscard]] std::size_t size() const noexcept { return S.size(); }
    
    void reserve(std::size_t n) {
        S.reserve(n); K.reserve(n); r.reserve(n);
        sigma.reserve(n); T.reserve(n); is_call.reserve(n);
        tickers.reserve(n);
    }
};

} // namespace mini_aladdin
