#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "csv_loader.hpp"
#include "bs_gpu_naive.cuh"
#include "benchmark_runner.hpp"
#include "bs_cpu_pricer.hpp"

using namespace mini_aladdin;
using namespace mini_aladdin::pricing;
using namespace mini_aladdin::bench;

int main() {
    auto batch = CsvLoader::load("data/input_options.csv");
    
    if (batch.size() == 0) {
        std::cerr << "No data loaded. Please check data/input_options.csv\n";
        return 1;
    }
    
    std::cout << "Loaded " << batch.size() << " options\n";
    
    // 1. Load CSV into AoS layout (std::vector<OptionContractGPU>)
    // Already done in price_batch_bs_gpu_naive
    
    std::vector<double> prices;
    
    // 2. Run benchmark: run_benchmark("black_scholes_gpu_naive", n, [&](){ ... }, 3, 10)
    auto result = run_benchmark(
        "black_scholes_gpu_naive",
        batch.size(),
        [&]() {
            prices = price_batch_bs_gpu_naive(
                batch.S, batch.K, batch.r, batch.sigma, batch.T, batch.is_call
            );
        },
        3,
        10
    );
    
    result.print();
    result.to_json("results/benchmark_bs_gpu_naive.json");
    
    // 3. Validate prices vs BlackScholes.CPU reference (tolerance 1e-6)
    std::vector<double> cpu_prices(batch.size());
    price_batch_bs_cpu(
        std::span<const double>(batch.S),
        std::span<const double>(batch.K),
        std::span<const double>(batch.r),
        std::span<const double>(batch.sigma),
        std::span<const double>(batch.T),
        std::span<const int8_t>(batch.is_call),
        std::span<double>(cpu_prices)
    );
    
    double max_error = 0.0;
    for (size_t i = 0; i < prices.size(); ++i) {
        double err = std::abs(prices[i] - cpu_prices[i]);
        if (err > max_error) max_error = err;
    }
    
    const double tolerance = 1e-6;
    if (max_error < tolerance) {
        std::cout << "Validation PASSED: max_error = " << max_error << "\n";
    } else {
        std::cout << "Validation FAILED: max_error = " << max_error << " (tolerance: " << tolerance << ")\n";
    }
    
    std::cout << "\nSample prices:\n";
    for (size_t i = 0; i < std::min(size_t(5), prices.size()); ++i) {
        std::cout << batch.tickers[i] << ": GPU=" << prices[i] 
                  << " CPU=" << cpu_prices[i] << "\n";
    }
    
    // 5. Print note: "Run profile_nsight.sh to compare coalescing efficiency vs Optimized"
    std::cout << "\nRun profile_nsight.sh to compare coalescing efficiency vs Optimized\n";
    
    return 0;
}
