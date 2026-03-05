#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "csv_loader.hpp"
#include "mc_gpu_naive.cuh"
#include "benchmark_runner.hpp"
#include "mc_cpu_pricer.hpp"

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
    
    const int n_sims = 10'000;
    std::cout << "Running Monte Carlo GPU Naive with " << n_sims << " simulations per option...\n";
    
    std::vector<double> prices;
    
    // 3. Run benchmark — note: we include the full operation including RNG in the benchmark
    // However, the RNG is done inside price_batch_mc_gpu_naive
    auto result = run_benchmark(
        "monte_carlo_gpu_naive",
        batch.size(),
        [&]() {
            prices = price_batch_mc_gpu_naive(
                batch.S, batch.K, batch.r, batch.sigma, batch.T, batch.is_call, n_sims
            );
        },
        1,
        5
    );
    
    result.print();
    result.to_json("results/benchmark_mc_gpu_naive.json");
    
    // 4. Validate prices vs MonteCarlo.CPU reference (tolerance 1e-2)
    std::cout << "Validating against CPU reference (Monte Carlo has statistical variance)...\n";
    std::vector<double> cpu_prices(batch.size());
    price_batch_mc_cpu(
        std::span<const double>(batch.S),
        std::span<const double>(batch.K),
        std::span<const double>(batch.r),
        std::span<const double>(batch.sigma),
        std::span<const double>(batch.T),
        std::span<const int8_t>(batch.is_call),
        std::span<double>(cpu_prices),
        n_sims
    );
    
    double max_error = 0.0;
    double avg_error = 0.0;
    for (size_t i = 0; i < prices.size(); ++i) {
        double err = std::abs(prices[i] - cpu_prices[i]);
        if (err > max_error) max_error = err;
        avg_error += err;
    }
    avg_error /= prices.size();
    
    const double tolerance = 1e-2;
    if (max_error < tolerance) {
        std::cout << "Validation PASSED: max_error = " << max_error 
                  << ", avg_error = " << avg_error << "\n";
    } else {
        std::cout << "Validation FAILED: max_error = " << max_error 
                  << " (tolerance: " << tolerance << ")\n";
    }
    
    std::cout << "\nSample prices (GPU vs CPU):\n";
    for (size_t i = 0; i < std::min(size_t(5), prices.size()); ++i) {
        std::cout << batch.tickers[i] << ": GPU=" << prices[i] 
                  << " CPU=" << cpu_prices[i] << "\n";
    }
    
    // 6. Print note
    std::cout << "\nRun profile_nsight.sh to compare vs Optimized\n";
    
    return 0;
}
