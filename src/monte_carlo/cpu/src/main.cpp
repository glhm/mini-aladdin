#include <iostream>
#include <vector>
#include "csv_loader.hpp"
#include "mc_cpu_pricer.hpp"
#include "benchmark_runner.hpp"

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
    std::cout << "Running Monte Carlo CPU (this may take a while)...\n";
    
    std::vector<double> prices(batch.size());
    const int n_sims = 10000; // Reduced for CPU
    
    auto result = run_benchmark(
        "monte_carlo_cpu",
        batch.size(),
        [&]() {
            price_batch_mc_cpu(
                batch.S, batch.K, batch.r, batch.sigma, batch.T, batch.is_call,
                std::span<double>(prices),
                n_sims
            );
        },
        1,
        3
    );
    
    result.print();
    result.to_json("results/benchmark_mc_cpu.json");
    
    std::cout << "\nSample prices:\n";
    for (size_t i = 0; i < std::min(size_t(5), prices.size()); ++i) {
        std::cout << batch.tickers[i] << ": " << prices[i] << "\n";
    }
    
    return 0;
}
