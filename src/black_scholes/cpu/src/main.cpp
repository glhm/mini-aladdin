#include <iostream>
#include <vector>
#include "csv_loader.hpp"
#include "bs_cpu_pricer.hpp"
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
    
    std::vector<double> prices(batch.size());
    
    auto result = run_benchmark(
        "black_scholes_cpu",
        batch.size(),
        [&]() {
            price_batch_bs_cpu(
                batch.S, batch.K, batch.r, batch.sigma, batch.T, batch.is_call,
                std::span<double>(prices)
            );
        },
        3,
        10
    );
    
    result.print();
    result.to_json("results/benchmark_bs_cpu.json");
    
    // Print first 5 prices as sample
    std::cout << "\nSample prices:\n";
    for (size_t i = 0; i < std::min(size_t(5), prices.size()); ++i) {
        std::cout << batch.tickers[i] << ": " << prices[i] << "\n";
    }
    
    return 0;
}
