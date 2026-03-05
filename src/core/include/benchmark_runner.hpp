#pragma once
#include "timer.hpp"
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>

namespace mini_aladdin::bench {

struct BenchmarkResult {
    std::string name;
    std::size_t n_options;
    double      mean_ms;
    double      stddev_ms;
    double      min_ms;
    double      max_ms;
    double      throughput;
    double      speedup;
    
    void print() const {
        std::cout << "===== " << name << " =====\n"
                  << "Options: " << n_options << "\n"
                  << std::fixed << std::setprecision(3)
                  << "Mean: " << mean_ms << " ms\n"
                  << "StdDev: " << stddev_ms << " ms\n"
                  << "Min: " << min_ms << " ms\n"
                  << "Max: " << max_ms << " ms\n"
                  << "Throughput: " << throughput << " opts/s\n"
                  << "Speedup: " << speedup << "x\n\n";
    }
    
    void to_json(const std::string& filepath) const {
        std::ofstream file(filepath);
        file << "{\n";
        file << "  \"name\": \"" << name << "\",\n";
        file << "  \"n_options\": " << n_options << ",\n";
        file << std::fixed << std::setprecision(6);
        file << "  \"mean_ms\": " << mean_ms << ",\n";
        file << "  \"stddev_ms\": " << stddev_ms << ",\n";
        file << "  \"min_ms\": " << min_ms << ",\n";
        file << "  \"max_ms\": " << max_ms << ",\n";
        file << "  \"throughput\": " << throughput << ",\n";
        file << "  \"speedup\": " << speedup << "\n";
        file << "}\n";
    }
};

template<typename Fn>
BenchmarkResult run_benchmark(
    std::string name,
    std::size_t n_options,
    Fn&&        fn,
    int         n_warmup = 3,
    int         n_runs   = 10
) {
    for (int i = 0; i < n_warmup; ++i) {
        fn();
    }
    
    std::vector<double> timings(n_runs);
    Timer timer;
    
    for (int i = 0; i < n_runs; ++i) {
        timer.start();
        fn();
        timer.stop();
        timings[i] = timer.elapsed_ms();
    }
    
    double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / n_runs;
    
    double variance = 0.0;
    for (double t : timings) variance += (t - mean) * (t - mean);
    variance /= n_runs;
    
    return BenchmarkResult{
        .name       = std::move(name),
        .n_options  = n_options,
        .mean_ms    = mean,
        .stddev_ms  = std::sqrt(variance),
        .min_ms     = *std::min_element(timings.begin(), timings.end()),
        .max_ms     = *std::max_element(timings.begin(), timings.end()),
        .throughput = static_cast<double>(n_options) / (mean * 1e-3),
        .speedup    = 1.0
    };
}

} // namespace mini_aladdin::bench
