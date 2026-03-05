#!/usr/bin/env python3
"""
Aggregates benchmark results from all 6 executables and prints comparison table.
"""

import json
import sys
from pathlib import Path
import subprocess

def load_benchmark(filepath):
    """Load a single benchmark JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
        return None

def format_number(num, decimals=1):
    """Format number with k/M suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}k"
    else:
        return f"{num:.{decimals}f}"

def format_time(ms):
    """Format time in ms with appropriate precision."""
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms:.0f}ms"

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    
    # Define all benchmark files
    benchmarks = [
        ("black_scholes_cpu", "benchmark_bs_cpu.json"),
        ("black_scholes_gpu_naive", "benchmark_bs_gpu_naive.json"),
        ("black_scholes_gpu_optimized", "benchmark_bs_gpu_optimized.json"),
        ("monte_carlo_cpu", "benchmark_mc_cpu.json"),
        ("monte_carlo_gpu_naive", "benchmark_mc_gpu_naive.json"),
        ("monte_carlo_gpu_optimized", "benchmark_mc_gpu_optimized.json"),
    ]
    
    # Load all results
    results = {}
    for name, filename in benchmarks:
        filepath = results_dir / filename
        data = load_benchmark(filepath)
        if data:
            results[name] = data
    
    if not results:
        print("No benchmark results found!")
        sys.exit(1)
    
    # Get GPU info
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip()
    except:
        gpu_name = "Unknown GPU"
    
    # Calculate speedups
    bs_cpu_time = results.get("black_scholes_cpu", {}).get("mean_ms", 1)
    mc_cpu_time = results.get("monte_carlo_cpu", {}).get("mean_ms", 1)
    
    # Print table
    print("=" * 80)
    print("                    Mini-Aladdin Benchmark Results")
    print(f"                    Hardware: {gpu_name}")
    print("=" * 80)
    print(f"{'Executable':<30} {'N Options':>12} {'Mean':>10} {'Throughput':>12} {'Speedup':>8}")
    print("-" * 80)
    
    # Black-Scholes group
    for name in ["black_scholes_cpu", "black_scholes_gpu_naive", "black_scholes_gpu_optimized"]:
        if name in results:
            r = results[name]
            n = r.get("n_options", 0)
            time_ms = r.get("mean_ms", 0)
            throughput = r.get("throughput", 0)
            speedup = bs_cpu_time / time_ms if time_ms > 0 else 0
            
            print(f"{name:<30} {n:>12,} {format_time(time_ms):>10} {format_number(throughput):>12}/s {speedup:>7.1f}x")
    
    print("-" * 80)
    
    # Monte Carlo group
    for name in ["monte_carlo_cpu", "monte_carlo_gpu_naive", "monte_carlo_gpu_optimized"]:
        if name in results:
            r = results[name]
            n = r.get("n_options", 0)
            time_ms = r.get("mean_ms", 0)
            throughput = r.get("throughput", 0)
            speedup = mc_cpu_time / time_ms if time_ms > 0 else 0
            
            print(f"{name:<30} {n:>12,} {format_time(time_ms):>10} {format_number(throughput):>12}/s {speedup:>7.1f}x")
    
    print("=" * 80)
    
    # Save summary
    summary = {
        "gpu": gpu_name,
        "benchmarks": results,
        "speedups": {
            "BlackScholes": {
                "CPU": 1.0,
                "GPU_Naive": bs_cpu_time / results.get("black_scholes_gpu_naive", {}).get("mean_ms", bs_cpu_time),
                "GPU_Optimized": bs_cpu_time / results.get("black_scholes_gpu_optimized", {}).get("mean_ms", bs_cpu_time),
            },
            "MonteCarlo": {
                "CPU": 1.0,
                "GPU_Naive": mc_cpu_time / results.get("monte_carlo_gpu_naive", {}).get("mean_ms", mc_cpu_time),
                "GPU_Optimized": mc_cpu_time / results.get("monte_carlo_gpu_optimized", {}).get("mean_ms", mc_cpu_time),
            }
        }
    }
    
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
