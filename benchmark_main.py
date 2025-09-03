#!/usr/bin/env python3
"""
Main entry point for the refactored Argument Mining Benchmark package.
"""

import sys
from pathlib import Path

# Add the benchmark package to the path
benchmark_path = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_path))

from benchmark.core.benchmark import ArgumentMiningBenchmark


def test_imports(max_samples: int = 100):
    """Test if all imports are working correctly."""
    print("Testing imports...")
    
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        print("Benchmark class created successfully")
        print(f"  - Loaded data for tasks: {list(benchmark.data.keys())}")
        print(f"  - Available implementations: {list(benchmark.implementations.keys())}")
        return True
    except Exception as e:
        print(f"Failed to create benchmark: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def run_full_benchmark(max_samples: int = 100):
    """Run the complete benchmark suite."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        results = benchmark.run_benchmark()
        print("Benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def run_single_task_benchmark(task_name: str, max_samples: int = 100, implementation_name: str = None):
    """Run benchmark for a single task."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        results = benchmark.run_single_task(task_name, implementation_name)
        print(f"Task '{task_name}' benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Task '{task_name}' benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def run_single_implementation_benchmark(implementation_name: str, max_samples: int = 100, task_name: str = None):
    """Run benchmark for a single implementation."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        results = benchmark.run_single_implementation(implementation_name, task_name)
        print(f"Implementation '{implementation_name}' benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Implementation '{implementation_name}' benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    # Test imports first
    if test_imports():
        # Run full benchmark
        print(run_full_benchmark())
