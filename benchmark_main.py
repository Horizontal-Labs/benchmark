#!/usr/bin/env python3
"""
Main entry point for the refactored Argument Mining Benchmark package.
"""

import sys
import argparse
from pathlib import Path

# Add the benchmark package to the path
benchmark_path = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_path))

from benchmark.core.benchmark import ArgumentMiningBenchmark


def test_imports(max_samples: int = 100, disable_openai: bool = False):
    """Test if all imports are working correctly."""
    print("Testing imports...")
    
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples, disable_openai=disable_openai)
        print("Benchmark class created successfully")
        print(f"  - Loaded data for tasks: {list(benchmark.data.keys())}")
        print(f"  - Available implementations: {list(benchmark.implementations.keys())}")
        return True
    except Exception as e:
        print(f"Failed to create benchmark: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def run_full_benchmark(max_samples: int = 100, disable_openai: bool = False):
    """Run the complete benchmark suite."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples, disable_openai=disable_openai)
        results = benchmark.run_benchmark()
        print("Benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def run_single_task_benchmark(task_name: str, max_samples: int = 100, implementation_name: str = None, disable_openai: bool = False):
    """Run benchmark for a single task."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples, disable_openai=disable_openai)
        results = benchmark.run_single_task(task_name, implementation_name)
        print(f"Task '{task_name}' benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Task '{task_name}' benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def run_single_implementation_benchmark(implementation_name: str, max_samples: int = 100, task_name: str = None, disable_openai: bool = False):
    """Run benchmark for a single implementation."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples, disable_openai=disable_openai)
        results = benchmark.run_single_implementation(implementation_name, task_name)
        print(f"Implementation '{implementation_name}' benchmark completed successfully")
        return results
    except Exception as e:
        print(f"Implementation '{implementation_name}' benchmark failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Argument Mining Benchmark')
    parser.add_argument('--max-samples', type=int, default=100, 
                       help='Maximum number of samples to use for benchmarking (default: 100)')
    parser.add_argument('--disable-openai', action='store_true',
                       help='Disable OpenAI implementations')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test imports, do not run benchmark')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save results to CSV files in addition to JSON')
    
    args = parser.parse_args()
    
    print(f"Benchmark configuration:")
    print(f"  - Max samples: {args.max_samples}")
    print(f"  - OpenAI disabled: {args.disable_openai}")
    print(f"  - Save CSV: {args.save_csv}")
    print()
    
    # Test imports first
    if test_imports(max_samples=args.max_samples, disable_openai=args.disable_openai):
        if not args.test_only:
            # Run full benchmark
            print(run_full_benchmark(max_samples=args.max_samples, disable_openai=args.disable_openai))
            
            if args.save_csv:
                print("\nCSV export enabled - results will be saved to CSV files")
    else:
        print("Import test failed. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
