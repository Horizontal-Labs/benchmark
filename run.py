import sys
import argparse
from app.benchmark import run_full_benchmark

def main(max_samples: int = 0, quick: bool = False):
    parser = argparse.ArgumentParser(description='Run the Argument Mining Benchmark')
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=100,
        help='Maximum number of samples to use for benchmarking (default: 100)'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run with fewer samples for quick testing (equivalent to --max-samples 10)'
    )
    
    args = parser.parse_args()
    
    # If quick mode is enabled, override max_samples
    if args.quick or quick:
        max_samples = 10
        print("🚀 Running in quick mode with 10 samples")
    else:
        if max_samples is None:
            max_samples = args.max_samples
        print(f"🚀 Running benchmark with {max_samples} samples")
    
    run_full_benchmark(max_samples=max_samples)

if __name__ == "__main__":
    main(max_samples=2, quick=False)