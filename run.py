import sys
import argparse
from app.benchmark import (
    run_full_benchmark, 
    run_single_task_benchmark, 
    run_single_implementation_benchmark
)

def main():
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
    parser.add_argument(
        '--task',
        type=str,
        choices=['adu_extraction', 'stance_classification', 'claim_premise_linking'],
        help='Run benchmark for a specific task only'
    )
    parser.add_argument(
        '--implementation',
        type=str,
        choices=['openai', 'tinyllama', 'modernbert', 'deberta'],
        help='Run benchmark for a specific implementation only'
    )
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='List available tasks and exit'
    )
    parser.add_argument(
        '--list-implementations',
        action='store_true',
        help='List available implementations and exit'
    )
    
    args = parser.parse_args()
    
    # If quick mode is enabled, override max_samples
    if args.quick:
        max_samples = 10
        print("Running in quick mode with 10 samples")
    else:
        max_samples = args.max_samples
        print(f"Running benchmark with {max_samples} samples")
    
    # Handle list options
    if args.list_tasks:
        print("Available tasks:")
        print("  - adu_extraction: Extract Argumentative Discourse Units (claims and premises)")
        print("  - stance_classification: Classify stance as pro/con/neutral")
        print("  - claim_premise_linking: Link claims to supporting/contradicting premises")
        return
    
    if args.list_implementations:
        print("Available implementations:")
        print("  - openai: OpenAI LLM Classifier")
        print("  - tinyllama: TinyLlama LLM Classifier")
        print("  - modernbert: ModernBERT (PeftEncoderModelLoader)")
        print("  - deberta: DeBERTa (NonTrainedEncoderModelLoader)")
        return
    
    # Determine what to run
    if args.task and args.implementation:
        # Run specific task with specific implementation
        print(f"Running task '{args.task}' with implementation '{args.implementation}'")
        results = run_single_task_benchmark(args.task, max_samples, args.implementation)
    elif args.task:
        # Run specific task with all implementations
        print(f"Running task '{args.task}' with all implementations")
        results = run_single_task_benchmark(args.task, max_samples)
    elif args.implementation:
        # Run specific implementation on all tasks
        print(f"Running implementation '{args.implementation}' on all tasks")
        results = run_single_implementation_benchmark(args.implementation, max_samples)
    else:
        # Run full benchmark
        print("Running full benchmark with all tasks and implementations")
        results = run_full_benchmark(max_samples)
    
    if results:
        print("Benchmark completed successfully!")
    else:
        print("Benchmark failed or returned no results.")

if __name__ == "__main__":
    main()