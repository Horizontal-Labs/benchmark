#!/usr/bin/env python3
"""
Examples of using the Argument Mining Benchmark with individual task and implementation execution

This script demonstrates how to:
1. Run a single task with all implementations
2. Run a single implementation on all tasks
3. Run a specific task with a specific implementation
4. Run the full benchmark
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.benchmark import (
    run_single_task_benchmark,
    run_single_implementation_benchmark,
    run_full_benchmark
)


def example_run_single_task():
    """Example: Run ADU extraction task with all implementations."""
    print("=" * 60)
    print("Example: Running ADU Extraction Task with All Implementations")
    print("=" * 60)
    
    try:
        results = run_single_task_benchmark('adu_extraction', max_samples=5)
        if results:
            print(f"✓ Successfully ran ADU extraction task")
            print(f"  - Generated {len(results)} results")
            
            # Group results by implementation
            implementations = set(r.implementation_name for r in results)
            print(f"  - Tested implementations: {list(implementations)}")
            
            # Show some metrics
            successful_results = [r for r in results if r.success]
            if successful_results:
                avg_f1 = sum(r.metrics.get('f1', 0) for r in successful_results) / len(successful_results)
                avg_time = sum(r.performance.get('inference_time', 0) for r in successful_results) / len(successful_results)
                print(f"  - Average F1: {avg_f1:.3f}")
                print(f"  - Average inference time: {avg_time:.3f}s")
        else:
            print("✗ Failed to run ADU extraction task")
    except Exception as e:
        print(f"✗ Error running ADU extraction task: {e}")


def example_run_single_implementation():
    """Example: Run OpenAI implementation on all tasks."""
    print("\n" + "=" * 60)
    print("Example: Running OpenAI Implementation on All Tasks")
    print("=" * 60)
    
    try:
        results = run_single_implementation_benchmark('openai', max_samples=5)
        if results:
            print(f"✓ Successfully ran OpenAI implementation")
            print(f"  - Generated {len(results)} results")
            
            # Group results by task
            tasks = set(r.task_name for r in results)
            print(f"  - Tested tasks: {list(tasks)}")
            
            # Show some metrics
            successful_results = [r for r in results if r.success]
            if successful_results:
                for task in tasks:
                    task_results = [r for r in successful_results if r.task_name == task]
                    if task_results:
                        if task == 'adu_extraction':
                            avg_f1 = sum(r.metrics.get('f1', 0) for r in task_results) / len(task_results)
                            print(f"  - {task}: Average F1 = {avg_f1:.3f}")
                        elif task == 'stance_classification':
                            avg_acc = sum(r.metrics.get('accuracy', 0) for r in task_results) / len(task_results)
                            print(f"  - {task}: Average Accuracy = {avg_acc:.3f}")
                        elif task == 'claim_premise_linking':
                            avg_acc = sum(r.metrics.get('accuracy', 0) for r in task_results) / len(task_results)
                            print(f"  - {task}: Average Accuracy = {avg_acc:.3f}")
        else:
            print("✗ Failed to run OpenAI implementation")
    except Exception as e:
        print(f"✗ Error running OpenAI implementation: {e}")


def example_run_specific_task_and_implementation():
    """Example: Run stance classification with TinyLlama only."""
    print("\n" + "=" * 60)
    print("Example: Running Stance Classification with TinyLlama Only")
    print("=" * 60)
    
    try:
        results = run_single_task_benchmark('stance_classification', max_samples=5, implementation_name='tinyllama')
        if results:
            print(f"✓ Successfully ran stance classification with TinyLlama")
            print(f"  - Generated {len(results)} results")
            
            # Show metrics
            successful_results = [r for r in results if r.success]
            if successful_results:
                avg_acc = sum(r.metrics.get('accuracy', 0) for r in successful_results) / len(successful_results)
                avg_f1 = sum(r.metrics.get('weighted_f1', 0) for r in successful_results) / len(successful_results)
                avg_time = sum(r.performance.get('inference_time', 0) for r in successful_results) / len(successful_results)
                print(f"  - Average Accuracy: {avg_acc:.3f}")
                print(f"  - Average F1: {avg_f1:.3f}")
                print(f"  - Average inference time: {avg_time:.3f}s")
        else:
            print("✗ Failed to run stance classification with TinyLlama")
    except Exception as e:
        print(f"✗ Error running stance classification with TinyLlama: {e}")


def example_run_full_benchmark():
    """Example: Run the full benchmark suite."""
    print("\n" + "=" * 60)
    print("Example: Running Full Benchmark Suite")
    print("=" * 60)
    
    try:
        results = run_full_benchmark(max_samples=5)
        if results:
            print(f"✓ Successfully ran full benchmark")
            print(f"  - Tasks completed: {list(results.keys())}")
            
            total_results = sum(len(task_results) for task_results in results.values())
            print(f"  - Total results generated: {total_results}")
            
            # Show summary for each task
            for task_name, task_results in results.items():
                successful_results = [r for r in task_results if r.success]
                if successful_results:
                    implementations = set(r.implementation_name for r in successful_results)
                    print(f"  - {task_name}: {len(successful_results)} results from {len(implementations)} implementations")
        else:
            print("✗ Failed to run full benchmark")
    except Exception as e:
        print(f"✗ Error running full benchmark: {e}")


def main():
    """Run all examples."""
    print("Argument Mining Benchmark - Individual Execution Examples")
    print("=" * 80)
    
    # Run examples
    example_run_single_task()
    example_run_single_implementation()
    example_run_specific_task_and_implementation()
    example_run_full_benchmark()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("\nYou can also use the command-line interface:")
    print("  python run.py --task adu_extraction")
    print("  python run.py --implementation openai")
    print("  python run.py --task stance_classification --implementation tinyllama")
    print("  python run.py --list-tasks")
    print("  python run.py --list-implementations")


if __name__ == "__main__":
    main()
