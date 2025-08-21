#!/usr/bin/env python3
"""
Simple test script to run the parameterized benchmark with real data.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Also add current directory to path
sys.path.insert(0, '.')

# Add external submodules to Python path
external_api = project_root / "external" / "api"
external_db = project_root / "external" / "db"

if external_api.exists():
    sys.path.insert(0, str(external_api))
    print(f"✓ Added argument-mining-api to Python path: {external_api}")

if external_db.exists():
    sys.path.insert(0, str(external_db))
    print(f"✓ Added argument-mining-db to Python path: {external_db}")

# Import the benchmark module
try:
    # Try importing from the app directory
    from app.benchmark import run_specific_benchmark
    print("✓ Successfully imported run_specific_benchmark from app.benchmark")
except ImportError as e:
    print(f"❌ Failed to import from app.benchmark: {e}")
    try:
        # Try importing directly from the benchmark file
        import app.benchmark
        run_specific_benchmark = app.benchmark.run_specific_benchmark
        print("✓ Successfully imported run_specific_benchmark from app.benchmark module")
    except ImportError as e2:
        print(f"❌ Failed to import from app.benchmark module: {e2}")
        sys.exit(1)

def test_parameterized_benchmark():
    """Test the parameterized benchmark with real data."""
    
    # Test cases
    test_cases = [
        ("adu_identification", "openai"),
        ("adu_classification", "openai"),
        ("stance_classification", "openai"),
        ("claim_premise_linking", "openai"),
    ]
    
    print("\n" + "="*60)
    print("RUNNING PARAMETERIZED BENCHMARK TESTS WITH REAL DATA")
    print("="*60)
    
    for task, implementation in test_cases:
        print(f"\nTesting {task} with {implementation}")
        print("-" * 40)
        
        try:
            # Run the specific benchmark with real data
            result = run_specific_benchmark(
                task=task,
                implementation=implementation,
                data_length=2
            )
            
            # Basic structure validation
            assert isinstance(result, dict), "Result should be a dictionary"
            assert 'task' in result, "Result should contain 'task'"
            assert 'implementation' in result, "Result should contain 'implementation'"
            assert 'data_length' in result, "Result should contain 'data_length'"
            assert 'results' in result, "Result should contain 'results'"
            assert 'summary' in result, "Result should contain 'summary'"
            assert 'success' in result, "Result should contain 'success'"
            assert 'error_message' in result, "Result should contain 'error_message'"
            
            # Validate specific values
            assert result['task'] == task, f"Task should be {task}"
            assert result['implementation'] == implementation, f"Implementation should be {implementation}"
            assert result['data_length'] == 2, "Data length should be 2"
            
            print(f"✓ Structure validation passed")
            
            # Check if the implementation was available
            if result['success']:
                # If successful, validate results structure
                assert isinstance(result['results'], list), "Results should be a list"
                assert len(result['results']) == 2, "Should have 2 results for data_length=2"
                
                # Validate each result
                for benchmark_result in result['results']:
                    assert hasattr(benchmark_result, 'task_name'), "Result should have task_name"
                    assert hasattr(benchmark_result, 'implementation_name'), "Result should have implementation_name"
                    assert hasattr(benchmark_result, 'sample_id'), "Result should have sample_id"
                    assert hasattr(benchmark_result, 'metrics'), "Result should have metrics"
                    assert hasattr(benchmark_result, 'performance'), "Result should have performance"
                    assert hasattr(benchmark_result, 'success'), "Result should have success"
                    
                    assert benchmark_result.task_name == task, f"Task name should be {task}"
                    assert benchmark_result.implementation_name == implementation, f"Implementation name should be {implementation}"
                
                # Validate summary
                summary = result['summary']
                assert isinstance(summary, dict), "Summary should be a dictionary"
                assert 'total_samples' in summary, "Summary should contain total_samples"
                assert 'successful_samples' in summary, "Summary should contain successful_samples"
                assert 'success_rate' in summary, "Summary should contain success_rate"
                assert summary['total_samples'] == 2, "Total samples should be 2"
                assert summary['successful_samples'] <= 2, "Successful samples should be <= 2"
                assert 0 <= summary['success_rate'] <= 1, "Success rate should be between 0 and 1"
                
                print(f"✓ {task} with {implementation}: {summary['successful_samples']}/{summary['total_samples']} successful")
                print(f"  Success rate: {summary['success_rate']:.2%}")
                
                # Print some metrics if available
                if 'avg_precision' in summary:
                    print(f"  Avg Precision: {summary['avg_precision']:.3f}")
                if 'avg_recall' in summary:
                    print(f"  Avg Recall: {summary['avg_recall']:.3f}")
                if 'avg_f1' in summary:
                    print(f"  Avg F1: {summary['avg_f1']:.3f}")
                
            else:
                # If not successful, check error message
                assert isinstance(result['error_message'], str), "Error message should be a string"
                assert len(result['error_message']) > 0, "Error message should not be empty"
                
                print(f"⚠️ {task} with {implementation} failed: {result['error_message'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error testing {task} with {implementation}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("PARAMETERIZED BENCHMARK TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_parameterized_benchmark()
