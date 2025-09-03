#!/usr/bin/env python3
"""
Simple test script to verify the refactored benchmark package works.
"""

import sys
from pathlib import Path

# Add the benchmark package to the path
benchmark_path = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_path))

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing package structure...")
    
    try:
        # Test importing the main package
        from benchmark import ArgumentMiningBenchmark, BenchmarkResult
        print("✓ Main package imports work")
        
        # Test importing submodules
        from benchmark.implementations import OpenAIImplementation
        print("✓ Implementations module imports work")
        
        from benchmark.tasks import ADUExtractionTask
        print("✓ Tasks module imports work")
        
        from benchmark.data import DataLoader
        print("✓ Data module imports work")
        
        from benchmark.metrics import MetricsEvaluator
        print("✓ Metrics module imports work")
        
        from benchmark.utils import setup_logging, save_results_to_csv
        print("✓ Utils module imports work")
        
        print("\n✓ All package imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_class_creation():
    """Test that classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        from benchmark.implementations.base import BaseImplementation
        from benchmark.tasks.base import BaseTask
        from benchmark.core.results import BenchmarkResult
        
        # Test abstract base classes (should not be instantiable)
        try:
            BaseImplementation("test")
            print("✗ BaseImplementation should be abstract")
            return False
        except TypeError:
            print("✓ BaseImplementation correctly abstract")
        
        try:
            BaseTask("test")
            print("✗ BaseTask should be abstract")
            return False
        except TypeError:
            print("✓ BaseTask correctly abstract")
        
        # Test concrete classes
        result = BenchmarkResult(
            task_name="test",
            implementation_name="test",
            sample_id="0",
            execution_date="2024-01-01",
            metrics={},
            performance={},
            predictions=None,
            ground_truth=None
        )
        print("✓ BenchmarkResult can be instantiated")
        
        print("\n✓ All class instantiation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Class instantiation test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from benchmark.utils.logging import setup_logging
        
        # Test logging setup
        logger = setup_logging("INFO")
        print("✓ Logging setup works")
        
        print("\n✓ Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Utility function test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("REFACTORED BENCHMARK PACKAGE TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all tests
    if not test_package_structure():
        all_tests_passed = False
    
    if not test_class_creation():
        all_tests_passed = False
    
    if not test_utility_functions():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The refactored package is working correctly.")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    print("=" * 60)
