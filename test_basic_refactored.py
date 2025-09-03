#!/usr/bin/env python3
"""
Basic functionality test for the refactored Argument Mining Benchmark package.

This script performs a quick check to ensure the package structure is correct
and basic functionality works.
"""

import sys
from pathlib import Path

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing package structure...")
    
    # Check if benchmark package exists
    benchmark_path = Path(__file__).parent / "benchmark"
    if not benchmark_path.exists():
        print("❌ Benchmark package directory not found")
        return False
    
    # Check required subdirectories
    required_dirs = ["core", "implementations", "tasks", "data", "metrics", "utils"]
    for dir_name in required_dirs:
        dir_path = benchmark_path / dir_name
        if not dir_path.exists():
            print(f"❌ Required directory not found: {dir_name}")
            return False
    
    print("✅ Package structure is correct")
    return True

def test_basic_imports():
    """Test basic package imports."""
    print("\nTesting basic imports...")
    
    try:
        # Add the benchmark package to the path
        benchmark_path = Path(__file__).parent / "benchmark"
        sys.path.insert(0, str(benchmark_path))
        
        # Test main package import
        from benchmark import ArgumentMiningBenchmark, BenchmarkResult
        print("✅ Main package imports work")
        
        # Test core module import
        from benchmark.core import benchmark, results
        print("✅ Core module imports work")
        
        # Test implementations module import
        from benchmark.implementations import base
        print("✅ Implementations module imports work")
        
        # Test tasks module import
        from benchmark.tasks import base as task_base
        print("✅ Tasks module imports work")
        
        # Test data module import
        from benchmark.data import loader
        print("✅ Data module imports work")
        
        # Test metrics module import
        from benchmark.metrics import evaluator
        print("✅ Metrics module imports work")
        
        # Test utils module import
        from benchmark.utils import logging, file_handlers
        print("✅ Utils module imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_class_instantiation():
    """Test that basic classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        from benchmark.core.results import BenchmarkResult
        
        # Test BenchmarkResult creation
        result = BenchmarkResult(
            task_name="test_task",
            implementation_name="test_impl",
            sample_id="0",
            execution_date="2024-01-01",
            metrics={},
            performance={},
            predictions=None,
            ground_truth=None
        )
        print("✅ BenchmarkResult can be instantiated")
        
        # Test that it has the expected attributes
        assert hasattr(result, 'task_name')
        assert hasattr(result, 'implementation_name')
        assert hasattr(result, 'sample_id')
        assert hasattr(result, 'execution_date')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'performance')
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'ground_truth')
        assert hasattr(result, 'error_message')
        assert hasattr(result, 'success')
        
        print("✅ BenchmarkResult has all expected attributes")
        return True
        
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        return False

def test_abstract_classes():
    """Test that abstract base classes cannot be instantiated."""
    print("\nTesting abstract base classes...")
    
    try:
        from benchmark.implementations.base import BaseImplementation
        from benchmark.tasks.base import BaseTask
        
        # Test that BaseImplementation is abstract
        try:
            BaseImplementation("test")
            print("❌ BaseImplementation should be abstract")
            return False
        except TypeError:
            print("✅ BaseImplementation correctly abstract")
        
        # Test that BaseTask is abstract
        try:
            BaseTask("test")
            print("❌ BaseTask should be abstract")
            return False
        except TypeError:
            print("✅ BaseTask correctly abstract")
        
        return True
        
    except Exception as e:
        print(f"❌ Abstract class test failed: {e}")
        return False

def test_utility_functions():
    """Test basic utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from benchmark.utils.logging import setup_logging
        
        # Test logging setup
        logger = setup_logging("INFO")
        if logger is not None:
            print("✅ Logging setup works")
        else:
            print("❌ Logging setup failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("BASIC REFACTORED BENCHMARK PACKAGE TEST")
    print("=" * 60)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Basic Imports", test_basic_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Abstract Classes", test_abstract_classes),
        ("Utility Functions", test_utility_functions),
    ]
    
    all_tests_passed = True
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if not success:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL BASIC TESTS PASSED! The refactored package structure is correct.")
        print("You can now run the comprehensive test suite with:")
        print("  cd tests")
        print("  python run_refactored_tests.py")
    else:
        print("❌ SOME BASIC TESTS FAILED. Please check the errors above.")
    print("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
