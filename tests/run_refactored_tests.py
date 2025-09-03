#!/usr/bin/env python3
"""
Test runner for the refactored Argument Mining Benchmark package.

This script runs all tests for the refactored package and provides a summary.
"""

import sys
import os
import subprocess
from pathlib import Path
import time

def run_pytest_tests(test_file, verbose=True):
    """Run pytest tests for a specific test file."""
    print(f"\n{'='*60}")
    print(f"Running tests: {test_file}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
    if verbose:
        cmd.append("-s")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests for {test_file}: {e}")
        return False

def run_simple_test(test_file):
    """Run a simple test script directly."""
    print(f"\n{'='*60}")
    print(f"Running simple test: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], capture_output=False, text=True, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running simple test {test_file}: {e}")
        return False

def main():
    """Run all refactored package tests."""
    print("REFACTORED BENCHMARK PACKAGE TEST SUITE")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Test directory: {Path(__file__).parent}")
    
    # List of test files to run
    test_files = [
        "test_refactored_benchmark.py",
        "test_refactored_components.py"
    ]
    
    # List of simple test scripts
    simple_tests = [
        "../test_basic_refactored.py"
    ]
    
    all_tests_passed = True
    test_results = []
    
    # Run pytest tests
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            success = run_pytest_tests(test_file)
            test_results.append((test_file, success, "pytest"))
            if not success:
                all_tests_passed = False
        else:
            print(f"Test file not found: {test_file}")
            test_results.append((test_file, False, "not found"))
            all_tests_passed = False
    
    # Run simple tests
    for test_file in simple_tests:
        test_path = Path(__file__).parent.parent / test_file
        if test_path.exists():
            success = run_simple_test(test_file)
            test_results.append((test_file, success, "simple"))
            if not success:
                all_tests_passed = False
        else:
            print(f"Test file not found: {test_file}")
            test_results.append((test_file, False, "not found"))
            all_tests_passed = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_file, success, test_type in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_file} ({test_type})")
    
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! The refactored package is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Please check the errors above.")
    
    print(f"\n{'='*60}")
    
    return all_tests_passed

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    print(f"Total test time: {end_time - start_time:.2f} seconds")
    
    sys.exit(0 if success else 1)
