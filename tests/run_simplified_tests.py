#!/usr/bin/env python3
"""
Simple test runner for the simplified Argument Mining Benchmark test suite
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_pytest_command(args, config_file="pytest_simplified.ini"):
    """Run pytest with specified arguments"""
    cmd = [
        sys.executable, "-m", "pytest",
        "-c", config_file
    ] + args
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

def run_import_tests():
    """Run only import tests"""
    return run_pytest_command([
        "test_simplified.py::TestImports",
        "-m", "unit"
    ])

def run_implementation_tests():
    """Run only implementation tests"""
    return run_pytest_command([
        "test_simplified.py::TestSingleImplementations",
        "-m", "unit"
    ])

def run_benchmark_tests():
    """Run only benchmark tests"""
    return run_pytest_command([
        "test_simplified.py::TestBenchmarking",
        "-m", "unit"
    ])

def run_integration_tests():
    """Run only integration tests"""
    return run_pytest_command([
        "test_simplified.py::TestIntegration",
        "-m", "integration"
    ])

def run_all_tests():
    """Run all simplified tests"""
    return run_pytest_command([
        "test_simplified.py"
    ])

def main():
    parser = argparse.ArgumentParser(description="Run simplified Argument Mining Benchmark tests")
    parser.add_argument(
        "--category", 
        choices=["imports", "implementations", "benchmark", "integration", "all"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Argument Mining Benchmark - Simplified Test Suite")
        print("=" * 50)
    
    if args.category == "imports":
        return run_import_tests()
    elif args.category == "implementations":
        return run_implementation_tests()
    elif args.category == "benchmark":
        return run_benchmark_tests()
    elif args.category == "integration":
        return run_integration_tests()
    else:  # all
        return run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
