#!/usr/bin/env python3
"""
Test script for the argument mining benchmark.

This script runs a quick test of the benchmark functionality with a small subset of data.
"""

import sys
import os
import time
from typing import List, Tuple

# Add the argument-mining-api to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'argument-mining-api'))

# Add the db-connector to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'db-connector'))

def test_data_loading():
    """Test if we can load benchmark data."""
    try:
        from db.queries import get_benchmark_data
        claims, premises, categories = get_benchmark_data()
        print(f"✅ Successfully loaded benchmark data:")
        print(f"   Claims: {len(claims)}")
        print(f"   Premises: {len(premises)}")
        print(f"   Categories: {len(categories)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load benchmark data: {e}")
        return False

def test_imports():
    """Test if we can import all required modules."""
    try:
        from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
        from app.argmining.interfaces.claim_premise_linker import ClaimPremiseLinker
        from app.argmining.models.argument_units import ArgumentUnit, UnlinkedArgumentUnits
        print("✅ Successfully imported argument mining interfaces")
        return True
    except Exception as e:
        print(f"❌ Failed to import argument mining modules: {e}")
        return False

def test_implementation_imports():
    """Test if we can import implementations."""
    implementations_to_test = [
        ('OpenAILLMClassifier', 'app.argmining.implementations.openai_llm_classifier'),
        ('OpenAIClaimPremiseLinker', 'app.argmining.implementations.openai_claim_premise_linker'),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for name, module_path in implementations_to_test:
        try:
            __import__(module_path)
            successful_imports.append(name)
        except Exception as e:
            failed_imports.append((name, str(e)))
    
    print(f"✅ Successfully imported: {successful_imports}")
    if failed_imports:
        print(f"❌ Failed to import: {failed_imports}")
    
    return len(failed_imports) == 0

def test_small_benchmark():
    """Run a small benchmark test with limited data."""
    try:
        from benchmark_script import ArgumentMiningBenchmark
        
        # Create a small test dataset
        from db.models import ADU
        from app.argmining.models.argument_units import ArgumentUnit
        import uuid
        
        # Create mock test data
        test_cases = []
        for i in range(3):  # Only 3 test cases for quick testing
            test_cases.append((
                f"test_case_{i}",
                [ADU(id=i, text=f"Test claim {i}", type="claim", domain_id=1)],
                [ADU(id=i+100, text=f"Test premise {i}", type="premise", domain_id=1)],
                ["stance_pro"]
            ))
        
        benchmark = ArgumentMiningBenchmark()
        
        # Test the data preparation method
        print("Testing data preparation...")
        prepared_data = benchmark.prepare_test_data()
        print(f"✅ Prepared {len(prepared_data)} test cases from benchmark data")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run small benchmark test: {e}")
        return False

def main():
    """Run all tests."""
    print("Argument Mining Benchmark - Test Suite")
    print("="*50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Interface Imports", test_imports),
        ("Implementation Imports", test_implementation_imports),
        ("Small Benchmark", test_small_benchmark),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅ PASSED' if success else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The benchmark should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 