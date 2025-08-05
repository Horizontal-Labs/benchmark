#!/usr/bin/env python3
"""
Test script to verify the integration of argument-mining-db and argument-mining-api
into the benchmarking project.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all components can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test database connector
        from app.db_connector.db.queries import get_benchmark_data_for_evaluation
        print("✓ Database connector import successful")
        
        # Test argument mining API
        from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
        from app.argmining.interfaces.claim_premise_linker import ClaimPremiseLinker
        print("✓ Argument mining API import successful")
        
        # Test comprehensive benchmark
        from app.comprehensive_benchmark import ArgumentMiningBenchmark, BenchmarkConfig
        print("✓ Comprehensive benchmark import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test that benchmark data can be loaded."""
    print("\nTesting data loading...")
    
    try:
        from app.db_connector.db.queries import get_benchmark_data_for_evaluation
        
        data = get_benchmark_data_for_evaluation()
        print(f"✓ Successfully loaded {len(data)} benchmark samples")
        
        # Check data structure
        if data and len(data) > 0:
            sample = data[0]
            required_keys = ['id', 'text', 'ground_truth', 'metadata']
            if all(key in sample for key in required_keys):
                print("✓ Data structure is correct")
                return True
            else:
                print("❌ Data structure is incorrect")
                return False
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_benchmark_initialization():
    """Test that the benchmark can be initialized."""
    print("\nTesting benchmark initialization...")
    
    try:
        from app.comprehensive_benchmark import ArgumentMiningBenchmark, BenchmarkConfig
        
        config = BenchmarkConfig(
            tasks=['adu_extraction'],
            implementations=['openai'],
            output_dir="test_results"
        )
        
        benchmark = ArgumentMiningBenchmark(config)
        print("✓ Benchmark initialized successfully")
        print(f"  - Loaded {len(benchmark.benchmark_data)} samples")
        print(f"  - Available implementations: {list(benchmark.implementations.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark initialization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Benchmark Initialization Test", test_benchmark_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 