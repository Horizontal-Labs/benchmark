#!/usr/bin/env python3
"""
Test script to verify the benchmark integration with different implementations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_benchmark_imports():
    """Test that the benchmark can be imported and initialized."""
    print("Testing benchmark imports...")
    
    try:
        from app.benchmark import ArgumentMiningBenchmark
        print("✓ Benchmark class imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import benchmark: {e}")
        return False

def test_benchmark_initialization():
    """Test that the benchmark can be initialized."""
    print("\nTesting benchmark initialization...")
    
    try:
        from app.benchmark import ArgumentMiningBenchmark
        
        benchmark = ArgumentMiningBenchmark()
        print("✓ Benchmark initialized successfully")
        print(f"  - Data samples: {len(benchmark.data)}")
        print(f"  - Available implementations: {list(benchmark.implementations.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize benchmark: {e}")
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
                
                # Check ground truth structure
                gt = sample['ground_truth']
                if 'adus' in gt and 'stance' in gt and 'relationships' in gt:
                    print("✓ Ground truth structure is correct")
                    print(f"  - ADUs: {len(gt['adus'])}")
                    print(f"  - Stance: {gt['stance']}")
                    print(f"  - Relationships: {len(gt['relationships'])}")
                    return True
                else:
                    print("❌ Ground truth structure is incorrect")
                    return False
            else:
                print("❌ Data structure is incorrect")
                return False
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_implementation_loading():
    """Test that implementations can be loaded."""
    print("\nTesting implementation loading...")
    
    try:
        from app.benchmark import ArgumentMiningBenchmark
        
        benchmark = ArgumentMiningBenchmark()
        
        if not benchmark.implementations:
            print("⚠️  No implementations loaded")
            return False
        
        print(f"✓ Loaded {len(benchmark.implementations)} implementations:")
        for impl_name, impl_config in benchmark.implementations.items():
            has_classifier = impl_config.get('adu_classifier') is not None
            has_linker = impl_config.get('linker') is not None
            print(f"  - {impl_name}: classifier={has_classifier}, linker={has_linker}")
        
        return True
        
    except Exception as e:
        print(f"❌ Implementation loading failed: {e}")
        return False

def test_single_benchmark_run():
    """Test running a single benchmark task."""
    print("\nTesting single benchmark run...")
    
    try:
        from app.benchmark import ArgumentMiningBenchmark
        
        benchmark = ArgumentMiningBenchmark()
        
        if not benchmark.implementations:
            print("⚠️  No implementations available for testing")
            return False
        
        # Test with first available implementation
        impl_name = list(benchmark.implementations.keys())[0]
        print(f"Testing with implementation: {impl_name}")
        
        # Run ADU extraction benchmark on first sample only
        results = benchmark.benchmark_adu_extraction(impl_name)
        
        if results:
            print(f"✓ Successfully ran benchmark with {len(results)} results")
            
            # Check first result
            first_result = results[0]
            print(f"  - Task: {first_result.task_name}")
            print(f"  - Implementation: {first_result.implementation_name}")
            print(f"  - Success: {first_result.success}")
            
            if first_result.success:
                print(f"  - Metrics: {first_result.metrics}")
                print(f"  - Performance: {first_result.performance}")
            
            return True
        else:
            print("❌ No results returned from benchmark")
            return False
        
    except Exception as e:
        print(f"❌ Benchmark run failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BENCHMARK INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_benchmark_imports),
        ("Initialization Test", test_benchmark_initialization),
        ("Data Loading Test", test_data_loading),
        ("Implementation Loading Test", test_implementation_loading),
        ("Single Benchmark Run Test", test_single_benchmark_run)
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
        print("\n🎉 All tests passed! Benchmark integration is working correctly.")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Run full benchmark: python app/benchmark.py")
        print("3. Check results in the results/ directory")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 