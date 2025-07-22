#!/usr/bin/env python3
"""
Benchmark script for argument mining implementations.

This script benchmarks all implementations of the AduAndStanceClassifier and ClaimPremiseLinker
interfaces using the benchmark data from db-connector.
"""

import sys
import os
import time
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add the argument-mining-api to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'argument-mining-api'))

# Add the db-connector to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'db-connector'))

try:
    from db.db import get_session
    from db.models import ADU, Relationship
    from db.queries import get_benchmark_data
except ImportError as e:
    print(f"Error importing db-connector modules: {e}")
    print("Make sure db-connector is properly set up")
    sys.exit(1)

try:
    from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
    from app.argmining.interfaces.claim_premise_linker import ClaimPremiseLinker
    from app.argmining.models.argument_units import (
        ArgumentUnit, 
        UnlinkedArgumentUnits, 
        LinkedArgumentUnits, 
        LinkedArgumentUnitsWithStance
    )
    from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
    from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
    from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    from app.argmining.implementations.encoder_model_loader import EncoderModelLoader
except ImportError as e:
    print(f"Error importing argument-mining-api modules: {e}")
    print("Make sure argument-mining-api is properly set up")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Represents the result of a single benchmark run."""
    implementation_name: str
    interface_type: str  # 'adu_classifier' or 'linker'
    test_case_id: str
    execution_time: float
    success: bool
    error_message: str = ""
    metrics: Dict[str, Any] = None


class ArgumentMiningBenchmark:
    """Benchmark class for argument mining implementations."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.implementations = {
            'adu_classifier': [
                ('OpenAILLMClassifier', OpenAILLMClassifier),
                ('TinyLLamaLLMClassifier', TinyLLamaLLMClassifier),
                ('EncoderModelLoader', EncoderModelLoader),
            ],
            'linker': [
                ('OpenAIClaimPremiseLinker', OpenAIClaimPremiseLinker),
            ]
        }
    
    def convert_adu_to_argument_unit(self, adu: ADU) -> ArgumentUnit:
        """Convert database ADU to ArgumentUnit."""
        return ArgumentUnit(
            uuid=uuid.uuid4(),
            text=adu.text,
            type=adu.type,
            start_pos=None,
            end_pos=None,
            confidence=None
        )
    
    def prepare_test_data(self) -> List[Tuple[str, List[ADU], List[ADU], List[str]]]:
        """Prepare test data from benchmark dataset."""
        print("Loading benchmark data...")
        claims, premises, categories = get_benchmark_data()
        
        # Group data by claim-premise relationships
        test_cases = []
        claim_premise_map = {}
        
        # Create a mapping of claim to premises
        for i, claim in enumerate(claims):
            if i < len(premises) and i < len(categories):
                claim_premise_map[claim.id] = {
                    'claim': claim,
                    'premises': [premises[i]] if premises[i] else [],
                    'categories': [categories[i]] if categories[i] else []
                }
        
        # Create test cases
        for claim_id, data in claim_premise_map.items():
            test_case_id = f"test_case_{claim_id}"
            test_cases.append((
                test_case_id,
                [data['claim']],
                data['premises'],
                data['categories']
            ))
        
        print(f"Prepared {len(test_cases)} test cases")
        return test_cases
    
    def benchmark_adu_classifier(self, classifier: AduAndStanceClassifier, test_cases: List[Tuple[str, List[ADU], List[ADU], List[str]]]) -> List[BenchmarkResult]:
        """Benchmark ADU classifier implementations."""
        results = []
        
        for test_case_id, claims, premises, categories in test_cases:
            # Combine all text for classification
            all_text = " ".join([claim.text for claim in claims] + [premise.text for premise in premises])
            
            if not all_text.strip():
                continue
                
            start_time = time.time()
            success = False
            error_message = ""
            
            try:
                # Test ADU classification
                argument_units = classifier.classify_adus(all_text)
                
                # Basic metrics
                metrics = {
                    'total_units_extracted': len(argument_units),
                    'claims_extracted': len([u for u in argument_units if u.type == 'claim']),
                    'premises_extracted': len([u for u in argument_units if u.type == 'premise']),
                    'avg_confidence': sum([u.confidence or 0 for u in argument_units]) / len(argument_units) if argument_units else 0
                }
                
                success = True
                
            except Exception as e:
                error_message = str(e)
                metrics = {}
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                implementation_name=classifier.__class__.__name__,
                interface_type='adu_classifier',
                test_case_id=test_case_id,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                metrics=metrics
            )
            
            results.append(result)
        
        return results
    
    def benchmark_linker(self, linker: ClaimPremiseLinker, test_cases: List[Tuple[str, List[ADU], List[ADU], List[str]]]) -> List[BenchmarkResult]:
        """Benchmark claim-premise linker implementations."""
        results = []
        
        for test_case_id, claims, premises, categories in test_cases:
            # Convert to ArgumentUnits
            claim_units = [self.convert_adu_to_argument_unit(claim) for claim in claims]
            premise_units = [self.convert_adu_to_argument_unit(premise) for premise in premises]
            
            if not claim_units or not premise_units:
                continue
            
            unlinked_units = UnlinkedArgumentUnits(
                claims=claim_units,
                premises=premise_units
            )
            
            start_time = time.time()
            success = False
            error_message = ""
            
            try:
                # Test linking
                linked_units = linker.link_claims_to_premises(unlinked_units, max_retries=3)
                
                # Basic metrics
                metrics = {
                    'total_claims': len(linked_units.claims),
                    'total_premises': len(linked_units.premises),
                    'total_relationships': len(linked_units.claims_premises_relationships),
                    'avg_premises_per_claim': len(linked_units.premises) / len(linked_units.claims) if linked_units.claims else 0
                }
                
                success = True
                
            except Exception as e:
                error_message = str(e)
                metrics = {}
            
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                implementation_name=linker.__class__.__name__,
                interface_type='linker',
                test_case_id=test_case_id,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                metrics=metrics
            )
            
            results.append(result)
        
        return results
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print("Starting argument mining benchmarks...")
        
        # Prepare test data
        test_cases = self.prepare_test_data()
        
        all_results = []
        
        # Benchmark ADU classifiers
        print("\nBenchmarking ADU classifiers...")
        for name, implementation_class in self.implementations['adu_classifier']:
            try:
                print(f"Testing {name}...")
                implementation = implementation_class()
                results = self.benchmark_adu_classifier(implementation, test_cases)
                all_results.extend(results)
                print(f"Completed {name}: {len(results)} test cases")
            except Exception as e:
                print(f"Error testing {name}: {e}")
        
        # Benchmark linkers
        print("\nBenchmarking claim-premise linkers...")
        for name, implementation_class in self.implementations['linker']:
            try:
                print(f"Testing {name}...")
                implementation = implementation_class()
                results = self.benchmark_linker(implementation, test_cases)
                all_results.extend(results)
                print(f"Completed {name}: {len(results)} test cases")
            except Exception as e:
                print(f"Error testing {name}: {e}")
        
        return all_results
    
    def save_results_to_csv(self, results: List[BenchmarkResult], filename: str = None):
        """Save benchmark results to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        print(f"\nSaving results to {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'implementation_name',
                'interface_type', 
                'test_case_id',
                'execution_time',
                'success',
                'error_message',
                'total_units_extracted',
                'claims_extracted',
                'premises_extracted',
                'avg_confidence',
                'total_claims',
                'total_premises',
                'total_relationships',
                'avg_premises_per_claim'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'implementation_name': result.implementation_name,
                    'interface_type': result.interface_type,
                    'test_case_id': result.test_case_id,
                    'execution_time': result.execution_time,
                    'success': result.success,
                    'error_message': result.error_message,
                    'total_units_extracted': result.metrics.get('total_units_extracted', 0),
                    'claims_extracted': result.metrics.get('claims_extracted', 0),
                    'premises_extracted': result.metrics.get('premises_extracted', 0),
                    'avg_confidence': result.metrics.get('avg_confidence', 0),
                    'total_claims': result.metrics.get('total_claims', 0),
                    'total_premises': result.metrics.get('total_premises', 0),
                    'total_relationships': result.metrics.get('total_relationships', 0),
                    'avg_premises_per_claim': result.metrics.get('avg_premises_per_claim', 0)
                }
                writer.writerow(row)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group results by implementation
        by_implementation = {}
        for result in results:
            key = result.implementation_name
            if key not in by_implementation:
                by_implementation[key] = []
            by_implementation[key].append(result)
        
        for impl_name, impl_results in by_implementation.items():
            successful = [r for r in impl_results if r.success]
            failed = [r for r in impl_results if not r.success]
            
            avg_time = sum(r.execution_time for r in successful) / len(successful) if successful else 0
            success_rate = len(successful) / len(impl_results) * 100 if impl_results else 0
            
            print(f"\n{impl_name}:")
            print(f"  Total tests: {len(impl_results)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average execution time: {avg_time:.3f}s")
            
            if failed:
                print(f"  Error messages: {[r.error_message for r in failed[:3]]}")


def main():
    """Main function to run the benchmark."""
    print("Argument Mining Benchmark Tool")
    print("="*40)
    
    # Check if required environment variables are set
    if not os.getenv('OPENAI_KEY'):
        print("Warning: OPENAI_KEY environment variable not set. OpenAI implementations may fail.")
    
    # Create and run benchmark
    benchmark = ArgumentMiningBenchmark()
    results = benchmark.run_benchmarks()
    
    # Save results
    benchmark.save_results_to_csv(results)
    
    # Print summary
    benchmark.print_summary(results)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main() 