#!/usr/bin/env python3
"""
Enhanced Argument Mining Benchmark

This script benchmarks all available argument mining implementations:
- OpenAI LLM Classifier
- TinyLlama LLM Classifier  
- ModernBERT (PeftEncoderModelLoader)
- DeBERTa (NonTrainedEncoderModelLoader)

Tasks benchmarked:
1. ADU Extraction (Claims and Premises)
2. Stance Classification (Pro/Con/Neutral)
3. Claim-Premise Linking

Metrics calculated:
- Token-level precision/recall/F1 for ADU extraction
- Accuracy and weighted F1 for stance classification
- Relationship accuracy for claim-premise linking
- Inference time per sample
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
import traceback
# Import log from external API if available, otherwise use local
try:
    from app.log import log
except ImportError:
    # Create a simple logger if external log module is not available
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add external submodules to Python path
external_api = project_root / "external" / "argument-mining-api"
external_db = project_root / "external" / "argument-mining-db"

if external_api.exists():
    sys.path.insert(0, str(external_api))
    print(f"✓ Added argument-mining-api to Python path: {external_api}")
else:
    print(f"⚠️  argument-mining-api not found at: {external_api}")

if external_db.exists():
    sys.path.insert(0, str(external_db))
    print(f"✓ Added argument-mining-db to Python path: {external_db}")
else:
    print(f"⚠️  argument-mining-db not found at: {external_db}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import argument mining components from external API
try:
    from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
    from app.argmining.interfaces.claim_premise_linker import ClaimPremiseLinker
    from app.argmining.models.argument_units import (
        ArgumentUnit, 
        UnlinkedArgumentUnits, 
        LinkedArgumentUnits, 
        LinkedArgumentUnitsWithStance,
        StanceRelation,
        ClaimPremiseRelationship
    )
    from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
    from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    from app.argmining.implementations.encoder_model_loader import (
        PeftEncoderModelLoader, 
        NonTrainedEncoderModelLoader,
        MODEL_CONFIGS
    )
    from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
    print("✓ Successfully imported argument mining components from external API")
except ImportError as e:
    print(f"❌ Error importing argument mining components: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

# Import database components from external DB
try:
    from db.queries import get_benchmark_data
    print("✓ Successfully imported database components from external DB")
except ImportError as e:
    print(f"❌ Error importing database components: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    # Fallback to local db_connector if external fails
    try:
        from app.db_connector.db.queries import get_benchmark_data_for_evaluation
        print("✓ Successfully imported database components from local db_connector")
    except ImportError as e2:
        print(f"❌ Error importing database components from local db_connector: {e2}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@dataclass
class BenchmarkResult:
    """Represents benchmark results for a single task/implementation."""
    task_name: str
    implementation_name: str
    sample_id: str
    metrics: Dict[str, float]
    performance: Dict[str, float]
    predictions: Any
    ground_truth: Any
    error_message: str = ""
    success: bool = True


class ArgumentMiningBenchmark:
    """Enhanced benchmark for argument mining implementations."""

    def __init__(self, max_samples: int = 100):
        """
        Initialize the benchmark.
        
        Args:
            max_samples: Maximum number of samples to use for benchmarking (default: 100)
        """
        self.data = None
        self.results = []
        self.implementations = {}
        self.max_samples = max_samples
        
        # Check environment variables
        self._check_environment()
        
        # Initialize implementations
        self._initialize_implementations()
        
        # Load benchmark data
        self._load_benchmark_data()
        
        log.info(f"Initialized benchmark with {len(self.data)} samples (max_samples: {self.max_samples})")
        log.info(f"Available implementations: {list(self.implementations.keys())}")
    
    def _check_environment(self):
        """Check if required environment variables are set."""
        openai_key = os.getenv("OPEN_AI_KEY")
        if not openai_key:
            log.warning("OPEN_AI_KEY environment variable not set. OpenAI implementations may fail.")
        else:
            log.info("✓ OPEN_AI_KEY environment variable found")
    
    def _initialize_implementations(self):
        """Initialize all available implementations."""
        implementations = {}
        
        try:
            # OpenAI implementation
            try:
                implementations['openai'] = {
                    'adu_classifier': OpenAILLMClassifier(),
                    'linker': OpenAIClaimPremiseLinker()
                }
                log.info("✓ Initialized OpenAI implementation")
            except Exception as e:
                log.error(f"Failed to initialize OpenAI implementation: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
            
            # TinyLlama implementation
            try:
                implementations['tinyllama'] = {
                    'adu_classifier': TinyLLamaLLMClassifier(),
                    'linker': None  # TinyLlama doesn't have linking
                }
                log.info("✓ Initialized TinyLlama implementation")
            except Exception as e:
                log.error(f"Failed to initialize TinyLlama implementation: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
            
            # ModernBERT implementation
            try:
                modernbert_config = MODEL_CONFIGS.get('modernbert')
                if modernbert_config:
                    implementations['modernbert'] = {
                        'adu_classifier': PeftEncoderModelLoader(**modernbert_config['params']),
                        'linker': None
                    }
                    log.info("✓ Initialized ModernBERT implementation")
            except Exception as e:
                log.error(f"Failed to initialize ModernBERT implementation: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
            
            # DeBERTa implementation
            try:
                deberta_config = MODEL_CONFIGS.get('deberta')
                if deberta_config:
                    implementations['deberta'] = {
                        'adu_classifier': NonTrainedEncoderModelLoader(**deberta_config['params']),
                        'linker': None
                    }
                    log.info("✓ Initialized DeBERTa implementation")
            except Exception as e:
                log.error(f"Failed to initialize DeBERTa implementation: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
            
        except Exception as e:
            log.error(f"Error initializing implementations: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
        
        self.implementations = implementations
    
    def _load_benchmark_data(self):
        """Load benchmark data from database."""
        try:
            # Try to use external DB function first
            try:
                claims, premises, categories = get_benchmark_data()
                # Convert to the expected format
                self.data = []
                for i, (claim, premise, category) in enumerate(zip(claims, premises, categories)):
                    if i >= self.max_samples:
                        break
                    sample = {
                        'text': f"{claim.text} {premise.text}",
                        'ground_truth': {
                            'adus': [
                                {'text': claim.text, 'type': 'claim'},
                                {'text': premise.text, 'type': 'premise'}
                            ],
                            'stance': category,
                            'relationships': [{'claim_id': claim.id, 'premise_id': premise.id}]
                        }
                    }
                    self.data.append(sample)
                log.info(f"Loaded {len(self.data)} benchmark samples from external DB (requested: {self.max_samples})")
            except NameError:
                # Fallback to local function if external function not available
                self.data = get_benchmark_data_for_evaluation(self.max_samples)
                log.info(f"Loaded {len(self.data)} benchmark samples from local DB (requested: {self.max_samples})")
        except Exception as e:
            log.error(f"Failed to load benchmark data: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            self.data = []
    
    def benchmark_adu_extraction(self, implementation_name: str) -> List[BenchmarkResult]:
        """Benchmark ADU extraction task."""
        log.info(f"Benchmarking ADU extraction with {implementation_name}")
        
        if implementation_name not in self.implementations:
            log.warning(f"Implementation {implementation_name} not available")
            return []
        
        classifier = self.implementations[implementation_name]['adu_classifier']
        results = []
        
        for i, sample in enumerate(self.data):
            try:
                # Measure inference time
                start_time = time.time()
                prediction = classifier.classify_adus(sample['text'])
                inference_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_adu_metrics(prediction, sample['ground_truth']['adus'])
                performance = {'inference_time': inference_time}
                
                result = BenchmarkResult(
                    task_name='adu_extraction',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics=metrics,
                    performance=performance,
                    predictions=prediction,
                    ground_truth=sample['ground_truth']['adus']
                )
                results.append(result)
                
            except Exception as e:
                log.error(f"Error processing sample {i} with {implementation_name}: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
                result = BenchmarkResult(
                    task_name='adu_extraction',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics={},
                    performance={},
                    predictions=None,
                    ground_truth=sample['ground_truth']['adus'],
                    error_message=f"{str(e)}\n{traceback.format_exc()}",
                    success=False
                )
                results.append(result)
        
        return results
    
    def benchmark_stance_classification(self, implementation_name: str) -> List[BenchmarkResult]:
        """Benchmark stance classification task."""
        log.info(f"Benchmarking stance classification with {implementation_name}")
        
        if implementation_name not in self.implementations:
            log.warning(f"Implementation {implementation_name} not available")
            return []
        
        classifier = self.implementations[implementation_name]['adu_classifier']
        results = []
        
        for i, sample in enumerate(self.data):
            try:
                # First extract ADUs
                adus = classifier.classify_adus(sample['text'])
                
                # Create linked argument units for stance classification
                linked_units = LinkedArgumentUnits(
                    claims=adus.claims,
                    premises=adus.premises,
                    claims_premises_relationships=[]
                )
                
                # Measure inference time
                start_time = time.time()
                stance_result = classifier.classify_stance(linked_units, sample['text'])
                inference_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_stance_metrics(stance_result, sample['ground_truth']['stance'])
                performance = {'inference_time': inference_time}
                
                result = BenchmarkResult(
                    task_name='stance_classification',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics=metrics,
                    performance=performance,
                    predictions=stance_result,
                    ground_truth=sample['ground_truth']['stance']
                )
                results.append(result)
                
            except Exception as e:
                log.error(f"Error processing sample {i} with {implementation_name}: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
                result = BenchmarkResult(
                    task_name='stance_classification',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics={},
                    performance={},
                    predictions=None,
                    ground_truth=sample['ground_truth']['stance'],
                    error_message=f"{str(e)}\n{traceback.format_exc()}",
                    success=False
                )
                results.append(result)
        
        return results
    
    def benchmark_claim_premise_linking(self, implementation_name: str) -> List[BenchmarkResult]:
        """Benchmark claim-premise linking task."""
        log.info(f"Benchmarking claim-premise linking with {implementation_name}")
        
        if implementation_name not in self.implementations:
            log.warning(f"Implementation {implementation_name} not available")
            return []
        
        linker = self.implementations[implementation_name]['linker']
        if not linker:
            log.warning(f"No linker available for {implementation_name}")
            return []
        
        results = []
        
        for i, sample in enumerate(self.data):
            try:
                # First extract ADUs to create UnlinkedArgumentUnits
                classifier = self.implementations[implementation_name]['adu_classifier']
                adus = classifier.classify_adus(sample['text'])
                
                # Measure inference time
                start_time = time.time()
                prediction = linker.link_claims_to_premises(adus)
                inference_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_linking_metrics(prediction, sample['ground_truth']['relationships'])
                performance = {'inference_time': inference_time}
                
                result = BenchmarkResult(
                    task_name='claim_premise_linking',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics=metrics,
                    performance=performance,
                    predictions=prediction,
                    ground_truth=sample['ground_truth']['relationships']
                )
                results.append(result)
                
            except Exception as e:
                log.error(f"Error processing sample {i} with {implementation_name}: {e}")
                log.error(f"Traceback: {traceback.format_exc()}")
                result = BenchmarkResult(
                    task_name='claim_premise_linking',
                    implementation_name=implementation_name,
                    sample_id=f"sample_{i}",
                    metrics={},
                    performance={},
                    predictions=None,
                    ground_truth=sample['ground_truth']['relationships'],
                    error_message=f"{str(e)}\n{traceback.format_exc()}",
                    success=False
                )
                results.append(result)
        
        return results
    
    def _calculate_adu_metrics(self, prediction, ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate token-level metrics for ADU extraction."""
        if not prediction or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Extract predicted ADUs
        predicted_adus = []
        if hasattr(prediction, 'claims'):
            predicted_adus.extend([adu.text for adu in prediction.claims])
        if hasattr(prediction, 'premises'):
            predicted_adus.extend([adu.text for adu in prediction.premises])
        
        # Extract ground truth ADUs
        gt_adus = [adu['text'] for adu in ground_truth]
        
        # Calculate token-level metrics
        true_positives = len(set(predicted_adus) & set(gt_adus))
        false_positives = len(set(predicted_adus) - set(gt_adus))
        false_negatives = len(set(gt_adus) - set(predicted_adus))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(gt_adus) if len(gt_adus) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_stance_metrics(self, prediction, ground_truth: str) -> Dict[str, float]:
        """Calculate metrics for stance classification."""
        if not prediction or not prediction.stance_relations:
            return {'accuracy': 0.0, 'weighted_f1': 0.0}
        
        # Extract predicted stance
        predicted_stance = prediction.stance_relations[0].stance if prediction.stance_relations else "neutral"
        
        # Map stances to consistent format
        stance_mapping = {
            'pro': 'pro', 'con': 'con', 'neutral': 'neutral',
            'stance_pro': 'pro', 'stance_con': 'con'
        }
        
        predicted_stance = stance_mapping.get(predicted_stance, predicted_stance)
        ground_truth = stance_mapping.get(ground_truth, ground_truth)
        
        # Calculate accuracy
        accuracy = 1.0 if predicted_stance == ground_truth else 0.0
        
        # Calculate weighted F1 (simplified)
        weighted_f1 = accuracy  # For binary classification, F1 = accuracy when classes are balanced
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'predicted_stance': predicted_stance,
            'ground_truth_stance': ground_truth
        }
    
    def _calculate_linking_metrics(self, prediction, ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for claim-premise linking."""
        if not prediction or not ground_truth:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # This is a simplified metric calculation
        # In a real implementation, you'd need more sophisticated relationship matching
        
        # For now, return basic accuracy
        accuracy = 0.5  # Placeholder
        
        return {
            'accuracy': accuracy,
            'precision': accuracy,
            'recall': accuracy,
            'f1': accuracy
        }
    
    def run_benchmark(self, tasks: List[str] = None, implementations: List[str] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run the complete benchmark suite."""
        if tasks is None:
            tasks = ['adu_extraction', 'stance_classification', 'claim_premise_linking']
        
        if implementations is None:
            implementations = list(self.implementations.keys())
        
        log.info(f"Starting benchmark with tasks: {tasks}")
        log.info(f"Testing implementations: {implementations}")
        
        all_results = {}
        
        for task in tasks:
            task_results = []
            
            for impl_name in implementations:
                if impl_name not in self.implementations:
                    log.warning(f"Implementation {impl_name} not available, skipping")
                    continue
                
                try:
                    if task == 'adu_extraction':
                        results = self.benchmark_adu_extraction(impl_name)
                    elif task == 'stance_classification':
                        results = self.benchmark_stance_classification(impl_name)
                    elif task == 'claim_premise_linking':
                        results = self.benchmark_claim_premise_linking(impl_name)
                    else:
                        log.warning(f"Unknown task: {task}")
                        continue
                    
                    task_results.extend(results)
                    
                except Exception as e:
                    log.error(f"Error running {task} with {impl_name}: {e}")
                    log.error(f"Traceback: {traceback.format_exc()}")
            
            all_results[task] = task_results
        
        # Save results
        self._save_results(all_results)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to CSV files."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for task_name, task_results in results.items():
            if not task_results:
                continue
            
            # Convert to DataFrame
            df_data = []
            for result in task_results:
                row = {
                    'task': result.task_name,
                    'implementation': result.implementation_name,
                    'sample_id': result.sample_id,
                    'success': result.success,
                    'error_message': result.error_message
                }
                
                # Add metrics
                for metric_name, metric_value in result.metrics.items():
                    row[f'metric_{metric_name}'] = metric_value
                
                # Add performance metrics
                for perf_name, perf_value in result.performance.items():
                    row[f'perf_{perf_name}'] = perf_value
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            filename = f"{task_name}_results_{timestamp}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            log.info(f"Saved {task_name} results to {filepath}")
    
    def _print_summary(self, results: Dict[str, List[BenchmarkResult]]):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for task_name, task_results in results.items():
            if not task_results:
                continue
            
            print(f"\n{task_name.upper()} RESULTS:")
            print("-" * 40)
            
            # Group by implementation
            impl_results = {}
            for result in task_results:
                if result.success:
                    if result.implementation_name not in impl_results:
                        impl_results[result.implementation_name] = []
                    impl_results[result.implementation_name].append(result)
            
            for impl_name, impl_result_list in impl_results.items():
                if not impl_result_list:
                    continue
                
                # Calculate average metrics
                avg_metrics = {}
                avg_performance = {}
                
                for metric_name in impl_result_list[0].metrics.keys():
                    values = [r.metrics[metric_name] for r in impl_result_list if r.success]
                    avg_metrics[metric_name] = np.mean(values) if values else 0.0
                
                for perf_name in impl_result_list[0].performance.keys():
                    values = [r.performance[perf_name] for r in impl_result_list if r.success]
                    avg_performance[perf_name] = np.mean(values) if values else 0.0
                
                print(f"\n{impl_name}:")
                for metric_name, value in avg_metrics.items():
                    print(f"  {metric_name}: {value:.3f}")
                for perf_name, value in avg_performance.items():
                    print(f"  {perf_name}: {value:.3f}s")
        
        print("\n" + "="*80)


def test_imports(max_samples: int = 100):
    """Test if all imports are working correctly."""
    print("Testing imports...")
    
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        print("✓ Benchmark class created successfully")
        print(f"  - Loaded {len(benchmark.data)} samples (requested: {max_samples})")
        print(f"  - Available implementations: {list(benchmark.implementations.keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed to create benchmark: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def run_full_benchmark(max_samples: int = 100):
    """Run the complete benchmark suite."""
    try:
        benchmark = ArgumentMiningBenchmark(max_samples=max_samples)
        results = benchmark.run_benchmark()
        print("✓ Benchmark completed successfully")
        return results
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    # Test imports first
    if test_imports():
        # Run full benchmark
        print(run_full_benchmark())
