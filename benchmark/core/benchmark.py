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
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add external API app to path to resolve imports
external_api_app = project_root / "external" / "argument-mining-api" / "app"
sys.path.insert(0, str(external_api_app))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Also check for OPENAI_API_KEY and set OPEN_AI_KEY if needed
if os.getenv('OPENAI_API_KEY') and not os.getenv('OPEN_AI_KEY'):
    os.environ['OPEN_AI_KEY'] = os.getenv('OPENAI_API_KEY')

# Import local modules
from ..implementations import (
    OpenAIImplementation,
    TinyLlamaImplementation,
    ModernBERTImplementation,
    DeBERTaImplementation
)
from ..tasks import (
    ADUExtractionTask,
    StanceClassificationTask,
    ClaimPremiseLinkingTask
)
from ..data import DataLoader
from ..utils.file_handlers import save_results_to_csv, save_comprehensive_results_csv
from .results import BenchmarkResult

print("Successfully imported argument mining components")


class ArgumentMiningBenchmark:
    """Enhanced benchmark for argument mining implementations with task-specific data preparation."""

    def __init__(self, max_samples: int = 100):
        """
        Initialize the benchmark.
        
        Args:
            max_samples: Maximum number of samples to use for benchmarking (default: 100)
        """
        self.data = {}
        self.results = []
        self.implementations = {}
        self.tasks = {}
        self.max_samples = max_samples
        self.execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check environment variables
        self._check_environment()
        
        # Initialize implementations
        self._initialize_implementations()
        
        # Initialize tasks
        self._initialize_tasks()
        
        # Load benchmark data for all tasks
        self._load_benchmark_data()
        
        print(f"Initialized benchmark with max_samples: {self.max_samples}")
        print(f"Available implementations: {list(self.implementations.keys())}")
        print(f"Available tasks: {list(self.tasks.keys())}")
    
    def _check_environment(self):
        """Check if required environment variables are set."""
        openai_key = os.getenv("OPEN_AI_KEY")
        if not openai_key:
            print("OPEN_AI_KEY environment variable not set. OpenAI implementations may fail.")
        else:
            print("OPEN_AI_KEY environment variable found")
    
    def _initialize_implementations(self):
        """Initialize all available implementations."""
        implementations = {}
        
        # OpenAI implementation
        try:
            openai_impl = OpenAIImplementation()
            if openai_impl.initialize():
                implementations['openai'] = openai_impl
                print("Initialized OpenAI implementation")
        except Exception as e:
            print(f"Failed to initialize OpenAI implementation: {e}")
        
        # TinyLlama implementation
        try:
            tinyllama_impl = TinyLlamaImplementation()
            if tinyllama_impl.initialize():
                implementations['tinyllama'] = tinyllama_impl
                print("Initialized TinyLlama implementation")
        except Exception as e:
            print(f"Failed to initialize TinyLlama implementation: {e}")
        
        # ModernBERT implementation
        try:
            modernbert_impl = ModernBERTImplementation()
            if modernbert_impl.initialize():
                implementations['modernbert'] = modernbert_impl
                print("Initialized ModernBERT implementation")
        except Exception as e:
            print(f"Failed to initialize ModernBERT implementation: {e}")
        
        # DeBERTa implementation
        try:
            deberta_impl = DeBERTaImplementation()
            if deberta_impl.initialize():
                implementations['deberta'] = deberta_impl
                print("Initialized DeBERTa implementation")
        except Exception as e:
            print(f"Failed to initialize DeBERTa implementation: {e}")
        
        self.implementations = implementations
    
    def _initialize_tasks(self):
        """Initialize all available tasks."""
        self.tasks = {
            'adu_extraction': ADUExtractionTask(),
            'stance_classification': StanceClassificationTask(),
            'claim_premise_linking': ClaimPremiseLinkingTask()
        }
    
    def _load_benchmark_data(self):
        """Load benchmark data for all tasks with task-specific preparation."""
        try:
            data_loader = DataLoader()
            claims, premises, topics = data_loader.load_benchmark_data()
            
            # Prepare data for each task
            self.data = {
                'adu_extraction': self.tasks['adu_extraction'].prepare_data((claims, premises, topics)),
                'stance_classification': self.tasks['stance_classification'].prepare_data((claims, premises, topics)),
                'claim_premise_linking': self.tasks['claim_premise_linking'].prepare_data((claims, premises, topics))
            }
            
            print(f"Loaded benchmark data for all tasks")
            for task, task_data in self.data.items():
                print(f"  {task}: {len(task_data)} samples")
                
        except Exception as e:
            print(f"Failed to load benchmark data: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    def run_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run the complete benchmark suite."""
        all_results = {}
        
        for task_name in self.tasks.keys():
            task_results = self.run_single_task(task_name)
            all_results[task_name] = task_results
        
        # Save results in organized subdirectories
        self.save_results(all_results, comprehensive=True, individual=True)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], 
                    comprehensive: bool = True, 
                    individual: bool = False,
                    output_dir: str = "results") -> str:
        """
        Save benchmark results to CSV files in organized subdirectories.
        
        Args:
            results: Dictionary of benchmark results
            comprehensive: Whether to save all results in one comprehensive CSV
            individual: Whether to save individual task results in separate CSVs
            output_dir: Output directory for CSV files
            
        Returns:
            Path to the output directory
        """
        # Create main results directory
        main_output_path = Path(output_dir)
        main_output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        inferences_dir = main_output_path / "inferences"
        metrics_dir = main_output_path / "metrics"
        
        inferences_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)
        
        if comprehensive:
            # Append to the three specific CSV files (no comprehensive results)
            from ..utils.file_handlers import (
                append_to_implementations_csv,
                append_to_tasks_csv,
                append_to_system_csv
            )
            
            # Append implementation metrics to implementations.csv
            append_to_implementations_csv(results, str(metrics_dir))
            
            # Append task metrics to tasks.csv
            append_to_tasks_csv(results, str(metrics_dir))
            
            # Append system metrics to system.csv
            append_to_system_csv(results, str(metrics_dir))
        
        if individual:
            # Save individual results to inferences subdirectory
            from ..utils.file_handlers import save_results_to_csv
            save_results_to_csv(results, str(inferences_dir))
        
        print(f"Results saved to:")
        print(f"  Inferences: {inferences_dir}")
        print(f"  Metrics: {metrics_dir}")
        print(f"    - implementations.csv (appended)")
        print(f"    - tasks.csv (appended)")
        print(f"    - system.csv (appended)")
        
        return str(main_output_path)
    
    def get_execution_timestamp(self) -> str:
        """Get the current execution timestamp."""
        return self.execution_date
    
    def run_single_task(self, task_name: str, implementation_name: str = None) -> List[BenchmarkResult]:
        """Run benchmark for a single task."""
        if task_name not in self.tasks:
            print(f"Unknown task: {task_name}")
            return []
        
        task = self.tasks[task_name]
        task_data = self.data.get(task_name, [])
        
        if not task_data:
            print(f"No data available for task: {task_name}")
            return []
        
        # Limit data to max_samples
        task_data = task_data[:self.max_samples]
        
        all_results = []
        
        # Run task with each implementation
        for impl_name, implementation in self.implementations.items():
            if implementation_name and impl_name != implementation_name:
                continue
            
            if not implementation.supports_task(task_name):
                print(f"Implementation {impl_name} does not support task {task_name}")
                continue
            
            print(f"Running {task_name} with {impl_name} implementation...")
            results = task.run_benchmark(implementation, task_data)
            all_results.extend(results)
        
        return all_results
    
    def run_single_implementation(self, implementation_name: str, task_name: str = None) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark for a single implementation."""
        if implementation_name not in self.implementations:
            print(f"Unknown implementation: {implementation_name}")
            return {}
        
        implementation = self.implementations[implementation_name]
        all_results = {}
        
        # Run all tasks or specific task
        tasks_to_run = [task_name] if task_name else self.tasks.keys()
        
        for task_name in tasks_to_run:
            if task_name not in self.tasks:
                continue
            
            if not implementation.supports_task(task_name):
                print(f"Implementation {implementation_name} does not support task {task_name}")
                continue
            
            print(f"Running {task_name} with {implementation_name} implementation...")
            results = self.run_single_task(task_name, implementation_name)
            all_results[task_name] = results
        
        return all_results
    
    def _print_summary(self, results: Dict[str, List[BenchmarkResult]]):
        """Print a summary of benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Execution Date: {self.execution_date}")
        
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
