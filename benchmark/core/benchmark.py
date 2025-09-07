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

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings
import traceback
from tqdm import tqdm

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
from ..utils.logging_utils import get_logger, log_initialization, log_benchmark_progress
from .results import BenchmarkResult

# Initialize logger
logger = get_logger()
logger.info("Successfully imported argument mining components")


class ArgumentMiningBenchmark:
    """Enhanced benchmark for argument mining implementations with task-specific data preparation."""

    def __init__(self, max_samples: int = 100, disable_openai: bool = False, 
                 disable_tinyllama: bool = False, disable_modernbert: bool = False, 
                 disable_deberta: bool = False):
        """
        Initialize the benchmark.
        
        Args:
            max_samples: Maximum number of samples to use for benchmarking (default: 100)
            disable_openai: If True, skip OpenAI implementation initialization (default: False)
            disable_tinyllama: If True, skip TinyLlama implementation initialization (default: False)
            disable_modernbert: If True, skip ModernBERT implementation initialization (default: False)
            disable_deberta: If True, skip DeBERTa implementation initialization (default: False)
        """
        self.data = {}
        self.results = []
        self.implementations = {}
        self.tasks = {}
        self.max_samples = max_samples
        self.disable_openai = disable_openai
        self.disable_tinyllama = disable_tinyllama
        self.disable_modernbert = disable_modernbert
        self.disable_deberta = disable_deberta
        self.execution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check environment variables
        self._check_environment()
        
        # Initialize implementations
        self._initialize_implementations()
        
        # Initialize tasks
        self._initialize_tasks()
        
        # Load benchmark data for all tasks
        self._load_benchmark_data()
        
        logger.info(f"Initialized benchmark with max_samples: {self.max_samples}")
        disabled_implementations = []
        if self.disable_openai:
            disabled_implementations.append("OpenAI")
        if self.disable_tinyllama:
            disabled_implementations.append("TinyLlama")
        if self.disable_modernbert:
            disabled_implementations.append("ModernBERT")
        if self.disable_deberta:
            disabled_implementations.append("DeBERTa")
        
        if disabled_implementations:
            logger.info(f"Disabled implementations: {', '.join(disabled_implementations)}")
        logger.info(f"Available implementations: {list(self.implementations.keys())}")
        logger.info(f"Available tasks: {list(self.tasks.keys())}")
    
    def _check_environment(self):
        """Check if required environment variables are set."""
        openai_key = os.getenv("OPEN_AI_KEY")
        if not openai_key:
            logger.warning("OPEN_AI_KEY environment variable not set. OpenAI implementations may fail.")
        else:
            logger.info("OPEN_AI_KEY environment variable found")
    
    def _initialize_implementations(self):
        """Initialize all available implementations."""
        implementations = {}
        
        # OpenAI implementation
        if not self.disable_openai:
            try:
                openai_impl = OpenAIImplementation()
                if openai_impl.initialize():
                    implementations['openai'] = openai_impl
                    log_initialization(logger, "OpenAI implementation", "success")
                else:
                    log_initialization(logger, "OpenAI implementation", "failed", "Initialization returned False")
            except Exception as e:
                log_initialization(logger, "OpenAI implementation", "failed", str(e))
        else:
            log_initialization(logger, "OpenAI implementation", "disabled")
        
        # TinyLlama implementation
        if not self.disable_tinyllama:
            try:
                tinyllama_impl = TinyLlamaImplementation()
                if tinyllama_impl.initialize():
                    implementations['tinyllama'] = tinyllama_impl
                    log_initialization(logger, "TinyLlama implementation", "success")
                else:
                    log_initialization(logger, "TinyLlama implementation", "failed", "Initialization returned False")
            except Exception as e:
                log_initialization(logger, "TinyLlama implementation", "failed", str(e))
        else:
            log_initialization(logger, "TinyLlama implementation", "disabled")
        
        # ModernBERT implementation
        if not self.disable_modernbert:
            try:
                modernbert_impl = ModernBERTImplementation()
                if modernbert_impl.initialize():
                    implementations['modernbert'] = modernbert_impl
                    log_initialization(logger, "ModernBERT implementation", "success")
                else:
                    log_initialization(logger, "ModernBERT implementation", "failed", "Initialization returned False")
            except Exception as e:
                log_initialization(logger, "ModernBERT implementation", "failed", str(e))
        else:
            log_initialization(logger, "ModernBERT implementation", "disabled")
        
        # DeBERTa implementation
        if not self.disable_deberta:
            try:
                deberta_impl = DeBERTaImplementation()
                if deberta_impl.initialize():
                    implementations['deberta'] = deberta_impl
                    log_initialization(logger, "DeBERTa implementation", "success")
                else:
                    log_initialization(logger, "DeBERTa implementation", "failed", "Initialization returned False")
            except Exception as e:
                log_initialization(logger, "DeBERTa implementation", "failed", str(e))
        else:
            log_initialization(logger, "DeBERTa implementation", "disabled")
        
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
            
            logger.info(f"Loaded benchmark data for all tasks")
            for task, task_data in self.data.items():
                logger.info(f"  {task}: {len(task_data)} samples")
                
        except Exception as e:
            logger.error(f"Failed to load benchmark data: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def run_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run the complete benchmark suite."""
        all_results = {}
        
        # Calculate total number of task-implementation combinations
        total_combinations = 0
        for task_name in self.tasks.keys():
            for impl_name, implementation in self.implementations.items():
                if implementation.supports_task(task_name):
                    total_combinations += 1
        
        logger.info(f"Running benchmark with {total_combinations} task-implementation combinations...")
        
        # Create overall progress bar
        with tqdm(total=total_combinations, desc="Overall Progress", unit="combination") as pbar:
            for task_name in self.tasks.keys():
                task_results = self.run_single_task(task_name, progress_bar=pbar)
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
        Save benchmark results to CSV and JSON files in organized subdirectories.
        
        Args:
            results: Dictionary of benchmark results
            comprehensive: Whether to save all results in comprehensive formats
            individual: Whether to save individual task results in separate files
            output_dir: Output directory for CSV and JSON files
            
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
            # Save to JSON files (append mode)
            from ..utils.file_handlers import (
                append_to_implementations_json,
                append_to_tasks_json,
                append_to_system_json
            )
            
            # Append implementation metrics to implementations.json
            append_to_implementations_json(results, str(metrics_dir))
            
            # Append task metrics to tasks.json
            append_to_tasks_json(results, str(metrics_dir))
            
            # Append system metrics to system.json
            append_to_system_json(results, str(metrics_dir))
            
            # Save to CSV files (comprehensive)
            from ..utils.file_handlers import save_comprehensive_results_csv
            save_comprehensive_results_csv(results, str(main_output_path))
        
        if individual:
            # Save individual results to inferences subdirectory
            from ..utils.file_handlers import save_results_to_json
            save_results_to_json(results, str(inferences_dir))
        
        logger.info(f"Results saved to:")
        logger.info(f"  Inferences: {inferences_dir}")
        logger.info(f"  Metrics: {metrics_dir}")
        logger.info(f"    - implementations.json (appended)")
        logger.info(f"    - tasks.json (appended)")
        logger.info(f"    - system.json (appended)")
        logger.info(f"    - CSV files (comprehensive)")
        
        return str(main_output_path)
    
    def get_execution_timestamp(self) -> str:
        """Get the current execution timestamp."""
        return self.execution_date
    
    def run_single_task(self, task_name: str, implementation_name: str = None, progress_bar=None) -> List[BenchmarkResult]:
        """Run benchmark for a single task."""
        if task_name not in self.tasks:
            logger.error(f"Unknown task: {task_name}")
            return []
        
        task = self.tasks[task_name]
        task_data = self.data.get(task_name, [])
        
        if not task_data:
            logger.warning(f"No data available for task: {task_name}")
            return []
        
        # Limit data to max_samples
        task_data = task_data[:self.max_samples]
        
        all_results = []
        
        # Run task with each implementation
        for impl_name, implementation in self.implementations.items():
            if implementation_name and impl_name != implementation_name:
                continue
            
            if not implementation.supports_task(task_name):
                if progress_bar:
                    progress_bar.update(1)
                continue
            
            log_benchmark_progress(logger, task_name, impl_name, "started")
            results = task.run_benchmark(implementation, task_data, progress_bar=progress_bar)
            all_results.extend(results)
            log_benchmark_progress(logger, task_name, impl_name, "completed")
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
        
        return all_results
    
    def run_single_implementation(self, implementation_name: str, task_name: str = None) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark for a single implementation."""
        if implementation_name not in self.implementations:
            logger.error(f"Unknown implementation: {implementation_name}")
            return {}
        
        implementation = self.implementations[implementation_name]
        all_results = {}
        
        # Run all tasks or specific task
        tasks_to_run = [task_name] if task_name else self.tasks.keys()
        
        for task_name in tasks_to_run:
            if task_name not in self.tasks:
                continue
            
            if not implementation.supports_task(task_name):
                logger.warning(f"Implementation {implementation_name} does not support task {task_name}")
                continue
            
            log_benchmark_progress(logger, task_name, implementation_name, "started")
            results = self.run_single_task(task_name, implementation_name)
            all_results[task_name] = results
            log_benchmark_progress(logger, task_name, implementation_name, "completed")
        
        return all_results
    
    def _print_summary(self, results: Dict[str, List[BenchmarkResult]]):
        """Print a summary of benchmark results."""
        logger.info("="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)
        logger.info(f"Execution Date: {self.execution_date}")
        
        for task_name, task_results in results.items():
            if not task_results:
                continue
            
            logger.info(f"{task_name.upper()} RESULTS:")
            logger.info("-" * 40)
            
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
                
                logger.info(f"{impl_name}:")
                for metric_name, value in avg_metrics.items():
                    logger.info(f"  {metric_name}: {value:.3f}")
                for perf_name, value in avg_performance.items():
                    logger.info(f"  {perf_name}: {value:.3f}s")
        
        logger.info("="*80)
