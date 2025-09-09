#!/usr/bin/env python3
"""
Enhanced run file for the Argument Mining Benchmark with progress bars and debugging options.

This file provides an easy way to execute and debug the benchmark with all available flags.
It includes progress bars for overall progress and individual implementation progress.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# DEFAULT CONFIGURATION - Modify these variables to change default behavior
# =============================================================================

# Core benchmark settings
DEFAULT_MAX_SAMPLES = 100
DEFAULT_DISABLE_OPENAI = True
DEFAULT_SAVE_CSV = True

# Output and debugging
DEFAULT_VERBOSE = False
DEFAULT_DEBUG = False
DEFAULT_OUTPUT_DIR = 'results'

# Task enable/disable flags (True = enabled by default, False = disabled by default)
DEFAULT_ENABLE_ADU_EXTRACTION = True
DEFAULT_ENABLE_STANCE_CLASSIFICATION = True
DEFAULT_ENABLE_CLAIM_PREMISE_LINKING = True
# Implementation enable/disable flags (True = enabled by default, False = disabled by default)
DEFAULT_ENABLE_OPENAI = True
DEFAULT_ENABLE_TINYLLAMA = True
DEFAULT_ENABLE_MODERNBERT = True
DEFAULT_ENABLE_DEBERTA = True
DEFAULT_ENABLE_GPT41 = True
DEFAULT_ENABLE_GPT5 = True
DEFAULT_ENABLE_GPT5_MINI = True
DEFAULT_ENABLE_LLAMA3_3B = True
DEFAULT_ENABLE_QWEN2_5B = True

# Quick presets
DEFAULT_QUICK_MAX_SAMPLES = 10
DEFAULT_FULL_MAX_SAMPLES = 1000

# =============================================================================

# Add the benchmark package to the path
benchmark_path = Path(__file__).parent / "benchmark"
sys.path.insert(0, str(benchmark_path))

# Import progress bar libraries
try:
    from tqdm import tqdm
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich import box
    PROGRESS_AVAILABLE = True
except ImportError:
    print("Warning: Rich and tqdm not available. Install with: pip install rich tqdm")
    PROGRESS_AVAILABLE = False

from benchmark.core.benchmark import ArgumentMiningBenchmark
from benchmark.utils.logging_utils import setup_logging, get_logger, log_initialization, log_benchmark_progress


def get_default_task_filter() -> Optional[List[str]]:
    """Get default task filter based on enabled tasks."""
    tasks = []
    if DEFAULT_ENABLE_ADU_EXTRACTION:
        tasks.append('adu_extraction')
    if DEFAULT_ENABLE_STANCE_CLASSIFICATION:
        tasks.append('stance_classification')
    if DEFAULT_ENABLE_CLAIM_PREMISE_LINKING:
        tasks.append('claim_premise_linking')
    return tasks if tasks else None


def get_default_implementation_filter() -> Optional[List[str]]:
    """Get default implementation filter based on enabled implementations."""
    implementations = []
    if DEFAULT_ENABLE_OPENAI:
        implementations.append('openai')
    if DEFAULT_ENABLE_TINYLLAMA:
        implementations.append('tinyllama')
    if DEFAULT_ENABLE_MODERNBERT:
        implementations.append('modernbert')
    if DEFAULT_ENABLE_DEBERTA:
        implementations.append('deberta')
    if DEFAULT_ENABLE_GPT41:
        implementations.append('gpt-4.1')
    if DEFAULT_ENABLE_GPT5:
        implementations.append('gpt-5')
    if DEFAULT_ENABLE_GPT5_MINI:
        implementations.append('gpt-5-mini')
    if DEFAULT_ENABLE_LLAMA3_3B:
        implementations.append('llama3-3b')
    if DEFAULT_ENABLE_QWEN2_5B:
        implementations.append('qwen2.5-1.5b')
    return implementations if implementations else None


class BenchmarkRunner:
    """Enhanced benchmark runner with progress bars and debugging capabilities."""
    
    def __init__(self, 
                 max_samples: int = 100,
                 disable_openai: bool = True,
                 disable_tinyllama: bool = False,
                 disable_modernbert: bool = False,
                 disable_deberta: bool = False,
                 disable_gpt41: bool = False,
                 disable_gpt5: bool = False,
                 disable_gpt5_mini: bool = False,
                 disable_llama3_3b: bool = False,
                 disable_qwen2_5b: bool = False,
                 save_csv: bool = True,
                 verbose: bool = False,
                 debug: bool = False,
                 task_filter: Optional[List[str]] = None,
                 implementation_filter: Optional[List[str]] = None,
                 output_dir: str = "results"):
        """
        Initialize the benchmark runner.
        
        Args:
            max_samples: Maximum number of samples to use for benchmarking
            disable_openai: If True, skip OpenAI implementation
            disable_tinyllama: If True, skip TinyLlama implementation
            disable_modernbert: If True, skip ModernBERT implementation
            disable_deberta: If True, skip DeBERTa implementation
            disable_llama3_3b: If True, skip Llama 3.2 3B implementation
            disable_qwen2_5b: If True, skip Qwen 2.5 1.5B implementation
            save_csv: Whether to save results to CSV files
            verbose: Enable verbose output
            debug: Enable debug mode with detailed logging
            task_filter: List of specific tasks to run (None for all)
            implementation_filter: List of specific implementations to run (None for all)
            output_dir: Output directory for results
        """
        self.max_samples = max_samples
        self.disable_openai = disable_openai
        self.disable_tinyllama = disable_tinyllama
        self.disable_modernbert = disable_modernbert
        self.disable_deberta = disable_deberta
        self.disable_gpt41 = disable_gpt41
        self.disable_gpt5 = disable_gpt5
        self.disable_gpt5_mini = disable_gpt5_mini
        self.disable_llama3_3b = disable_llama3_3b
        self.disable_qwen2_5b = disable_qwen2_5b
        self.save_csv = save_csv
        self.verbose = verbose
        self.debug = debug
        self.task_filter = task_filter
        self.implementation_filter = implementation_filter
        self.output_dir = output_dir
        
        # Initialize logging
        log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
        self.logger = setup_logging(
            log_level=log_level,
            log_file=os.path.join(output_dir, "benchmark.log") if debug else None,
            progress_bar_compatible=True
        )
        
        # Initialize console for rich output
        self.console = Console() if PROGRESS_AVAILABLE else None
        
        # Benchmark instance
        self.benchmark = None
        
    def print_configuration(self):
        """Print the current configuration."""
        if self.console:
            config_table = Table(title="Benchmark Configuration", box=box.ROUNDED)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="green")
            
            config_table.add_row("Max Samples", str(self.max_samples))
            config_table.add_row("OpenAI Disabled", str(self.disable_openai))
            config_table.add_row("TinyLlama Disabled", str(self.disable_tinyllama))
            config_table.add_row("ModernBERT Disabled", str(self.disable_modernbert))
            config_table.add_row("DeBERTa Disabled", str(self.disable_deberta))
            config_table.add_row("Llama 3.2 3B Disabled", str(self.disable_llama3_3b))
            config_table.add_row("Qwen 2.5 1.5B Disabled", str(self.disable_qwen2_5b))
            config_table.add_row("Save CSV", str(self.save_csv))
            config_table.add_row("Verbose", str(self.verbose))
            config_table.add_row("Debug", str(self.debug))
            config_table.add_row("Task Filter", str(self.task_filter) if self.task_filter else "All tasks")
            config_table.add_row("Implementation Filter", str(self.implementation_filter) if self.implementation_filter else "All implementations")
            config_table.add_row("Output Directory", self.output_dir)
            
            self.console.print(config_table)
        else:
            print("Benchmark Configuration:")
            print(f"  Max Samples: {self.max_samples}")
            print(f"  OpenAI Disabled: {self.disable_openai}")
            print(f"  TinyLlama Disabled: {self.disable_tinyllama}")
            print(f"  ModernBERT Disabled: {self.disable_modernbert}")
            print(f"  DeBERTa Disabled: {self.disable_deberta}")
            print(f"  Llama 3.2 3B Disabled: {self.disable_llama3_3b}")
            print(f"  Qwen 2.5 1.5B Disabled: {self.disable_qwen2_5b}")
            print(f"  Save CSV: {self.save_csv}")
            print(f"  Verbose: {self.verbose}")
            print(f"  Debug: {self.debug}")
            print(f"  Task Filter: {self.task_filter if self.task_filter else 'All tasks'}")
            print(f"  Implementation Filter: {self.implementation_filter if self.implementation_filter else 'All implementations'}")
            print(f"  Output Directory: {self.output_dir}")
    
    def initialize_benchmark(self) -> bool:
        """Initialize the benchmark and return success status."""
        try:
            if self.verbose:
                if self.console:
                    self.console.print("\n[bold blue]Initializing benchmark...[/bold blue]")
                else:
                    print("Initializing benchmark...")
            
            self.benchmark = ArgumentMiningBenchmark(
                max_samples=self.max_samples,
                disable_openai=self.disable_openai,
                disable_tinyllama=self.disable_tinyllama,
                disable_modernbert=self.disable_modernbert,
                disable_deberta=self.disable_deberta,
                disable_llama3_3b=self.disable_llama3_3b,
                disable_qwen2_5b=self.disable_qwen2_5b
            )
            
            if self.verbose:
                if self.console:
                    self.console.print("[green]✓ Benchmark initialized successfully[/green]")
                    self.console.print(f"Available implementations: {list(self.benchmark.implementations.keys())}")
                    self.console.print(f"Available tasks: {list(self.benchmark.tasks.keys())}")
                else:
                    print("✓ Benchmark initialized successfully")
                    print(f"Available implementations: {list(self.benchmark.implementations.keys())}")
                    print(f"Available tasks: {list(self.benchmark.tasks.keys())}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize benchmark: {e}"
            if self.console:
                self.console.print(f"[red]✗ {error_msg}[/red]")
            else:
                print(f"✗ {error_msg}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            return False
    
    def run_full_benchmark_with_progress(self) -> Dict[str, Any]:
        """Run the complete benchmark with progress bars."""
        if not self.benchmark:
            return {}
        
        # Determine which tasks to run
        tasks_to_run = self.task_filter if self.task_filter else list(self.benchmark.tasks.keys())
        
        # Determine which implementations to run
        implementations_to_run = self.implementation_filter if self.implementation_filter else list(self.benchmark.implementations.keys())
        
        # Filter out disabled implementations
        if self.disable_openai:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'openai']
        if self.disable_tinyllama:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'tinyllama']
        if self.disable_modernbert:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'modernbert']
        if self.disable_deberta:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'deberta']
        if self.disable_gpt41:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'gpt-4.1']
        if self.disable_gpt5:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'gpt-5']
        if self.disable_gpt5_mini:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'gpt-5-mini']
        if self.disable_llama3_3b:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'llama3-3b']
        if self.disable_qwen2_5b:
            implementations_to_run = [impl for impl in implementations_to_run if impl != 'qwen2.5-1.5b']
        
        # Calculate total work
        total_combinations = len(tasks_to_run) * len(implementations_to_run)
        
        if self.console:
            # Create progress display
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TextColumn("[green]{task.completed}/{task.total}"),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                console=self.console,
                expand=True
            ) as progress:
                
                # Overall progress
                overall_task = progress.add_task("Overall Progress", total=total_combinations)
                
                all_results = {}
                
                for task_idx, task_name in enumerate(tasks_to_run):
                    if task_name not in self.benchmark.tasks:
                        continue
                    
                    # Task-specific progress
                    task_progress = progress.add_task(f"Task: {task_name}", total=len(implementations_to_run))
                    
                    task_results = []
                    
                    for impl_idx, impl_name in enumerate(implementations_to_run):
                        if impl_name not in self.benchmark.implementations:
                            continue
                        
                        implementation = self.benchmark.implementations[impl_name]
                        
                        if not implementation.supports_task(task_name):
                            if self.verbose:
                                self.console.print(f"[yellow]Skipping {impl_name} for {task_name} (not supported)[/yellow]")
                            progress.update(task_progress, advance=1)
                            progress.update(overall_task, advance=1)
                            continue
                        
                        # Get task data first
                        task_data = self.benchmark.data.get(task_name, [])
                        if not task_data:
                            if self.verbose:
                                self.console.print(f"[yellow]No data available for {task_name}[/yellow]")
                            progress.update(task_progress, advance=1)
                            progress.update(overall_task, advance=1)
                            continue
                        
                        # Limit data to max_samples
                        task_data = task_data[:self.benchmark.max_samples]
                        
                        # Create a custom progress bar wrapper that works with Rich
                        class RichProgressWrapper:
                            def __init__(self, progress_context, task_id, total):
                                self.progress_context = progress_context
                                self.task_id = task_id
                                self.total = total
                                self.current = 0
                            
                            def update(self, n=1):
                                self.current += n
                                self.progress_context.update(self.task_id, completed=self.current)
                            
                            def close(self):
                                pass  # Rich progress bars don't need to be closed
                        
                        impl_progress = progress.add_task(f"  {impl_name}", total=len(task_data))
                        progress_wrapper = RichProgressWrapper(progress, impl_progress, len(task_data))
                        
                        try:
                            if self.verbose:
                                self.console.print(f"[cyan]Running {task_name} with {impl_name}...[/cyan]")
                            
                            # Run the task
                            task_obj = self.benchmark.tasks[task_name]
                            results = task_obj.run_benchmark(implementation, task_data, progress_bar=progress_wrapper)
                            task_results.extend(results)
                            
                            if self.verbose:
                                self.console.print(f"[green]✓ Completed {task_name} with {impl_name}[/green]")
                            
                        except Exception as e:
                            error_msg = f"Failed to run {task_name} with {impl_name}: {e}"
                            if self.console:
                                self.console.print(f"[red]✗ {error_msg}[/red]")
                            else:
                                print(f"✗ {error_msg}")
                            
                            if self.debug:
                                import traceback
                                traceback.print_exc()
                        
                        progress.update(impl_progress, advance=1)
                        progress.update(task_progress, advance=1)
                        progress.update(overall_task, advance=1)
                    
                    all_results[task_name] = task_results
                    progress.remove_task(task_progress)
                
                return all_results
        
        else:
            # Fallback to simple progress without rich
            print(f"Running benchmark with {total_combinations} task-implementation combinations...")
            
            all_results = {}
            completed = 0
            
            for task_name in tasks_to_run:
                if task_name not in self.benchmark.tasks:
                    continue
                
                print(f"\nRunning task: {task_name}")
                task_results = []
                
                for impl_name in implementations_to_run:
                    if impl_name not in self.benchmark.implementations:
                        continue
                    
                    implementation = self.benchmark.implementations[impl_name]
                    
                    if not implementation.supports_task(task_name):
                        print(f"  Skipping {impl_name} (not supported)")
                        completed += 1
                        continue
                    
                    try:
                        print(f"  Running with {impl_name}...")
                        
                        # Get task data
                        task_data = self.benchmark.data.get(task_name, [])
                        if not task_data:
                            print(f"  No data available for {task_name}")
                            completed += 1
                            continue
                        
                        # Limit data to max_samples
                        task_data = task_data[:self.benchmark.max_samples]
                        
                        # Run the task
                        task_obj = self.benchmark.tasks[task_name]
                        results = task_obj.run_benchmark(implementation, task_data)
                        task_results.extend(results)
                        
                        print(f"  ✓ Completed with {impl_name}")
                        
                    except Exception as e:
                        print(f"  ✗ Failed with {impl_name}: {e}")
                        if self.debug:
                            import traceback
                            traceback.print_exc()
                    
                    completed += 1
                    print(f"  Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
                
                all_results[task_name] = task_results
            
            return all_results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save benchmark results."""
        if not results:
            return ""
        
        try:
            if self.verbose:
                if self.console:
                    self.console.print("\n[bold blue]Saving results...[/bold blue]")
                else:
                    print("Saving results...")
            
            output_path = self.benchmark.save_results(
                results, 
                comprehensive=True, 
                individual=True,
                output_dir=self.output_dir
            )
            
            if self.verbose:
                if self.console:
                    self.console.print(f"[green]✓ Results saved to: {output_path}[/green]")
                else:
                    print(f"✓ Results saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to save results: {e}"
            if self.console:
                self.console.print(f"[red]✗ {error_msg}[/red]")
            else:
                print(f"✗ {error_msg}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            return ""
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """Print detailed benchmark results with confusion matrix and metrics in neat tables."""
        if not results:
            return
        
        if self.console:
            for task_name, task_results in results.items():
                if not task_results:
                    continue
                
                # Group results by implementation
                impl_results = {}
                for result in task_results:
                    if result.success:
                        if result.implementation_name not in impl_results:
                            impl_results[result.implementation_name] = []
                        impl_results[result.implementation_name].append(result)
                
                if not impl_results:
                    continue
                
                # Create detailed results table for this task
                task_table = Table(title=f"{task_name.replace('_', ' ').title()} Results", box=box.ROUNDED)
                task_table.add_column("Implementation", style="cyan", width=15)
                task_table.add_column("Accuracy", style="green", justify="right")
                task_table.add_column("Precision", style="green", justify="right")
                task_table.add_column("Recall", style="green", justify="right")
                task_table.add_column("F1-Score", style="green", justify="right")
                task_table.add_column("TP", style="yellow", justify="right")
                task_table.add_column("FP", style="red", justify="right")
                task_table.add_column("FN", style="red", justify="right")
                task_table.add_column("Samples", style="blue", justify="right")
                
                for impl_name, impl_result_list in impl_results.items():
                    if not impl_result_list:
                        continue
                    
                    # Calculate average metrics
                    import numpy as np
                    avg_metrics = {}
                    for metric_name in impl_result_list[0].metrics.keys():
                        values = [r.metrics[metric_name] for r in impl_result_list if r.success]
                        avg_metrics[metric_name] = np.mean(values) if values else 0.0
                    
                    # Add row to table
                    task_table.add_row(
                        impl_name,
                        f"{avg_metrics.get('accuracy', 0.0):.3f}",
                        f"{avg_metrics.get('precision', 0.0):.3f}",
                        f"{avg_metrics.get('recall', 0.0):.3f}",
                        f"{avg_metrics.get('f1', 0.0):.3f}",
                        str(avg_metrics.get('tp', 0)),
                        str(avg_metrics.get('fp', 0)),
                        str(avg_metrics.get('fn', 0)),
                        str(len(impl_result_list))
                    )
                
                self.console.print(task_table)
                
                # Print confusion matrix for each implementation
                for impl_name, impl_result_list in impl_results.items():
                    if not impl_result_list:
                        continue
                    
                    # Create confusion matrix table with task-specific labels
                    cm_table = Table(title=f"{impl_name} - Confusion Matrix", box=box.ROUNDED)
                    
                    # Get task-specific labels
                    if task_name == 'stance_classification':
                        label_0, label_1 = "Refute", "Support"
                    elif task_name == 'adu_extraction':
                        label_0, label_1 = "Not ADU", "ADU"
                    elif task_name == 'claim_premise_linking':
                        label_0, label_1 = "No Link", "Linked"
                    else:
                        label_0, label_1 = "Class 0", "Class 1"
                    
                    cm_table.add_column("", style="cyan", width=8)
                    cm_table.add_column(f"Predicted: {label_0}", style="yellow", justify="right")
                    cm_table.add_column(f"Predicted: {label_1}", style="yellow", justify="right")
                    
                    # Calculate confusion matrix values
                    tp = sum(r.metrics.get('tp', 0) for r in impl_result_list)
                    fp = sum(r.metrics.get('fp', 0) for r in impl_result_list)
                    fn = sum(r.metrics.get('fn', 0) for r in impl_result_list)
                    tn = sum(r.metrics.get('tn', 0) for r in impl_result_list)
                    
                    cm_table.add_row(f"Actual: {label_0}", str(tn), str(fp))
                    cm_table.add_row(f"Actual: {label_1}", str(fn), str(tp))
                    
                    self.console.print(cm_table)
            
            # Print aggregated confusion matrix across all tasks
            self.print_aggregated_confusion_matrix(results)
        else:
            # Fallback for when Rich is not available
            print("\nDetailed Benchmark Results:")
            print("=" * 80)
            
            for task_name, task_results in results.items():
                if not task_results:
                    continue
                
                print(f"\n{task_name.replace('_', ' ').title()} Results:")
                print("-" * 50)
                
                # Group results by implementation
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
                    import numpy as np
                    avg_metrics = {}
                    for metric_name in impl_result_list[0].metrics.keys():
                        values = [r.metrics[metric_name] for r in impl_result_list if r.success]
                        avg_metrics[metric_name] = np.mean(values) if values else 0.0
                    
                    print(f"\n{impl_name}:")
                    print(f"  Accuracy:  {avg_metrics.get('accuracy', 0.0):.3f}")
                    print(f"  Precision: {avg_metrics.get('precision', 0.0):.3f}")
                    print(f"  Recall:    {avg_metrics.get('recall', 0.0):.3f}")
                    print(f"  F1-Score:  {avg_metrics.get('f1', 0.0):.3f}")
                    print(f"  Samples:   {len(impl_result_list)}")
                    
                    # Confusion matrix
                    tp = sum(r.metrics.get('tp', 0) for r in impl_result_list)
                    fp = sum(r.metrics.get('fp', 0) for r in impl_result_list)
                    fn = sum(r.metrics.get('fn', 0) for r in impl_result_list)
                    tn = sum(r.metrics.get('tn', 0) for r in impl_result_list)
                    
                    # Get task-specific labels
                    if task_name == 'stance_classification':
                        label_0, label_1 = "Refute", "Support"
                    elif task_name == 'adu_extraction':
                        label_0, label_1 = "Not ADU", "ADU"
                    elif task_name == 'claim_premise_linking':
                        label_0, label_1 = "No Link", "Linked"
                    else:
                        label_0, label_1 = "Class 0", "Class 1"
                    
                    print(f"  Confusion Matrix:")
                    print(f"    Actual: {label_0} -> Predicted: {label_0} = {tn}, Predicted: {label_1} = {fp}")
                    print(f"    Actual: {label_1} -> Predicted: {label_0} = {fn}, Predicted: {label_1} = {tp}")
            
            # Print aggregated confusion matrix across all tasks
            self.print_aggregated_confusion_matrix_fallback(results)

    def print_aggregated_confusion_matrix(self, results: Dict[str, Any]):
        """Print aggregated confusion matrix across all tasks and implementations."""
        if not results:
            return
        
        # Collect all results by implementation
        impl_results = {}
        for task_name, task_results in results.items():
            for result in task_results:
                if result.success:
                    if result.implementation_name not in impl_results:
                        impl_results[result.implementation_name] = []
                    impl_results[result.implementation_name].append(result)
        
        if not impl_results:
            return
        
        # Create aggregated confusion matrix table
        agg_table = Table(title="Aggregated Confusion Matrix Across All Tasks", box=box.ROUNDED)
        agg_table.add_column("Implementation", style="cyan", width=15)
        agg_table.add_column("Predicted: Negative", style="yellow", justify="right")
        agg_table.add_column("Predicted: Positive", style="yellow", justify="right")
        agg_table.add_column("Total Samples", style="blue", justify="right")
        
        for impl_name, impl_result_list in impl_results.items():
            if not impl_result_list:
                continue
            
            # Calculate aggregated confusion matrix values
            total_tp = sum(r.metrics.get('tp', 0) for r in impl_result_list)
            total_fp = sum(r.metrics.get('fp', 0) for r in impl_result_list)
            total_fn = sum(r.metrics.get('fn', 0) for r in impl_result_list)
            total_tn = sum(r.metrics.get('tn', 0) for r in impl_result_list)
            total_samples = len(impl_result_list)
            
            # Add row to aggregated table
            agg_table.add_row(
                impl_name,
                str(total_tn + total_fn),  # Predicted: Negative (TN + FN)
                str(total_tp + total_fp),  # Predicted: Positive (TP + FP)
                str(total_samples)
            )
        
        self.console.print(agg_table)
        
        # Create detailed aggregated confusion matrix
        detailed_table = Table(title="Detailed Aggregated Confusion Matrix", box=box.ROUNDED)
        detailed_table.add_column("Implementation", style="cyan", width=15)
        detailed_table.add_column("", style="cyan", width=8)
        detailed_table.add_column("Predicted: Negative", style="yellow", justify="right")
        detailed_table.add_column("Predicted: Positive", style="yellow", justify="right")
        
        for impl_name, impl_result_list in impl_results.items():
            if not impl_result_list:
                continue
            
            # Calculate aggregated confusion matrix values
            total_tp = sum(r.metrics.get('tp', 0) for r in impl_result_list)
            total_fp = sum(r.metrics.get('fp', 0) for r in impl_result_list)
            total_fn = sum(r.metrics.get('fn', 0) for r in impl_result_list)
            total_tn = sum(r.metrics.get('tn', 0) for r in impl_result_list)
            
            # Add rows for this implementation
            detailed_table.add_row(
                impl_name,
                "Actual: Negative",
                str(total_tn),
                str(total_fp)
            )
            detailed_table.add_row(
                "",
                "Actual: Positive",
                str(total_fn),
                str(total_tp)
            )
        
        self.console.print(detailed_table)

    def print_aggregated_confusion_matrix_fallback(self, results: Dict[str, Any]):
        """Print aggregated confusion matrix fallback for when Rich is not available."""
        if not results:
            return
        
        # Collect all results by implementation
        impl_results = {}
        for task_name, task_results in results.items():
            for result in task_results:
                if result.success:
                    if result.implementation_name not in impl_results:
                        impl_results[result.implementation_name] = []
                    impl_results[result.implementation_name].append(result)
        
        if not impl_results:
            return
        
        print("\nAggregated Confusion Matrix Across All Tasks:")
        print("=" * 60)
        
        for impl_name, impl_result_list in impl_results.items():
            if not impl_result_list:
                continue
            
            # Calculate aggregated confusion matrix values
            total_tp = sum(r.metrics.get('tp', 0) for r in impl_result_list)
            total_fp = sum(r.metrics.get('fp', 0) for r in impl_result_list)
            total_fn = sum(r.metrics.get('fn', 0) for r in impl_result_list)
            total_tn = sum(r.metrics.get('tn', 0) for r in impl_result_list)
            total_samples = len(impl_result_list)
            
            print(f"\n{impl_name}:")
            print(f"  Total Samples: {total_samples}")
            print(f"  Confusion Matrix:")
            print(f"    Actual: Negative -> Predicted: Negative = {total_tn}, Predicted: Positive = {total_fp}")
            print(f"    Actual: Positive -> Predicted: Negative = {total_fn}, Predicted: Positive = {total_tp}")
            
            # Calculate aggregated metrics
            if total_tp + total_fp > 0:
                precision = total_tp / (total_tp + total_fp)
            else:
                precision = 0.0
                
            if total_tp + total_fn > 0:
                recall = total_tp / (total_tp + total_fn)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0.0
            
            print(f"  Aggregated Metrics:")
            print(f"    Accuracy:  {accuracy:.3f}")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall:    {recall:.3f}")
            print(f"    F1-Score:  {f1:.3f}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the benchmark results."""
        if not results:
            return
        
        if self.console:
            # Create summary table
            summary_table = Table(title="Benchmark Summary", box=box.ROUNDED)
            summary_table.add_column("Task", style="cyan")
            summary_table.add_column("Implementations", style="green")
            summary_table.add_column("Total Results", style="yellow")
            summary_table.add_column("Successful", style="green")
            summary_table.add_column("Failed", style="red")
            
            for task_name, task_results in results.items():
                if not task_results:
                    continue
                
                successful = sum(1 for r in task_results if r.success)
                failed = len(task_results) - successful
                implementations = len(set(r.implementation_name for r in task_results))
                
                summary_table.add_row(
                    task_name,
                    str(implementations),
                    str(len(task_results)),
                    str(successful),
                    str(failed)
                )
            
            self.console.print(summary_table)
        else:
            print("\nBenchmark Summary:")
            print("-" * 50)
            for task_name, task_results in results.items():
                if not task_results:
                    continue
                
                successful = sum(1 for r in task_results if r.success)
                failed = len(task_results) - successful
                implementations = len(set(r.implementation_name for r in task_results))
                
                print(f"{task_name}:")
                print(f"  Implementations: {implementations}")
                print(f"  Total Results: {len(task_results)}")
                print(f"  Successful: {successful}")
                print(f"  Failed: {failed}")
    
    def run(self) -> bool:
        """Run the complete benchmark process."""
        try:
            # Print configuration
            self.print_configuration()
            
            # Initialize benchmark
            if not self.initialize_benchmark():
                return False
            
            # Run benchmark with progress
            if self.console:
                self.console.print("\n[bold green]Starting benchmark execution...[/bold green]")
            else:
                print("\nStarting benchmark execution...")
            
            start_time = time.time()
            results = self.run_full_benchmark_with_progress()
            end_time = time.time()
            
            if not results:
                if self.console:
                    self.console.print("[red]No results generated[/red]")
                else:
                    print("No results generated")
                return False
            
            # Save results
            output_path = self.save_results(results)
            
            # Print detailed results with confusion matrix and metrics
            self.print_detailed_results(results)
            
            # Print summary
            self.print_summary(results)
            
            # Print execution time
            execution_time = end_time - start_time
            if self.console:
                self.console.print(f"\n[bold green]Benchmark completed in {execution_time:.2f} seconds[/bold green]")
                if output_path:
                    self.console.print(f"[bold blue]Results saved to: {output_path}[/bold blue]")
            else:
                print(f"\nBenchmark completed in {execution_time:.2f} seconds")
                if output_path:
                    print(f"Results saved to: {output_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Benchmark execution failed: {e}"
            if self.console:
                self.console.print(f"[red]✗ {error_msg}[/red]")
            else:
                print(f"✗ {error_msg}")
            
            if self.debug:
                import traceback
                traceback.print_exc()
            
            return False


def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description='Enhanced Argument Mining Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                    # Run with default settings (OpenAI disabled)
  python run.py --max-samples 50                   # Limit to 50 samples
  python run.py --enable-openai                    # Enable OpenAI implementations
  python run.py --task-filter adu_extraction       # Run only ADU extraction
  python run.py --impl-filter tinyllama modernbert # Run only specific implementations
  python run.py --verbose --debug                  # Enable verbose and debug output
  python run.py --no-save-csv                      # Skip CSV output
        """
    )
    
    # Core benchmark settings
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES,
                       help=f'Maximum number of samples to use for benchmarking (default: {DEFAULT_MAX_SAMPLES})')
    parser.add_argument('--disable-openai', action='store_true', default=DEFAULT_DISABLE_OPENAI,
                       help=f'Disable OpenAI implementations (default: {DEFAULT_DISABLE_OPENAI})')
    parser.add_argument('--disable-tinyllama', action='store_true', default=not DEFAULT_ENABLE_TINYLLAMA,
                       help=f'Disable TinyLlama implementations (default: {not DEFAULT_ENABLE_TINYLLAMA})')
    parser.add_argument('--disable-modernbert', action='store_true', default=not DEFAULT_ENABLE_MODERNBERT,
                       help=f'Disable ModernBERT implementations (default: {not DEFAULT_ENABLE_MODERNBERT})')
    parser.add_argument('--disable-deberta', action='store_true', default=not DEFAULT_ENABLE_DEBERTA,
                       help=f'Disable DeBERTa implementations (default: {not DEFAULT_ENABLE_DEBERTA})')
    parser.add_argument('--disable-gpt41', action='store_true', default=not DEFAULT_ENABLE_GPT41,
                       help=f'Disable GPT-4.1 implementations (default: {not DEFAULT_ENABLE_GPT41})')
    parser.add_argument('--disable-gpt5', action='store_true', default=not DEFAULT_ENABLE_GPT5,
                       help=f'Disable GPT-5 implementations (default: {not DEFAULT_ENABLE_GPT5})')
    parser.add_argument('--disable-gpt5-mini', action='store_true', default=not DEFAULT_ENABLE_GPT5_MINI,
                       help=f'Disable GPT-5 Mini implementations (default: {not DEFAULT_ENABLE_GPT5_MINI})')
    parser.add_argument('--disable-llama3-3b', action='store_true', default=not DEFAULT_ENABLE_LLAMA3_3B,
                       help=f'Disable Llama 3.2 3B implementations (default: {not DEFAULT_ENABLE_LLAMA3_3B})')
    parser.add_argument('--disable-qwen2-5b', action='store_true', default=not DEFAULT_ENABLE_QWEN2_5B,
                       help=f'Disable Qwen 2.5 1.5B implementations (default: {not DEFAULT_ENABLE_QWEN2_5B})')
    parser.add_argument('--enable-openai', action='store_true',
                       help='Enable OpenAI implementations (overrides --disable-openai)')
    parser.add_argument('--no-save-csv', action='store_true',
                       help='Disable CSV output (results still saved to JSON)')
    
    # Output and debugging
    parser.add_argument('--verbose', '-v', action='store_true', default=DEFAULT_VERBOSE,
                       help=f'Enable verbose output (default: {DEFAULT_VERBOSE})')
    parser.add_argument('--debug', '-d', action='store_true', default=DEFAULT_DEBUG,
                       help=f'Enable debug mode with detailed error information (default: {DEFAULT_DEBUG})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
    
    # Filtering options
    parser.add_argument('--task-filter', nargs='+', 
                       choices=['adu_extraction', 'stance_classification', 'claim_premise_linking'],
                       help='Run only specific tasks')
    parser.add_argument('--impl-filter', nargs='+',
                       choices=['openai', 'tinyllama', 'modernbert', 'deberta'],
                       help='Run only specific implementations')
    
    # Quick presets
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run (10 samples, no OpenAI)')
    parser.add_argument('--full', action='store_true',
                       help='Full benchmark run (all samples, all implementations)')
    
    args = parser.parse_args()
    
    # Handle OpenAI enable/disable logic
    if args.enable_openai:
        args.disable_openai = False
    
    # Apply presets
    if args.quick:
        args.max_samples = DEFAULT_QUICK_MAX_SAMPLES
        args.disable_openai = True
        args.verbose = True
        print(f"Quick test mode enabled: {DEFAULT_QUICK_MAX_SAMPLES} samples, no OpenAI, verbose output")
    
    if args.full:
        args.max_samples = DEFAULT_FULL_MAX_SAMPLES
        args.disable_openai = False
        print(f"Full benchmark mode enabled: {DEFAULT_FULL_MAX_SAMPLES} samples, all implementations")
    
    # Use default filters if none specified
    task_filter = args.task_filter if args.task_filter else get_default_task_filter()
    implementation_filter = args.impl_filter if args.impl_filter else get_default_implementation_filter()
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        max_samples=args.max_samples,
        disable_openai=args.disable_openai,
        disable_tinyllama=args.disable_tinyllama,
        disable_modernbert=args.disable_modernbert,
        disable_deberta=args.disable_deberta,
        disable_gpt41=args.disable_gpt41,
        disable_gpt5=args.disable_gpt5,
        disable_gpt5_mini=args.disable_gpt5_mini,
        disable_llama3_3b=args.disable_llama3_3b,
        disable_qwen2_5b=args.disable_qwen2_5b,
        save_csv=not args.no_save_csv,
        verbose=args.verbose,
        debug=args.debug,
        task_filter=task_filter,
        implementation_filter=implementation_filter,
        output_dir=args.output_dir
    )
    
    success = runner.run()
    
    if success:
        print("\n✓ Benchmark completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Benchmark failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()