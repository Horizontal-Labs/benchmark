"""
Utility module for saving and appending benchmark results to JSON files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from benchmark.core.results import BenchmarkResult
from .logging_utils import get_logger


def save_results_to_json(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Save individual benchmark results to JSON files.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, task_results in results.items():
        if not task_results:
            continue
            
        # Convert BenchmarkResult objects to dictionaries
        results_data = []
        for result in task_results:
            result_dict = {
                'timestamp': result.execution_date,
                'implementation': result.implementation_name,
                'task': result.task_name,
                'sample_id': result.sample_id,
                'predictions': str(result.predictions),  # Convert to string for JSON serialization
                'ground_truth': str(result.ground_truth),  # Convert to string for JSON serialization
                'metrics': result.metrics,
                'performance': result.performance
            }
            results_data.append(result_dict)
        
        # Save to JSON file
        filename = f"benchmark_results_{task_name}_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger = get_logger()
        logger.info(f"Saved {len(results_data)} results to {filepath}")


def append_to_json(filepath: Path, new_data: List[Dict], mode: str = 'append') -> None:
    """
    Append new data to an existing JSON file or create a new one.
    
    Args:
        filepath: Path to the JSON file
        new_data: List of dictionaries to append
        mode: 'append' to add to existing file, 'overwrite' to replace
    """
    filepath = Path(filepath)
    
    if mode == 'append' and filepath.exists():
        # Read existing data
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []
        
        # Append new data
        if isinstance(existing_data, list):
            existing_data.extend(new_data)
        else:
            existing_data = [existing_data] + new_data
    else:
        existing_data = new_data
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


def append_to_implementations_json(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Aggregate and append implementation metrics to implementations.json.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    implementations_file = metrics_dir / "implementations.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate metrics by implementation
    implementation_metrics = {}
    
    for task_name, task_results in results.items():
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in implementation_metrics:
                implementation_metrics[impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0.0,
                    'tn_sum': 0.0,
                    'fp_sum': 0.0,
                    'fn_sum': 0.0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            implementation_metrics[impl_name]['total_samples'] += 1
            implementation_metrics[impl_name]['successful_samples'] += 1
            implementation_metrics[impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            implementation_metrics[impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            implementation_metrics[impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            implementation_metrics[impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            implementation_metrics[impl_name]['tp_sum'] += metrics.get('tp', 0.0)
            implementation_metrics[impl_name]['tn_sum'] += metrics.get('tn', 0.0)
            implementation_metrics[impl_name]['fp_sum'] += metrics.get('fp', 0.0)
            implementation_metrics[impl_name]['fn_sum'] += metrics.get('fn', 0.0)
            # Get inference time from performance metrics if available
            inference_time = result.performance.get('inference_time', 0.0)
            implementation_metrics[impl_name]['inference_time_sum'] += inference_time
    
    # Calculate averages and prepare data for JSON
    append_data = []
    for impl_name, metrics in implementation_metrics.items():
        total_samples = metrics['total_samples']
        if total_samples > 0:
            avg_data = {
                'execution_timestamp': timestamp,
                'implementation': impl_name,
                'metrics': {
                    'samples': {
                        'total': total_samples,
                        'successful': metrics['successful_samples'],
                        'success_rate': metrics['successful_samples'] / total_samples
                    },
                    'performance': {
                        'avg_accuracy': metrics['accuracy_sum'] / total_samples,
                        'avg_precision': metrics['precision_sum'] / total_samples,
                        'avg_recall': metrics['recall_sum'] / total_samples,
                        'avg_f1': metrics['f1_sum'] / total_samples
                    },
                    'confusion_matrix': {
                        'avg_tp': metrics['tp_sum'] / total_samples,
                        'avg_tn': metrics['tn_sum'] / total_samples,
                        'avg_fp': metrics['fp_sum'] / total_samples,
                        'avg_fn': metrics['fn_sum'] / total_samples
                    },
                    'timing': {
                        'avg_inference_time': metrics['inference_time_sum'] / total_samples
                    }
                }
            }
            append_data.append(avg_data)
    
    # Append to JSON file
    append_to_json(implementations_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended implementation metrics to {implementations_file}")


def append_to_tasks_json(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Aggregate and append task metrics to tasks.json.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_file = metrics_dir / "tasks.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate metrics by task and implementation
    task_metrics = {}
    
    for task_name, task_results in results.items():
        if task_name not in task_metrics:
            task_metrics[task_name] = {}
        
        # Aggregate by implementation
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in task_metrics[task_name]:
                task_metrics[task_name][impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0.0,
                    'tn_sum': 0.0,
                    'fp_sum': 0.0,
                    'fn_sum': 0.0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            task_metrics[task_name][impl_name]['total_samples'] += 1
            task_metrics[task_name][impl_name]['successful_samples'] += 1
            task_metrics[task_name][impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            task_metrics[task_name][impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            task_metrics[task_name][impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            task_metrics[task_name][impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            task_metrics[task_name][impl_name]['tp_sum'] += metrics.get('tp', 0.0)
            task_metrics[task_name][impl_name]['tn_sum'] += metrics.get('tn', 0.0)
            task_metrics[task_name][impl_name]['fp_sum'] += metrics.get('fp', 0.0)
            task_metrics[task_name][impl_name]['fn_sum'] += metrics.get('fn', 0.0)
            # Get inference time from performance metrics if available
            inference_time = result.performance.get('inference_time', 0.0)
            task_metrics[task_name][impl_name]['inference_time_sum'] += inference_time
    
    # Calculate averages and prepare data for JSON
    append_data = []
    
    # Add aggregated metrics across all implementations for each task
    for task_name, implementations in task_metrics.items():
        all_impl_metrics = {
            'total_samples': 0,
            'successful_samples': 0,
            'accuracy_sum': 0.0,
            'precision_sum': 0.0,
            'recall_sum': 0.0,
            'f1_sum': 0.0,
            'tp_sum': 0.0,
            'tn_sum': 0.0,
            'fp_sum': 0.0,
            'fn_sum': 0.0,
            'inference_time_sum': 0.0
        }
        
        # Aggregate across implementations
        for impl_metrics in implementations.values():
            all_impl_metrics['total_samples'] += impl_metrics['total_samples']
            all_impl_metrics['successful_samples'] += impl_metrics['successful_samples']
            all_impl_metrics['accuracy_sum'] += impl_metrics['accuracy_sum']
            all_impl_metrics['precision_sum'] += impl_metrics['precision_sum']
            all_impl_metrics['recall_sum'] += impl_metrics['recall_sum']
            all_impl_metrics['f1_sum'] += impl_metrics['f1_sum']
            all_impl_metrics['tp_sum'] += impl_metrics['tp_sum']
            all_impl_metrics['tn_sum'] += impl_metrics['tn_sum']
            all_impl_metrics['fp_sum'] += impl_metrics['fp_sum']
            all_impl_metrics['fn_sum'] += impl_metrics['fn_sum']
            all_impl_metrics['inference_time_sum'] += impl_metrics['inference_time_sum']
        
        # Calculate overall averages for the task
        if all_impl_metrics['total_samples'] > 0:
            overall_avg_data = {
                'execution_timestamp': timestamp,
                'task': task_name,
                'implementation': 'ALL',
                'metrics': {
                    'samples': {
                        'total': all_impl_metrics['total_samples'],
                        'successful': all_impl_metrics['successful_samples'],
                        'success_rate': all_impl_metrics['successful_samples'] / all_impl_metrics['total_samples']
                    },
                    'performance': {
                        'avg_accuracy': all_impl_metrics['accuracy_sum'] / all_impl_metrics['total_samples'],
                        'avg_precision': all_impl_metrics['precision_sum'] / all_impl_metrics['total_samples'],
                        'avg_recall': all_impl_metrics['recall_sum'] / all_impl_metrics['total_samples'],
                        'avg_f1': all_impl_metrics['f1_sum'] / all_impl_metrics['total_samples']
                    },
                    'confusion_matrix': {
                        'avg_tp': all_impl_metrics['tp_sum'] / all_impl_metrics['total_samples'],
                        'avg_tn': all_impl_metrics['tn_sum'] / all_impl_metrics['total_samples'],
                        'avg_fp': all_impl_metrics['fp_sum'] / all_impl_metrics['total_samples'],
                        'avg_fn': all_impl_metrics['fn_sum'] / all_impl_metrics['total_samples']
                    },
                    'timing': {
                        'avg_inference_time': all_impl_metrics['inference_time_sum'] / all_impl_metrics['total_samples']
                    }
                }
            }
            append_data.append(overall_avg_data)
        
        # Add individual implementation metrics for the task
        for impl_name, impl_metrics in implementations.items():
            total_samples = impl_metrics['total_samples']
            if total_samples > 0:
                impl_avg_data = {
                    'execution_timestamp': timestamp,
                    'task': task_name,
                    'implementation': impl_name,
                    'metrics': {
                        'samples': {
                            'total': total_samples,
                            'successful': impl_metrics['successful_samples'],
                            'success_rate': impl_metrics['successful_samples'] / total_samples
                        },
                        'performance': {
                            'avg_accuracy': impl_metrics['accuracy_sum'] / total_samples,
                            'avg_precision': impl_metrics['precision_sum'] / total_samples,
                            'avg_recall': impl_metrics['recall_sum'] / total_samples,
                            'avg_f1': impl_metrics['f1_sum'] / total_samples
                        },
                        'confusion_matrix': {
                            'avg_tp': impl_metrics['tp_sum'] / total_samples,
                            'avg_tn': impl_metrics['tn_sum'] / total_samples,
                            'avg_fp': impl_metrics['fp_sum'] / total_samples,
                            'avg_fn': impl_metrics['fn_sum'] / total_samples
                        },
                        'timing': {
                            'avg_inference_time': impl_metrics['inference_time_sum'] / total_samples
                        }
                    }
                }
                append_data.append(impl_avg_data)
    
    # Append to JSON file
    append_to_json(tasks_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended task metrics to {tasks_file}")


def append_to_system_json(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Aggregate and append combined system metrics to system.json.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    system_file = metrics_dir / "system.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate metrics by implementation, task, and overall
    system_metrics = {
        'implementations': {},
        'tasks': {},
        'overall': {
            'total_samples': 0,
            'successful_samples': 0,
            'accuracy_sum': 0.0,
            'precision_sum': 0.0,
            'recall_sum': 0.0,
            'f1_sum': 0.0,
            'tp_sum': 0.0,
            'tn_sum': 0.0,
            'fp_sum': 0.0,
            'fn_sum': 0.0,
            'inference_time_sum': 0.0
        }
    }
    
    # Process all results
    for task_name, task_results in results.items():
        for result in task_results:
            impl_name = result.implementation_name
            metrics = result.metrics
            
            # Initialize implementation metrics if not exists
            if impl_name not in system_metrics['implementations']:
                system_metrics['implementations'][impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0.0,
                    'tn_sum': 0.0,
                    'fp_sum': 0.0,
                    'fn_sum': 0.0,
                    'inference_time_sum': 0.0
                }
            
            # Initialize task metrics if not exists
            if task_name not in system_metrics['tasks']:
                system_metrics['tasks'][task_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0.0,
                    'tn_sum': 0.0,
                    'fp_sum': 0.0,
                    'fn_sum': 0.0,
                    'inference_time_sum': 0.0
                }
            
            # Update implementation metrics
            system_metrics['implementations'][impl_name]['total_samples'] += 1
            system_metrics['implementations'][impl_name]['successful_samples'] += 1
            system_metrics['implementations'][impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            system_metrics['implementations'][impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            system_metrics['implementations'][impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            system_metrics['implementations'][impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            system_metrics['implementations'][impl_name]['tp_sum'] += metrics.get('tp', 0.0)
            system_metrics['implementations'][impl_name]['tn_sum'] += metrics.get('tn', 0.0)
            system_metrics['implementations'][impl_name]['fp_sum'] += metrics.get('fp', 0.0)
            system_metrics['implementations'][impl_name]['fn_sum'] += metrics.get('fn', 0.0)
            # Get inference time from performance metrics if available
            inference_time = result.performance.get('inference_time', 0.0)
            system_metrics['implementations'][impl_name]['inference_time_sum'] += inference_time
            
            # Update task metrics
            system_metrics['tasks'][task_name]['total_samples'] += 1
            system_metrics['tasks'][task_name]['successful_samples'] += 1
            system_metrics['tasks'][task_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            system_metrics['tasks'][task_name]['precision_sum'] += metrics.get('precision', 0.0)
            system_metrics['tasks'][task_name]['recall_sum'] += metrics.get('recall', 0.0)
            system_metrics['tasks'][task_name]['f1_sum'] += metrics.get('f1', 0.0)
            system_metrics['tasks'][task_name]['tp_sum'] += metrics.get('tp', 0.0)
            system_metrics['tasks'][task_name]['tn_sum'] += metrics.get('tn', 0.0)
            system_metrics['tasks'][task_name]['fp_sum'] += metrics.get('fp', 0.0)
            system_metrics['tasks'][task_name]['fn_sum'] += metrics.get('fn', 0.0)
            # Get inference time from performance metrics if available
            inference_time = result.performance.get('inference_time', 0.0)
            system_metrics['tasks'][task_name]['inference_time_sum'] += inference_time
            
            # Update overall metrics
            system_metrics['overall']['total_samples'] += 1
            system_metrics['overall']['successful_samples'] += 1
            system_metrics['overall']['accuracy_sum'] += metrics.get('accuracy', 0.0)
            system_metrics['overall']['precision_sum'] += metrics.get('precision', 0.0)
            system_metrics['overall']['recall_sum'] += metrics.get('recall', 0.0)
            system_metrics['overall']['f1_sum'] += metrics.get('f1', 0.0)
            system_metrics['overall']['tp_sum'] += metrics.get('tp', 0.0)
            system_metrics['overall']['tn_sum'] += metrics.get('tn', 0.0)
            system_metrics['overall']['fp_sum'] += metrics.get('fp', 0.0)
            system_metrics['overall']['fn_sum'] += metrics.get('fn', 0.0)
            # Get inference time from performance metrics if available
            inference_time = result.performance.get('inference_time', 0.0)
            system_metrics['overall']['inference_time_sum'] += inference_time
    
    # Calculate averages and prepare data for JSON
    append_data = []
    
    # Add implementation-level metrics
    for impl_name, impl_metrics in system_metrics['implementations'].items():
        total_samples = impl_metrics['total_samples']
        if total_samples > 0:
            impl_data = {
                'execution_timestamp': timestamp,
                'metric_type': 'implementation',
                'implementation': impl_name,
                'task': 'ALL',
                'metrics': {
                    'samples': {
                        'total': total_samples,
                        'successful': impl_metrics['successful_samples'],
                        'success_rate': impl_metrics['successful_samples'] / total_samples
                    },
                    'performance': {
                        'avg_accuracy': impl_metrics['accuracy_sum'] / total_samples,
                        'avg_precision': impl_metrics['precision_sum'] / total_samples,
                        'avg_recall': impl_metrics['recall_sum'] / total_samples,
                        'avg_f1': impl_metrics['f1_sum'] / total_samples
                    },
                    'confusion_matrix': {
                        'avg_tp': impl_metrics['tp_sum'] / total_samples,
                        'avg_tn': impl_metrics['tn_sum'] / total_samples,
                        'avg_fp': impl_metrics['fp_sum'] / total_samples,
                        'avg_fn': impl_metrics['fn_sum'] / total_samples
                    },
                    'timing': {
                        'avg_inference_time': impl_metrics['inference_time_sum'] / total_samples
                    }
                }
            }
            append_data.append(impl_data)
    
    # Add task-level metrics
    for task_name, task_metrics in system_metrics['tasks'].items():
        total_samples = task_metrics['total_samples']
        if total_samples > 0:
            task_data = {
                'execution_timestamp': timestamp,
                'metric_type': 'task',
                'implementation': 'ALL',
                'task': task_name,
                'metrics': {
                    'samples': {
                        'total': total_samples,
                        'successful': task_metrics['successful_samples'],
                        'success_rate': task_metrics['successful_samples'] / total_samples
                    },
                    'performance': {
                        'avg_accuracy': task_metrics['accuracy_sum'] / total_samples,
                        'avg_precision': task_metrics['precision_sum'] / total_samples,
                        'avg_recall': task_metrics['recall_sum'] / total_samples,
                        'avg_f1': task_metrics['f1_sum'] / total_samples
                    },
                    'confusion_matrix': {
                        'avg_tp': task_metrics['tp_sum'] / total_samples,
                        'avg_tn': task_metrics['tn_sum'] / total_samples,
                        'avg_fp': task_metrics['fp_sum'] / total_samples,
                        'avg_fn': task_metrics['fn_sum'] / total_samples
                    },
                    'timing': {
                        'avg_inference_time': task_metrics['inference_time_sum'] / total_samples
                    }
                }
            }
            append_data.append(task_data)
    
    # Add task-implementation combination metrics
    for task_name, task_results in results.items():
        for result in task_results:
            impl_name = result.implementation_name
            metrics = result.metrics
            
            task_impl_data = {
                'execution_timestamp': timestamp,
                'metric_type': 'task_implementation',
                'implementation': impl_name,
                'task': task_name,
                'metrics': {
                    'samples': {
                        'total': 1,
                        'successful': 1,
                        'success_rate': 1.0
                    },
                    'performance': {
                        'avg_accuracy': metrics.get('accuracy', 0.0),
                        'avg_precision': metrics.get('precision', 0.0),
                        'avg_recall': metrics.get('recall', 0.0),
                        'avg_f1': metrics.get('f1', 0.0)
                    },
                    'confusion_matrix': {
                        'avg_tp': metrics.get('tp', 0.0),
                        'avg_tn': metrics.get('tn', 0.0),
                        'avg_fp': metrics.get('fp', 0.0),
                        'avg_fn': metrics.get('fn', 0.0)
                    },
                    'timing': {
                        'avg_inference_time': result.performance.get('inference_time', 0.0)
                    }
                }
            }
            append_data.append(task_impl_data)
    
    # Append to JSON file
    append_to_json(system_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended system metrics to {system_file}")


# Keep the old CSV functions for backward compatibility but mark them as deprecated
def save_comprehensive_results_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Save comprehensive benchmark results to CSV files.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
        
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
        main_results_dir = output_path.parent
    else:
        # We received the main results directory
        main_results_dir = output_path
        metrics_dir = output_path / "metrics"
    
    # Save individual results to CSV in the main results directory
    save_results_to_csv(results, str(main_results_dir))
    
    # Append aggregated metrics to existing CSV files in the metrics directory
    append_to_tasks_csv(results, str(metrics_dir))
    append_to_implementations_csv(results, str(metrics_dir))
    append_to_system_csv(results, str(metrics_dir))
    
    logger = get_logger()
    logger.info(f"Saved comprehensive results to CSV in {main_results_dir}")


def save_results_to_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Save individual benchmark results to CSV files.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for task_name, task_results in results.items():
        for result in task_results:
            row = {
                'timestamp': result.execution_date,
                'task': result.task_name,
                'implementation': result.implementation_name,
                'sample_id': result.sample_id,
                'success': result.success,
                'error_message': result.error_message,
                'accuracy': result.metrics.get('accuracy', 0.0),
                'precision': result.metrics.get('precision', 0.0),
                'recall': result.metrics.get('recall', 0.0),
                'f1': result.metrics.get('f1', 0.0),
                'tp': result.metrics.get('tp', 0),
                'tn': result.metrics.get('tn', 0),
                'fp': result.metrics.get('fp', 0),
                'fn': result.metrics.get('fn', 0),
                'inference_time': result.performance.get('inference_time', 0.0),
                'predictions': str(result.predictions),
                'ground_truth': str(result.ground_truth)
            }
            csv_data.append(row)
    
    if csv_data:
        # Append to existing CSV file or create new one
        filename = "benchmark_results.csv"
        filepath = output_path / filename
        
        append_to_csv(filepath, csv_data, mode='append')
        
        logger = get_logger()
        logger.info(f"Appended {len(csv_data)} individual results to {filepath}")


def save_aggregated_metrics_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Save aggregated metrics to CSV files for tasks and implementations.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to
    """
    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save task metrics to CSV
    save_tasks_csv(results, metrics_dir, timestamp)
    
    # Save implementation metrics to CSV
    save_implementations_csv(results, metrics_dir, timestamp)
    
    # Save system metrics to CSV
    save_system_csv(results, metrics_dir, timestamp)


def save_tasks_csv(results: Dict[str, List[BenchmarkResult]], metrics_dir: Path, timestamp: str) -> None:
    """Save task metrics to CSV with absolute confusion matrix numbers."""
    csv_data = []
    
    # Aggregate metrics by task and implementation
    task_metrics = {}
    
    for task_name, task_results in results.items():
        if task_name not in task_metrics:
            task_metrics[task_name] = {}
        
        # Aggregate by implementation
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in task_metrics[task_name]:
                task_metrics[task_name][impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0,
                    'tn_sum': 0,
                    'fp_sum': 0,
                    'fn_sum': 0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            task_metrics[task_name][impl_name]['total_samples'] += 1
            task_metrics[task_name][impl_name]['successful_samples'] += 1
            task_metrics[task_name][impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            task_metrics[task_name][impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            task_metrics[task_name][impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            task_metrics[task_name][impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            task_metrics[task_name][impl_name]['tp_sum'] += metrics.get('tp', 0)
            task_metrics[task_name][impl_name]['tn_sum'] += metrics.get('tn', 0)
            task_metrics[task_name][impl_name]['fp_sum'] += metrics.get('fp', 0)
            task_metrics[task_name][impl_name]['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            task_metrics[task_name][impl_name]['inference_time_sum'] += inference_time
    
    # Prepare CSV data
    for task_name, implementations in task_metrics.items():
        for impl_name, impl_metrics in implementations.items():
            total_samples = impl_metrics['total_samples']
            if total_samples > 0:
                row = {
                    'execution_timestamp': timestamp,
                    'task': task_name,
                    'implementation': impl_name,
                    'total_samples': total_samples,
                    'successful_samples': impl_metrics['successful_samples'],
                    'success_rate': impl_metrics['successful_samples'] / total_samples,
                    'avg_accuracy': impl_metrics['accuracy_sum'] / total_samples,
                    'avg_precision': impl_metrics['precision_sum'] / total_samples,
                    'avg_recall': impl_metrics['recall_sum'] / total_samples,
                    'avg_f1': impl_metrics['f1_sum'] / total_samples,
                    'total_tp': impl_metrics['tp_sum'],
                    'total_tn': impl_metrics['tn_sum'],
                    'total_fp': impl_metrics['fp_sum'],
                    'total_fn': impl_metrics['fn_sum'],
                    'avg_tp': impl_metrics['tp_sum'] / total_samples,
                    'avg_tn': impl_metrics['tn_sum'] / total_samples,
                    'avg_fp': impl_metrics['fp_sum'] / total_samples,
                    'avg_fn': impl_metrics['fn_sum'] / total_samples,
                    'avg_inference_time': impl_metrics['inference_time_sum'] / total_samples
                }
                csv_data.append(row)
    
    if csv_data:
        filename = f"tasks_{timestamp}.csv"
        filepath = metrics_dir / filename
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger = get_logger()
        logger.info(f"Saved task metrics to {filepath}")


def save_implementations_csv(results: Dict[str, List[BenchmarkResult]], metrics_dir: Path, timestamp: str) -> None:
    """Save implementation metrics to CSV with absolute confusion matrix numbers."""
    csv_data = []
    
    # Aggregate metrics by implementation
    implementation_metrics = {}
    
    for task_name, task_results in results.items():
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in implementation_metrics:
                implementation_metrics[impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0,
                    'tn_sum': 0,
                    'fp_sum': 0,
                    'fn_sum': 0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            implementation_metrics[impl_name]['total_samples'] += 1
            implementation_metrics[impl_name]['successful_samples'] += 1
            implementation_metrics[impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            implementation_metrics[impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            implementation_metrics[impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            implementation_metrics[impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            implementation_metrics[impl_name]['tp_sum'] += metrics.get('tp', 0)
            implementation_metrics[impl_name]['tn_sum'] += metrics.get('tn', 0)
            implementation_metrics[impl_name]['fp_sum'] += metrics.get('fp', 0)
            implementation_metrics[impl_name]['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            implementation_metrics[impl_name]['inference_time_sum'] += inference_time
    
    # Prepare CSV data
    for impl_name, metrics in implementation_metrics.items():
        total_samples = metrics['total_samples']
        if total_samples > 0:
            row = {
                'execution_timestamp': timestamp,
                'implementation': impl_name,
                'total_samples': total_samples,
                'successful_samples': metrics['successful_samples'],
                'success_rate': metrics['successful_samples'] / total_samples,
                'avg_accuracy': metrics['accuracy_sum'] / total_samples,
                'avg_precision': metrics['precision_sum'] / total_samples,
                'avg_recall': metrics['recall_sum'] / total_samples,
                'avg_f1': metrics['f1_sum'] / total_samples,
                'total_tp': metrics['tp_sum'],
                'total_tn': metrics['tn_sum'],
                'total_fp': metrics['fp_sum'],
                'total_fn': metrics['fn_sum'],
                'avg_tp': metrics['tp_sum'] / total_samples,
                'avg_tn': metrics['tn_sum'] / total_samples,
                'avg_fp': metrics['fp_sum'] / total_samples,
                'avg_fn': metrics['fn_sum'] / total_samples,
                'avg_inference_time': metrics['inference_time_sum'] / total_samples
            }
            csv_data.append(row)
    
    if csv_data:
        filename = f"implementations_{timestamp}.csv"
        filepath = metrics_dir / filename
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger = get_logger()
        logger.info(f"Saved implementation metrics to {filepath}")


def save_system_csv(results: Dict[str, List[BenchmarkResult]], metrics_dir: Path, timestamp: str) -> None:
    """Save system-wide metrics to CSV with absolute confusion matrix numbers."""
    csv_data = []
    
    # Aggregate overall metrics
    overall_metrics = {
        'total_samples': 0,
        'successful_samples': 0,
        'accuracy_sum': 0.0,
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'f1_sum': 0.0,
        'tp_sum': 0,
        'tn_sum': 0,
        'fp_sum': 0,
        'fn_sum': 0,
        'inference_time_sum': 0.0
    }
    
    # Process all results
    for task_name, task_results in results.items():
        for result in task_results:
            metrics = result.metrics
            
            overall_metrics['total_samples'] += 1
            overall_metrics['successful_samples'] += 1
            overall_metrics['accuracy_sum'] += metrics.get('accuracy', 0.0)
            overall_metrics['precision_sum'] += metrics.get('precision', 0.0)
            overall_metrics['recall_sum'] += metrics.get('recall', 0.0)
            overall_metrics['f1_sum'] += metrics.get('f1', 0.0)
            overall_metrics['tp_sum'] += metrics.get('tp', 0)
            overall_metrics['tn_sum'] += metrics.get('tn', 0)
            overall_metrics['fp_sum'] += metrics.get('fp', 0)
            overall_metrics['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            overall_metrics['inference_time_sum'] += inference_time
    
    # Add overall system metrics
    if overall_metrics['total_samples'] > 0:
        total_samples = overall_metrics['total_samples']
        row = {
            'execution_timestamp': timestamp,
            'metric_type': 'system_overall',
            'total_samples': total_samples,
            'successful_samples': overall_metrics['successful_samples'],
            'success_rate': overall_metrics['successful_samples'] / total_samples,
            'avg_accuracy': overall_metrics['accuracy_sum'] / total_samples,
            'avg_precision': overall_metrics['precision_sum'] / total_samples,
            'avg_recall': overall_metrics['recall_sum'] / total_samples,
            'avg_f1': overall_metrics['f1_sum'] / total_samples,
            'total_tp': overall_metrics['tp_sum'],
            'total_tn': overall_metrics['tn_sum'],
            'total_fp': overall_metrics['fp_sum'],
            'total_fn': overall_metrics['fn_sum'],
            'avg_tp': overall_metrics['tp_sum'] / total_samples,
            'avg_tn': overall_metrics['tn_sum'] / total_samples,
            'avg_fp': overall_metrics['fp_sum'] / total_samples,
            'avg_fn': overall_metrics['fn_sum'] / total_samples,
            'avg_inference_time': overall_metrics['inference_time_sum'] / total_samples
        }
        csv_data.append(row)
    
    if csv_data:
        filename = f"system_{timestamp}.csv"
        filepath = metrics_dir / filename
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger = get_logger()
        logger.info(f"Saved system metrics to {filepath}")


def append_to_csv(filepath: Path, new_data: List[Dict], mode: str = 'append') -> None:
    """
    Append new data to an existing CSV file or create a new one.
    
    Args:
        filepath: Path to the CSV file
        new_data: List of dictionaries to append
        mode: 'append' to add to existing file, 'overwrite' to replace
    """
    filepath = Path(filepath)
    
    if mode == 'append' and filepath.exists():
        # Read existing data
        try:
            existing_df = pd.read_csv(filepath)
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined_df = pd.DataFrame(new_data)
    else:
        combined_df = pd.DataFrame(new_data)
    
    # Write back to file
    combined_df.to_csv(filepath, index=False, encoding='utf-8')


def append_to_implementations_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Append implementation metrics to implementations.csv.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    implementations_file = metrics_dir / "implementations.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate metrics by implementation
    implementation_metrics = {}
    
    for task_name, task_results in results.items():
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in implementation_metrics:
                implementation_metrics[impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0,
                    'tn_sum': 0,
                    'fp_sum': 0,
                    'fn_sum': 0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            implementation_metrics[impl_name]['total_samples'] += 1
            implementation_metrics[impl_name]['successful_samples'] += 1
            implementation_metrics[impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            implementation_metrics[impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            implementation_metrics[impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            implementation_metrics[impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            implementation_metrics[impl_name]['tp_sum'] += metrics.get('tp', 0)
            implementation_metrics[impl_name]['tn_sum'] += metrics.get('tn', 0)
            implementation_metrics[impl_name]['fp_sum'] += metrics.get('fp', 0)
            implementation_metrics[impl_name]['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            implementation_metrics[impl_name]['inference_time_sum'] += inference_time
    
    # Prepare data for CSV
    append_data = []
    for impl_name, metrics in implementation_metrics.items():
        total_samples = metrics['total_samples']
        if total_samples > 0:
            row = {
                'execution_timestamp': timestamp,
                'implementation': impl_name,
                'total_samples': total_samples,
                'successful_samples': metrics['successful_samples'],
                'success_rate': metrics['successful_samples'] / total_samples,
                'avg_accuracy': metrics['accuracy_sum'] / total_samples,
                'avg_precision': metrics['precision_sum'] / total_samples,
                'avg_recall': metrics['recall_sum'] / total_samples,
                'avg_f1': metrics['f1_sum'] / total_samples,
                'total_tp': metrics['tp_sum'],
                'total_tn': metrics['tn_sum'],
                'total_fp': metrics['fp_sum'],
                'total_fn': metrics['fn_sum'],
                'avg_tp': metrics['tp_sum'] / total_samples,
                'avg_tn': metrics['tn_sum'] / total_samples,
                'avg_fp': metrics['fp_sum'] / total_samples,
                'avg_fn': metrics['fn_sum'] / total_samples,
                'avg_inference_time': metrics['inference_time_sum'] / total_samples
            }
            append_data.append(row)
    
    # Append to CSV file
    append_to_csv(implementations_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended implementation metrics to {implementations_file}")


def append_to_tasks_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Append task metrics to tasks.csv.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_file = metrics_dir / "tasks.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate metrics by task and implementation
    task_metrics = {}
    
    for task_name, task_results in results.items():
        if task_name not in task_metrics:
            task_metrics[task_name] = {}
        
        # Aggregate by implementation
        for result in task_results:
            impl_name = result.implementation_name
            
            if impl_name not in task_metrics[task_name]:
                task_metrics[task_name][impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'accuracy_sum': 0.0,
                    'precision_sum': 0.0,
                    'recall_sum': 0.0,
                    'f1_sum': 0.0,
                    'tp_sum': 0,
                    'tn_sum': 0,
                    'fp_sum': 0,
                    'fn_sum': 0,
                    'inference_time_sum': 0.0
                }
            
            metrics = result.metrics
            task_metrics[task_name][impl_name]['total_samples'] += 1
            task_metrics[task_name][impl_name]['successful_samples'] += 1
            task_metrics[task_name][impl_name]['accuracy_sum'] += metrics.get('accuracy', 0.0)
            task_metrics[task_name][impl_name]['precision_sum'] += metrics.get('precision', 0.0)
            task_metrics[task_name][impl_name]['recall_sum'] += metrics.get('recall', 0.0)
            task_metrics[task_name][impl_name]['f1_sum'] += metrics.get('f1', 0.0)
            task_metrics[task_name][impl_name]['tp_sum'] += metrics.get('tp', 0)
            task_metrics[task_name][impl_name]['tn_sum'] += metrics.get('tn', 0)
            task_metrics[task_name][impl_name]['fp_sum'] += metrics.get('fp', 0)
            task_metrics[task_name][impl_name]['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            task_metrics[task_name][impl_name]['inference_time_sum'] += inference_time
    
    # Prepare data for CSV
    append_data = []
    for task_name, implementations in task_metrics.items():
        for impl_name, impl_metrics in implementations.items():
            total_samples = impl_metrics['total_samples']
            if total_samples > 0:
                row = {
                    'execution_timestamp': timestamp,
                    'task': task_name,
                    'implementation': impl_name,
                    'total_samples': total_samples,
                    'successful_samples': impl_metrics['successful_samples'],
                    'success_rate': impl_metrics['successful_samples'] / total_samples,
                    'avg_accuracy': impl_metrics['accuracy_sum'] / total_samples,
                    'avg_precision': impl_metrics['precision_sum'] / total_samples,
                    'avg_recall': impl_metrics['recall_sum'] / total_samples,
                    'avg_f1': impl_metrics['f1_sum'] / total_samples,
                    'total_tp': impl_metrics['tp_sum'],
                    'total_tn': impl_metrics['tn_sum'],
                    'total_fp': impl_metrics['fp_sum'],
                    'total_fn': impl_metrics['fn_sum'],
                    'avg_tp': impl_metrics['tp_sum'] / total_samples,
                    'avg_tn': impl_metrics['tn_sum'] / total_samples,
                    'avg_fp': impl_metrics['fp_sum'] / total_samples,
                    'avg_fn': impl_metrics['fn_sum'] / total_samples,
                    'avg_inference_time': impl_metrics['inference_time_sum'] / total_samples
                }
                append_data.append(row)
    
    # Append to CSV file
    append_to_csv(tasks_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended task metrics to {tasks_file}")


def append_to_system_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results") -> None:
    """
    Append system metrics to system.csv.
    
    Args:
        results: Dictionary mapping task names to lists of BenchmarkResult objects
        output_dir: Directory to save results to (can be main results dir or metrics dir)
    """
    output_path = Path(output_dir)
    
    # Determine if we received the metrics directory or main results directory
    if output_path.name == "metrics":
        # We received the metrics directory directly
        metrics_dir = output_path
    else:
        # We received the main results directory
        metrics_dir = output_path / "metrics"
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    system_file = metrics_dir / "system.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aggregate overall metrics
    overall_metrics = {
        'total_samples': 0,
        'successful_samples': 0,
        'accuracy_sum': 0.0,
        'precision_sum': 0.0,
        'recall_sum': 0.0,
        'f1_sum': 0.0,
        'tp_sum': 0,
        'tn_sum': 0,
        'fp_sum': 0,
        'fn_sum': 0,
        'inference_time_sum': 0.0
    }
    
    # Process all results
    for task_name, task_results in results.items():
        for result in task_results:
            metrics = result.metrics
            
            overall_metrics['total_samples'] += 1
            overall_metrics['successful_samples'] += 1
            overall_metrics['accuracy_sum'] += metrics.get('accuracy', 0.0)
            overall_metrics['precision_sum'] += metrics.get('precision', 0.0)
            overall_metrics['recall_sum'] += metrics.get('recall', 0.0)
            overall_metrics['f1_sum'] += metrics.get('f1', 0.0)
            overall_metrics['tp_sum'] += metrics.get('tp', 0)
            overall_metrics['tn_sum'] += metrics.get('tn', 0)
            overall_metrics['fp_sum'] += metrics.get('fp', 0)
            overall_metrics['fn_sum'] += metrics.get('fn', 0)
            inference_time = result.performance.get('inference_time', 0.0)
            overall_metrics['inference_time_sum'] += inference_time
    
    # Prepare data for CSV
    append_data = []
    if overall_metrics['total_samples'] > 0:
        total_samples = overall_metrics['total_samples']
        row = {
            'execution_timestamp': timestamp,
            'metric_type': 'system_overall',
            'total_samples': total_samples,
            'successful_samples': overall_metrics['successful_samples'],
            'success_rate': overall_metrics['successful_samples'] / total_samples,
            'avg_accuracy': overall_metrics['accuracy_sum'] / total_samples,
            'avg_precision': overall_metrics['precision_sum'] / total_samples,
            'avg_recall': overall_metrics['recall_sum'] / total_samples,
            'avg_f1': overall_metrics['f1_sum'] / total_samples,
            'total_tp': overall_metrics['tp_sum'],
            'total_tn': overall_metrics['tn_sum'],
            'total_fp': overall_metrics['fp_sum'],
            'total_fn': overall_metrics['fn_sum'],
            'avg_tp': overall_metrics['tp_sum'] / total_samples,
            'avg_tn': overall_metrics['tn_sum'] / total_samples,
            'avg_fp': overall_metrics['fp_sum'] / total_samples,
            'avg_fn': overall_metrics['fn_sum'] / total_samples,
            'avg_inference_time': overall_metrics['inference_time_sum'] / total_samples
        }
        append_data.append(row)
    
    # Append to CSV file
    append_to_csv(system_file, append_data, mode='append')
    logger = get_logger()
    logger.info(f"Appended system metrics to {system_file}")
