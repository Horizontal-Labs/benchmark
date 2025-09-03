"""
File handling utilities for benchmark results.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from ..core.results import BenchmarkResult


def save_results_to_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results"):
    """Save benchmark results to CSV files with improved format."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, task_results in results.items():
        if not task_results:
            continue
        
        # Convert to DataFrame
        df_data = []
        for result in task_results:
            row = {
                'execution_date': result.execution_date,
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
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {task_name} results to {filepath}")
    
    return output_path


def save_comprehensive_results_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results"):
    """
    Save all benchmark results in one comprehensive CSV with execution timestamp,
    task, implementation, and all metrics including accuracy, precision, recall, f1, tp, tn, fp, fn.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect all results in one list
    all_results = []
    
    for task_name, task_results in results.items():
        if not task_results:
            continue
        
        for result in task_results:
            row = {
                'execution_timestamp': result.execution_date,
                'task': result.task_name,
                'implementation': result.implementation_name,
                'sample_id': result.sample_id,
                'success': result.success,
                'error_message': result.error_message
            }
            
            # Add standard metrics (accuracy, precision, recall, f1)
            standard_metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in standard_metrics:
                row[metric] = result.metrics.get(metric, 0.0)
            
            # Add confusion matrix metrics (tp, tn, fp, fn)
            confusion_metrics = ['tp', 'tn', 'fp', 'fn']
            for metric in confusion_metrics:
                row[metric] = result.metrics.get(metric, 0)
            
            # Add any additional metrics
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in standard_metrics and metric_name not in confusion_metrics:
                    row[f'metric_{metric_name}'] = metric_value
            
            # Add performance metrics
            for perf_name, perf_value in result.performance.items():
                row[f'perf_{perf_name}'] = perf_value
            
            all_results.append(row)
    
    if all_results:
        # Create comprehensive DataFrame
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        filename = f"comprehensive_benchmark_results_{timestamp}.csv"
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        print(f"Saved comprehensive benchmark results to {filepath}")
        
        # Also save a summary CSV with aggregated metrics per task/implementation
        summary_df = create_summary_dataframe(df)
        summary_filename = f"benchmark_summary_{timestamp}.csv"
        summary_filepath = output_path / summary_filename
        summary_df.to_csv(summary_filepath, index=False)
        print(f"Saved benchmark summary to {summary_filepath}")
        
        return output_path
    else:
        print("No results to save")
        return output_path


def create_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary DataFrame with aggregated metrics per task/implementation."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by task and implementation
    summary_data = []
    
    for (task, implementation), group in df.groupby(['task', 'implementation']):
        # Filter successful results
        successful_results = group[group['success'] == True]
        
        if successful_results.empty:
            continue
        
        summary_row = {
            'task': task,
            'implementation': implementation,
            'total_samples': len(group),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(group)
        }
        
        # Calculate average metrics for successful results
        numeric_columns = ['accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn']
        for col in numeric_columns:
            if col in successful_results.columns:
                summary_row[f'avg_{col}'] = successful_results[col].mean()
                summary_row[f'std_{col}'] = successful_results[col].std()
        
        # Calculate average performance metrics
        perf_columns = [col for col in successful_results.columns if col.startswith('perf_')]
        for col in perf_columns:
            summary_row[f'avg_{col}'] = successful_results[col].mean()
            summary_row[f'std_{col}'] = successful_results[col].std()
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def append_to_csv(filepath: Path, new_data: List[Dict], mode: str = 'append'):
    """
    Append new data to existing CSV file or create new one.
    
    Args:
        filepath: Path to CSV file
        new_data: List of dictionaries with new data
        mode: 'append' to add to existing file, 'overwrite' to replace
    """
    if not new_data:
        return
    
    if mode == 'append' and filepath.exists():
        # Read existing data
        try:
            existing_df = pd.read_csv(filepath)
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing CSV, creating new file: {e}")
            combined_df = pd.DataFrame(new_data)
    else:
        combined_df = pd.DataFrame(new_data)
    
    # Save combined data
    combined_df.to_csv(filepath, index=False)
    print(f"Updated {filepath} with {len(new_data)} new records")


def append_to_implementations_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results"):
    """
    Append implementation metrics to implementations.csv file.
    Aggregates results by implementation across all tasks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group results by implementation
    impl_metrics = {}
    
    for task_name, task_results in results.items():
        if not task_results:
            continue
        
        for result in task_results:
            if not result.success:
                continue
                
            impl_name = result.implementation_name
            if impl_name not in impl_metrics:
                impl_metrics[impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'metrics_sum': {},
                    'performance_sum': {}
                }
            
            impl_metrics[impl_name]['total_samples'] += 1
            impl_metrics[impl_name]['successful_samples'] += 1
            
            # Aggregate metrics
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in impl_metrics[impl_name]['metrics_sum']:
                    impl_metrics[impl_name]['metrics_sum'][metric_name] = 0.0
                impl_metrics[impl_name]['metrics_sum'][metric_name] += metric_value
            
            # Aggregate performance metrics
            for perf_name, perf_value in result.performance.items():
                if perf_name not in impl_metrics[impl_name]['performance_sum']:
                    impl_metrics[impl_name]['performance_sum'][perf_name] = 0.0
                impl_metrics[impl_name]['performance_sum'][perf_name] += perf_value
    
    # Create data for appending
    append_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for impl_name, metrics in impl_metrics.items():
        if metrics['successful_samples'] == 0:
            continue
            
        row = {
            'execution_timestamp': timestamp,
            'implementation': impl_name,
            'total_samples': metrics['total_samples'],
            'successful_samples': metrics['successful_samples'],
            'success_rate': metrics['successful_samples'] / metrics['total_samples']
        }
        
        # Calculate average metrics
        for metric_name, metric_sum in metrics['metrics_sum'].items():
            row[f'avg_{metric_name}'] = metric_sum / metrics['successful_samples']
        
        # Calculate average performance metrics
        for perf_name, perf_sum in metrics['performance_sum'].items():
            row[f'avg_{perf_name}'] = perf_sum / metrics['successful_samples']
        
        append_data.append(row)
    
    if append_data:
        # Append to implementations.csv
        implementations_file = output_path / "implementations.csv"
        append_to_csv(implementations_file, append_data, mode='append')
        print(f"Appended implementation metrics to {implementations_file}")
    else:
        print("No implementation metrics to append")


def append_to_tasks_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results"):
    """
    Append task metrics to tasks.csv file.
    Aggregates results by task across all implementations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group results by task
    task_metrics = {}
    
    for task_name, task_results in results.items():
        if not task_results:
            continue
        
        if task_name not in task_metrics:
            task_metrics[task_name] = {
                'total_samples': 0,
                'successful_samples': 0,
                'metrics_sum': {},
                'performance_sum': {},
                'implementation_results': {}
            }
        
        for result in task_results:
            if not result.success:
                continue
                
            impl_name = result.implementation_name
            task_metrics[task_name]['total_samples'] += 1
            task_metrics[task_name]['successful_samples'] += 1
            
            # Initialize implementation results for this task
            if impl_name not in task_metrics[task_name]['implementation_results']:
                task_metrics[task_name]['implementation_results'][impl_name] = {
                    'samples': 0,
                    'metrics_sum': {},
                    'performance_sum': {}
                }
            
            task_metrics[task_name]['implementation_results'][impl_name]['samples'] += 1
            
            # Aggregate metrics
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in task_metrics[task_name]['metrics_sum']:
                    task_metrics[task_name]['metrics_sum'][metric_name] = 0.0
                task_metrics[task_name]['metrics_sum'][metric_name] += metric_value
                
                if metric_name not in task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum']:
                    task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum'][metric_name] = 0.0
                task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum'][metric_name] += metric_value
            
            # Aggregate performance metrics
            for perf_name, perf_value in result.performance.items():
                if perf_name not in task_metrics[task_name]['performance_sum']:
                    task_metrics[task_name]['performance_sum'][perf_name] = 0.0
                task_metrics[task_name]['performance_sum'][perf_name] += perf_value
                
                if perf_name not in task_metrics[task_name]['implementation_results'][impl_name]['performance_sum']:
                    task_metrics[task_name]['implementation_results'][impl_name]['performance_sum'][perf_name] = 0.0
                task_metrics[task_name]['implementation_results'][impl_name]['performance_sum'][perf_name] += perf_value
    
    # Create data for appending
    append_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, metrics in task_metrics.items():
        if metrics['successful_samples'] == 0:
            continue
            
        # Task-level summary
        task_row = {
            'execution_timestamp': timestamp,
            'task': task_name,
            'implementation': 'ALL',
            'total_samples': metrics['total_samples'],
            'successful_samples': metrics['successful_samples'],
            'success_rate': metrics['successful_samples'] / metrics['total_samples']
        }
        
        # Calculate overall task averages
        for metric_name, metric_sum in metrics['metrics_sum'].items():
            task_row[f'avg_{metric_name}'] = metric_sum / metrics['successful_samples']
        
        for perf_name, perf_sum in metrics['performance_sum'].items():
            task_row[f'avg_{perf_name}'] = perf_sum / metrics['successful_samples']
        
        append_data.append(task_row)
        
        # Implementation-level results for this task
        for impl_name, impl_metrics in metrics['implementation_results'].items():
            if impl_metrics['samples'] == 0:
                continue
                
            impl_row = {
                'execution_timestamp': timestamp,
                'task': task_name,
                'implementation': impl_name,
                'total_samples': impl_metrics['samples'],
                'successful_samples': impl_metrics['samples'],
                'success_rate': 1.0
            }
            
            # Calculate implementation-specific averages
            for metric_name, metric_sum in impl_metrics['metrics_sum'].items():
                impl_row[f'avg_{metric_name}'] = metric_sum / impl_metrics['samples']
            
            for perf_name, perf_sum in impl_metrics['performance_sum'].items():
                impl_row[f'avg_{perf_name}'] = perf_sum / impl_metrics['samples']
            
            append_data.append(impl_row)
    
    if append_data:
        # Append to tasks.csv
        tasks_file = output_path / "tasks.csv"
        append_to_csv(tasks_file, append_data, mode='append')
        print(f"Appended task metrics to {tasks_file}")
    else:
        print("No task metrics to append")


def append_to_system_csv(results: Dict[str, List[BenchmarkResult]], output_dir: str = "results"):
    """
    Append system-level metrics to system.csv file.
    Combines implementation and task metrics with additional aggregated statistics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get implementation and task metrics data directly
    impl_metrics = {}
    task_metrics = {}
    
    # Process results to get both implementation and task metrics
    for task_name, task_results in results.items():
        if not task_results:
            continue
        
        for result in task_results:
            if not result.success:
                continue
                
            impl_name = result.implementation_name
            
            # Implementation metrics
            if impl_name not in impl_metrics:
                impl_metrics[impl_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'metrics_sum': {},
                    'performance_sum': {}
                }
            
            impl_metrics[impl_name]['total_samples'] += 1
            impl_metrics[impl_name]['successful_samples'] += 1
            
            # Task metrics
            if task_name not in task_metrics:
                task_metrics[task_name] = {
                    'total_samples': 0,
                    'successful_samples': 0,
                    'metrics_sum': {},
                    'performance_sum': {},
                    'implementation_results': {}
                }
            
            task_metrics[task_name]['total_samples'] += 1
            task_metrics[task_name]['successful_samples'] += 1
            
            # Initialize implementation results for this task
            if impl_name not in task_metrics[task_name]['implementation_results']:
                task_metrics[task_name]['implementation_results'][impl_name] = {
                    'samples': 0,
                    'metrics_sum': {},
                    'performance_sum': {}
                }
            
            task_metrics[task_name]['implementation_results'][impl_name]['samples'] += 1
            
            # Aggregate metrics for both
            for metric_name, metric_value in result.metrics.items():
                # Implementation level
                if metric_name not in impl_metrics[impl_name]['metrics_sum']:
                    impl_metrics[impl_name]['metrics_sum'][metric_name] = 0.0
                impl_metrics[impl_name]['metrics_sum'][metric_name] += metric_value
                
                # Task level
                if metric_name not in task_metrics[task_name]['metrics_sum']:
                    task_metrics[task_name]['metrics_sum'][metric_name] = 0.0
                task_metrics[task_name]['metrics_sum'][metric_name] += metric_value
                
                # Implementation within task
                if metric_name not in task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum']:
                    task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum'][metric_name] = 0.0
                task_metrics[task_name]['implementation_results'][impl_name]['metrics_sum'][metric_name] += metric_value
            
            # Aggregate performance metrics
            for perf_name, perf_value in result.performance.items():
                # Implementation level
                if perf_name not in impl_metrics[impl_name]['performance_sum']:
                    impl_metrics[impl_name]['performance_sum'][perf_name] = 0.0
                impl_metrics[impl_name]['performance_sum'][perf_name] += perf_value
                
                # Task level
                if perf_name not in task_metrics[task_name]['performance_sum']:
                    task_metrics[task_name]['performance_sum'][perf_name] = 0.0
                task_metrics[task_name]['performance_sum'][perf_name] += perf_value
                
                # Implementation within task
                if perf_name not in task_metrics[task_name]['implementation_results'][impl_name]['performance_sum']:
                    task_metrics[task_name]['implementation_results'][impl_name]['performance_sum'][perf_name] = 0.0
                task_metrics[task_name]['implementation_results'][impl_name]['performance_sum'][perf_name] += perf_value
    
    # Create combined data for appending
    append_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add implementation metrics
    for impl_name, metrics in impl_metrics.items():
        if metrics['successful_samples'] == 0:
            continue
            
        row = {
            'execution_timestamp': timestamp,
            'metric_type': 'implementation',
            'implementation': impl_name,
            'task': 'ALL',
            'total_samples': metrics['total_samples'],
            'successful_samples': metrics['successful_samples'],
            'success_rate': metrics['successful_samples'] / metrics['total_samples']
        }
        
        # Calculate average metrics
        for metric_name, metric_sum in metrics['metrics_sum'].items():
            row[f'avg_{metric_name}'] = metric_sum / metrics['successful_samples']
        
        # Calculate average performance metrics
        for perf_name, perf_sum in metrics['performance_sum'].items():
            row[f'avg_{perf_name}'] = perf_sum / metrics['successful_samples']
        
        append_data.append(row)
    
    # Add task metrics
    for task_name, metrics in task_metrics.items():
        if metrics['successful_samples'] == 0:
            continue
            
        # Task-level summary
        task_row = {
            'execution_timestamp': timestamp,
            'metric_type': 'task',
            'implementation': 'ALL',
            'task': task_name,
            'total_samples': metrics['total_samples'],
            'successful_samples': metrics['successful_samples'],
            'success_rate': metrics['successful_samples'] / metrics['total_samples']
        }
        
        # Calculate overall task averages
        for metric_name, metric_sum in metrics['metrics_sum'].items():
            task_row[f'avg_{metric_name}'] = metric_sum / metrics['successful_samples']
        
        for perf_name, perf_sum in metrics['performance_sum'].items():
            task_row[f'avg_{perf_name}'] = perf_sum / metrics['successful_samples']
        
        append_data.append(task_row)
        
        # Implementation-level results for this task
        for impl_name, impl_metrics in metrics['implementation_results'].items():
            if impl_metrics['samples'] == 0:
                continue
                
            impl_row = {
                'execution_timestamp': timestamp,
                'metric_type': 'task_implementation',
                'task': task_name,
                'implementation': impl_name,
                'total_samples': impl_metrics['samples'],
                'successful_samples': impl_metrics['samples'],
                'success_rate': 1.0
            }
            
            # Calculate implementation-specific averages
            for metric_name, metric_sum in impl_metrics['metrics_sum'].items():
                impl_row[f'avg_{metric_name}'] = metric_sum / impl_metrics['samples']
            
            for perf_name, perf_sum in impl_metrics['performance_sum'].items():
                impl_row[f'avg_{perf_name}'] = perf_sum / impl_metrics['samples']
            
            append_data.append(impl_row)
    
    if append_data:
        # Append to system.csv
        system_file = output_path / "system.csv"
        append_to_csv(system_file, append_data, mode='append')
        print(f"Appended system metrics to {system_file}")
    else:
        print("No system metrics to append")
