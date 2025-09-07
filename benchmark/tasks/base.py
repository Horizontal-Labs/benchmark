"""
Base task interface for benchmark tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..core.results import BenchmarkResult


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def prepare_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Prepare data specifically for this task."""
        pass
    
    @abstractmethod
    def run_benchmark(self, implementation, data: List[Dict[str, Any]], progress_bar=None) -> List[BenchmarkResult]:
        """Run the benchmark for this task with the given implementation."""
        pass
    
    @abstractmethod
    def calculate_metrics(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Calculate metrics for this task."""
        pass
    
    def get_task_name(self) -> str:
        """Get the task name."""
        return self.name
