"""
Benchmark result data structures.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkResult:
    """Represents benchmark results for a single task/implementation."""
    task_name: str
    implementation_name: str
    sample_id: str
    execution_date: str
    metrics: Dict[str, float]
    performance: Dict[str, float]
    predictions: Any
    ground_truth: Any
    error_message: str = ""
    success: bool = True
