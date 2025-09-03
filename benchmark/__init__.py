"""
Argument Mining Benchmark Package

A comprehensive benchmarking tool for argument mining implementations.
"""

__version__ = "1.0.0"
__author__ = "Horizontal Labs"

from .core.benchmark import ArgumentMiningBenchmark
from .core.results import BenchmarkResult

__all__ = [
    "ArgumentMiningBenchmark",
    "BenchmarkResult",
]
