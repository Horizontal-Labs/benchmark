#!/usr/bin/env python3
"""
Tests for individual components of the refactored Argument Mining Benchmark package.

This module tests each component in isolation to ensure they work correctly.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Import the base classes for testing inheritance
from benchmark.implementations.base import BaseImplementation
from benchmark.tasks.base import BaseTask


class TestCoreComponents:
    """Test the core components of the benchmark package."""
    
    def test_benchmark_result_dataclass(self):
        """Test the BenchmarkResult dataclass functionality."""
        from benchmark.core.results import BenchmarkResult
        
        # Test basic creation
        result = BenchmarkResult(
            task_name="test_task",
            implementation_name="test_impl",
            sample_id="0",
            execution_date="2024-01-01",
            metrics={"accuracy": 0.95},
            performance={"inference_time": 0.1},
            predictions=["test_prediction"],
            ground_truth=["test_ground_truth"]
        )
        
        assert result.task_name == "test_task"
        assert result.implementation_name == "test_impl"
        assert result.sample_id == "0"
        assert result.execution_date == "2024-01-01"
        assert result.metrics["accuracy"] == 0.95
        assert result.performance["inference_time"] == 0.1
        assert result.predictions == ["test_prediction"]
        assert result.ground_truth == ["test_ground_truth"]
        assert result.error_message == ""
        assert result.success is True
        
        # Test with error
        error_result = BenchmarkResult(
            task_name="test_task",
            implementation_name="test_impl",
            sample_id="0",
            execution_date="2024-01-01",
            metrics={},
            performance={},
            predictions=None,
            ground_truth=["test_ground_truth"],
            error_message="Test error",
            success=False
        )
        
        assert error_result.success is False
        assert error_result.error_message == "Test error"
        assert error_result.predictions is None


class TestImplementationComponents:
    """Test the implementation components."""
    
    def test_base_implementation_interface(self):
        """Test the BaseImplementation abstract interface."""
        from benchmark.implementations.base import BaseImplementation
        
        # Test that it's abstract
        with pytest.raises(TypeError):
            BaseImplementation("test")
        
        # Test the interface methods exist
        assert hasattr(BaseImplementation, 'initialize')
        assert hasattr(BaseImplementation, 'is_available')
        assert hasattr(BaseImplementation, 'get_adu_classifier')
        assert hasattr(BaseImplementation, 'get_linker')
        assert hasattr(BaseImplementation, 'supports_task')
    
    def test_openai_implementation_structure(self):
        """Test the OpenAI implementation structure."""
        from benchmark.implementations.openai import OpenAIImplementation
        
        # Test class inheritance
        assert issubclass(OpenAIImplementation, BaseImplementation)
        
        # Test initialization
        impl = OpenAIImplementation()
        assert impl.name == "openai"
        assert impl.adu_classifier is None
        assert impl.linker is None
    
    def test_tinyllama_implementation_structure(self):
        """Test the TinyLlama implementation structure."""
        from benchmark.implementations.tinyllama import TinyLlamaImplementation
        
        # Test class inheritance
        assert issubclass(TinyLlamaImplementation, BaseImplementation)
        
        # Test initialization
        impl = TinyLlamaImplementation()
        assert impl.name == "tinyllama"
        assert impl.adu_classifier is None
        assert impl.linker is None
    
    def test_modernbert_implementation_structure(self):
        """Test the ModernBERT implementation structure."""
        from benchmark.implementations.modernbert import ModernBERTImplementation
        
        # Test class inheritance
        assert issubclass(ModernBERTImplementation, BaseImplementation)
        
        # Test initialization
        impl = ModernBERTImplementation()
        assert impl.name == "modernbert"
        assert impl.adu_classifier is None
        assert impl.linker is None
    
    def test_deberta_implementation_structure(self):
        """Test the DeBERTa implementation structure."""
        from benchmark.implementations.deberta import DeBERTaImplementation
        
        # Test class inheritance
        assert issubclass(DeBERTaImplementation, BaseImplementation)
        
        # Test initialization
        impl = DeBERTaImplementation()
        assert impl.name == "deberta"
        assert impl.adu_classifier is None
        assert impl.linker is None


class TestTaskComponents:
    """Test the task components."""
    
    def test_base_task_interface(self):
        """Test the BaseTask abstract interface."""
        from benchmark.tasks.base import BaseTask
        
        # Test that it's abstract
        with pytest.raises(TypeError):
            BaseTask("test")
        
        # Test the interface methods exist
        assert hasattr(BaseTask, 'prepare_data')
        assert hasattr(BaseTask, 'run_benchmark')
        assert hasattr(BaseTask, 'calculate_metrics')
        assert hasattr(BaseTask, 'get_task_name')
    
    def test_adu_extraction_task_structure(self):
        """Test the ADU extraction task structure."""
        from benchmark.tasks.adu_extraction import ADUExtractionTask
        
        # Test class inheritance
        assert issubclass(ADUExtractionTask, BaseTask)
        
        # Test initialization
        task = ADUExtractionTask()
        assert task.name == "adu_extraction"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert all('text' in sample for sample in prepared_data)
        assert all('ground_truth' in sample for sample in prepared_data)
        assert all('adus' in sample['ground_truth'] for sample in prepared_data)
    
    def test_stance_classification_task_structure(self):
        """Test the stance classification task structure."""
        from benchmark.tasks.stance_classification import StanceClassificationTask
        
        # Test class inheritance
        assert issubclass(StanceClassificationTask, BaseTask)
        
        # Test initialization
        task = StanceClassificationTask()
        assert task.name == "stance_classification"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert all('text' in sample for sample in prepared_data)
        assert all('ground_truth' in sample for sample in prepared_data)
        assert all('stance' in sample['ground_truth'] for sample in prepared_data)
    
    def test_claim_premise_linking_task_structure(self):
        """Test the claim-premise linking task structure."""
        from benchmark.tasks.claim_premise_linking import ClaimPremiseLinkingTask
        
        # Test class inheritance
        assert issubclass(ClaimPremiseLinkingTask, BaseTask)
        
        # Test initialization
        task = ClaimPremiseLinkingTask()
        assert task.name == "claim_premise_linking"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert all('text' in sample for sample in prepared_data)
        assert all('ground_truth' in sample for sample in prepared_data)
        assert all('relationships' in sample['ground_truth'] for sample in prepared_data)


class TestDataComponents:
    """Test the data components."""
    
    def test_data_loader_structure(self):
        """Test the DataLoader structure."""
        from benchmark.data.loader import DataLoader
        
        # Test initialization
        loader = DataLoader()
        assert hasattr(loader, 'db_available')
        assert hasattr(loader, 'load_benchmark_data')
        assert hasattr(loader, 'is_database_available')
        
        # Test methods exist
        assert callable(loader.load_benchmark_data)
        assert callable(loader.is_database_available)


class TestMetricsComponents:
    """Test the metrics components."""
    
    def test_metrics_evaluator_structure(self):
        """Test the MetricsEvaluator structure."""
        from benchmark.metrics.evaluator import MetricsEvaluator
        
        # Test initialization
        evaluator = MetricsEvaluator()
        assert hasattr(evaluator, 'calculate_classification_metrics')
        assert hasattr(evaluator, 'calculate_token_level_metrics')
        
        # Test methods exist
        assert callable(evaluator.calculate_classification_metrics)
        assert callable(evaluator.calculate_token_level_metrics)
    
    def test_classification_metrics_calculation(self):
        """Test classification metrics calculation."""
        from benchmark.metrics.evaluator import MetricsEvaluator
        
        evaluator = MetricsEvaluator()
        
        # Test with valid data
        y_true = ["pro", "con", "pro", "neutral"]
        y_pred = ["pro", "con", "pro", "pro"]
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        
        # Test with empty data
        empty_metrics = evaluator.calculate_classification_metrics([], [])
        assert empty_metrics["accuracy"] == 0.0
    
    def test_token_level_metrics_calculation(self):
        """Test token-level metrics calculation."""
        from benchmark.metrics.evaluator import MetricsEvaluator
        
        evaluator = MetricsEvaluator()
        
        # Test with valid data
        pred_tokens = ["climate", "change", "real"]
        gt_tokens = ["climate", "change", "is", "real"]
        
        metrics = evaluator.calculate_token_level_metrics(pred_tokens, gt_tokens)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        
        # Test with empty data
        empty_metrics = evaluator.calculate_token_level_metrics([], [])
        assert empty_metrics["precision"] == 0.0


class TestUtilityComponents:
    """Test the utility components."""
    
    def test_logging_setup(self):
        """Test the logging utility setup."""
        from benchmark.utils.logging import setup_logging
        
        # Test basic setup
        logger = setup_logging("INFO")
        assert logger is not None
        assert logger.level == 20  # INFO level
        
        # Test with different levels
        debug_logger = setup_logging("DEBUG")
        assert debug_logger.level == 10  # DEBUG level
    
    def test_file_handlers_structure(self):
        """Test the file handlers utility structure."""
        from benchmark.utils.file_handlers import save_results_to_csv
        
        # Test function exists
        assert callable(save_results_to_csv)
        
        # Test function signature
        import inspect
        sig = inspect.signature(save_results_to_csv)
        assert 'results' in sig.parameters
        assert 'output_dir' in sig.parameters


class TestPackageIntegration:
    """Test the integration between package components."""
    
    def test_package_import_chain(self):
        """Test that the package import chain works correctly."""
        # Test main package import
        from benchmark import ArgumentMiningBenchmark, BenchmarkResult
        assert ArgumentMiningBenchmark is not None
        assert BenchmarkResult is not None
        
        # Test that core components are accessible
        from benchmark.core import benchmark, results
        assert benchmark is not None
        assert results is not None
        
        # Test that submodules are accessible
        from benchmark.implementations import base
        from benchmark.tasks import base as task_base
        from benchmark.data import loader
        from benchmark.metrics import evaluator
        from benchmark.utils import logging, file_handlers
        
        assert base is not None
        assert task_base is not None
        assert loader is not None
        assert evaluator is not None
        assert logging is not None
        assert file_handlers is not None
    
    def test_component_relationships(self):
        """Test that components can work together."""
        from benchmark.core.results import BenchmarkResult
        from benchmark.tasks.adu_extraction import ADUExtractionTask
        
        # Test that task can create results
        task = ADUExtractionTask()
        result = BenchmarkResult(
            task_name=task.name,
            implementation_name="test_impl",
            sample_id="0",
            execution_date="2024-01-01",
            metrics={},
            performance={},
            predictions=None,
            ground_truth=None
        )
        
        assert result.task_name == task.name
        assert isinstance(result, BenchmarkResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
