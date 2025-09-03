#!/usr/bin/env python3
"""
Tests for the refactored Argument Mining Benchmark package.

This module tests the new modular structure and ensures all components work correctly.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Import the refactored benchmark package
from benchmark import ArgumentMiningBenchmark, BenchmarkResult
from benchmark.core.results import BenchmarkResult as CoreBenchmarkResult
from benchmark.implementations.base import BaseImplementation
from benchmark.tasks.base import BaseTask
from benchmark.data import DataLoader
from benchmark.metrics import MetricsEvaluator
from benchmark.utils import setup_logging, save_results_to_csv


class TestRefactoredBenchmarkPackage:
    """Test the refactored benchmark package structure and functionality."""
    
    def test_package_imports(self):
        """Test that all package modules can be imported correctly."""
        # Test main package
        from benchmark import ArgumentMiningBenchmark, BenchmarkResult
        assert ArgumentMiningBenchmark is not None
        assert BenchmarkResult is not None
        
        # Test core module
        from benchmark.core import ArgumentMiningBenchmark as CoreBenchmark
        assert CoreBenchmark is not None
        
        # Test implementations module
        from benchmark.implementations import (
            BaseImplementation,
            OpenAIImplementation,
            TinyLlamaImplementation,
            ModernBERTImplementation,
            DeBERTaImplementation
        )
        assert BaseImplementation is not None
        
        # Test tasks module
        from benchmark.tasks import (
            BaseTask,
            ADUExtractionTask,
            StanceClassificationTask,
            ClaimPremiseLinkingTask
        )
        assert BaseTask is not None
        
        # Test data module
        from benchmark.data import DataLoader
        assert DataLoader is not None
        
        # Test metrics module
        from benchmark.metrics import MetricsEvaluator
        assert MetricsEvaluator is not None
        
        # Test utils module
        from benchmark.utils import setup_logging, save_results_to_csv
        assert setup_logging is not None
        assert save_results_to_csv is not None
    
    def test_benchmark_result_creation(self):
        """Test that BenchmarkResult can be created correctly."""
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
        assert result.metrics["accuracy"] == 0.95
        assert result.performance["inference_time"] == 0.1
        assert result.success is True
        assert result.error_message == ""
    
    def test_benchmark_result_with_error(self):
        """Test that BenchmarkResult can handle errors correctly."""
        result = BenchmarkResult(
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
        
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.predictions is None
    
    def test_base_implementation_abstract(self):
        """Test that BaseImplementation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseImplementation("test")
    
    def test_base_task_abstract(self):
        """Test that BaseTask cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTask("test")
    
    def test_metrics_evaluator(self):
        """Test the MetricsEvaluator functionality."""
        evaluator = MetricsEvaluator()
        
        # Test classification metrics
        y_true = ["pro", "con", "pro", "neutral"]
        y_pred = ["pro", "con", "pro", "pro"]
        
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Test token-level metrics
        pred_tokens = ["climate", "change", "real"]
        gt_tokens = ["climate", "change", "is", "real"]
        
        token_metrics = evaluator.calculate_token_level_metrics(pred_tokens, gt_tokens)
        assert "precision" in token_metrics
        assert "recall" in token_metrics
        assert "f1" in token_metrics
    
    def test_logging_setup(self):
        """Test the logging utility setup."""
        logger = setup_logging("INFO")
        assert logger is not None
        assert logger.level == 20  # INFO level
    
    @patch('benchmark.utils.file_handlers.Path')
    @patch('benchmark.utils.file_handlers.pd.DataFrame')
    def test_save_results_to_csv(self, mock_df, mock_path):
        """Test the CSV saving functionality."""
        # Mock the results
        mock_results = {
            'test_task': [
                BenchmarkResult(
                    task_name="test_task",
                    implementation_name="test_impl",
                    sample_id="0",
                    execution_date="2024-01-01",
                    metrics={"accuracy": 0.95},
                    performance={"inference_time": 0.1},
                    predictions=["test_prediction"],
                    ground_truth=["test_ground_truth"]
                )
            ]
        }
        
        # Mock the Path operations
        mock_output_path = Mock()
        mock_path.return_value = mock_output_path
        mock_output_path.mkdir.return_value = None
        mock_output_path.__truediv__ = lambda self, other: Mock()  # Mock the / operator
        
        # Mock the DataFrame operations
        mock_dataframe = Mock()
        mock_df.return_value = mock_dataframe
        mock_dataframe.to_csv.return_value = None
        
        # Test the function
        result = save_results_to_csv(mock_results)
        assert result == mock_output_path


class TestRefactoredBenchmarkClass:
    """Test the main ArgumentMiningBenchmark class."""
    
    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock benchmark instance for testing."""
        with patch('benchmark.core.benchmark.DataLoader') as mock_loader, \
             patch('benchmark.core.benchmark.OpenAIImplementation') as mock_openai, \
             patch('benchmark.core.benchmark.TinyLlamaImplementation') as mock_tinyllama, \
             patch('benchmark.core.benchmark.ModernBERTImplementation') as mock_modernbert, \
             patch('benchmark.core.benchmark.DeBERTaImplementation') as mock_deberta:
            
            # Mock the data loader
            mock_loader_instance = Mock()
            mock_loader_instance.load_benchmark_data.return_value = ([], [], [])
            mock_loader.return_value = mock_loader_instance
            
            # Mock the implementations
            mock_openai_instance = Mock()
            mock_openai_instance.initialize.return_value = True
            mock_openai_instance.name = "openai"
            mock_openai_instance.supports_task.return_value = True
            mock_openai.return_value = mock_openai_instance
            
            mock_tinyllama_instance = Mock()
            mock_tinyllama_instance.initialize.return_value = True
            mock_tinyllama_instance.name = "tinyllama"
            mock_tinyllama_instance.supports_task.return_value = True
            mock_tinyllama.return_value = mock_tinyllama_instance
            
            mock_modernbert_instance = Mock()
            mock_modernbert_instance.initialize.return_value = True
            mock_modernbert_instance.name = "modernbert"
            mock_modernbert_instance.supports_task.return_value = True
            mock_modernbert.return_value = mock_modernbert_instance
            
            mock_deberta_instance = Mock()
            mock_deberta_instance.initialize.return_value = True
            mock_deberta_instance.name = "deberta"
            mock_deberta_instance.supports_task.return_value = True
            mock_deberta.return_value = mock_deberta_instance
            
            # Create benchmark instance
            benchmark = ArgumentMiningBenchmark(max_samples=10)
            return benchmark
    
    def test_benchmark_initialization(self, mock_benchmark):
        """Test that the benchmark initializes correctly."""
        assert mock_benchmark.max_samples == 10
        assert mock_benchmark.execution_date is not None
        assert len(mock_benchmark.implementations) > 0
        assert len(mock_benchmark.tasks) > 0
    
    def test_benchmark_data_loading(self, mock_benchmark):
        """Test that the benchmark loads data correctly."""
        assert 'adu_extraction' in mock_benchmark.data
        assert 'stance_classification' in mock_benchmark.data
        assert 'claim_premise_linking' in mock_benchmark.data
    
    def test_benchmark_task_execution(self, mock_benchmark):
        """Test that individual tasks can be executed."""
        # Mock the task execution
        with patch.object(mock_benchmark.tasks['adu_extraction'], 'run_benchmark') as mock_run:
            mock_run.return_value = []
            results = mock_benchmark.run_single_task('adu_extraction')
            assert isinstance(results, list)
    
    def test_benchmark_implementation_execution(self, mock_benchmark):
        """Test that individual implementations can be executed."""
        # Mock the implementation execution
        with patch.object(mock_benchmark, 'run_single_task') as mock_run:
            mock_run.return_value = []
            results = mock_benchmark.run_single_implementation('openai')
            assert isinstance(results, dict)


class TestRefactoredTaskClasses:
    """Test the individual task classes."""
    
    def test_adu_extraction_task(self):
        """Test the ADU extraction task."""
        from benchmark.tasks.adu_extraction import ADUExtractionTask
        
        task = ADUExtractionTask()
        assert task.name == "adu_extraction"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert 'text' in prepared_data[0]
        assert 'ground_truth' in prepared_data[0]
    
    def test_stance_classification_task(self):
        """Test the stance classification task."""
        from benchmark.tasks.stance_classification import StanceClassificationTask
        
        task = StanceClassificationTask()
        assert task.name == "stance_classification"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert 'text' in prepared_data[0]
        assert 'ground_truth' in prepared_data[0]
    
    def test_claim_premise_linking_task(self):
        """Test the claim-premise linking task."""
        from benchmark.tasks.claim_premise_linking import ClaimPremiseLinkingTask
        
        task = ClaimPremiseLinkingTask()
        assert task.name == "claim_premise_linking"
        
        # Test data preparation
        raw_data = (["claim1", "claim2"], ["premise1"], ["topic1"])
        prepared_data = task.prepare_data(raw_data)
        assert len(prepared_data) == 2
        assert 'text' in prepared_data[0]
        assert 'ground_truth' in prepared_data[0]


class TestRefactoredImplementationClasses:
    """Test the individual implementation classes."""
    
    def test_openai_implementation(self):
        """Test the OpenAI implementation."""
        from benchmark.implementations.openai import OpenAIImplementation
        
        # Mock the OpenAI components
        with patch('benchmark.implementations.openai.OPENAI_AVAILABLE', True), \
             patch('benchmark.implementations.openai.OpenAILLMClassifier') as mock_classifier, \
             patch('benchmark.implementations.openai.OpenAIClaimPremiseLinker') as mock_linker, \
             patch.dict(os.environ, {'OPEN_AI_KEY': 'test_key'}):
            
            mock_classifier.return_value = Mock()
            mock_linker.return_value = Mock()
            
            impl = OpenAIImplementation()
            assert impl.name == "openai"
            assert impl.initialize() is True
            assert impl.is_available() is True
    
    def test_tinyllama_implementation(self):
        """Test the TinyLlama implementation."""
        from benchmark.implementations.tinyllama import TinyLlamaImplementation
        
        # Mock the TinyLlama components
        with patch('benchmark.implementations.tinyllama.TINYLLAMA_AVAILABLE', True), \
             patch('benchmark.implementations.tinyllama.TinyLLamaLLMClassifier') as mock_classifier:
            
            mock_classifier.return_value = Mock()
            
            impl = TinyLlamaImplementation()
            assert impl.name == "tinyllama"
            assert impl.initialize() is True
            assert impl.is_available() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

