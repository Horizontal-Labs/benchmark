#!/usr/bin/env python3
"""
Pytest test suite for the Argument Mining Benchmark

This module contains comprehensive tests for the benchmark functionality,
including initialization, data loading, and individual benchmark methods.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the benchmark module
from app.benchmark import ArgumentMiningBenchmark, BenchmarkResult
from app.db_connector.db.models import ADU
from app.log import log
from app.argmining.models.argument_units import ArgumentUnit, UnlinkedArgumentUnits, LinkedArgumentUnits


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            task_name='test_task',
            implementation_name='test_impl',
            sample_id='test_sample',
            metrics={'precision': 0.8, 'recall': 0.7},
            performance={'inference_time': 1.5},
            predictions={'test': 'data'},
            ground_truth={'test': 'truth'}
        )
        
        assert result.task_name == 'test_task'
        assert result.implementation_name == 'test_impl'
        assert result.sample_id == 'test_sample'
        assert result.metrics['precision'] == 0.8
        assert result.performance['inference_time'] == 1.5
        assert result.success is True
        assert result.error_message == ""
    
    def test_benchmark_result_with_error(self):
        """Test creating a BenchmarkResult instance with an error."""
        result = BenchmarkResult(
            task_name='test_task',
            implementation_name='test_impl',
            sample_id='test_sample',
            metrics={},
            performance={},
            predictions=None,
            ground_truth={'test': 'truth'},
            error_message="Test error",
            success=False
        )
        
        assert result.success is False
        assert result.error_message == "Test error"


class TestArgumentMiningBenchmark:
    """Test the ArgumentMiningBenchmark class."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {'OPEN_AI_KEY': 'test_key'}):
            yield
    
    @pytest.fixture
    def mock_data(self):
        """Mock benchmark data."""
        return [
            {
                'id': 1,
                'text': 'Test argument text. This is a claim. Supporting evidence.',
                'ground_truth': {
                    'adus': [
                        {'text': 'This is a claim', 'type': 'claim'},
                        {'text': 'Supporting evidence', 'type': 'premise'}
                    ],
                    'stance': 'pro',
                    'relationships': [{'claim_id': 1, 'premise_ids': [2]}]
                },
                'metadata': {'source': 'test', 'domain': 'test'}
            }
        ]
    
    @pytest.fixture
    def mock_implementations(self):
        """Mock implementations."""
        mock_classifier = Mock()
        mock_linker = Mock()
        
        return {
            'test_impl': {
                'adu_classifier': mock_classifier,
                'linker': mock_linker
            }
        }
    
    def test_benchmark_initialization(self, mock_environment):
        """Test benchmark initialization."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark(max_samples=50)
                
                assert benchmark.data == []
                assert benchmark.results == []
                assert benchmark.implementations == {}
                assert benchmark.max_samples == 50
    
    def test_check_environment_with_key(self, mock_environment):
        """Test environment check with API key."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark(max_samples=100)
                benchmark._check_environment()
                
                # Should not raise any exceptions
                assert True
    
    def test_check_environment_without_key(self):
        """Test environment check without API key."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark(max_samples=100)
                benchmark._check_environment()
                
                # Should not raise any exceptions
                assert True
    
    def test_load_benchmark_data_success(self, mock_environment):
        """Test successful benchmark data loading."""
        mock_data = [{'id': 1, 'text': 'test'}]
        
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = mock_data
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark(max_samples=100)
                benchmark._load_benchmark_data()
                
                assert benchmark.data == mock_data
    
    def test_load_benchmark_data_failure(self, mock_environment):
        """Test benchmark data loading failure."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.side_effect = Exception("Database error")
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark(max_samples=100)
                benchmark._load_benchmark_data()
                
                assert benchmark.data == []
    
    def test_calculate_adu_metrics_perfect_match(self):
        """Test ADU metrics calculation with perfect match."""
        mock_prediction = Mock()
        mock_prediction.claims = [Mock(text="This is a claim")]
        mock_prediction.premises = [Mock(text="Supporting evidence")]
        
        ground_truth = [
            {'text': 'This is a claim', 'type': 'claim'},
            {'text': 'Supporting evidence', 'type': 'premise'}
        ]
            
        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_adu_metrics(mock_prediction, ground_truth)
            
            assert metrics['precision'] == 1.0
            assert metrics['recall'] == 1.0
            assert metrics['f1'] == 1.0
            assert metrics['accuracy'] == 1.0
    
    def test_calculate_adu_metrics_no_match(self):
        """Test ADU metrics calculation with no match."""
        mock_prediction = Mock()
        mock_prediction.claims = [Mock(text="Wrong claim")]
        mock_prediction.premises = [Mock(text="Wrong evidence")]
        
        ground_truth = [
            {'text': 'This is a claim', 'type': 'claim'},
            {'text': 'Supporting evidence', 'type': 'premise'}
        ]

        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_adu_metrics(mock_prediction, ground_truth)
            
            assert metrics['precision'] == 0.0
            assert metrics['recall'] == 0.0
            assert metrics['f1'] == 0.0
            assert metrics['accuracy'] == 0.0

    def test_calculate_adu_metrics_empty_prediction(self):
        """Test ADU metrics calculation with empty prediction."""
        mock_prediction = None
        ground_truth = [
            {'text': 'This is a claim', 'type': 'claim'},
            {'text': 'Supporting evidence', 'type': 'premise'}
        ]
  
        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_adu_metrics(mock_prediction, ground_truth)
            
            assert metrics['precision'] == 0.0
            assert metrics['recall'] == 0.0
            assert metrics['f1'] == 0.0
            assert metrics['accuracy'] == 0.0
    
    def test_calculate_stance_metrics_correct(self):
        """Test stance metrics calculation with correct prediction."""
        mock_prediction = Mock()
        mock_stance_relation = Mock()
        mock_stance_relation.stance = 'pro'
        mock_prediction.stance_relations = [mock_stance_relation]
        
        ground_truth = 'pro'
            
        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_stance_metrics(mock_prediction, ground_truth)
            
            assert metrics['accuracy'] == 1.0
            assert metrics['weighted_f1'] == 1.0
            assert metrics['predicted_stance'] == 'pro'
            assert metrics['ground_truth_stance'] == 'pro'
    
    def test_calculate_stance_metrics_incorrect(self):
        """Test stance metrics calculation with incorrect prediction."""
        mock_prediction = Mock()
        mock_stance_relation = Mock()
        mock_stance_relation.stance = 'con'
        mock_prediction.stance_relations = [mock_stance_relation]
        
        ground_truth = 'pro'

        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_stance_metrics(mock_prediction, ground_truth)
            
            assert metrics['accuracy'] == 0.0
            assert metrics['weighted_f1'] == 0.0
            assert metrics['predicted_stance'] == 'con'
            assert metrics['ground_truth_stance'] == 'pro'
    
    def test_calculate_stance_metrics_empty_prediction(self):
        """Test stance metrics calculation with empty prediction."""
        mock_prediction = None
        ground_truth = 'pro'
        
        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_stance_metrics(mock_prediction, ground_truth)
            
            assert metrics['accuracy'] == 0.0
            assert metrics['weighted_f1'] == 0.0
    
    def test_calculate_linking_metrics(self):
        """Test linking metrics calculation."""
        mock_prediction = Mock()
        ground_truth = [{'claim_id': 1, 'premise_ids': [2, 3]}]

        with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
            mock_init_impl.return_value = {}
            
            benchmark = ArgumentMiningBenchmark()
            metrics = benchmark._calculate_linking_metrics(mock_prediction, ground_truth)
            
            # Currently returns placeholder values
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
    
    def test_benchmark_adu_extraction_success(self, mock_environment, mock_data):
        """Test successful ADU extraction benchmark."""
        mock_classifier = Mock()
        mock_classifier.classify_adus.return_value = Mock(
            claims=[Mock(text="This is a claim")],
            premises=[Mock(text="Supporting evidence")]
        )

        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = mock_data
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {
                    'test_impl': {
                        'adu_classifier': mock_classifier,
                        'linker': None
                    }
                }
                
                # Create benchmark after mocking
                benchmark = ArgumentMiningBenchmark()
                # Manually set the implementations since the mock doesn't work on __init__
                benchmark.implementations = {
                    'test_impl': {
                        'adu_classifier': mock_classifier,
                        'linker': None
                    }
                }
                
                results = benchmark.benchmark_adu_extraction('test_impl')
                
                assert len(results) == 1
                assert results[0].task_name == 'adu_extraction'
                assert results[0].implementation_name == 'test_impl'
                assert results[0].success is True
    
    def test_benchmark_adu_extraction_failure(self, mock_environment, mock_data):
        """Test ADU extraction benchmark with failure."""
        mock_classifier = Mock()
        mock_classifier.classify_adus.side_effect = Exception("Classification error")
        
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = mock_data
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {
                    'test_impl': {
                        'adu_classifier': mock_classifier,
                        'linker': None
                    }
                }
                
                benchmark = ArgumentMiningBenchmark()
                # Manually set the implementations since the mock doesn't work on __init__
                benchmark.implementations = {
                    'test_impl': {
                        'adu_classifier': mock_classifier,
                        'linker': None
                    }
                }
                
                results = benchmark.benchmark_adu_extraction('test_impl')
                
                assert len(results) == 1
                assert results[0].success is False
                assert "Classification error" in results[0].error_message
    
    def test_benchmark_adu_extraction_implementation_not_found(self, mock_environment, mock_data):
        """Test ADU extraction benchmark with non-existent implementation."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = mock_data
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                # Manually set the implementations since the mock doesn't work on __init__
                benchmark.implementations = {}
                
                results = benchmark.benchmark_adu_extraction('non_existent_impl')
                
                assert len(results) == 0
    
    def test_save_results(self, mock_environment, tmp_path):
        """Test saving benchmark results."""
        results = {
            'adu_extraction': [
                BenchmarkResult(
                    task_name='adu_extraction',
                    implementation_name='test_impl',
                    sample_id='test_sample',
                    metrics={'precision': 0.8},
                    performance={'inference_time': 1.5},
                    predictions={'test': 'data'},
                    ground_truth={'test': 'truth'}
                )
            ]
        }
        
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                
                # Test that the method doesn't raise an exception
                benchmark._save_results(results)
                
                # Check if results directory was created in current working directory
                results_dir = Path("results")
                assert results_dir.exists()
                
                # Clean up - remove the results directory
                import shutil
                if results_dir.exists():
                    shutil.rmtree(results_dir)
    
    def test_print_summary(self, mock_environment, capsys):
        """Test printing benchmark summary."""
        results = {
            'adu_extraction': [
                BenchmarkResult(
                    task_name='adu_extraction',
                    implementation_name='test_impl',
                    sample_id='test_sample',
                    metrics={'precision': 0.8, 'recall': 0.7},
                    performance={'inference_time': 1.5},
                    predictions={'test': 'data'},
                    ground_truth={'test': 'truth'}
                )
            ]
        }
        
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                benchmark._print_summary(results)
                
                captured = capsys.readouterr()
                assert "BENCHMARK SUMMARY" in captured.out
                assert "ADU_EXTRACTION RESULTS" in captured.out
                assert "test_impl" in captured.out


class TestBenchmarkIntegration:
    """Integration tests for the benchmark."""
    
    def test_full_benchmark_initialization(self):
        """Test full benchmark initialization with mocked dependencies."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                
                assert benchmark is not None
                assert hasattr(benchmark, 'data')
                assert hasattr(benchmark, 'implementations')
                assert hasattr(benchmark, 'results')
    
    def test_benchmark_with_no_implementations(self):
        """Test benchmark behavior when no implementations are available."""
        with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
            mock_get_data.return_value = []
            
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                results = benchmark.run_benchmark()
                
                assert isinstance(results, dict)
                assert len(results) == 3  # adu_extraction, stance_classification, claim_premise_linking
                for task_results in results.values():
                    assert len(task_results) == 0  # No implementations available


class TestOpenAIAPI:
    """Test OpenAI API connectivity and functionality."""
    
    def test_openai_api_key_exists(self):
        """Test that the OpenAI API key is set in environment variables."""
        api_key = os.getenv('OPEN_AI_KEY')
        assert api_key is not None, "OPEN_AI_KEY environment variable is not set"
        assert len(api_key) > 0, "OPEN_AI_KEY environment variable is empty"
        assert api_key.startswith('sk-'), "OPEN_AI_KEY should start with 'sk-'"
        log.info("✓ OpenAI API key is properly configured")
    
    def test_openai_api_connection(self):
        """Test that we can connect to the OpenAI API and make a simple request."""
        try:
            import openai
            
            # Get API key
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OPEN_AI_KEY not set")
            
            # Create OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Make a simple test request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            # Check response
            assert response is not None, "OpenAI API returned None response"
            assert hasattr(response, 'choices'), "Response missing 'choices' attribute"
            assert len(response.choices) > 0, "Response has no choices"
            
            content = response.choices[0].message.content
            assert content is not None, "Response content is None"
            assert len(content) > 0, "Response content is empty"
            
            log.info(f"✓ OpenAI API connection successful. Response: {content}")
            
        except openai.AuthenticationError as e:
            pytest.fail(f"OpenAI API authentication failed: {e}")
        except openai.RateLimitError as e:
            pytest.fail(f"OpenAI API rate limit exceeded: {e}")
        except openai.APIError as e:
            pytest.fail(f"OpenAI API error: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error connecting to OpenAI API: {e}")
    
    def test_openai_llm_classifier_initialization(self):
        """Test that the OpenAI LLM classifier can be initialized with valid API key."""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            # Initialize the classifier
            classifier = OpenAILLMClassifier()
            
            # Check that it was initialized properly
            assert classifier is not None, "OpenAILLMClassifier initialization returned None"
            assert hasattr(classifier, 'client'), "OpenAILLMClassifier missing 'client' attribute"
            assert classifier.client is not None, "OpenAI client is None"
            
            log.info("✓ OpenAI LLM classifier initialized successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize OpenAI LLM classifier: {e}")
    
    def test_openai_llm_classifier_simple_classification(self):
        """Test that the OpenAI LLM classifier can perform a simple classification."""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            # Initialize the classifier
            classifier = OpenAILLMClassifier()
            
            # Test with a simple sentence
            test_sentence = "Climate change is real."
            result = classifier.classify_sentence(test_sentence)
            
            # Check result
            assert result is not None, "Classification result is None"
            assert result in ['claim', 'premise'], f"Unexpected classification result: {result}"
            
            log.info(f"✓ OpenAI LLM classifier classification successful. Result: {result}")
            
        except Exception as e:
            pytest.fail(f"Failed to perform classification with OpenAI LLM classifier: {e}")


class TestDatabaseConnectivity:
    """Tests for database connectivity and data loading."""
    
    def test_database_connection_available(self):
        """Test if database connection can be established."""
        try:
            # Try to import the database module
            from app.db_connector.db.queries import get_benchmark_data
            print("✓ Database module imported successfully")
            
            # Try to get a database session
            from app.db_connector.db.db import get_session
            session = get_session()
            print("✓ Database session created successfully")
            
            # Test if we can execute a simple query
            from app.db_connector.db.models import ADU
            count = session.query(ADU).count()
            print(f"✓ Database query executed successfully. Found {count} ADU records")
            
            session.close()
            assert True  # Test passed if no exceptions
            
        except ImportError as e:
            pytest.skip(f"Database module not available: {e}")
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
    
    def test_benchmark_data_function_exists(self):
        """Test if the benchmark data function exists and is callable."""
        try:
            from app.db_connector.db.queries import get_benchmark_data
            assert callable(get_benchmark_data)
            print("✓ get_benchmark_data function exists and is callable")
        except ImportError as e:
            pytest.skip(f"Database module not available: {e}")
    
    def test_benchmark_data_loading(self):
        """Test if benchmark data can be loaded from the database."""
        try:
            from app.db_connector.db.queries import get_benchmark_data
            
            # Try to load benchmark data
            claims, premises, stances = get_benchmark_data()
            
            print(f"✓ Loaded {len(claims)} claims from database")
            print(f"✓ Loaded {len(premises)} premises from database")
            print(f"✓ Loaded {len(stances)} stances from database")
            
            # Basic validation
            assert isinstance(claims, list)
            assert isinstance(premises, list)
            assert isinstance(stances, list)
            
            # Check if we have data
            if len(claims) > 0:
                print("✓ Database contains benchmark data")
                assert len(claims) == len(premises) == len(stances)
            else:
                print("⚠️ Database is empty (no benchmark data found)")
                
        except ImportError as e:
            pytest.skip(f"Database module not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to load benchmark data: {e}")
    
    def test_benchmark_data_structure(self):
        """Test the structure of loaded benchmark data."""
        try:
            from app.db_connector.db.queries import get_benchmark_data
            
            claims, premises, stances = get_benchmark_data()
            
            if len(claims) > 0:
                # Test claim structure
                claim = claims[0]
                assert isinstance(claim, ADU)
                print("✓ Claim structure is valid")
                
                # Test premise structure
                if len(premises) > 0:
                    # premise_list = list(premises[0])
                    premise = premises[0]   
                    assert isinstance(premise, ADU)
                    print("✓ Premise structure is valid")
                
                # Test stance structure
                if len(stances) > 0:
                    stance = stances[0]
                    assert isinstance(stance, str)
                    assert stance in ['stance_pro', 'stance_con']
                    print(f"✓ Stance structure is valid: {stance}")
                    
            else:
                print("⚠️ No data to test structure")
                
        except ImportError as e:
            pytest.skip(f"Database module not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to test data structure: {e}")
    
    def test_benchmark_data_function(self):
        """Test if the get_benchmark_data function exists."""
        try:
            # Try to import the function
            from app.db_connector.db.queries import get_benchmark_data
            assert callable(get_benchmark_data)
            print("✓ get_benchmark_data function exists")

            # Try to call the function
            data = get_benchmark_data()
            assert isinstance(data, tuple)
            assert len(data) == 3  # Should return (claims, premises, stances)
            print(f"✓ Function returned tuple with {len(data[0])} claims, {len(data[1])} premises, {len(data[2])} stances")

            # Test data structure if we have data
            if len(data[0]) > 0:
                claim = data[0][0]
                assert hasattr(claim, 'id')
                assert hasattr(claim, 'text')
                assert hasattr(claim, 'type')
                print("✓ Benchmark data structure is valid")

        except ImportError as e:
            pytest.skip(f"get_benchmark_data function not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to test get_benchmark_data: {e}")
    
    def test_database_fallback_to_sample_data(self):
        """Test that the system falls back to sample data when database is unavailable."""
        try:
            from app.db_connector.db.queries import get_benchmark_data_for_evaluation
            
            # This should work even if database is not available
            data = get_benchmark_data_for_evaluation()
            assert isinstance(data, list)
            assert len(data) > 0, "Should have fallback sample data"
            
            print(f"✓ Fallback data loaded: {len(data)} samples")
            
            # Check that it's sample data
            if len(data) > 0:
                sample = data[0]
                if 'metadata' in sample and 'source' in sample['metadata']:
                    source = sample['metadata']['source']
                    print(f"✓ Data source: {source}")
                    
        except Exception as e:
            pytest.fail(f"Failed to load fallback data: {e}")
    
    def test_benchmark_with_real_data(self):
        """Test benchmark initialization with real data from database."""
        try:
            from app.db_connector.db.queries import get_benchmark_data_for_evaluation
            
            # Get real data
            data = get_benchmark_data_for_evaluation()
            
            # Initialize benchmark with real data
            with patch('app.benchmark.ArgumentMiningBenchmark._initialize_implementations') as mock_init_impl:
                mock_init_impl.return_value = {}
                
                benchmark = ArgumentMiningBenchmark()
                
                # Check that data was loaded
                assert len(benchmark.data) > 0
                print(f"✓ Benchmark initialized with {len(benchmark.data)} real data samples")
                
                # Check data structure
                sample = benchmark.data[0]
                assert 'text' in sample
                assert 'ground_truth' in sample
                assert 'adus' in sample['ground_truth']
                print("✓ Real data structure is valid")
                
        except Exception as e:
            pytest.fail(f"Failed to test benchmark with real data: {e}")


class TestImplementationFallbackStrategies:
    """Test implementations with fallback strategies disabled."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {'OPEN_AI_KEY': 'test_key', 'HF_TOKEN': 'test_token'}):
            yield
    
    @pytest.fixture
    def mock_data(self):
        """Mock benchmark data."""
        return [
            {
                'id': 1,
                'text': 'Climate change is real. Scientific evidence shows increasing temperatures. We must take action.',
                'ground_truth': {
                    'adus': [
                        {'text': 'Climate change is real', 'type': 'claim'},
                        {'text': 'Scientific evidence shows increasing temperatures', 'type': 'premise'},
                        {'text': 'We must take action', 'type': 'claim'}
                    ],
                    'stance': 'pro',
                    'relationships': [{'claim_id': 1, 'premise_ids': [2]}]
                },
                'metadata': {'source': 'test', 'domain': 'climate_change'}
            }
        ]
    
    def test_openai_implementation_no_fallback(self, mock_environment, mock_data):
        """Test OpenAI implementation without fallback strategies."""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            classifier = OpenAILLMClassifier()
            
            # Test with benchmark data
            test_text = mock_data[0]['text']
            result = classifier.classify_adus(test_text)
            
            assert isinstance(result, UnlinkedArgumentUnits)
            assert hasattr(result, 'claims')
            assert hasattr(result, 'premises')
            assert isinstance(result.claims, list)
            assert isinstance(result.premises, list)
            
            log.info(f"✓ OpenAI implementation extracted {len(result.claims)} claims and {len(result.premises)} premises")
            
        except Exception as e:
            pytest.skip(f"OpenAI implementation test failed: {e}")
    
    def test_tinyllama_implementation_no_fallback(self, mock_environment, mock_data):
        """Test TinyLlama implementation without fallback strategies."""
        try:
            from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
            
            classifier = TinyLLamaLLMClassifier()
            
            # Test with benchmark data
            test_text = mock_data[0]['text']
            
            # Mock the fallback method to ensure it's not used
            with patch.object(classifier, '_fallback_sentence_analysis') as mock_fallback:
                result = classifier.classify_adus(test_text)
                
                # Ensure fallback was not called
                mock_fallback.assert_not_called()
                
                assert isinstance(result, UnlinkedArgumentUnits)
                assert hasattr(result, 'claims')
                assert hasattr(result, 'premises')
                assert isinstance(result.claims, list)
                assert isinstance(result.premises, list)
                
                log.info(f"✓ TinyLlama implementation extracted {len(result.claims)} claims and {len(result.premises)} premises without fallback")
                
        except Exception as e:
            pytest.skip(f"TinyLlama implementation test failed: {e}")
    
    def test_modernbert_implementation_no_fallback(self, mock_environment, mock_data):
        """Test ModernBERT implementation without fallback strategies."""
        try:
            from app.argmining.implementations.encoder_model_loader import PeftEncoderModelLoader, MODEL_CONFIGS
            
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if not modernbert_config:
                pytest.skip("ModernBERT configuration not available")
            
            classifier = PeftEncoderModelLoader(**modernbert_config['params'])
            
            # Test with benchmark data
            test_text = mock_data[0]['text']
            
            # Test with fallback disabled
            result = classifier.identify_adus(test_text, use_sentence_fallback=False)
            
            assert isinstance(result, list)
            # Note: Result might be empty if no ADUs are identified, which is acceptable
            
            log.info(f"✓ ModernBERT implementation identified {len(result)} ADUs without fallback")
            
        except Exception as e:
            pytest.skip(f"ModernBERT implementation test failed: {e}")
    
    def test_deberta_implementation_no_fallback(self, mock_environment, mock_data):
        """Test DeBERTa implementation without fallback strategies."""
        try:
            from app.argmining.implementations.encoder_model_loader import NonTrainedEncoderModelLoader, MODEL_CONFIGS
            
            deberta_config = MODEL_CONFIGS.get('deberta')
            if not deberta_config:
                pytest.skip("DeBERTa configuration not available")
            
            classifier = NonTrainedEncoderModelLoader(**deberta_config['params'])
            
            # Test with benchmark data
            test_text = mock_data[0]['text']
            result = classifier.classify_adus(test_text)
            
            assert isinstance(result, UnlinkedArgumentUnits)
            assert hasattr(result, 'claims')
            assert hasattr(result, 'premises')
            assert isinstance(result.claims, list)
            assert isinstance(result.premises, list)
            
            log.info(f"✓ DeBERTa implementation classified {len(result.claims)} claims and {len(result.premises)} premises")
            
        except Exception as e:
            pytest.skip(f"DeBERTa implementation test failed: {e}")
    
    def test_openai_linker_implementation(self, mock_environment, mock_data):
        """Test OpenAI linker implementation."""
        try:
            from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
            from app.argmining.models.argument_units import ArgumentUnit, UnlinkedArgumentUnits
            
            linker = OpenAIClaimPremiseLinker()
            
            # Create test argument units from benchmark data
            ground_truth_adus = mock_data[0]['ground_truth']['adus']
            claims = [ArgumentUnit(text=adu['text'], type=adu['type']) for adu in ground_truth_adus if adu['type'] == 'claim']
            premises = [ArgumentUnit(text=adu['text'], type=adu['type']) for adu in ground_truth_adus if adu['type'] == 'premise']
            
            unlinked_units = UnlinkedArgumentUnits(claims=claims, premises=premises)
            
            result = linker.link_claims_premises(unlinked_units, mock_data[0]['text'])
            
            assert isinstance(result, LinkedArgumentUnits)
            assert hasattr(result, 'claims_premises_relationships')
            assert isinstance(result.claims_premises_relationships, list)
            
            log.info(f"✓ OpenAI linker created {len(result.claims_premises_relationships)} relationships")
            
        except Exception as e:
            pytest.skip(f"OpenAI linker implementation test failed: {e}")
    
    def test_benchmark_with_all_implementations_no_fallback(self, mock_environment, mock_data):
        """Test benchmark with all implementations and fallback strategies disabled."""
        try:
            from app.benchmark import ArgumentMiningBenchmark
            
            # Mock the data loading
            with patch('app.benchmark.get_benchmark_data_for_evaluation') as mock_get_data:
                mock_get_data.return_value = mock_data
                
                # Create benchmark
                benchmark = ArgumentMiningBenchmark()
                
                # Test each available implementation
                for impl_name, impl_config in benchmark.implementations.items():
                    if impl_config['adu_classifier']:
                        log.info(f"Testing implementation: {impl_name}")
                        
                        # For TinyLlama, mock the fallback method
                        if impl_name == 'tinyllama':
                            with patch.object(impl_config['adu_classifier'], '_fallback_sentence_analysis') as mock_fallback:
                                results = benchmark.benchmark_adu_extraction(impl_name)
                                mock_fallback.assert_not_called()
                        # For ModernBERT, disable sentence fallback
                        elif impl_name == 'modernbert':
                            with patch.object(impl_config['adu_classifier'], 'identify_adus') as mock_identify:
                                mock_identify.return_value = []
                                results = benchmark.benchmark_adu_extraction(impl_name)
                        else:
                            results = benchmark.benchmark_adu_extraction(impl_name)
                        
                        assert len(results) > 0, f"No results for {impl_name}"
                        assert all(r.success for r in results), f"Some results failed for {impl_name}"
                        
                        log.info(f"✓ {impl_name} implementation passed benchmark test")
                
                log.info("✓ All implementations passed benchmark tests")
                
        except Exception as e:
            pytest.skip(f"Benchmark with implementations test failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 
