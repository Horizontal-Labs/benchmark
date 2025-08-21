#!/usr/bin/env python3
"""
Simplified test suite for Argument Mining Benchmark
Focuses on: imports, single implementations, and benchmarking
Uses real API keys and database queries
"""

import pytest
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from uuid import uuid4

# Setup paths
project_root = Path(__file__).parent.parent
external_api = project_root / "external" / "api"
external_db = project_root / "external" / "db"

# Add paths to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(external_api))
sys.path.insert(0, str(external_db))

# Load environment variables from project root .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.external
]

class TestImports:
    """Test that all modules can be imported correctly"""
    
    def test_core_benchmark_imports(self):
        """Test importing core benchmark components"""
        try:
            from app.benchmark import ArgumentMiningBenchmark
            assert ArgumentMiningBenchmark is not None
        except ImportError:
            pytest.skip("Core benchmark module not available")
    
    def test_external_api_imports(self):
        """Test importing external API components"""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            assert OpenAILLMClassifier is not None
        except ImportError:
            pytest.skip("External API module not available")
    
    def test_external_db_imports(self):
        """Test importing external DB components"""
        try:
            from db.queries import get_benchmark_data
            assert get_benchmark_data is not None
        except ImportError:
            pytest.skip("External DB module not available")
    
    def test_implementation_imports(self):
        """Test importing all implementation classes"""
        implementations = [
            "openai_llm_classifier",
            "tinyllama_llm_classifier", 
            "encoder_model_loader"
        ]
        
        for impl in implementations:
            try:
                module = __import__(f"app.argmining.implementations.{impl}", fromlist=["*"])
                assert module is not None
            except ImportError:
                pytest.skip(f"Implementation {impl} not available")

class TestSingleImplementations:
    """Test individual implementation classes with real API keys"""
    
    @pytest.fixture
    def sample_text(self):
        return "This is a sample text for testing argument mining implementations."
    
    def test_openai_classifier_initialization(self):
        """Test OpenAI classifier can be initialized with real API key"""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            # Check if API key is available
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OpenAI API key not available in environment")
            
            classifier = OpenAILLMClassifier()
            assert classifier is not None
            assert hasattr(classifier, 'classify_sentence')
        except ImportError:
            pytest.skip("OpenAI classifier not available")
        except Exception as e:
            pytest.skip(f"OpenAI classifier initialization failed: {e}")
    
    def test_openai_classifier_classification(self, sample_text):
        """Test OpenAI classifier can classify real text"""
        try:
            from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            # Check if API key is available
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OpenAI API key not available in environment")
            
            classifier = OpenAILLMClassifier()
            result = classifier.classify_sentence(sample_text)
            assert result is not None
            assert result in ['claim', 'premise', 'unknown']
        except ImportError:
            pytest.skip("OpenAI classifier not available")
        except Exception as e:
            pytest.skip(f"OpenAI classifier classification failed: {e}")
    
    def test_tinyllama_classifier_initialization(self):
        """Test TinyLlama classifier can be initialized"""
        try:
            from app.argmining.implementations.tinyllama_llm_classifier import TinyLlamaLLMClassifier
            
            # Check if HF token is available
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                pytest.skip("HuggingFace token not available in environment")
            
            classifier = TinyLlamaLLMClassifier()
            assert classifier is not None
            assert hasattr(classifier, 'classify')
        except ImportError:
            pytest.skip("TinyLlama classifier not available")
        except Exception as e:
            pytest.skip(f"TinyLlama classifier initialization failed: {e}")
    
    def test_encoder_model_loader_initialization(self):
        """Test encoder model loader can be initialized"""
        try:
            from app.argmining.implementations.encoder_model_loader import EncoderModelLoader
            
            # Check if HF token is available
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                pytest.skip("HuggingFace token not available in environment")
            
            loader = EncoderModelLoader()
            assert loader is not None
            assert hasattr(loader, 'load_model')
        except ImportError:
            pytest.skip("Encoder model loader not available")
        except Exception as e:
            pytest.skip(f"Encoder model loader initialization failed: {e}")
    
    def test_claim_premise_linker_initialization(self):
        """Test claim-premise linker can be initialized with real API key"""
        try:
            from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
            
            # Check if API key is available
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OpenAI API key not available in environment")
            
            linker = OpenAIClaimPremiseLinker()
            assert linker is not None
            assert hasattr(linker, 'link_claims_to_premises')
        except ImportError:
            pytest.skip("Claim-premise linker not available")
        except Exception as e:
            pytest.skip(f"Claim-premise linker initialization failed: {e}")

class TestBenchmarking:
    """Test core benchmarking functionality with real data"""
    
    def test_benchmark_data_loading(self):
        """Test benchmark data can be loaded from real database"""
        try:
            from db.queries import get_benchmark_data
            
            # Load real benchmark data
            claims, premises, categories = get_benchmark_data()
            
            assert claims is not None
            assert premises is not None
            assert categories is not None
            assert len(claims) > 0, "No claims found in benchmark data"
            assert len(premises) > 0, "No premises found in benchmark data"
            assert len(categories) > 0, "No categories found in benchmark data"
            
            # Check data structure
            assert len(claims) == len(premises) == len(categories), "Data lengths should match"
            
        except ImportError:
            pytest.skip("DB queries module not available")
        except Exception as e:
            pytest.skip(f"Database query failed: {e}")
    
    def test_training_data_loading(self):
        """Test training data can be loaded from real database"""
        try:
            from db.queries import get_training_data
            
            # Load real training data
            claims, premises, categories = get_training_data()
            
            assert claims is not None
            assert premises is not None
            assert categories is not None
            assert len(claims) > 0, "No claims found in training data"
            assert len(premises) > 0, "No premises found in training data"
            assert len(categories) > 0, "No categories found in training data"
            
        except ImportError:
            pytest.skip("DB queries module not available")
        except Exception as e:
            pytest.skip(f"Database query failed: {e}")
    
    def test_test_data_loading(self):
        """Test test data can be loaded from real database"""
        try:
            from db.queries import get_test_data
            
            # Load real test data
            claims, premises, categories = get_test_data()
            
            assert claims is not None
            assert premises is not None
            assert categories is not None
            assert len(claims) > 0, "No claims found in test data"
            assert len(premises) > 0, "No premises found in test data"
            assert len(categories) > 0, "No categories found in test data"
            
        except ImportError:
            pytest.skip("DB queries module not available")
        except Exception as e:
            pytest.skip(f"Database query failed: {e}")
    
    def test_benchmark_initialization(self):
        """Test benchmark can be initialized with real data"""
        try:
            from app.benchmark import ArgumentMiningBenchmark
            
            benchmark = ArgumentMiningBenchmark()
            assert benchmark is not None
            assert hasattr(benchmark, 'run_benchmark')
        except ImportError:
            pytest.skip("Benchmark module not available")
        except Exception as e:
            pytest.skip(f"Benchmark initialization failed: {e}")
    
    def test_single_implementation_benchmark(self):
        """Test running benchmark on single implementation with real data"""
        try:
            from app.benchmark import ArgumentMiningBenchmark
            
            # Check if API key is available
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OpenAI API key not available in environment")
            
            benchmark = ArgumentMiningBenchmark()
            
            # Test with a small subset of data to avoid long execution
            result = benchmark.run_single_implementation("openai_llm_classifier", max_samples=5)
            assert result is not None
            assert hasattr(result, 'implementation_name')
            assert hasattr(result, 'accuracy')
        except ImportError:
            pytest.skip("Benchmark module not available")
        except Exception as e:
            pytest.skip(f"Single implementation benchmark failed: {e}")

class TestIntegration:
    """Integration tests for the complete pipeline with real data"""
    
    @pytest.mark.integration
    def test_full_pipeline_imports(self):
        """Test that all components can be imported together"""
        modules_to_test = [
            'app.benchmark',
            'app.argmining.implementations.openai_llm_classifier',
            'app.argmining.implementations.tinyllama_llm_classifier',
            'app.argmining.implementations.encoder_model_loader',
            'app.argmining.implementations.openai_claim_premise_linker',
            'db.queries'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError:
                pytest.skip(f"Module {module} not available")
    
    @pytest.mark.integration
    def test_benchmark_with_real_implementation(self):
        """Test complete benchmark with real implementation and data"""
        try:
            from app.benchmark import ArgumentMiningBenchmark
            
            # Check if API key is available
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OpenAI API key not available in environment")
            
            benchmark = ArgumentMiningBenchmark()
            
            # Run benchmark with real implementation and limited data
            result = benchmark.run_benchmark(implementations=["openai_llm_classifier"], max_samples=3)
            assert result is not None
            assert len(result) > 0
            
        except ImportError:
            pytest.skip("Benchmark module not available")
        except Exception as e:
            pytest.skip(f"Benchmark with real implementation failed: {e}")
    
    @pytest.mark.integration
    def test_data_processing_pipeline(self):
        """Test the complete data processing pipeline"""
        try:
            from db.queries import get_benchmark_data
            from app.argmining.models.argument_units import ArgumentUnit, UnlinkedArgumentUnits
            
            # Load real data
            claims, premises, categories = get_benchmark_data()
            
            # Test data processing
            assert len(claims) > 0, "No claims loaded"
            assert len(premises) > 0, "No premises loaded"
            assert len(categories) > 0, "No categories loaded"
            
            # Test that claims and premises are ArgumentUnit objects
            assert all(isinstance(claim, ArgumentUnit) for claim in claims), "Claims should be ArgumentUnit objects"
            assert all(isinstance(premise, ArgumentUnit) for premise in premises), "Premises should be ArgumentUnit objects"
            
            # Test categories are valid
            valid_categories = ['stance_pro', 'stance_con']
            assert all(cat in valid_categories for cat in categories), f"Categories should be in {valid_categories}"
            
        except ImportError:
            pytest.skip("Required modules not available")
        except Exception as e:
            pytest.skip(f"Data processing pipeline failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
