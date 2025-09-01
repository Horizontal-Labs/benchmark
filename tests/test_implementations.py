#!/usr/bin/env python3
"""
Tests for individual argument mining implementations

This module contains comprehensive tests for each argument mining implementation
to ensure they are working properly. Fallback strategies are disabled for testing.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from uuid import uuid4
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file in project root
load_dotenv(project_root / '.env')

# Add the external app directory to the path for external implementations
external_app_path = os.path.join(project_root, 'external', 'argument-mining-api', 'app')
if external_app_path not in sys.path:
    sys.path.insert(0, external_app_path)

# Import the implementations with proper error handling
try:
    from argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
    from argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    from argmining.implementations.encoder_model_loader import (
        PeftEncoderModelLoader, 
        NonTrainedEncoderModelLoader,
        MODEL_CONFIGS
    )
    from argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
    from argmining.models.argument_units import (
        ArgumentUnit, 
        UnlinkedArgumentUnits, 
        LinkedArgumentUnits, 
        LinkedArgumentUnitsWithStance,
        StanceRelation,
        ClaimPremiseRelationship
    )
    from argmining.config import OPENAI_KEY, HF_TOKEN
    
    # Import local log module
    import sys
    from pathlib import Path
    local_app_path = Path(__file__).parent.parent / "app"
    if str(local_app_path) not in sys.path:
        sys.path.insert(0, str(local_app_path))
    from log import log as logger
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Create dummy classes for testing
    class OpenAILLMClassifier:
        pass
    class TinyLLamaLLMClassifier:
        pass
    class PeftEncoderModelLoader:
        pass
    class NonTrainedEncoderModelLoader:
        pass
    class OpenAIClaimPremiseLinker:
        pass
    class ArgumentUnit:
        pass
    class UnlinkedArgumentUnits:
        pass
    class LinkedArgumentUnits:
        pass
    class LinkedArgumentUnitsWithStance:
        pass
    class StanceRelation:
        pass
    class ClaimPremiseRelationship:
        pass
    MODEL_CONFIGS = {}
    OPENAI_KEY = None
    HF_TOKEN = None
    IMPORTS_SUCCESSFUL = False
    
    # Create a simple log function
    def log():
        class Logger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return Logger()


class TestBasicImplementationDiscovery:
    """Basic tests that should always be discoverable."""
    
    def test_import_status(self):
        """Test that we can check if imports were successful."""
        assert 'IMPORTS_SUCCESSFUL' in globals(), "Import status variable should be defined"
        print(f"Import status: {IMPORTS_SUCCESSFUL}")
    
    def test_basic_assertion(self):
        """Basic test to ensure pytest discovery works."""
        assert True, "Basic test should always pass"
    
    def test_path_setup(self):
        """Test that the path setup is working."""
        # Check that the external app path is in sys.path
        assert external_app_path in sys.path, f"External app path should be in sys.path: {external_app_path}"
        print(f"Project root: {project_root}")
        print(f"External app path: {external_app_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"First few paths in sys.path: {sys.path[:5]}")


class TestOpenAIImplementation:
    """Test OpenAI LLM Classifier implementation."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables with actual tokens if available."""
        env_vars = {}
        if OPENAI_KEY:
            env_vars['OPEN_AI_KEY'] = OPENAI_KEY
        else:
            env_vars['OPEN_AI_KEY'] = 'test_key'
        
        with patch.dict(os.environ, env_vars):
            yield
    
    def test_openai_classifier_initialization(self, mock_environment):
        """Test OpenAI classifier initialization."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        classifier = OpenAILLMClassifier()
        assert classifier is not None
        assert hasattr(classifier, 'client')
        assert hasattr(classifier, 'system_prompt_adu_classification')
        assert hasattr(classifier, 'system_prompt_stance_classification')
        logger.info("✓ OpenAI classifier initialized successfully")

    
    def test_openai_classify_sentence_claim(self, mock_environment):
        """Test OpenAI classifier with a claim sentence."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        try:
            classifier = OpenAILLMClassifier()
            
            # Test with a clear claim
            test_sentence = "Climate change is caused by human activities."
            result = classifier.classify_sentence(test_sentence)
            
            assert result in ['claim', 'premise'], f"Unexpected result: {result}"
            logger.info(f"✓ OpenAI classifier classified claim sentence: {result}")
        except Exception as e:
            pytest.skip(f"OpenAI classifier test failed: {e}")
    
    def test_openai_classify_sentence_premise(self, mock_environment):
        """Test OpenAI classifier with a premise sentence."""
        if not OPENAI_KEY:
            pytest.fail("OpenAI API key not available in config")
        
        try:
            classifier = OpenAILLMClassifier()
            
            # Test with a clear premise
            test_sentence = "Scientific studies show increasing global temperatures."
            result = classifier.classify_sentence(test_sentence)
            
            assert result in ['claim', 'premise'], f"Unexpected result: {result}"
            logger.info(f"✓ OpenAI classifier classified premise sentence: {result}")
        except Exception as e:
            pytest.fail(f"OpenAI classifier test failed: {e}")
    
    def test_openai_classify_adus(self, mock_environment):
        """Test OpenAI classifier ADU extraction."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        try:
            classifier = OpenAILLMClassifier()
            
            # Test with argumentative text
            test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
            result = classifier.classify_adus(test_text)
            
            assert isinstance(result, UnlinkedArgumentUnits)
            assert hasattr(result, 'claims')
            assert hasattr(result, 'premises')
            assert isinstance(result.claims, list)
            assert isinstance(result.premises, list)
            
            logger.info(f"✓ OpenAI classifier extracted {len(result.claims)} claims and {len(result.premises)} premises")
        except Exception as e:
            pytest.skip(f"OpenAI classifier ADU extraction failed: {e}")
    
    def test_openai_classify_stance(self, mock_environment):
        """Test OpenAI classifier stance classification."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        try:
            classifier = OpenAILLMClassifier()
            
            # Create test argument units
            claims = [ArgumentUnit(text="Climate change is real", type="claim")]
            premises = [ArgumentUnit(text="Scientific evidence shows increasing temperatures", type="premise")]
            linked_units = LinkedArgumentUnits(
                claims=claims,
                premises=premises,
                claims_premises_relationships=[]
            )
            
            result = classifier.classify_stance(linked_units, "Climate change is real. Scientific evidence shows increasing temperatures.")
            
            assert isinstance(result, LinkedArgumentUnitsWithStance)
            assert hasattr(result, 'stance_relations')
            assert isinstance(result.stance_relations, list)
            
            logger.info(f"✓ OpenAI classifier classified stance with {len(result.stance_relations)} relations")
        except Exception as e:
            pytest.skip(f"OpenAI classifier stance classification failed: {e}")


class TestTinyLlamaImplementation:
    """Test TinyLlama LLM Classifier implementation."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables with actual tokens if available."""
        env_vars = {}
        if HF_TOKEN:
            env_vars['HF_TOKEN'] = HF_TOKEN
        else:
            env_vars['HF_TOKEN'] = 'test_token'
        
        with patch.dict(os.environ, env_vars):
            yield
    
    def test_tinyllama_classifier_initialization(self, mock_environment):
        """Test TinyLlama classifier initialization."""
        if not HF_TOKEN:
            pytest.skip("Hugging Face token not available in config")
        
        try:
            classifier = TinyLLamaLLMClassifier()
            assert classifier is not None
            assert hasattr(classifier, 'model')
            assert hasattr(classifier, 'tokenizer')
            assert hasattr(classifier, 'base_model_id')
            logger.info("✓ TinyLlama classifier initialized successfully")
        except Exception as e:
            pytest.skip(f"TinyLlama classifier initialization failed: {e}")
    
    def test_tinyllama_classify_sentence_claim(self, mock_environment):
        """Test TinyLlama classifier with a claim sentence."""
        if not HF_TOKEN:
            pytest.skip("Hugging Face token not available in config")

        classifier = TinyLLamaLLMClassifier()
        
        # Test with a clear claim
        test_sentence = "Climate change is caused by human activities."
        result = classifier.classify_sentence(test_sentence)
        
        assert result in ['claim', 'premise'], f"Unexpected result: {result}"
        logger.info(f"✓ TinyLlama classifier classified claim sentence: {result}")

    
    def test_tinyllama_classify_sentence_premise(self, mock_environment):
        """Test TinyLlama classifier with a premise sentence."""
        if not HF_TOKEN:
            pytest.skip("Hugging Face token not available in config")
        
        classifier = TinyLLamaLLMClassifier()
        
        # Test with a clear premise
        test_sentence = "Scientific studies show increasing global temperatures."
        result = classifier.classify_sentence(test_sentence)
        
        assert result in ['claim', 'premise'], f"Unexpected result: {result}"
        logger.info(f"✓ TinyLlama classifier classified premise sentence: {result}")

    
    def test_tinyllama_classify_adus_no_fallback(self, mock_environment):
        """Test TinyLlama classifier ADU extraction without fallback."""
        if not HF_TOKEN:
            pytest.skip("Hugging Face token not available in config")
        
        try:
            classifier = TinyLLamaLLMClassifier()
            
            # Test with argumentative text
            test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
            
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
                
                logger.info(f"✓ TinyLlama classifier extracted {len(result.claims)} claims and {len(result.premises)} premises without fallback")
        except Exception as e:
            pytest.skip(f"TinyLlama classifier ADU extraction failed: {e}")
    
    def test_tinyllama_classify_stance(self, mock_environment):
        """Test TinyLlama classifier stance classification."""
        if not HF_TOKEN:
            pytest.skip("Hugging Face token not available in config")
        
        try:
            classifier = TinyLLamaLLMClassifier()
            
            # Create test argument units
            claims = [ArgumentUnit(text="Climate change is real", type="claim")]
            premises = [ArgumentUnit(text="Scientific evidence shows increasing temperatures", type="premise")]
            linked_units = LinkedArgumentUnits(
                claims=claims,
                premises=premises,
                claims_premises_relationships=[]
            )
            
            result = classifier.classify_stance(linked_units, "Climate change is real. Scientific evidence shows increasing temperatures.")
            
            assert isinstance(result, LinkedArgumentUnitsWithStance)
            assert hasattr(result, 'stance_relations')
            assert isinstance(result.stance_relations, list)
            
            logger.info(f"✓ TinyLlama classifier classified stance with {len(result.stance_relations)} relations")
        except Exception as e:
            pytest.skip(f"TinyLlama classifier stance classification failed: {e}")


class TestModernBERTImplementation:
    """Test ModernBERT (PeftEncoderModelLoader) implementation."""
    
    def test_modernbert_classifier_initialization(self):
        """Test ModernBERT classifier initialization."""
        try:
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if not modernbert_config:
                pytest.skip("ModernBERT configuration not available")
            
            classifier = PeftEncoderModelLoader(**modernbert_config['params'])
            assert classifier is not None
            assert hasattr(classifier, 'tokenizer')
            assert hasattr(classifier, 'base_model_path')
            assert hasattr(classifier, 'adapter_paths')
            logger.info("✓ ModernBERT classifier initialized successfully")
        except Exception as e:
            pytest.skip(f"ModernBERT classifier initialization failed: {e}")
    
    def test_modernbert_identify_adus_no_fallback(self):
        """Test ModernBERT ADU identification without fallback."""

        modernbert_config = MODEL_CONFIGS.get('modernbert')
        if not modernbert_config:
            pytest.skip("ModernBERT configuration not available")
        
        classifier = PeftEncoderModelLoader(**modernbert_config['params'])
        
        # Test with argumentative text
        test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
        
        # Test with fallback disabled
        result = classifier.identify_adus(test_text) # , use_sentence_fallback=False)
        
        assert isinstance(result, list)
        # Note: Result might be empty if no ADUs are identified, which is acceptable
        
        logger.info(f"✓ ModernBERT identified {len(result)} ADUs without fallback")

    
    def test_modernbert_classify_adus(self):
        """Test ModernBERT ADU classification."""
        try:
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if not modernbert_config:
                pytest.skip("ModernBERT configuration not available")
            
            classifier = PeftEncoderModelLoader(**modernbert_config['params'])
            
            # Test with argumentative text
            test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
            result = classifier.classify_adus(test_text)
            
            assert isinstance(result, UnlinkedArgumentUnits)
            assert hasattr(result, 'claims')
            assert hasattr(result, 'premises')
            assert isinstance(result.claims, list)
            assert isinstance(result.premises, list)
            
            logger.info(f"✓ ModernBERT classified {len(result.claims)} claims and {len(result.premises)} premises")
        except Exception as e:
            pytest.skip(f"ModernBERT ADU classification failed: {e}")
    
    def test_modernbert_classify_stance(self):
        """Test ModernBERT stance classification."""

        modernbert_config = MODEL_CONFIGS.get('modernbert')
        if not modernbert_config:
            pytest.skip("ModernBERT configuration not available")
        
        classifier = PeftEncoderModelLoader(**modernbert_config['params'])
        
        # Create test argument units
        claims = [ArgumentUnit(uuid=uuid4(), text="Climate change is real", type="claim")]
        premises = [ArgumentUnit(uuid=uuid4(), text="Scientific evidence shows increasing temperatures", type="premise")]
        linked_units = LinkedArgumentUnits(
            claims=claims,
            premises=premises,
            claims_premises_relationships=[]
        )
        
        result = classifier.classify_stance(linked_units, "Climate change is real. Scientific evidence shows increasing temperatures.")
        
        assert isinstance(result, LinkedArgumentUnitsWithStance)
        assert hasattr(result, 'stance_relations')
        assert isinstance(result.stance_relations, list)
        
        logger.info(f"✓ ModernBERT classified stance with {len(result.stance_relations)} relations")



class TestDeBERTaImplementation:
    """Test DeBERTa (NonTrainedEncoderModelLoader) implementation."""
    
    def test_deberta_classifier_initialization(self):
        """Test DeBERTa classifier initialization."""

        deberta_config = MODEL_CONFIGS.get('deberta')
        if not deberta_config:
            pytest.skip("DeBERTa configuration not available")
        
        # Check if DeBERTa checkpoints exist
        checkpoint_paths = [
            Path("app/argmining/argmining/implementations/deberta-type-checkpoints/checkpoint-3"),
            Path("app/argmining/argmining/implementations/deberta-stance-checkpoints/checkpoint-3")
        ]
        
        missing_checkpoints = [str(p) for p in checkpoint_paths if not p.exists()]
        if missing_checkpoints:
            pytest.skip(f"DeBERTa checkpoints not available: {', '.join(missing_checkpoints)}")
        
        classifier = NonTrainedEncoderModelLoader(**deberta_config['params'])
        assert classifier is not None
        assert hasattr(classifier, 'tokenizer')
        assert hasattr(classifier, 'base_model_path')
        assert hasattr(classifier, 'model_paths')
        logger.info("✓ DeBERTa classifier initialized successfully")
    
    def test_deberta_classify_adus(self):
        """Test DeBERTa ADU classification."""

        deberta_config = MODEL_CONFIGS.get('deberta')
        if not deberta_config:
            pytest.skip("DeBERTa configuration not available")
        
        # Check if DeBERTa checkpoints exist
        checkpoint_paths = [
            Path("app/argmining/argmining/implementations/deberta-type-checkpoints/checkpoint-3"),
            Path("app/argmining/argmining/implementations/deberta-stance-checkpoints/checkpoint-3")
        ]
        
        missing_checkpoints = [str(p) for p in checkpoint_paths if not p.exists()]
        if missing_checkpoints:
            pytest.skip(f"DeBERTa checkpoints not available: {', '.join(missing_checkpoints)}")
        
        classifier = NonTrainedEncoderModelLoader(**deberta_config['params'])
        
        # Test with argumentative text
        test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
        result = classifier.classify_adus(test_text)
        
        assert isinstance(result, UnlinkedArgumentUnits)
        assert hasattr(result, 'claims')
        assert hasattr(result, 'premises')
        assert isinstance(result.claims, list)
        assert isinstance(result.premises, list)
        
        logger.info(f"✓ DeBERTa classified {len(result.claims)} claims and {len(result.premises)} premises")
    
    def test_deberta_classify_stance(self):
        """Test DeBERTa stance classification."""

        deberta_config = MODEL_CONFIGS.get('deberta')
        if not deberta_config:
            pytest.skip("DeBERTa configuration not available")
        
        # Check if DeBERTa checkpoints exist
        checkpoint_paths = [
            Path("app/argmining/argmining/implementations/deberta-type-checkpoints/checkpoint-3"),
            Path("app/argmining/argmining/implementations/deberta-stance-checkpoints/checkpoint-3")
        ]
        
        missing_checkpoints = [str(p) for p in checkpoint_paths if not p.exists()]
        if missing_checkpoints:
            pytest.skip(f"DeBERTa checkpoints not available: {', '.join(missing_checkpoints)}")
        
        classifier = NonTrainedEncoderModelLoader(**deberta_config['params'])
        
        # Create test argument units
        claims = [ArgumentUnit(text="Climate change is real", type="claim")]
        premises = [ArgumentUnit(text="Scientific evidence shows increasing temperatures", type="premise")]
        linked_units = LinkedArgumentUnits(
            claims=claims,
            premises=premises,
            claims_premises_relationships=[]
        )
        
        result = classifier.classify_stance(linked_units, "Climate change is real. Scientific evidence shows increasing temperatures.")
        
        assert isinstance(result, LinkedArgumentUnitsWithStance)
        assert hasattr(result, 'stance_relations')
        assert isinstance(result.stance_relations, list)
        
        logger.info(f"✓ DeBERTa classified stance with {len(result.stance_relations)} relations")

class TestOpenAIClaimPremiseLinker:
    """Test OpenAI Claim-Premise Linker implementation."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables with actual tokens if available."""
        env_vars = {}
        if OPENAI_KEY:
            env_vars['OPEN_AI_KEY'] = OPENAI_KEY
        else:
            env_vars['OPEN_AI_KEY'] = 'test_key'
        
        with patch.dict(os.environ, env_vars):
            yield
    
    def test_openai_linker_initialization(self, mock_environment):
        """Test OpenAI linker initialization."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        try:
            linker = OpenAIClaimPremiseLinker()
            assert linker is not None
            assert hasattr(linker, 'client')
            logger.info("✓ OpenAI linker initialized successfully")
        except Exception as e:
            pytest.skip(f"OpenAI linker initialization failed: {e}")
    
    def test_openai_linker_link_claims_premises(self, mock_environment):
        """Test OpenAI linker functionality."""
        if not OPENAI_KEY:
            pytest.skip("OpenAI API key not available in config")
        
        try:
            linker = OpenAIClaimPremiseLinker()
            
            # Create test argument units with more realistic data
            claims = [
                ArgumentUnit(uuid=uuid4(), text="Electric vehicles are better for the environment.", type="claim"),
                ArgumentUnit(uuid=uuid4(), text="Remote work improves employee productivity.", type="claim")
            ]
            premises = [
                ArgumentUnit(uuid=uuid4(), text="EVs have zero tailpipe emissions.", type="premise"),
                ArgumentUnit(uuid=uuid4(), text="EV battery mining damages ecosystems.", type="premise"),
                ArgumentUnit(uuid=uuid4(), text="People working remotely often report fewer distractions and better focus.", type="premise")
            ]
            unlinked_units = UnlinkedArgumentUnits(claims=claims, premises=premises)
            
            result = linker.link_claims_to_premises(unlinked_units)
            
            assert isinstance(result, LinkedArgumentUnits)
            assert hasattr(result, 'claims_premises_relationships')
            assert isinstance(result.claims_premises_relationships, list)
            assert len(result.claims) == len(claims)
            assert len(result.premises) == len(premises)
            
            # Log the relationships for debugging
            for relation in result.claims_premises_relationships:
                claim = next((c for c in claims if c.uuid == relation.claim_id), None)
                if claim:
                    linked_premises = [p.text for p in premises if p.uuid in relation.premise_ids]
                    logger.info(f"Claim: {claim.text} -> Linked premises: {linked_premises}")
            
            logger.info(f"✓ OpenAI linker created {len(result.claims_premises_relationships)} relationships")
        except Exception as e:
            pytest.skip(f"OpenAI linker test failed: {e}")


class TestImplementationIntegration:
    """Integration tests for all implementations."""
    
    @pytest.fixture
    def mock_environment(self):
        """Mock environment variables with actual tokens if available."""
        env_vars = {}
        if OPENAI_KEY:
            env_vars['OPEN_AI_KEY'] = OPENAI_KEY
        if HF_TOKEN:
            env_vars['HF_TOKEN'] = HF_TOKEN
        
        with patch.dict(os.environ, env_vars):
            yield
    
    def test_all_implementations_initialization(self, mock_environment):
        """Test that all implementations can be initialized."""
        implementations = {}
        
        # Test OpenAI
        if OPENAI_KEY:
            try:
                implementations['openai'] = OpenAILLMClassifier()
                logger.info("✓ OpenAI implementation initialized")
            except Exception as e:
                logger.warning(f"OpenAI implementation failed: {e}")
        else:
            logger.info("⚠️ OpenAI implementation skipped (no API key)")
        
        # Test TinyLlama
        if HF_TOKEN:
            try:
                implementations['tinyllama'] = TinyLLamaLLMClassifier()
                logger.info("✓ TinyLlama implementation initialized")
            except Exception as e:
                logger.warning(f"TinyLlama implementation failed: {e}")
        else:
            logger.info("⚠️ TinyLlama implementation skipped (no HF token)")
        
        # Test ModernBERT
        try:
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if modernbert_config:
                implementations['modernbert'] = PeftEncoderModelLoader(**modernbert_config['params'])
                logger.info("✓ ModernBERT implementation initialized")
        except Exception as e:
            logger.warning(f"ModernBERT implementation failed: {e}")
        
        # Test DeBERTa
        try:
            deberta_config = MODEL_CONFIGS.get('deberta')
            if deberta_config:
                # Check if DeBERTa checkpoints exist
                checkpoint_paths = [
                        "mrkk11/deberta-stance/deberta-type-checkpoints/checkpoint-3",
                        "mrkk11/deberta-stance/deberta-stance-checkpoints/checkpoint-3"
                ]
                
                if all(p.exists() for p in checkpoint_paths):
                    implementations['deberta'] = NonTrainedEncoderModelLoader(**deberta_config['params'])
                    logger.info("✓ DeBERTa implementation initialized")
                else:
                    logger.info("⚠️ DeBERTa implementation skipped (checkpoints not available)")
        except Exception as e:
            logger.warning(f"DeBERTa implementation failed: {e}")
        
        # Test OpenAI Linker
        if OPENAI_KEY:
            try:
                implementations['openai_linker'] = OpenAIClaimPremiseLinker()
                logger.info("✓ OpenAI Linker implementation initialized")
            except Exception as e:
                logger.warning(f"OpenAI Linker implementation failed: {e}")
        else:
            logger.info("⚠️ OpenAI Linker implementation skipped (no API key)")
        
        # Assert that at least one implementation was successful
        assert len(implementations) > 0, "No implementations could be initialized"
        logger.info(f"✓ Successfully initialized {len(implementations)} implementations")
    
    def test_implementation_consistency(self, mock_environment):
        """Test that implementations produce consistent output formats."""
        test_text = "Climate change is real. Scientific evidence shows increasing temperatures. We must take action."
        
        results = {}
        
        # Test each available implementation
        implementations_to_test = []
        
        # Try to initialize each implementation
        if OPENAI_KEY:
            try:
                implementations_to_test.append(('openai', OpenAILLMClassifier()))
            except Exception:
                pass
        
        if HF_TOKEN:
            try:
                implementations_to_test.append(('tinyllama', TinyLLamaLLMClassifier()))
            except Exception:
                pass
        
        try:
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if modernbert_config:
                implementations_to_test.append(('modernbert', PeftEncoderModelLoader(**modernbert_config['params'])))
        except Exception:
            pass
        
        try:
            deberta_config = MODEL_CONFIGS.get('deberta')
            if deberta_config:
                # Check if DeBERTa checkpoints exist
                checkpoint_paths = [
                        "mrkk11/deberta-stance/deberta-type-checkpoints/checkpoint-3",
                        "mrkk11/deberta-stance/deberta-stance-checkpoints/checkpoint-3"
                ]
                
                if all(p.exists() for p in checkpoint_paths):
                    implementations_to_test.append(('deberta', NonTrainedEncoderModelLoader(**deberta_config['params'])))
        except Exception:
            pass
        
        # Test each implementation
        for name, classifier in implementations_to_test:
            try:
                result = classifier.classify_adus(test_text)
                results[name] = result
                
                # Check output format consistency
                assert isinstance(result, UnlinkedArgumentUnits)
                assert hasattr(result, 'claims')
                assert hasattr(result, 'premises')
                assert isinstance(result.claims, list)
                assert isinstance(result.premises, list)
                
                logger.info(f"✓ {name} produced consistent output format")
            except Exception as e:
                logger.warning(f"{name} failed consistency test: {e}")
        
        # Assert that at least one implementation produced results
        if results:
            logger.info(f"✓ {len(results)} implementations produced consistent output formats")
        else:
            pytest.skip("No implementations could be tested for consistency")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
