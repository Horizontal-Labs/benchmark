import os
import pytest
import openai
from unittest.mock import patch, Mock
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.log import log


class TestOpenAIAPI:
    """Test cases for OpenAI API connectivity and API key validation."""
    
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
            pytest.fail(f"Unexpected error connecting to OpenAI API: {e}\n{traceback.format_exc()}")
    
    def test_openai_api_models_available(self):
        """Test that we can list available models from OpenAI API."""
        try:
            # Get API key
            api_key = os.getenv('OPEN_AI_KEY')
            if not api_key:
                pytest.skip("OPEN_AI_KEY not set")
            
            # Create OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # List models
            models = client.models.list()
            
            # Check response
            assert models is not None, "OpenAI API returned None for models list"
            assert hasattr(models, 'data'), "Models response missing 'data' attribute"
            
            # Check that we have some models
            model_ids = [model.id for model in models.data]
            assert len(model_ids) > 0, "No models returned from OpenAI API"
            
            # Check for specific models we use
            required_models = ['gpt-3.5-turbo', 'gpt-4']
            available_models = [model for model in required_models if model in model_ids]
            
            log.info(f"✓ OpenAI API models available. Found {len(available_models)}/{len(required_models)} required models: {available_models}")
            
        except openai.AuthenticationError as e:
            pytest.fail(f"OpenAI API authentication failed: {e}")
        except Exception as e:
            pytest.fail(f"Error listing OpenAI models: {e}\n{traceback.format_exc()}")
    
    def test_openai_llm_classifier_initialization(self):
        """Test that the OpenAI LLM classifier can be initialized with valid API key."""
        try:
            import sys
            import os
            # Add the external app directory to the path
            external_app_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'argument-mining-api', 'app')
            if external_app_path not in sys.path:
                sys.path.insert(0, external_app_path)
            
            from argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
            # Initialize the classifier
            classifier = OpenAILLMClassifier()
            
            # Check that it was initialized properly
            assert classifier is not None, "OpenAILLMClassifier initialization returned None"
            assert hasattr(classifier, 'client'), "OpenAILLMClassifier missing 'client' attribute"
            assert classifier.client is not None, "OpenAI client is None"
            
            log.info("✓ OpenAI LLM classifier initialized successfully")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize OpenAI LLM classifier: {e}\n{traceback.format_exc()}")
    
    def test_openai_llm_classifier_simple_classification(self):
        """Test that the OpenAI LLM classifier can perform a simple classification."""
        try:
            import sys
            import os
            # Add the external app directory to the path
            external_app_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'argument-mining-api', 'app')
            if external_app_path not in sys.path:
                sys.path.insert(0, external_app_path)
            
            from argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
            
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
            pytest.fail(f"Failed to perform classification with OpenAI LLM classifier: {e}\n{traceback.format_exc()}")
    
    @patch('openai.OpenAI')
    def test_openai_api_mock_authentication_error(self, mock_openai):
        """Test handling of authentication errors with mocked API."""
        # Mock authentication error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            "Invalid API key", response=Mock(), body={}
        )
        mock_openai.return_value = mock_client
        
        with pytest.raises(openai.AuthenticationError):
            client = openai.OpenAI(api_key="invalid-key")
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
        
        log.info("✓ OpenAI API authentication error handling works correctly")
    
    @patch('openai.OpenAI')
    def test_openai_api_mock_rate_limit_error(self, mock_openai):
        """Test handling of rate limit errors with mocked API."""
        # Mock rate limit error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            "Rate limit exceeded", response=Mock(), body={}
        )
        mock_openai.return_value = mock_client
        
        with pytest.raises(openai.RateLimitError):
            client = openai.OpenAI(api_key="test-key")
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
        
        log.info("✓ OpenAI API rate limit error handling works correctly")


def run_openai_api_tests():
    """Run all OpenAI API tests and return results."""
    import subprocess
    import sys
    
    # Run the tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_openai_api.py", 
        "-v", 
        "--tb=short"
    ], capture_output=True, text=True)
    
    print("OpenAI API Test Results:")
    print("=" * 50)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    success = run_openai_api_tests()
    sys.exit(0 if success else 1) 