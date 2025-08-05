#!/usr/bin/env python3
"""
Simple script to test OpenAI API connectivity and API key validation.
This script can be run independently to check if the OpenAI API is working.
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_openai_api_key():
    """Test that the OpenAI API key is properly configured."""
    print("🔑 Testing OpenAI API key configuration...")
    
    api_key = os.getenv('OPEN_AI_KEY')
    if not api_key:
        print("❌ OPEN_AI_KEY environment variable is not set")
        return False
    
    if len(api_key) == 0:
        print("❌ OPEN_AI_KEY environment variable is empty")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ OPEN_AI_KEY should start with 'sk-'")
        return False
    
    print(f"✅ OpenAI API key is properly configured (starts with: {api_key[:7]}...)")
    return True

def test_openai_api_connection():
    """Test that we can connect to the OpenAI API."""
    print("\n🌐 Testing OpenAI API connection...")
    
    try:
        import openai
        
        # Get API key
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            print("❌ OPEN_AI_KEY not set")
            return False
        
        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple test request
        print("📡 Making test request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        # Check response
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            print("❌ OpenAI API returned invalid response")
            return False
        
        content = response.choices[0].message.content
        if not content:
            print("❌ OpenAI API returned empty content")
            return False
        
        print(f"✅ OpenAI API connection successful! Response: '{content.strip()}'")
        return True
        
    except ImportError:
        print("❌ OpenAI library not installed. Run: pip install openai")
        return False
    except openai.AuthenticationError as e:
        print(f"❌ OpenAI API authentication failed: {e}")
        return False
    except openai.RateLimitError as e:
        print(f"❌ OpenAI API rate limit exceeded: {e}")
        return False
    except openai.APIError as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error connecting to OpenAI API: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_openai_models():
    """Test that we can list available models from OpenAI API."""
    print("\n🤖 Testing OpenAI API models availability...")
    
    try:
        import openai
        
        # Get API key
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            print("❌ OPEN_AI_KEY not set")
            return False
        
        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # List models
        print("📋 Fetching available models...")
        models = client.models.list()
        
        # Check response
        if not models or not hasattr(models, 'data'):
            print("❌ OpenAI API returned invalid models response")
            return False
        
        # Check that we have some models
        model_ids = [model.id for model in models.data]
        if len(model_ids) == 0:
            print("❌ No models returned from OpenAI API")
            return False
        
        # Check for specific models we use
        required_models = ['gpt-3.5-turbo', 'gpt-4']
        available_models = [model for model in required_models if model in model_ids]
        
        print(f"✅ Found {len(model_ids)} total models")
        print(f"✅ Available required models: {len(available_models)}/{len(required_models)}")
        print(f"   Available: {available_models}")
        
        if len(available_models) == 0:
            print("⚠️  Warning: No required models available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing OpenAI models: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_openai_classifier():
    """Test that our OpenAI classifier can be initialized."""
    print("\n🧠 Testing OpenAI LLM classifier initialization...")
    
    try:
        # Add app directory to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
        
        from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
        
        # Initialize the classifier
        classifier = OpenAILLMClassifier()
        
        # Check that it was initialized properly
        if not classifier:
            print("❌ OpenAILLMClassifier initialization returned None")
            return False
        
        if not hasattr(classifier, 'client') or classifier.client is None:
            print("❌ OpenAILLMClassifier missing or invalid 'client' attribute")
            return False
        
        print("✅ OpenAI LLM classifier initialized successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import OpenAILLMClassifier: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI LLM classifier: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all OpenAI API tests."""
    print("🚀 OpenAI API Connectivity Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("API Key Configuration", test_openai_api_key),
        ("API Connection", test_openai_api_connection),
        ("Models Availability", test_openai_models),
        ("Classifier Initialization", test_openai_classifier),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenAI API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check your OpenAI API configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 