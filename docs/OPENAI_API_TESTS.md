# OpenAI API Test Cases

This document describes the test cases that have been added to validate OpenAI API connectivity and functionality.

## Overview

The test suite includes comprehensive tests to ensure that:
1. The OpenAI API key is properly configured
2. The OpenAI API is accessible and responding
3. Required models are available
4. Our OpenAI LLM classifier can be initialized and used

## Test Files

### 1. `tests/test_openai_api.py`
A dedicated pytest test file with comprehensive OpenAI API tests.

### 2. `test_openai_connection.py`
A standalone script that can be run independently to test OpenAI API connectivity.

### 3. `tests/test_benchmark_pytest.py::TestOpenAIAPI`
Integrated tests within the main benchmark test suite.

## Test Cases

### 1. API Key Configuration Test
**File**: All test files  
**Test**: `test_openai_api_key_exists`

Validates that:
- The `OPEN_AI_KEY` environment variable is set
- The API key is not empty
- The API key starts with `sk-` (OpenAI format)

### 2. API Connection Test
**File**: All test files  
**Test**: `test_openai_api_connection`

Validates that:
- We can create an OpenAI client
- We can make a simple API request
- The API returns a valid response
- The response contains the expected content

### 3. Models Availability Test
**File**: `tests/test_openai_api.py`, `test_openai_connection.py`  
**Test**: `test_openai_api_models_available`

Validates that:
- We can list available models from the API
- Required models (`gpt-3.5-turbo`, `gpt-4`) are available
- The API returns a valid models list

### 4. Classifier Initialization Test
**File**: All test files  
**Test**: `test_openai_llm_classifier_initialization`

Validates that:
- Our `OpenAILLMClassifier` can be imported
- The classifier can be initialized with the API key
- The classifier has a valid OpenAI client

### 5. Simple Classification Test
**File**: All test files  
**Test**: `test_openai_llm_classifier_simple_classification`

Validates that:
- The classifier can perform a simple sentence classification
- The result is either 'claim' or 'premise'
- The classification process completes successfully

### 6. Error Handling Tests
**File**: `tests/test_openai_api.py`  
**Tests**: 
- `test_openai_api_mock_authentication_error`
- `test_openai_api_mock_rate_limit_error`

Validates that:
- Authentication errors are properly handled
- Rate limit errors are properly handled
- Error handling works with mocked API responses

## Running the Tests

### Standalone Script
```bash
python test_openai_connection.py
```

### Pytest Tests
```bash
# Run all OpenAI API tests
python -m pytest tests/test_openai_api.py -v

# Run integrated OpenAI API tests
python -m pytest tests/test_benchmark_pytest.py::TestOpenAIAPI -v

# Run all tests including OpenAI API tests
python -m pytest tests/ -v
```

## Test Output Examples

### Successful Test Run
```
🚀 OpenAI API Connectivity Test
==================================================
🔑 Testing OpenAI API key configuration...
✅ OpenAI API key is properly configured (starts with: sk-1234...)

🌐 Testing OpenAI API connection...
📡 Making test request to OpenAI API...
✅ OpenAI API connection successful! Response: 'Hello, World!'

🤖 Testing OpenAI API models availability...
📋 Fetching available models...
✅ Found 73 total models
✅ Available required models: 2/2
   Available: ['gpt-3.5-turbo', 'gpt-4']

🧠 Testing OpenAI LLM classifier initialization...
✅ OpenAI LLM classifier initialized successfully

==================================================
📊 Test Results Summary:
==================================================
✅ PASS - API Key Configuration
✅ PASS - API Connection
✅ PASS - Models Availability
✅ PASS - Classifier Initialization

Overall: 4/4 tests passed
🎉 All tests passed! OpenAI API is working correctly.
```

### Pytest Output
```
tests/test_openai_api.py::TestOpenAIAPI::test_openai_api_key_exists PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_api_connection PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_api_models_available PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_llm_classifier_initialization PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_llm_classifier_simple_classification PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_api_mock_authentication_error PASSED
tests/test_openai_api.py::TestOpenAIAPI::test_openai_api_mock_rate_limit_error PASSED

7 passed in 6.13s
```

## Error Scenarios

The tests handle various error scenarios:

1. **Missing API Key**: Tests will skip or fail with clear error messages
2. **Invalid API Key**: Authentication errors are caught and reported
3. **Rate Limiting**: Rate limit errors are caught and reported
4. **Network Issues**: Connection errors are caught and reported
5. **Model Unavailability**: Tests check for required model availability

## Dependencies

The tests require:
- `openai` library installed
- `python-dotenv` for loading environment variables
- Valid OpenAI API key in `.env` file or environment variables
- Internet connection for API calls

## Configuration

Make sure your `.env` file contains:
```
OPEN_AI_KEY=sk-your-actual-api-key-here
```

## Integration with CI/CD

These tests can be integrated into CI/CD pipelines to:
- Validate API connectivity before deployments
- Ensure API key configuration is correct
- Monitor API availability and model access
- Catch configuration issues early in the development process 