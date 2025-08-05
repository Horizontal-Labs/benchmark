# Implementation Testing Guide

This document describes the comprehensive testing strategy for all argument mining implementations in the benchmark.

## Overview

The testing suite includes tests for all available argument mining implementations:

1. **OpenAI LLM Classifier** - Uses GPT models for classification
2. **TinyLlama LLM Classifier** - Uses a fine-tuned TinyLlama model
3. **ModernBERT (PeftEncoderModelLoader)** - Uses ModernBERT with PEFT adapters
4. **DeBERTa (NonTrainedEncoderModelLoader)** - Uses DeBERTa models
5. **OpenAI Claim-Premise Linker** - Links claims and premises using GPT

## Test Files

### 1. `tests/test_implementations.py`
Comprehensive tests for each implementation individually, including:
- Initialization tests
- Sentence classification tests
- ADU extraction tests
- Stance classification tests
- Integration tests

### 2. `tests/test_benchmark_pytest.py::TestImplementationFallbackStrategies`
Tests that specifically disable fallback strategies to ensure implementations work without relying on fallbacks.

## Fallback Strategies Disabled

The tests are designed to disable fallback strategies to ensure the primary implementations work correctly:

### TinyLlama Implementation
- **Fallback**: `_fallback_sentence_analysis()` method
- **Test Strategy**: Mock the fallback method to ensure it's not called
- **Code**: `with patch.object(classifier, '_fallback_sentence_analysis') as mock_fallback:`

### ModernBERT Implementation
- **Fallback**: Sentence-level fallback in `identify_adus()`
- **Test Strategy**: Use `use_sentence_fallback=False` parameter
- **Code**: `classifier.identify_adus(test_text, use_sentence_fallback=False)`

### OpenAI Implementation
- **Fallback**: Model fallback from GPT-4 to GPT-3.5-turbo
- **Test Strategy**: Tests ensure primary model works without fallback

## Running Tests

### Prerequisites

To run the full test suite, you need:

1. **OpenAI API Key** (for OpenAI implementations):
   ```bash
   export OPEN_AI_KEY="your-openai-api-key"
   ```

2. **Hugging Face Token** (for TinyLlama and other HF models):
   ```bash
   export HF_TOKEN="your-huggingface-token"
   ```

3. **Model Files** (for ModernBERT and DeBERTa):
   - Ensure the model checkpoints are available in the expected paths
   - Check `app/argmining/implementations/encoder_model_loader.py` for paths

### Running Individual Implementation Tests

```bash
# Test OpenAI implementation
python -m pytest tests/test_implementations.py::TestOpenAIImplementation -v

# Test TinyLlama implementation
python -m pytest tests/test_implementations.py::TestTinyLlamaImplementation -v

# Test ModernBERT implementation
python -m pytest tests/test_implementations.py::TestModernBERTImplementation -v

# Test DeBERTa implementation
python -m pytest tests/test_implementations.py::TestDeBERTaImplementation -v

# Test OpenAI Linker
python -m pytest tests/test_implementations.py::TestOpenAIClaimPremiseLinker -v
```

### Running Fallback Strategy Tests

```bash
# Test implementations with fallback strategies disabled
python -m pytest tests/test_benchmark_pytest.py::TestImplementationFallbackStrategies -v
```

### Running All Implementation Tests

```bash
# Run all implementation tests
python -m pytest tests/test_implementations.py -v

# Run all tests including fallback strategy tests
python -m pytest tests/test_implementations.py tests/test_benchmark_pytest.py::TestImplementationFallbackStrategies -v
```

## Test Results Interpretation

### Expected Results

- **Passed Tests**: Implementation is working correctly
- **Skipped Tests**: Missing credentials or model files (expected in CI/CD)
- **Failed Tests**: Implementation has issues that need fixing

### Common Skip Reasons

1. **Missing API Keys**: Tests requiring OpenAI API or HF token will be skipped
2. **Missing Model Files**: Tests requiring local model files will be skipped
3. **Network Issues**: Tests requiring model downloads will be skipped

### Test Coverage

Each implementation is tested for:

1. **Initialization**: Can the implementation be instantiated?
2. **Basic Functionality**: Can it classify individual sentences?
3. **ADU Extraction**: Can it extract claims and premises from text?
4. **Stance Classification**: Can it classify stance relationships?
5. **Output Format**: Does it produce the expected output structure?
6. **Fallback Prevention**: Does it work without relying on fallback strategies?

## Implementation-Specific Notes

### OpenAI LLM Classifier
- Requires valid OpenAI API key
- Tests both GPT-4 and GPT-3.5-turbo models
- Includes retry mechanism testing

### TinyLlama LLM Classifier
- Requires Hugging Face token
- Tests both base model and fine-tuned adapter
- Fallback to base model if adapter fails

### ModernBERT (PeftEncoderModelLoader)
- Requires model checkpoints in specific paths
- Tests token classification and sequence classification
- Disables sentence-level fallback for testing

### DeBERTa (NonTrainedEncoderModelLoader)
- Requires DeBERTa model checkpoints
- Tests type classification and stance classification
- No fallback strategies implemented

### OpenAI Claim-Premise Linker
- Requires valid OpenAI API key
- Tests relationship creation between claims and premises

## Continuous Integration

In CI/CD environments:
- Tests will be skipped if credentials are not available
- This is expected behavior and doesn't indicate test failure
- Focus on tests that pass to verify implementation correctness

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Loading Errors**: Check model file paths and permissions
3. **API Errors**: Verify API keys are valid and have sufficient credits
4. **Memory Issues**: Some models require significant RAM/VRAM

### Debug Mode

To run tests with more verbose output:

```bash
python -m pytest tests/test_implementations.py -v -s --tb=short
```

### Environment Setup

For local development:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPEN_AI_KEY="your-key"
export HF_TOKEN="your-token"

# Run tests
python -m pytest tests/test_implementations.py -v
```

## Contributing

When adding new implementations:

1. Add tests to `tests/test_implementations.py`
2. Include initialization, basic functionality, and integration tests
3. Disable any fallback strategies in tests
4. Update this documentation with implementation-specific notes
5. Ensure tests handle missing dependencies gracefully 