# Integration Summary: Argument Mining API and Database

## Overview

Successfully integrated both repositories from Horizontal Labs into your benchmarking project:

1. **argument-mining-db** (https://github.com/Horizontal-Labs/argument-mining-db)
2. **argument-mining-api** (https://github.com/Horizontal-Labs/argument-mining-api)

## What Was Done

### 1. Repository Cloning
- Cloned both repositories into your project directory
- `argument-mining-db/` - Contains database models, queries, and data management
- `argument-mining-api/` - Contains the argument mining implementations and interfaces

### 2. Code Integration
- **Database Connector**: Copied `argument-mining-db/db/` → `app/db_connector/db/`
- **Argument Mining API**: Copied `argument-mining-api/app/argmining/` → `app/argmining/`
- Created proper Python package structure with `__init__.py` files

### 3. Dependencies Integration
- Updated `requirements.txt` with all dependencies from both repositories
- Combined and deduplicated dependencies
- Installed all required packages

### 4. Data Access Layer
- Created `get_benchmark_data_for_evaluation()` function in `app/db_connector/db/queries.py`
- Provides 100 sample benchmark items when database is not available
- Falls back to sample data if database connection fails
- Returns data in the format expected by the comprehensive benchmark

### 5. Environment Configuration
- Created `.env` file template with required environment variables
- Configured `OPEN_AI_KEY` (with underscore as specified)
- Added optional `HF_TOKEN` for Hugging Face models

## Project Structure

```
benchmark/
├── app/
│   ├── argmining/                    # Argument Mining API (from argument-mining-api)
│   │   ├── interfaces/               # Abstract base classes
│   │   ├── implementations/          # Concrete implementations
│   │   ├── models/                   # Data models
│   │   └── config.py                 # Configuration
│   ├── db_connector/                 # Database Connector (from argument-mining-db)
│   │   └── db/
│   │       ├── queries.py            # Database queries
│   │       ├── models.py             # Database models
│   │       ├── db.py                 # Database connection
│   │       └── config.py             # Database config
│   └── comprehensive_benchmark.py    # Main benchmark script
├── argument-mining-db/               # Original repository (cloned)
├── argument-mining-api/              # Original repository (cloned)
├── requirements.txt                  # Updated with all dependencies
├── .env                             # Environment variables
└── test_integration.py              # Integration test script
```

## Available Components

### Argument Mining Implementations
- **OpenAI LLM Classifier**: Uses OpenAI API for ADU extraction and stance classification
- **TinyLlama LLM Classifier**: Local TinyLlama model for classification
- **ModernBERT**: PeftEncoderModelLoader for transformer-based classification
- **DeBERTa**: NonTrainedEncoderModelLoader for DeBERTa models

### Database Components
- **Database Models**: ADU, Relationship, and other data models
- **Query Functions**: `get_benchmark_data()`, `get_benchmark_data_for_evaluation()`
- **Connection Management**: Session handling and database configuration

## Usage

### 1. Set Up Environment Variables
Edit `.env` file:
```env
OPEN_AI_KEY=your-actual-openai-api-key
HF_TOKEN=your-huggingface-token-here
```

### 2. Run Integration Test
```bash
python test_integration.py
```

### 3. Run Comprehensive Benchmark
```bash
python app/comprehensive_benchmark.py
```

### 4. Import Components in Your Code
```python
# Import database components
from app.db_connector.db.queries import get_benchmark_data_for_evaluation

# Import argument mining components
from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier

# Import benchmark framework
from app.comprehensive_benchmark import ArgumentMiningBenchmark, BenchmarkConfig
```

## Benchmark Features

The comprehensive benchmark provides:

1. **ADU Extraction**: Token-level precision/recall/F1 metrics
2. **Stance Classification**: Accuracy and weighted F1 scores
3. **Claim-Premise Linking**: Relationship accuracy metrics
4. **Performance Measurement**: Inference time per sample
5. **Multiple Implementations**: Compare all available models
6. **Results Export**: CSV output with detailed metrics

## Data Sources

- **Primary**: Database at `argumentmining.ddns.net` (requires credentials)
- **Fallback**: 100 sample items with claims, premises, and stance annotations
- **Format**: Structured data with ground truth for all three tasks

## Next Steps

1. **Configure Database**: Add actual database credentials to `.env` if needed
2. **Add OpenAI Key**: Replace placeholder with actual OpenAI API key
3. **Run Benchmarks**: Execute comprehensive benchmark on all implementations
4. **Analyze Results**: Review metrics and performance comparisons
5. **Extend**: Add new implementations or modify existing ones as needed

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
2. **Environment Variables**: Check `.env` file exists and has correct format
3. **Database Connection**: Falls back to sample data if database is unavailable
4. **Model Loading**: Some implementations may require additional setup (e.g., model downloads)

### Test Integration
Run the integration test to verify everything is working:
```bash
python test_integration.py
```

## Files Modified/Created

### New Files
- `app/argmining/` (entire directory)
- `app/db_connector/` (entire directory)
- `test_integration.py`
- `.env`
- `INTEGRATION_SUMMARY.md`

### Modified Files
- `requirements.txt` (updated with all dependencies)
- `app/comprehensive_benchmark.py` (updated imports and data loading)

The integration is complete and ready for use! 🎉 