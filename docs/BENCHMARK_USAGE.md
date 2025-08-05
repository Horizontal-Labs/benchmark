# Argument Mining Benchmark Usage Guide

## Overview

This benchmark integrates multiple argument mining implementations and provides comprehensive evaluation metrics for:

1. **ADU Extraction** - Identifying claims and premises in text
2. **Stance Classification** - Determining pro/con/neutral stance
3. **Claim-Premise Linking** - Connecting claims to supporting premises

## Python Version Compatibility

### Python 3.9 Support

This benchmark is compatible with Python 3.9. For Python 3.9 users:

```bash
# Use the Python 3.9 specific requirements
pip install -r requirements_python39.txt

# Or use the main requirements (already Python 3.9 compatible)
pip install -r requirements.txt
```

### Python 3.9 Compatibility Notes

- **Pandas**: Uses version 1.5.x (compatible with Python 3.9)
- **Numpy**: Uses version 1.24.x (compatible with Python 3.9)
- **Pydantic**: Uses version 1.x (compatible with Python 3.9)
- **PyTorch**: Uses version 1.13.x (compatible with Python 3.9)
- **Transformers**: Uses version 4.29.x (compatible with Python 3.9)

### Testing Python 3.9 Compatibility

```bash
# Run the compatibility test
python test_python39_compatibility.py
```

## Available Implementations

The benchmark supports the following implementations:

- **OpenAI LLM Classifier** - Uses OpenAI API for classification
- **TinyLlama LLM Classifier** - Local TinyLlama model
- **ModernBERT** - PeftEncoderModelLoader for transformer models
- **DeBERTa** - NonTrainedEncoderModelLoader for DeBERTa models

## Quick Start

### 1. Setup Environment

```bash
# For Python 3.9 users
pip install -r requirements_python39.txt

# For Python 3.10+ users
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Run Basic Tests

```bash
# Test the integration
python test_benchmark_integration.py

# Test the comprehensive benchmark
python test_integration.py

# Test Python 3.9 compatibility (if using Python 3.9)
python test_python39_compatibility.py
```

### 3. Run the Benchmark

```bash
# Run the main benchmark
python app/benchmark.py

# Or use the example script
python run_benchmark_example.py
```

## Usage Examples

### Basic Usage

```python
from app.benchmark import ArgumentMiningBenchmark

# Initialize benchmark
benchmark = ArgumentMiningBenchmark()

# Run all tasks with all implementations
results = benchmark.run_benchmark()
```

### Custom Configuration

```python
# Run specific tasks only
results = benchmark.run_benchmark(
    tasks=['adu_extraction', 'stance_classification']
)

# Run with specific implementations only
results = benchmark.run_benchmark(
    implementations=['openai', 'tinyllama']
)

# Custom configuration
results = benchmark.run_benchmark(
    tasks=['adu_extraction'],
    implementations=['openai']
)
```

### Individual Task Benchmarking

```python
# Benchmark ADU extraction only
adu_results = benchmark.benchmark_adu_extraction('openai')

# Benchmark stance classification only
stance_results = benchmark.benchmark_stance_classification('openai')

# Benchmark claim-premise linking only
linking_results = benchmark.benchmark_claim_premise_linking('openai')
```

## Data Structure

The benchmark uses data from `get_benchmark_data_for_evaluation()` with the following structure:

```python
{
    "id": 1,
    "text": "Climate change is caused by human activities...",
    "ground_truth": {
        "adus": [
            {"text": "Climate change is caused by human activities", "type": "claim"},
            {"text": "The burning of fossil fuels releases greenhouse gases", "type": "premise"}
        ],
        "stance": "pro",
        "relationships": [
            {"claim_id": 1, "premise_ids": [2]}
        ]
    },
    "metadata": {"source": "database", "domain": "environmental"}
}
```

## Metrics Calculated

### ADU Extraction Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Accuracy**: True positives / Total ground truth ADUs

### Stance Classification Metrics
- **Accuracy**: Correct predictions / Total predictions
- **Weighted F1**: F1-score weighted by class frequency

### Claim-Premise Linking Metrics
- **Accuracy**: Correct relationships / Total relationships
- **Precision**: Correct relationships / Predicted relationships
- **Recall**: Correct relationships / Ground truth relationships
- **F1-Score**: Harmonic mean of precision and recall

### Performance Metrics
- **Inference Time**: Time per sample in seconds

## Output Files

Results are saved to the `results/` directory with timestamps:

```
results/
├── adu_extraction_results_20241204_143022.csv
├── stance_classification_results_20241204_143022.csv
└── claim_premise_linking_results_20241204_143022.csv
```

Each CSV file contains:
- Task name
- Implementation name
- Sample ID
- Success status
- Error messages (if any)
- All calculated metrics
- Performance metrics

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required for OpenAI implementations
OPEN_AI_KEY=your-openai-api-key-here

# Optional for Hugging Face models
HF_TOKEN=your-huggingface-token-here
```

### Benchmark Configuration

You can customize the benchmark behavior:

```python
# Initialize with specific configuration
benchmark = ArgumentMiningBenchmark()

# Run with custom parameters
results = benchmark.run_benchmark(
    tasks=['adu_extraction'],           # Specific tasks
    implementations=['openai']          # Specific implementations
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # For Python 3.9 users
   pip install -r requirements_python39.txt
   
   # Check if submodules are initialized
   git submodule update --init --recursive
   ```

2. **OpenAI API Errors**
   ```bash
   # Check your API key
   echo $OPEN_AI_KEY
   
   # Ensure .env file exists and has correct key
   cat .env
   ```

3. **No Implementations Loaded**
   ```bash
   # Check implementation availability
   python -c "from app.benchmark import ArgumentMiningBenchmark; b = ArgumentMiningBenchmark(); print(b.implementations.keys())"
   ```

4. **Database Connection Issues**
   - The benchmark falls back to sample data if database is unavailable
   - Check logs for specific error messages

5. **Python 3.9 Compatibility Issues**
   ```bash
   # Run compatibility test
   python test_python39_compatibility.py
   
   # Check Python version
   python --version
   
   # Ensure using correct requirements file
   pip install -r requirements_python39.txt
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from app.benchmark import ArgumentMiningBenchmark
benchmark = ArgumentMiningBenchmark()
```

## Advanced Usage

### Adding Custom Implementations

```python
from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier

class CustomClassifier(AduAndStanceClassifier):
    def classify_adus(self, text):
        # Your implementation here
        pass
    
    def classify_stance(self, linked_units, text):
        # Your implementation here
        pass

# Add to benchmark
benchmark.implementations['custom'] = {
    'adu_classifier': CustomClassifier(),
    'linker': None
}
```

### Custom Metrics

```python
def custom_metric_calculation(prediction, ground_truth):
    # Your custom metric calculation
    return {'custom_metric': 0.85}

# Override in benchmark
benchmark._calculate_adu_metrics = custom_metric_calculation
```

### Batch Processing

```python
# Process data in batches
for i in range(0, len(benchmark.data), 10):
    batch = benchmark.data[i:i+10]
    # Process batch
    pass
```

## Performance Tips

1. **Use Specific Implementations**: Only test the implementations you need
2. **Run Individual Tasks**: Test one task at a time for faster results
3. **Use Sample Data**: For quick testing, use the sample data instead of full database
4. **Parallel Processing**: Consider running implementations in parallel for large datasets

## Integration with Student Repositories

To integrate fellow students' implementations:

```python
# Add student repositories as submodules
git submodule add https://github.com/student1/repo.git student-repos/student1
git submodule add https://github.com/student2/repo.git student-repos/student2

# Install student implementations
pip install -e student-repos/student1/
pip install -e student-repos/student2/

# Use in benchmark
from student1.implementation import Student1Classifier
benchmark.implementations['student1'] = {
    'adu_classifier': Student1Classifier(),
    'linker': None
}
```

## Contributing

To add new implementations or improve the benchmark:

1. Follow the interface contracts in `app/argmining/interfaces/`
2. Add proper error handling and logging
3. Include unit tests for your implementation
4. Update this documentation
5. Ensure Python 3.9 compatibility if targeting that version

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test scripts for examples
3. Check the logs for detailed error messages
4. Ensure all dependencies are properly installed
5. Run compatibility tests if using Python 3.9 