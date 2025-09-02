# Individual Task and Implementation Execution

The Argument Mining Benchmark now supports running individual tasks and implementations independently, allowing for more flexible and targeted benchmarking.

## Overview

The benchmark has been enhanced with the following new capabilities:

1. **Individual Task Execution**: Run a specific task (ADU extraction, stance classification, or claim-premise linking) with all available implementations
2. **Individual Implementation Execution**: Run a specific implementation on all available tasks
3. **Combined Execution**: Run a specific task with a specific implementation
4. **Task-Specific Data Preparation**: Each task now has optimized data preparation for better benchmarking accuracy
5. **Enhanced CSV Output**: Results now include execution date, task, and implementation information

## Available Tasks

- **`adu_extraction`**: Extract Argumentative Discourse Units (claims and premises)
- **`stance_classification`**: Classify stance as pro/con/neutral
- **`claim_premise_linking`**: Link claims to supporting/contradicting premises

## Available Implementations

- **`openai`**: OpenAI LLM Classifier
- **`tinyllama`**: TinyLlama LLM Classifier
- **`modernbert`**: ModernBERT (PeftEncoderModelLoader)
- **`deberta`**: DeBERTa (NonTrainedEncoderModelLoader)

## Command-Line Usage

### List Available Options

```bash
# List available tasks
python run.py --list-tasks

# List available implementations
python run.py --list-implementations
```

### Run Individual Tasks

```bash
# Run ADU extraction with all implementations
python run.py --task adu_extraction

# Run stance classification with all implementations
python run.py --task stance_classification

# Run claim-premise linking with all implementations
python run.py --task claim_premise_linking
```

### Run Individual Implementations

```bash
# Run OpenAI implementation on all tasks
python run.py --implementation openai

# Run TinyLlama implementation on all tasks
python run.py --implementation tinyllama

# Run ModernBERT implementation on all tasks
python run.py --implementation modernbert

# Run DeBERTa implementation on all tasks
python run.py --implementation deberta
```

### Run Specific Combinations

```bash
# Run stance classification with TinyLlama only
python run.py --task stance_classification --implementation tinyllama

# Run ADU extraction with OpenAI only
python run.py --task adu_extraction --implementation openai
```

### Additional Options

```bash
# Quick mode (10 samples)
python run.py --task adu_extraction --quick

# Custom sample size
python run.py --task stance_classification --max-samples 50

# Full benchmark (default behavior)
python run.py
```

## Programmatic Usage

### Run Individual Task

```python
from app.benchmark import run_single_task_benchmark

# Run ADU extraction with all implementations
results = run_single_task_benchmark('adu_extraction', max_samples=100)

# Run stance classification with specific implementation
results = run_single_task_benchmark('stance_classification', max_samples=100, implementation_name='openai')
```

### Run Individual Implementation

```python
from app.benchmark import run_single_implementation_benchmark

# Run OpenAI implementation on all tasks
results = run_single_implementation_benchmark('openai', max_samples=100)

# Run TinyLlama implementation on specific task
results = run_single_implementation_benchmark('tinyllama', max_samples=100, task_name='adu_extraction')
```

### Run Full Benchmark

```python
from app.benchmark import run_full_benchmark

# Run complete benchmark suite
results = run_full_benchmark(max_samples=100)
```

## CSV Output Format

The benchmark now generates CSV files with enhanced information:

| Column | Description |
|--------|-------------|
| `execution_date` | Date and time when the benchmark was executed |
| `task` | Name of the task (adu_extraction, stance_classification, claim_premise_linking) |
| `implementation` | Name of the implementation tested |
| `sample_id` | Identifier for the sample |
| `success` | Whether the execution was successful |
| `error_message` | Error message if execution failed |
| `metric_*` | Task-specific metrics (precision, recall, F1, accuracy, etc.) |
| `perf_*` | Performance metrics (inference time, etc.) |

### Example CSV Output

```csv
execution_date,task,implementation,sample_id,success,error_message,metric_precision,metric_recall,metric_f1,perf_inference_time
2024-01-15 14:30:25,adu_extraction,openai,sample_0,True,,1.0,1.0,1.0,0.914
2024-01-15 14:30:25,adu_extraction,openai,sample_1,True,,1.0,1.0,1.0,0.538
2024-01-15 14:30:25,adu_extraction,tinyllama,sample_0,True,,0.0,0.0,0.0,1.701
```

## Task-Specific Data Preparation

Each task now has optimized data preparation:

### ADU Extraction
- Prepares ground truth ADUs from claims and premises
- Optimizes text format for extraction algorithms

### Stance Classification
- Determines stance from topic information or uses alternating patterns
- Ensures balanced stance distribution for testing

### Claim-Premise Linking
- Creates relationship mappings between claims and premises
- Supports different relationship types (supports, contradicts)

## Benefits

1. **Focused Testing**: Test specific components without running the entire benchmark
2. **Faster Iteration**: Run smaller, targeted benchmarks during development
3. **Resource Efficiency**: Test only what you need
4. **Better Debugging**: Isolate issues to specific tasks or implementations
5. **Flexible Workflows**: Integrate into CI/CD pipelines with specific requirements

## Examples

See `examples/individual_benchmark_examples.py` for complete usage examples.

## Testing

Run the test suite to verify individual execution capabilities:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_benchmark_individual.py -v

# Run specific test class
pytest tests/test_benchmark_individual.py::TestBenchmarkIndividualExecution -v
```

## Migration from Previous Version

The new capabilities are backward compatible. Existing code will continue to work:

```python
# Old way (still works)
from app.benchmark import run_full_benchmark
results = run_full_benchmark(max_samples=100)

# New way (more flexible)
from app.benchmark import run_single_task_benchmark
results = run_single_task_benchmark('adu_extraction', max_samples=100)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Database Connection**: Check database configuration if using external data
3. **Implementation Initialization**: Verify API keys and model paths for specific implementations

### Debug Mode

Enable debug logging by setting the log level in your environment:

```bash
export LOG_LEVEL=DEBUG
python run.py --task adu_extraction
```

## Contributing

When adding new tasks or implementations:

1. Update the task list in `run.py`
2. Add data preparation methods in `ArgumentMiningBenchmark`
3. Update the test suite
4. Document the new capabilities

## Support

For issues or questions about individual execution capabilities, please refer to the main project documentation or create an issue in the project repository.
