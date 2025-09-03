# Argument Mining Benchmark Package

This package provides a refactored, modular structure for benchmarking argument mining implementations.

## Package Structure

```
benchmark/
├── __init__.py                 # Main package entry point
├── core/                       # Core benchmark components
│   ├── __init__.py
│   ├── benchmark.py            # Main benchmark orchestrator
│   └── results.py              # BenchmarkResult data structure
├── implementations/            # Implementation interfaces and concrete implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract base class for implementations
│   ├── openai.py               # OpenAI implementation
│   ├── tinyllama.py            # TinyLlama implementation
│   ├── modernbert.py           # ModernBERT implementation
│   └── deberta.py              # DeBERTa implementation
├── tasks/                      # Task-specific benchmark implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract base class for tasks
│   ├── adu_extraction.py       # ADU extraction task
│   ├── stance_classification.py # Stance classification task
│   └── claim_premise_linking.py # Claim-premise linking task
├── data/                       # Data loading and preparation utilities
│   ├── __init__.py
│   └── loader.py               # Data loading utilities
├── metrics/                    # Metrics calculation and evaluation utilities
│   ├── __init__.py
│   └── evaluator.py            # Metrics evaluation utilities
└── utils/                      # Utility functions and helpers
    ├── __init__.py
    ├── logging.py              # Logging utilities
    └── file_handlers.py        # File I/O operations
```

## Key Benefits of the Refactored Structure

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Maintainability**: Easier to modify individual components without affecting others
3. **Testability**: Each component can be tested independently
4. **Reusability**: Components can be imported and used separately
5. **Extensibility**: Easy to add new implementations, tasks, or metrics
6. **Cleaner Imports**: More organized import structure

## Usage

### Basic Usage

```python
from benchmark import ArgumentMiningBenchmark

# Create benchmark instance
benchmark = ArgumentMiningBenchmark(max_samples=100)

# Run full benchmark
results = benchmark.run_benchmark()

# Run single task
task_results = benchmark.run_single_task('adu_extraction')

# Run single implementation
impl_results = benchmark.run_single_implementation('openai')
```

### Adding New Implementations

1. Create a new class in `benchmark/implementations/`
2. Inherit from `BaseImplementation`
3. Implement required abstract methods
4. Add to the implementations list in the main benchmark class

### Adding New Tasks

1. Create a new class in `benchmark/tasks/`
2. Inherit from `BaseTask`
3. Implement required abstract methods
4. Add to the tasks list in the main benchmark class

## Migration from Old Structure

The old monolithic `benchmark.py` file has been split into logical components:

- **Main benchmark logic** → `benchmark/core/benchmark.py`
- **Implementation management** → `benchmark/implementations/`
- **Task-specific logic** → `benchmark/tasks/`
- **Data handling** → `benchmark/data/`
- **Metrics calculation** → `benchmark/metrics/`
- **Utility functions** → `benchmark/utils/`

## Testing

Run the test script to verify the package works correctly:

```bash
python test_refactored_benchmark.py
```

## Entry Points

- **`benchmark_main.py`**: New main entry point using the refactored package
- **`app/benchmark.py`**: Original monolithic benchmark file (kept for reference)

## Dependencies

The package maintains the same dependencies as the original implementation but organizes them better through the modular structure.
