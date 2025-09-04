# Refactored Benchmark Package Test Suite

This directory contains comprehensive tests for the refactored Argument Mining Benchmark package.

## Test Structure

### New Test Files

- **`test_refactored_benchmark.py`** - Comprehensive tests for the refactored benchmark package
- **`test_refactored_components.py`** - Tests for individual components in isolation
- **`run_refactored_tests.py`** - Test runner script for all refactored package tests
- **`conftest_refactored.py`** - Pytest configuration for the refactored package

### Test Categories

#### 1. Package Structure Tests (`test_refactored_benchmark.py`)
- Package import functionality
- Main benchmark class functionality
- Component integration tests
- Mock-based testing for external dependencies

#### 2. Component Tests (`test_refactored_components.py`)
- Core components (BenchmarkResult, etc.)
- Implementation classes (OpenAI, TinyLlama, etc.)
- Task classes (ADU extraction, stance classification, etc.)
- Data and metrics components
- Utility functions

#### 3. Integration Tests
- Component relationships
- Package import chains
- End-to-end functionality

## Running Tests

### Option 1: Run All Tests
```bash
cd tests
python run_refactored_tests.py
```

### Option 2: Run Individual Test Files
```bash
# Run pytest tests
pytest test_refactored_benchmark.py -v
pytest test_refactored_components.py -v

# Run simple tests
python ../test_refactored_benchmark.py
```

### Option 3: Run with Pytest
```bash
# Run all refactored tests
pytest test_refactored_*.py -v

# Run specific test class
pytest test_refactored_components.py::TestCoreComponents -v

# Run specific test method
pytest test_refactored_components.py::TestCoreComponents::test_benchmark_result_dataclass -v
```

## Test Configuration

### Pytest Configuration
The `pytest.ini` file is configured to:
- Automatically discover test files
- Use verbose output
- Handle warnings appropriately
- Support test markers

### Test Fixtures
The `conftest_refactored.py` file provides:
- Path configuration for the refactored package
- Environment setup
- Common test fixtures

## Test Coverage

### Core Components
- ✅ BenchmarkResult dataclass
- ✅ ArgumentMiningBenchmark class
- ✅ Abstract base classes

### Implementation Classes
- ✅ BaseImplementation interface
- ✅ OpenAI implementation
- ✅ TinyLlama implementation
- ✅ ModernBERT implementation
- ✅ DeBERTa implementation

### Task Classes
- ✅ BaseTask interface
- ✅ ADU extraction task
- ✅ Stance classification task
- ✅ Claim-premise linking task

### Utility Components
- ✅ Data loading utilities
- ✅ Metrics evaluation
- ✅ Logging setup
- ✅ File handling

## Mocking Strategy

The tests use extensive mocking to:
- Isolate components for unit testing
- Avoid external dependencies
- Test error conditions
- Ensure consistent test behavior

### Key Mocked Components
- External API calls
- Database connections
- File system operations
- External model dependencies

## Test Data

### Mock Data Structures
- Sample claims and premises
- Test metrics and performance data
- Error scenarios
- Edge cases

### Data Validation
- Structure validation
- Type checking
- Boundary condition testing
- Error handling verification

## Best Practices

### Test Organization
- Clear test class names
- Descriptive test method names
- Logical grouping of related tests
- Consistent assertion patterns

### Error Handling
- Test both success and failure scenarios
- Verify error messages and types
- Test edge cases and boundary conditions
- Ensure graceful degradation

### Performance
- Fast test execution
- Minimal external dependencies
- Efficient mocking strategies
- Parallel test execution support

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure the benchmark package is in the Python path
   - Check that all required dependencies are installed
   - Verify the package structure is correct

2. **Mock Configuration Issues**
   - Check that mocks are properly configured
   - Ensure mock return values match expected types
   - Verify mock method calls are correct

3. **Test Environment Issues**
   - Check Python version compatibility
   - Verify pytest installation
   - Ensure working directory is correct

### Debug Mode
Run tests with increased verbosity:
```bash
pytest test_refactored_*.py -v -s --tb=long
```

## Future Enhancements

### Planned Test Improvements
- Performance benchmarking tests
- Memory usage tests
- Stress testing for large datasets
- Integration with CI/CD pipelines

### Test Data Expansion
- More diverse test scenarios
- Real-world data samples
- Performance regression tests
- Compatibility testing

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate test categories
3. Include comprehensive mocking
4. Add proper documentation
5. Ensure backward compatibility

## Dependencies

### Required Packages
- pytest
- unittest.mock
- pandas (for data handling tests)
- numpy (for metrics tests)

### Optional Packages
- coverage (for test coverage reporting)
- pytest-cov (for coverage integration)
- pytest-xdist (for parallel execution)
