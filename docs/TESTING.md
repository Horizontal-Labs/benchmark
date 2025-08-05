# Testing Guide for Argument Mining Benchmark

This document provides comprehensive information about testing the Argument Mining Benchmark.

## Overview

The benchmark includes a comprehensive test suite built with pytest that covers:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Mock Tests**: Testing with mocked dependencies
- **Error Handling Tests**: Testing error scenarios and edge cases

## Test Structure

### Test Files

- `test_benchmark_pytest.py` - Main test suite
- `pytest.ini` - Pytest configuration
- `requirements_test.txt` - Test dependencies
- `run_tests.py` - Test runner script

### Test Classes

1. **TestBenchmarkResult** - Tests for the BenchmarkResult dataclass
2. **TestArgumentMiningBenchmark** - Tests for the main benchmark class
3. **TestBenchmarkIntegration** - Integration tests

## Installation

### Install Test Dependencies

```bash
# Install test dependencies
pip install -r requirements_test.txt

# Or use the test runner
python run_tests.py --install-deps
```

### Python 3.9 Compatibility

For Python 3.9 users, the test dependencies are automatically configured to use compatible versions.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Run tests in parallel
python run_tests.py --parallel
```

### Test Types

```bash
# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run fast tests (exclude slow tests)
python run_tests.py --type fast

# Run all tests
python run_tests.py --type all
```

### Direct Pytest Usage

```bash
# Run specific test file
pytest test_benchmark_pytest.py -v

# Run specific test class
pytest test_benchmark_pytest.py::TestBenchmarkResult -v

# Run specific test method
pytest test_benchmark_pytest.py::TestBenchmarkResult::test_benchmark_result_creation -v

# Run with coverage
pytest test_benchmark_pytest.py --cov=app --cov-report=html

# Run in parallel
pytest test_benchmark_pytest.py -n auto
```

## Test Coverage

### What's Tested

1. **BenchmarkResult Dataclass**
   - Creation with valid data
   - Creation with error information
   - Field validation

2. **ArgumentMiningBenchmark Class**
   - Initialization
   - Environment checking
   - Data loading (success and failure scenarios)
   - Implementation initialization
   - Metrics calculation
   - Benchmark execution
   - Results saving and printing

3. **Metrics Calculation**
   - ADU extraction metrics (perfect match, no match, empty prediction)
   - Stance classification metrics (correct, incorrect, empty prediction)
   - Claim-premise linking metrics

4. **Error Handling**
   - Database connection failures
   - Implementation initialization failures
   - Sample processing errors
   - Missing implementations

5. **Integration Scenarios**
   - Full benchmark initialization
   - Benchmark with no implementations
   - End-to-end workflow testing

### Mock Testing

The test suite uses extensive mocking to:

- **Mock Dependencies**: External APIs, database connections, file operations
- **Mock Implementations**: Argument mining implementations
- **Mock Data**: Sample benchmark data
- **Mock Environment**: Environment variables

## Test Configuration

### Pytest Configuration (pytest.ini)

```ini
[tool:pytest]
testpaths = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests

## Test Fixtures

### Available Fixtures

1. **mock_environment** - Mocks environment variables
2. **mock_data** - Provides sample benchmark data
3. **mock_implementations** - Mocks argument mining implementations
4. **tmp_path** - Provides temporary directory for file operations

### Usage Example

```python
def test_with_mock_data(mock_data, mock_environment):
    """Test using mock fixtures."""
    # Test implementation here
    pass
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements_test.txt
    
    - name: Run tests
      run: |
        python run_tests.py --coverage --verbose
```

## Debugging Tests

### Running Failed Tests

```bash
# Run only failed tests
pytest --lf

# Run last failed test with more detail
pytest --lf -vvv

# Run with full traceback
pytest --tb=long
```

### Debug Mode

```bash
# Run with debugger
pytest --pdb

# Run specific test with debugger
pytest test_benchmark_pytest.py::TestBenchmarkResult::test_benchmark_result_creation --pdb
```

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

## Best Practices

### Writing Tests

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests with clear sections
3. **Mock External Dependencies**: Don't rely on external services in tests
4. **Test Edge Cases**: Include tests for error conditions and edge cases
5. **Use Fixtures**: Reuse common test setup with fixtures

### Example Test Structure

```python
def test_benchmark_initialization_success():
    """Test successful benchmark initialization."""
    # Arrange
    mock_data = [{'id': 1, 'text': 'test'}]
    
    # Act
    with patch('app.benchmark.get_benchmark_data') as mock_get_data:
        mock_get_data.return_value = mock_data
        benchmark = ArgumentMiningBenchmark()
    
    # Assert
    assert benchmark.data == mock_data
    assert benchmark.implementations == {}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install -r requirements_test.txt
   ```

3. **Mock Issues**
   ```python
   # Ensure mocks are properly configured
   with patch('module.function') as mock_func:
       mock_func.return_value = expected_value
   ```

4. **Environment Issues**
   ```bash
   # Set up test environment
   export OPEN_AI_KEY="test_key"
   ```

### Performance Optimization

1. **Parallel Execution**: Use `-n auto` for parallel test execution
2. **Test Selection**: Use markers to run specific test types
3. **Caching**: Pytest caches test results automatically
4. **Minimal Setup**: Keep test setup minimal and focused

## Contributing

### Adding New Tests

1. Follow the existing test structure
2. Use appropriate test markers
3. Include both positive and negative test cases
4. Add documentation for complex tests
5. Ensure tests are deterministic

### Test Review Checklist

- [ ] Tests cover the main functionality
- [ ] Edge cases are tested
- [ ] Error conditions are handled
- [ ] Tests are independent and isolated
- [ ] Mocking is used appropriately
- [ ] Test names are descriptive
- [ ] Documentation is updated

## Support

For test-related issues:

1. Check the test output for specific error messages
2. Review the test configuration
3. Ensure all dependencies are installed
4. Check Python version compatibility
5. Review the test documentation

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Mock Documentation](https://pytest-mock.readthedocs.io/)
- [Pytest Coverage Documentation](https://pytest-cov.readthedocs.io/) 