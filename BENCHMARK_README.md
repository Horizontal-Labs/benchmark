# Argument Mining Benchmark Tool

This benchmark tool evaluates the performance of different argument mining implementations using the benchmark data from the db-connector.

## Overview

The benchmark tests implementations of two main interfaces:
1. **AduAndStanceClassifier** - For extracting and classifying argumentative units (ADUs)
2. **ClaimPremiseLinker** - For linking claims to premises

## Features

- **Time Measurement**: Tracks execution time for each implementation
- **Success Rate Tracking**: Monitors success/failure rates
- **Basic Metrics**: Extracts key performance indicators
- **CSV Export**: Saves detailed results to CSV files
- **Summary Reports**: Provides overview of benchmark results

## Prerequisites

1. **Database Setup**: Ensure the db-connector is properly configured and the database is accessible
2. **Environment Variables**: Set required API keys (e.g., `OPENAI_KEY` for OpenAI implementations)
3. **Dependencies**: Install required packages

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r benchmark_requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export OPENAI_KEY="your-openai-api-key"
   ```

3. **Ensure database connectivity**:
   - Make sure the db-connector is properly configured
   - Verify database connection settings in `db-connector/db/config.py`

## Usage

### Basic Usage

Run the benchmark with default settings:

```bash
python benchmark_script.py
```

### What the Benchmark Does

1. **Loads Benchmark Data**: Uses `get_benchmark_data()` from db-connector
2. **Tests ADU Classifiers**: 
   - OpenAILLMClassifier
   - TinyLlamaLLMClassifier  
   - EncoderModelLoader
3. **Tests Claim-Premise Linkers**:
   - OpenAIClaimPremiseLinker
4. **Measures Performance**: Time, success rate, and basic metrics
5. **Saves Results**: Exports to timestamped CSV file

### Output Files

The benchmark generates:
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Detailed results for each test case
- Console output with summary statistics

### CSV Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `implementation_name` | Name of the implementation being tested |
| `interface_type` | Type of interface ('adu_classifier' or 'linker') |
| `test_case_id` | Unique identifier for each test case |
| `execution_time` | Time taken for the operation (seconds) |
| `success` | Whether the operation succeeded (True/False) |
| `error_message` | Error message if operation failed |
| `total_units_extracted` | Total ADUs extracted (for classifiers) |
| `claims_extracted` | Number of claims extracted |
| `premises_extracted` | Number of premises extracted |
| `avg_confidence` | Average confidence score |
| `total_claims` | Total claims processed (for linkers) |
| `total_premises` | Total premises processed |
| `total_relationships` | Number of claim-premise relationships created |
| `avg_premises_per_claim` | Average premises per claim |

## Metrics Explained

### For ADU Classifiers:
- **total_units_extracted**: Total number of argumentative units identified
- **claims_extracted**: Number of claims identified
- **premises_extracted**: Number of premises identified  
- **avg_confidence**: Average confidence score across all extracted units

### For Claim-Premise Linkers:
- **total_claims**: Number of claims processed
- **total_premises**: Number of premises processed
- **total_relationships**: Number of claim-premise relationships established
- **avg_premises_per_claim**: Average number of premises linked to each claim

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Ensure argument-mining-api and db-connector are in the correct paths
   - Check that all dependencies are installed

2. **Database Connection Issues**:
   - Verify database configuration in db-connector
   - Check database server is running

3. **API Key Issues**:
   - Ensure OPENAI_KEY is set for OpenAI implementations
   - Check API key validity and quota

4. **Model Loading Issues**:
   - Some implementations may require downloading models on first run
   - Check internet connectivity for model downloads

### Debug Mode

To run with more verbose output, you can modify the script to add debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extending the Benchmark

### Adding New Implementations

To add a new implementation to the benchmark:

1. **Add to implementations dictionary** in `ArgumentMiningBenchmark.__init__()`:
   ```python
   self.implementations['adu_classifier'].append(
       ('YourNewClassifier', YourNewClassifierClass)
   )
   ```

2. **Ensure the implementation follows the interface**:
   - ADU classifiers must implement `AduAndStanceClassifier`
   - Linkers must implement `ClaimPremiseLinker`

### Adding New Metrics

To add new metrics:

1. **Update the BenchmarkResult dataclass** to include new fields
2. **Modify the benchmark methods** to calculate new metrics
3. **Update the CSV output** to include new columns

## Performance Considerations

- **Large Datasets**: The benchmark uses the full benchmark dataset which may be large
- **API Costs**: OpenAI implementations will incur API costs
- **Memory Usage**: Some implementations may require significant memory
- **Time**: Full benchmark may take several minutes to complete

## Contributing

When adding new implementations or metrics:

1. Follow the existing code structure
2. Add appropriate error handling
3. Include documentation for new features
4. Test with a subset of data first

## License

This benchmark tool is part of the Argument Mining project and follows the same license terms. 