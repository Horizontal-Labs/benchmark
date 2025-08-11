# Argument Mining Benchmark

This repository contains a benchmarking tool to compare different argument mining methods. It integrates with external repositories using Git submodules for the argument mining API and database components.

## Project Structure

```
benchmark/
├── external/                          # External submodules
│   ├── argument-mining-api/           # Argument mining implementations
│   └── argument-mining-db/            # Database and data management
├── app/
│   ├── benchmark.py                   # Main benchmark script
│   ├── config.py                      # Configuration management
│   └── db_connector/                  # Local database connector
├── results/                           # Benchmark results output
├── tests/                             # Test files
└── check_submodules.py               # Submodule status checker
```

## Prerequisites

- Python 3.8 or higher
- Git with submodule support
- Access to the external repositories

## Setup

### 1. Clone the Repository with Submodules

```bash
# Clone the repository and initialize submodules
git clone --recursive <repository-url>
cd benchmark

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 2. Check Submodule Status

Run the submodule checker to verify everything is set up correctly:

```bash
python check_submodules.py
```

This script will:
- Verify that all submodules are properly initialized
- Check if the repositories are valid Git repositories
- Test Python imports from the submodules
- Provide recommendations if issues are found

### 3. Update Submodules (Optional)

To update to the latest versions of the external repositories:

```bash
# Update all submodules to latest commits
git submodule update --remote

# Or update specific submodules
git submodule update --remote external/argument-mining-api
git submodule update --remote external/argument-mining-db

# Commit the updated submodule references
git add external/
git commit -m "Update submodules to latest versions"
```

### 4. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development
pip install -r requirements.txt[dev]
```

## Usage

### Running the Benchmark

```bash
# Run the full benchmark
python app/benchmark.py

# Or import and use programmatically
from app.benchmark import ArgumentMiningBenchmark

benchmark = ArgumentMiningBenchmark(max_samples=100)
results = benchmark.run_benchmark()
```

### Available Implementations

The benchmark supports multiple argument mining implementations:

- **OpenAI LLM Classifier**: Uses OpenAI's API for ADU classification and stance detection
- **TinyLlama LLM Classifier**: Local LLM-based classification
- **ModernBERT**: Fine-tuned BERT models for argument mining tasks
- **DeBERTa**: Pre-trained DeBERTa models

### Benchmark Tasks

1. **ADU Extraction**: Identify claims and premises in text
2. **Stance Classification**: Classify stance as Pro/Con/Neutral
3. **Claim-Premise Linking**: Link claims to supporting premises

## Configuration

The project uses a centralized configuration system in `app/config.py`:

```python
from app.config import config

# Check submodule status
status = config.get_submodule_status()

# Validate all components
config.validate_components()
```

## Troubleshooting

### Submodule Issues

If you encounter issues with submodules:

1. **Check submodule status**:
   ```bash
   python check_submodules.py
   ```

2. **Reinitialize submodules**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Reset submodules**:
   ```bash
   git submodule foreach --recursive git reset --hard
   git submodule update --init --recursive
   ```

### Import Issues

If Python imports fail:

1. Verify submodules are properly initialized
2. Check that the external repositories contain the expected code structure
3. Ensure the Python path includes the external submodule directories

### Database Connection Issues

The benchmark uses the database connector from the external `argument-mining-db` repository. Ensure:

1. Database credentials are properly configured
2. The database is accessible
3. Required tables exist

## Development

### Adding New Implementations

1. Add your implementation to the external `argument-mining-api` repository
2. Update the benchmark to import and use your implementation
3. Add appropriate tests

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update submodules if needed
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## External Dependencies

This project integrates with:

- [argument-mining-api](https://github.com/Horizontal-Labs/argument-mining-api): Argument mining implementations
- [argument-mining-db](https://github.com/Horizontal-Labs/argument-mining-db): Database and data management

These are included as Git submodules and should be kept up to date for the latest features and bug fixes.
