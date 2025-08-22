# Argument Mining Benchmark

This repository contains a comprehensive benchmarking tool to compare different argument mining methods and implementations.

## Overview

The benchmark evaluates various argument mining models on tasks including:
- **ADU Extraction**: Identifying argumentative discourse units (claims and premises)
- **Stance Classification**: Determining relationships between arguments (pro/con/neutral)
- **Claim-Premise Linking**: Connecting claims with their supporting or opposing premises

## Supported Models

- **OpenAI LLM**: GPT-based models for argument mining
- **TinyLlama**: Lightweight LLM fine-tuned for argument tasks
- **ModernBERT**: PEFT-adapted encoder model
- **DeBERTa**: Pre-trained encoder for stance classification

## Installation

For detailed installation instructions, please see [INSTALLATION.md](INSTALLATION.md).

### Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/Horizontal-Labs/benchmark.git
cd benchmark

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -e external/argument-mining-api
pip install -e external/argument-mining-db

# Create .env file in the benchmark root directory
# Add your API keys (see INSTALLATION.md for details)

# Run benchmark
python run.py
```

## Project Structure

```
benchmark/
├── run.py                     # Main entry point
├── requirements.txt           # Benchmark dependencies
├── INSTALLATION.md           # Detailed installation guide
├── .env                      # Environment variables (create this)
├── app/                      # Benchmark application code
│   ├── benchmark.py          # Core benchmarking logic
│   └── log.py               # Logging configuration
├── external/                 # Git submodules
│   ├── argument-mining-api/  # API implementations
│   └── argument-mining-db/   # Database layer
└── results/                  # Benchmark outputs (CSV files)
```

## Usage

### Basic Usage

Run the benchmark with default settings (2 samples):
```bash
python run.py
```

### Advanced Usage

Modify `run.py` to customize:
- Number of samples to benchmark
- Which models to evaluate
- Which tasks to run

### Environment Variables

Create a `.env` file in the benchmark root directory:

```env
# Required for OpenAI models
OPENAI_API_KEY=your-api-key-here

# Optional: HuggingFace private models
HF_TOKEN=your-huggingface-token

# Optional: Database connection (defaults to cached data)
DATABASE_URL=mysql+pymysql://user:password@host:port/dbname
# Or use individual variables:
# DB_HOST=localhost
# DB_PORT=3306
# DB_NAME=argument-mining
# DB_USER=username
# DB_PASSWORD=password
```

## Submodules

This project uses two git submodules that can also work standalone:

- **argument-mining-api**: Contains model implementations and interfaces
- **argument-mining-db**: Database models and queries for benchmark data

Each submodule has its own `requirements.txt` and can be used independently. See [INSTALLATION.md](INSTALLATION.md) for standalone usage instructions.

## Results

Benchmark results are saved as CSV files in the `results/` directory:
- `adu_extraction_results_[timestamp].csv`
- `stance_classification_results_[timestamp].csv`
- `claim_premise_linking_results_[timestamp].csv`

## Requirements

- Python 3.9+ (3.12 recommended)
- CUDA-capable GPU (optional, for faster inference)
- 8GB+ RAM recommended

## Troubleshooting

See [INSTALLATION.md](INSTALLATION.md#troubleshooting) for common issues and solutions.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Citation

[Add citation information if applicable]