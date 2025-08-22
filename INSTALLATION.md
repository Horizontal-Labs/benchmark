# Installation Guide

This guide explains how to install all dependencies for the Argument Mining Benchmark project.

## Prerequisites

- Python 3.9+ (Python 3.12 recommended)
- pip package manager
- Git (for cloning submodules)

## Installation Steps

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/Horizontal-Labs/benchmark.git
cd benchmark
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install the benchmark dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the submodules as editable packages

This step installs the argument-mining-api and argument-mining-db as Python packages, along with their dependencies:

```bash
pip install -e external/argument-mining-api
pip install -e external/argument-mining-db
```

Note: This will automatically install all dependencies including protobuf, torch, transformers, etc.

### 5. Set up environment variables

Create a `.env` file in the **benchmark root directory** (the main project folder where `run.py` is located):

```bash
# Navigate to the benchmark root directory
cd benchmark  # if not already there

# Create the .env file
# On Windows: use notepad or any text editor
# On Linux/Mac: use nano, vim, or any text editor
```

Add the following content to the `.env` file:

```env
# OpenAI API Key (required for OpenAI-based models)
OPENAI_API_KEY=your-api-key-here

# HuggingFace token (optional, for private models)
HF_TOKEN=your-hf-token-here

# Database configuration (optional - uses cached data if not provided)
# Option 1: Use a complete database URL
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/dbname

# Option 2: Or configure individual components
DB_HOST=localhost
DB_PORT=3306
DB_NAME=argument-mining
DB_USER=your-username
DB_PASSWORD=your-password

# Cache configuration (optional, defaults to True)
CACHE_ENABLED=True
```

**Important:** The `.env` file must be placed in the benchmark root directory (where `run.py` is located), not in any subdirectory.

## Dependency Overview

### Benchmark (`requirements.txt`)
- **Core**: pandas, numpy, scikit-learn for data processing and metrics
- **Utilities**: python-dotenv for environment variables, tqdm for progress bars
- **Optional**: plotly, rich for visualization and formatting

### Argument Mining API (`external/argument-mining-api/requirements.txt`)
- **ML/AI**: torch, transformers, peft for model loading and inference
- **API**: openai for GPT models, huggingface-hub for model downloads
- **Web**: fastapi, uvicorn for the REST API server
- **Optimization**: bitsandbytes for 8-bit quantization

### Argument Mining DB (`external/argument-mining-db/requirements.txt`)
- **Database**: SQLAlchemy ORM, PyMySQL/mysql-connector-python for MySQL
- **Data**: pandas, numpy for data processing
- **Migrations**: alembic for database schema management

## Standalone Usage of Submodules

Each submodule can also be used independently:

### Using argument-mining-api standalone

```bash
cd external/argument-mining-api
python -m venv venv
venv\Scripts\activate  # On Windows
# or: source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt

# Test the installation
python -c "from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier; print('Success')"
```

### Using argument-mining-db standalone

```bash
cd external/argument-mining-db
python -m venv venv
venv\Scripts\activate  # On Windows
# or: source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt

# Test the installation
python -c "from db.queries import get_benchmark_data; print('Success')"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed the submodules as editable packages (step 4)

2. **CUDA/GPU Issues**: If you don't have a GPU, the models will run on CPU (slower but functional)

3. **bitsandbytes on Windows**: The 8-bit optimizer warning is normal on Windows. Models will still work.

4. **OpenAI API Errors**: Ensure your OPENAI_API_KEY is set in the `.env` file in the benchmark root directory

5. **Database Connection**: 
   - The benchmark uses cached data by default if database is unavailable
   - If you see "No module named 'mariadb'", this is normal - PyMySQL will be used instead
   - To use a custom database, set DATABASE_URL in your `.env` file

6. **Missing .env file**: The `.env` file must be in the benchmark root directory (same folder as `run.py`)

### Minimal Installation

For a minimal installation without all features:

```bash
# Just core benchmark dependencies
pip install pandas numpy scikit-learn python-dotenv

# Then install submodules
pip install -e external/argument-mining-api
pip install -e external/argument-mining-db
```

## Verifying Installation

Run the benchmark with a small sample to verify everything works:

```bash
python run.py
```

This should run the benchmark with 2 samples using available models.

## Project Structure

```
benchmark/
├── run.py                     # Main entry point
├── requirements.txt           # Benchmark dependencies
├── .env                      # Environment variables (create this file here!)
├── app/                      # Benchmark application code
├── external/
│   ├── argument-mining-api/  # API submodule (can be used standalone)
│   │   ├── requirements.txt
│   │   └── app/
│   └── argument-mining-db/   # Database submodule (can be used standalone)
│       ├── requirements.txt
│       └── db/
└── results/                  # Benchmark results will be saved here
```