"""
Pytest configuration for the refactored benchmark package tests.
"""

import sys
import os
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the benchmark package to the path
benchmark_path = project_root / "benchmark"
sys.path.insert(0, str(benchmark_path))

# Add external API app to path to resolve imports
external_api_app = project_root / "external" / "argument-mining-api" / "app"
if external_api_app.exists():
    sys.path.insert(0, str(external_api_app))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    pass


@pytest.fixture(scope="session")
def benchmark_package_path():
    """Return the path to the benchmark package."""
    return benchmark_path


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture(scope="session")
def external_api_path():
    """Return the external API path."""
    return external_api_app
