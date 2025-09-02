#!/usr/bin/env python3
"""
Pytest configuration and fixtures for the argument mining benchmark tests.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / '.env')

# Add external paths
external_db_path = project_root / "external" / "argument-mining-db"
external_api_app = project_root / "external" / "argument-mining-api" / "app"

if str(external_db_path) not in sys.path:
    sys.path.insert(0, str(external_db_path))
if str(external_api_app) not in sys.path:
    sys.path.insert(0, str(external_api_app))


@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path for tests."""
    return project_root


@pytest.fixture(scope="session")
def external_db_path():
    """Provide the external database path for tests."""
    return project_root / "external" / "argument-mining-db"


@pytest.fixture(scope="session")
def external_api_path():
    """Provide the external API path for tests."""
    return project_root / "external" / "argument-mining-api" / "app"


@pytest.fixture(scope="session")
def test_data_path():
    """Provide the test data path."""
    return project_root / "test_results"


@pytest.fixture(scope="session")
def results_path():
    """Provide the results path."""
    return project_root / "results"


@pytest.fixture(scope="function")
def mock_database_connection():
    """Mock database connection for testing without real database."""
    with patch('db.db.get_session') as mock_session:
        # Create a mock session
        mock_db = Mock()
        mock_db.execute.return_value = Mock()
        mock_db.query.return_value.count.return_value = 100
        mock_db.close = Mock()
        
        mock_session.return_value = mock_db
        yield mock_session


@pytest.fixture(scope="function")
def mock_database_data():
    """Mock database data for testing."""
    mock_claims = [
        Mock(text="This is a test claim 1", id=1, type="claim"),
        Mock(text="This is a test claim 2", id=2, type="claim"),
        Mock(text="This is a test claim 3", id=3, type="claim"),
    ]
    
    mock_premises = [
        Mock(text="This is a test premise 1", id=101, type="premise"),
        Mock(text="This is a test premise 2", id=102, type="premise"),
    ]
    
    mock_topics = ["topic1", "topic2"]
    
    return mock_claims, mock_premises, mock_topics


@pytest.fixture(scope="function")
def mock_benchmark_data():
    """Mock benchmark data structure for testing."""
    return [
        {
            'text': 'This is a test claim about video games.',
            'ground_truth': {
                'adus': ['This is a test claim about video games.'],
                'stance': 'pro',
                'links': [],
                'relationships': []
            }
        },
        {
            'text': 'Another test claim about education.',
            'ground_truth': {
                'adus': ['Another test claim about education.'],
                'stance': 'con',
                'links': [],
                'relationships': []
            }
        }
    ]


@pytest.fixture(scope="function")
def mock_environment_variables():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'DB_HOST': 'testhost',
        'DB_PORT': '3306',
        'DB_NAME': 'testdb',
        'DB_USER': 'testuser',
        'DB_PASSWORD': 'testpass',
        'CACHE_ENABLED': 'True',
        'OPEN_AI_KEY': 'test-openai-key',
        'HF_TOKEN': 'test-hf-token'
    }):
        yield


@pytest.fixture(scope="function")
def cleanup_test_files():
    """Cleanup test files after tests."""
    yield
    
    # Cleanup any test files that might have been created
    test_files = [
        project_root / "test_results" / "test_benchmark_results.csv",
        project_root / "results" / "test_results.csv"
    ]
    
    for file_path in test_files:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "database: marks tests as database tests (deselect with '-m \"not database\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark database connection tests
        if "database" in item.name.lower() or "db" in item.name.lower():
            item.add_marker(pytest.mark.database)
        
        # Mark performance tests as slow
        if "performance" in item.name.lower() or "perf" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip database tests if database imports are not available
    if item.get_closest_marker('database'):
        try:
            from db.db import get_session
            from db.queries import get_benchmark_data
        except ImportError:
            pytest.skip("Database tests require database imports to be available")
    
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker('slow') and not item.config.getoption("--runslow"):
        pytest.skip("Slow tests are skipped by default. Use --runslow to run them.")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--database-only", 
        action="store_true", 
        default=False, 
        help="run only database tests"
    )
    parser.addoption(
        "--skip-database", 
        action="store_true", 
        default=False, 
        help="skip database tests"
    )

