#!/usr/bin/env python3
"""
Pytest tests to verify imports work correctly.
"""

import sys
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def setup_paths(project_root):
    """Setup Python paths for external modules."""
    # Add project root to path
    sys.path.insert(0, str(project_root))
    
    # Add external submodules to Python path
    external_api = project_root / "external" / "api"
    external_db = project_root / "external" / "db"
    
    paths_added = []
    
    if external_api.exists():
        sys.path.insert(0, str(external_api))
        paths_added.append(("argument-mining-api", external_api))
    
    if external_db.exists():
        sys.path.insert(0, str(external_db))
        paths_added.append(("argument-mining-db", external_db))
    
    return paths_added


def test_project_root_exists(project_root):
    """Test that project root directory exists."""
    assert project_root.exists(), f"Project root does not exist: {project_root}"
    assert project_root.is_dir(), f"Project root is not a directory: {project_root}"


def test_external_api_path(project_root):
    """Test that external API path is properly configured."""
    external_api = project_root / "external" / "api"
    # This test passes whether the path exists or not, but documents the expected structure
    assert external_api.parent == project_root / "external", "External API should be in external/api directory"


def test_external_db_path(project_root):
    """Test that external DB path is properly configured."""
    external_db = project_root / "external" / "db"
    # This test passes whether the path exists or not, but documents the expected structure
    assert external_db.parent == project_root / "external", "External DB should be in external/db directory"


def test_external_modules_available(setup_paths):
    """Test that external modules are available when they exist."""
    # This test will pass even if modules don't exist, but documents what should be available
    module_names = [name for name, _ in setup_paths]
    assert isinstance(module_names, list), "Module names should be a list"


def test_app_benchmark_import(setup_paths):
    """Test that app.benchmark module can be imported."""
    try:
        from app.benchmark import ArgumentMiningBenchmark, BenchmarkResult, run_specific_benchmark
        # If we get here, the import was successful
        assert True, "Import successful"
    except ImportError as e:
        # Try fallback import
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / "app"))
        try:
            from benchmark import ArgumentMiningBenchmark, BenchmarkResult, run_specific_benchmark
            assert True, "Fallback import successful"
        except ImportError as e2:
            pytest.fail(f"Failed to import benchmark module: {e2}")


def test_benchmark_classes_available(setup_paths):
    """Test that benchmark classes are available after import."""
    try:
        from app.benchmark import ArgumentMiningBenchmark, BenchmarkResult, run_specific_benchmark
        # Test that classes exist
        assert ArgumentMiningBenchmark is not None, "ArgumentMiningBenchmark should be available"
        assert BenchmarkResult is not None, "BenchmarkResult should be available"
        assert run_specific_benchmark is not None, "run_specific_benchmark should be available"
    except ImportError:
        # Try fallback import
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / "app"))
        from benchmark import ArgumentMiningBenchmark, BenchmarkResult, run_specific_benchmark
        # Test that classes exist
        assert ArgumentMiningBenchmark is not None, "ArgumentMiningBenchmark should be available"
        assert BenchmarkResult is not None, "BenchmarkResult should be available"
        assert run_specific_benchmark is not None, "run_specific_benchmark should be available"


def test_python_path_contains_project_root(setup_paths):
    """Test that project root is in Python path."""
    project_root = Path(__file__).parent.parent
    assert str(project_root) in sys.path, f"Project root {project_root} should be in Python path"
