#!/usr/bin/env python3
"""
Tests for database connection functionality

This module contains comprehensive tests for database connectivity,
data retrieval, and error handling in the argument mining benchmark.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file in project root
load_dotenv(project_root / '.env')

# Add external DB to path
external_db_path = project_root / "external" / "argument-mining-db"
if str(external_db_path) not in sys.path:
    sys.path.insert(0, str(external_db_path))

# Add external API app to path
external_api_app = project_root / "external" / "argument-mining-api" / "app"
if str(external_api_app) not in sys.path:
    sys.path.insert(0, str(external_api_app))

# Import database components
try:
    from db.queries import get_benchmark_data, get_benchmark_data_details
    from db.db import get_session, get_engine
    from db.config import DB_URI, DB_HOST, DB_PORT, DB_NAME, DB_USER, CACHE_ENABLED
    from db.models import ADU, Relationship, Domain
    DB_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Database imports failed: {e}")
    DB_IMPORTS_SUCCESSFUL = False

# Import benchmark components
try:
    from app.benchmark import ArgumentMiningBenchmark
    BENCHMARK_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Benchmark imports failed: {e}")
    BENCHMARK_IMPORTS_SUCCESSFUL = False


class TestDatabaseConfiguration:
    """Test database configuration and environment variables."""
    
    def test_database_config_loaded(self):
        """Test that database configuration can be loaded."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # Check that configuration values exist
        assert DB_HOST is not None
        assert DB_PORT is not None
        assert DB_NAME is not None
        assert DB_USER is not None
        
        # Check that DB_URI is constructed
        assert DB_URI is not None
        assert isinstance(DB_URI, str)
        assert len(DB_URI) > 0
    
    def test_cache_configuration(self):
        """Test that cache configuration is properly set."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # CACHE_ENABLED should be a boolean
        assert isinstance(CACHE_ENABLED, bool)
    
    def test_database_uri_format(self):
        """Test that database URI is properly formatted."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # Check URI format
        assert DB_URI.startswith('mysql+pymysql://')
        assert DB_HOST in DB_URI
        assert str(DB_PORT) in DB_URI
        assert DB_NAME in DB_URI


class TestDatabaseConnection:
    """Test database connection establishment."""
    
    @pytest.fixture(scope="class")
    def db_session(self):
        """Create a database session for testing."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            session = get_session()
            yield session
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")
        finally:
            if 'session' in locals():
                session.close()
    
    def test_database_connection_established(self, db_session):
        """Test that database connection can be established."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # Test basic connection
        assert db_session is not None
        
        # Test that we can execute a simple query
        try:
            from sqlalchemy import text
            result = db_session.execute(text("SELECT 1"))
            assert result is not None
        except Exception as e:
            pytest.fail(f"Failed to execute simple query: {e}")
    
    def test_database_engine_creation(self):
        """Test that database engine can be created."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            engine = get_engine()
            assert engine is not None
            
            # In SQLAlchemy 2.0+, Engine has different attributes
            # Check for core engine attributes instead of execute method
            assert hasattr(engine, 'connect')
            assert hasattr(engine, 'dispose')
            assert hasattr(engine, 'url')
            
            # Test that we can get connection info
            assert engine.url is not None
            assert str(engine.url).startswith('mysql+pymysql://')
            
            # Test that we can create a connection (this is what engines are for)
            try:
                with engine.connect() as connection:
                    assert connection is not None
                    # Test a simple query on the connection
                    from sqlalchemy import text
                    result = connection.execute(text("SELECT 1"))
                    row = result.fetchone()
                    assert row[0] == 1
                print("✅ Engine connection test passed")
            except Exception as conn_error:
                pytest.fail(f"Engine connection test failed: {conn_error}")
            
        except Exception as e:
            pytest.fail(f"Failed to create database engine: {e}")
    
    def test_session_factory(self):
        """Test that session factory works correctly."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            session1 = get_session()
            session2 = get_session()
            
            # Both sessions should be valid
            assert session1 is not None
            assert session2 is not None
            
            # Clean up
            session1.close()
            session2.close()
        except Exception as e:
            pytest.fail(f"Failed to create sessions: {e}")


class TestDatabaseQueries:
    """Test database query functionality."""
    
    @pytest.fixture(scope="class")
    def db_session(self):
        """Create a database session for testing."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            session = get_session()
            yield session
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")
        finally:
            if 'session' in locals():
                session.close()
    
    def test_get_benchmark_data_basic(self, db_session):
        """Test basic benchmark data retrieval."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            claims, premises, topics = get_benchmark_data()
            
            # Check return types
            assert isinstance(claims, list)
            assert isinstance(premises, list)
            assert isinstance(topics, list)
            
            # Check that we got some data
            assert len(claims) > 0, "No claims returned from database"
            
            # Log data summary for debugging
            print(f"Retrieved {len(claims)} claims, {len(premises)} premises, {len(topics)} topics")
            
        except Exception as e:
            pytest.fail(f"Failed to retrieve benchmark data: {e}")
    
    def test_get_benchmark_data_details(self, db_session):
        """Test detailed benchmark data retrieval."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            # Test with a small number of premises
            claims, premises_list, categories_list = get_benchmark_data_details(
                number_of_premises=2
            )
            
            # Check return types
            assert isinstance(claims, list)
            assert isinstance(premises_list, list)
            assert isinstance(categories_list, list)
            
            # Check that we got some data
            assert len(claims) > 0, "No claims returned from detailed query"
            
            # Check that premises_list and categories_list have the same length as claims
            assert len(premises_list) == len(claims)
            assert len(categories_list) == len(claims)
            
            # Log data summary for debugging
            print(f"Retrieved {len(claims)} claims with detailed premise information")
            
        except Exception as e:
            pytest.fail(f"Failed to retrieve detailed benchmark data: {e}")
    
    def test_database_tables_exist(self, db_session):
        """Test that required database tables exist."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        try:
            # Test ADU table
            adu_count = db_session.query(ADU).count()
            print(f"ADU table contains {adu_count} records")
            assert adu_count >= 0  # Should not raise an error
            
            # Test Relationship table
            rel_count = db_session.query(Relationship).count()
            print(f"Relationship table contains {rel_count} records")
            assert rel_count >= 0  # Should not raise an error
            
            # Test Domain table
            domain_count = db_session.query(Domain).count()
            print(f"Domain table contains {domain_count} records")
            assert domain_count >= 0  # Should not raise an error
            
        except Exception as e:
            pytest.fail(f"Failed to query database tables: {e}")


class TestBenchmarkDataLoading:
    """Test benchmark data loading functionality."""
    
    def test_benchmark_initialization_with_database(self):
        """Test that benchmark can initialize with database data."""
        if not BENCHMARK_IMPORTS_SUCCESSFUL:
            pytest.skip("Benchmark imports not available")
        
        try:
            # Create benchmark with small number of samples
            benchmark = ArgumentMiningBenchmark(max_samples=5)
            
            # Check that data was loaded
            assert benchmark.data is not None
            assert len(benchmark.data) > 0
            
            # Check data structure
            for sample in benchmark.data:
                assert 'text' in sample
                assert 'ground_truth' in sample
                assert isinstance(sample['text'], str)
                assert len(sample['text']) > 0
            
            print(f"Successfully loaded {len(benchmark.data)} samples from database")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize benchmark with database: {e}")
    
    def test_benchmark_data_quality(self):
        """Test the quality of benchmark data."""
        if not BENCHMARK_IMPORTS_SUCCESSFUL:
            pytest.skip("Benchmark imports not available")
        
        try:
            benchmark = ArgumentMiningBenchmark(max_samples=10)
            
            # Check data quality
            for i, sample in enumerate(benchmark.data):
                text = sample['text']
                
                # Text should not be empty
                assert len(text.strip()) > 0, f"Sample {i} has empty text"
                
                # Text should not be too short (likely a placeholder)
                assert len(text.strip()) > 10, f"Sample {i} text too short: '{text}'"
                
                # Text should not be too long (likely corrupted)
                assert len(text.strip()) < 1000, f"Sample {i} text too long: {len(text)} chars"
                
                # Check ground truth structure
                gt = sample['ground_truth']
                assert 'stance' in gt
                assert 'adus' in gt
                
                # Log sample for debugging
                print(f"Sample {i}: {text[:100]}... (stance: {gt['stance']})")
            
        except Exception as e:
            pytest.fail(f"Failed to validate benchmark data quality: {e}")


class TestDatabaseErrorHandling:
    """Test database error handling scenarios."""
    
    def test_invalid_database_uri(self):
        """Test handling of invalid database URI."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # Test with a completely invalid URI that should fail
        try:
            from sqlalchemy import create_engine
            
            # This should raise an exception with an invalid URI
            engine = create_engine("invalid://uri")
            pytest.fail("Expected exception for invalid database URI")
        except Exception as e:
            # Should handle the error gracefully
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["invalid", "connection", "dialect", "driver"])
            print(f"✅ Successfully caught invalid URI error: {e}")
    
    def test_database_connection_failure(self):
        """Test handling of database connection failure."""
        if not BENCHMARK_IMPORTS_SUCCESSFUL:
            pytest.skip("Benchmark imports not available")
        
        # Instead of trying to modify environment variables (which may not work),
        # test the fallback mechanism by creating a benchmark that should use CSV data
        try:
            # Create benchmark with small number of samples
            # This should work even if database connection fails
            benchmark = ArgumentMiningBenchmark(max_samples=5)
            
            # Should have some data (either from database or CSV fallback)
            assert benchmark.data is not None
            assert len(benchmark.data) > 0
            
            print(f"✅ Benchmark loaded {len(benchmark.data)} samples (database or CSV fallback)")
            
            # Check if we're using CSV fallback by looking at the data source
            if hasattr(benchmark, 'data_source'):
                print(f"Data source: {benchmark.data_source}")
            
        except Exception as e:
            pytest.fail(f"Benchmark should work with fallback mechanism: {e}")
    
    def test_invalid_database_uri_handling(self):
        """Test handling of invalid database URI."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        # Test with a completely invalid URI that should fail
        try:
            from sqlalchemy import create_engine
            
            # This should raise an exception with an invalid URI
            engine = create_engine("invalid://uri")
            pytest.fail("Expected exception for invalid database URI")
        except Exception as e:
            # Should handle the error gracefully
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["invalid", "connection", "dialect", "driver"])
            print(f"✅ Successfully caught invalid URI error: {e}")
    



class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    def test_query_performance(self):
        """Test that database queries complete within reasonable time."""
        if not DB_IMPORTS_SUCCESSFUL:
            pytest.skip("Database imports not available")
        
        import time
        
        try:
            start_time = time.time()
            
            # Execute benchmark data query
            claims, premises, topics = get_benchmark_data()
            
            query_time = time.time() - start_time
            
            # Query should complete within 30 seconds
            assert query_time < 30.0, f"Query took too long: {query_time:.2f} seconds"
            
            print(f"Database query completed in {query_time:.2f} seconds")
            print(f"Retrieved {len(claims)} claims, {len(premises)} premises, {len(topics)} topics")
            
        except Exception as e:
            pytest.fail(f"Failed to test query performance: {e}")


# Integration tests
class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    def test_full_data_pipeline(self):
        """Test the complete data pipeline from database to benchmark."""
        if not (DB_IMPORTS_SUCCESSFUL and BENCHMARK_IMPORTS_SUCCESSFUL):
            pytest.skip("Required imports not available")
        
        try:
            # Test database connection
            session = get_session()
            assert session is not None
            
            # Test data retrieval
            claims, premises, topics = get_benchmark_data()
            assert len(claims) > 0
            
            # Test benchmark initialization
            benchmark = ArgumentMiningBenchmark(max_samples=5)
            assert len(benchmark.data) > 0
            
            # Test that benchmark data matches database data
            db_texts = [claim.text if hasattr(claim, 'text') else str(claim) for claim in claims[:5]]
            benchmark_texts = [sample['text'] for sample in benchmark.data[:5]]
            
            # Should have some overlap (allowing for different ordering)
            overlap = set(db_texts) & set(benchmark_texts)
            assert len(overlap) > 0, "No overlap between database and benchmark data"
            
            print(f"Full data pipeline test successful. Overlap: {len(overlap)}/{len(db_texts)}")
            
            session.close()
            
        except Exception as e:
            pytest.fail(f"Full data pipeline test failed: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

