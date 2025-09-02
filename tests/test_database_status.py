#!/usr/bin/env python3
"""
Database status checker for quick diagnostics.

This module provides simple tests to check database connectivity
and provide diagnostic information about the database state.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / '.env')

# Add external DB to path
external_db_path = project_root / "external" / "argument-mining-db"
if str(external_db_path) not in sys.path:
    sys.path.insert(0, str(external_db_path))


def check_database_status():
    """Check database connection status and provide diagnostic information."""
    print("=" * 60)
    print("DATABASE CONNECTION STATUS CHECK")
    print("=" * 60)
    
    # Check environment variables
    print("\n1. Environment Variables:")
    print("-" * 30)
    
    env_vars = {
        'DB_HOST': os.getenv('DB_HOST'),
        'DB_PORT': os.getenv('DB_PORT'),
        'DB_NAME': os.getenv('DB_NAME'),
        'DB_USER': os.getenv('DB_USER'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', '***HIDDEN***'),
        'CACHE_ENABLED': os.getenv('CACHE_ENABLED'),
        'DATABASE_URL': os.getenv('DATABASE_URL', 'Not set')
    }
    
    for var, value in env_vars.items():
        status = "✅" if value else "❌"
        print(f"{status} {var}: {value}")
    
    # Check database imports
    print("\n2. Database Imports:")
    print("-" * 30)
    
    try:
        from db.config import DB_URI, DB_HOST, DB_PORT, DB_NAME, DB_USER, CACHE_ENABLED
        print("✅ Database configuration imported successfully")
        print(f"   DB_URI: {DB_URI[:50]}..." if len(str(DB_URI)) > 50 else f"   DB_URI: {DB_URI}")
        print(f"   CACHE_ENABLED: {CACHE_ENABLED}")
    except ImportError as e:
        print(f"❌ Failed to import database configuration: {e}")
        return False
    
    # Check database connection
    print("\n3. Database Connection:")
    print("-" * 30)
    
    try:
        from db.db import get_session, get_engine
        
        # Test engine creation
        print("🔄 Creating database engine...")
        engine = get_engine()
        print("✅ Database engine created successfully")
        
        # Test session creation
        print("🔄 Creating database session...")
        session = get_session()
        print("✅ Database session created successfully")
        
        # Test simple query
        print("🔄 Testing simple query...")
        from sqlalchemy import text
        result = session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        if row and row.test == 1:
            print("✅ Simple query executed successfully")
        else:
            print("⚠️  Query executed but returned unexpected result")
        
        # Close session
        session.close()
        print("✅ Database session closed successfully")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # Check database queries
    print("\n4. Database Queries:")
    print("-" * 30)
    
    try:
        from db.queries import get_benchmark_data
        
        print("🔄 Testing benchmark data query...")
        claims, premises, topics = get_benchmark_data()
        
        print(f"✅ Benchmark data retrieved successfully")
        print(f"   Claims: {len(claims)}")
        print(f"   Premises: {len(premises)}")
        print(f"   Topics: {len(topics)}")
        
        if len(claims) > 0:
            print(f"   Sample claim: {claims[0].text[:100]}..." if hasattr(claims[0], 'text') and len(str(claims[0].text)) > 100 else f"   Sample claim: {claims[0]}")
        
    except Exception as e:
        print(f"❌ Benchmark data query failed: {e}")
        return False
    
    # Check database models
    print("\n5. Database Models:")
    print("-" * 30)
    
    try:
        from db.models import ADU, Relationship, Domain
        
        print("✅ Database models imported successfully")
        
        # Test table queries
        print("🔄 Testing table queries...")
        
        session = get_session()
        
        # Count records in each table
        adu_count = session.query(ADU).count()
        rel_count = session.query(Relationship).count()
        domain_count = session.query(Domain).count()
        
        print(f"   ADU table: {adu_count} records")
        print(f"   Relationship table: {rel_count} records")
        print(f"   Domain table: {domain_count} records")
        
        session.close()
        
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ DATABASE STATUS CHECK COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return True


def check_benchmark_integration():
    """Check if benchmark can integrate with database."""
    print("\n" + "=" * 60)
    print("BENCHMARK DATABASE INTEGRATION CHECK")
    print("=" * 60)
    
    try:
        from app.benchmark import ArgumentMiningBenchmark
        
        print("🔄 Testing benchmark initialization with database...")
        benchmark = ArgumentMiningBenchmark(max_samples=5)
        
        print(f"✅ Benchmark initialized successfully")
        print(f"   Data samples loaded: {len(benchmark.data)}")
        
        if len(benchmark.data) > 0:
            print(f"   Sample text: {benchmark.data[0]['text'][:100]}...")
            print(f"   Sample stance: {benchmark.data[0]['ground_truth']['stance']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark integration failed: {e}")
        return False


def main():
    """Main function to run database status checks."""
    print("Starting database status checks...")
    
    # Check basic database status
    db_status = check_database_status()
    
    if db_status:
        # Check benchmark integration
        benchmark_status = check_benchmark_integration()
        
        if benchmark_status:
            print("\n🎉 All checks passed! Database is working correctly.")
            return 0
        else:
            print("\n⚠️  Database is working but benchmark integration has issues.")
            return 1
    else:
        print("\n❌ Database connection failed. Check your configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

