#!/usr/bin/env python3
"""
Database test runner script.

This script provides an easy way to run database connection tests
with various options and configurations.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_database_tests(verbose=True, markers=None, skip_slow=False, database_only=False):
    """
    Run database tests with specified options.
    
    Args:
        verbose (bool): Run tests in verbose mode
        markers (str): Comma-separated list of pytest markers to run
        skip_slow (bool): Skip slow tests
        database_only (bool): Run only database tests
    """
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test file
    cmd.append("test_database_connection.py")
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    elif database_only:
        cmd.extend(["-m", "database"])
    
    # Skip slow tests if requested
    if skip_slow:
        cmd.extend(["-m", "not slow"])
    
    # Add output options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings"  # Disable warnings for cleaner output
    ])
    
    print(f"Running database tests with command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Run the tests
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ All database tests passed!")
        else:
            print("\n" + "=" * 60)
            print("❌ Some database tests failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_quick_database_tests():
    """Run a quick subset of database tests."""
    print("Running quick database tests...")
    return run_database_tests(
        verbose=True,
        markers="database and not slow",
        skip_slow=True
    )


def run_full_database_tests():
    """Run all database tests including slow ones."""
    print("Running full database tests (including slow tests)...")
    return run_database_tests(
        verbose=True,
        markers="database",
        skip_slow=False
    )


def run_database_connection_tests_only():
    """Run only database connection tests."""
    print("Running database connection tests only...")
    return run_database_tests(
        verbose=True,
        markers="database and not integration",
        skip_slow=True
    )


def run_integration_tests():
    """Run database integration tests."""
    print("Running database integration tests...")
    return run_database_tests(
        verbose=True,
        markers="integration",
        skip_slow=False
    )


def main():
    """Main function to run database tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run database connection tests')
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick database tests (skip slow tests)'
    )
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run all database tests including slow ones'
    )
    parser.add_argument(
        '--connection-only', 
        action='store_true',
        help='Run only database connection tests'
    )
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run database integration tests'
    )
    parser.add_argument(
        '--markers', 
        type=str,
        help='Comma-separated list of pytest markers to run'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Run tests in quiet mode (not verbose)'
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.quick:
        success = run_quick_database_tests()
    elif args.full:
        success = run_full_database_tests()
    elif args.connection_only:
        success = run_database_connection_tests_only()
    elif args.integration:
        success = run_integration_tests()
    elif args.markers:
        success = run_database_tests(
            verbose=not args.quiet,
            markers=args.markers
        )
    else:
        # Default: run quick tests
        success = run_quick_database_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

