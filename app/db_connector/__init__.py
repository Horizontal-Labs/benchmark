"""
Database connector package for the argument mining benchmark.
This integrates the argument-mining-db repository functionality.
"""

from .db import queries, models, db

__all__ = ['queries', 'models', 'db']
