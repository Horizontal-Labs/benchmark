"""
Data loading utilities for benchmark data.
"""

import traceback
from typing import Tuple, Any, Optional
from pathlib import Path

# Try to import database components
try:
    from db.queries import get_benchmark_data, get_benchmark_data_details
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


class DataLoader:
    """Data loader for benchmark data."""
    
    def __init__(self):
        self.db_available = DB_AVAILABLE
    
    def load_benchmark_data(self) -> Tuple[Any, Any, Any]:
        """Load benchmark data from database or fallback to local CSV."""
        if self.db_available:
            try:
                # Get benchmark data - returns (claims, premises, topics)
                claims, premises, topics = get_benchmark_data()
                print("Successfully loaded database components")
                return claims, premises, topics
            except Exception as e:
                print(f"Error importing database components: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Attempting to load local CSV data as fallback...")
        
        # Fallback to local CSV data
        return self._load_local_csv_data()
    
    def _load_local_csv_data(self) -> Tuple[Any, Any, Any]:
        """Load local CSV data as fallback."""
        # This would implement loading from local CSV files
        # For now, return empty data structures
        print("Local CSV loading not implemented yet")
        return [], [], []
    
    def is_database_available(self) -> bool:
        """Check if database is available."""
        return self.db_available
