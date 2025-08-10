"""
Configuration for the argument mining benchmark project.
Centralizes submodule paths and other configuration settings.
"""

from pathlib import Path
import os


class Config:
    """Configuration class for the benchmark project."""
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Integrated component paths (no longer using submodules)
    DB_CONNECTOR_PATH = PROJECT_ROOT / "app" / "db_connector"
    API_PATH = PROJECT_ROOT / "app" / "argument_mining_api"
    
    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Ensure directories exist
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.DATA_DIR, cls.RESULTS_DIR]:
            directory.mkdir(exist_ok=True)
    
    # Component validation
    @classmethod
    def validate_components(cls):
        """Validate that all required components exist."""
        missing = []
        
        if not cls.DB_CONNECTOR_PATH.exists():
            missing.append("db_connector")
        
        if not cls.API_PATH.exists():
            missing.append("argument_mining_api")
        
        if missing:
            raise RuntimeError(
                f"Missing components: {', '.join(missing)}. "
                "These should be part of the integrated project structure."
            )
    
    # Environment detection
    @classmethod
    def is_development(cls):
        """Check if running in development mode."""
        return os.getenv("ENVIRONMENT", "development") == "development"
    
    @classmethod
    def is_production(cls):
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT") == "production"


# Global configuration instance
config = Config() 