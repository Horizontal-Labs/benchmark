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
    
    # External submodule paths
    EXTERNAL_DIR = PROJECT_ROOT / "external"
    ARGUMENT_MINING_API_PATH = EXTERNAL_DIR / "argument-mining-api"
    ARGUMENT_MINING_DB_PATH = EXTERNAL_DIR / "argument-mining-db"
    
    # Local component paths
    DB_CONNECTOR_PATH = PROJECT_ROOT / "app" / "db_connector"
    
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
        
        if not cls.ARGUMENT_MINING_API_PATH.exists():
            missing.append("argument-mining-api submodule")
        
        if not cls.ARGUMENT_MINING_DB_PATH.exists():
            missing.append("argument-mining-db submodule")
        
        if not cls.DB_CONNECTOR_PATH.exists():
            missing.append("local db_connector")
        
        if missing:
            raise RuntimeError(
                f"Missing components: {', '.join(missing)}. "
                "Please ensure all submodules are properly initialized with: "
                "git submodule update --init --recursive"
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
    
    # Submodule status
    @classmethod
    def get_submodule_status(cls):
        """Get status of external submodules."""
        status = {}
        
        status['argument_mining_api'] = {
            'path': cls.ARGUMENT_MINING_API_PATH,
            'exists': cls.ARGUMENT_MINING_API_PATH.exists(),
            'is_git_repo': (cls.ARGUMENT_MINING_API_PATH / '.git').exists()
        }
        
        status['argument_mining_db'] = {
            'path': cls.ARGUMENT_MINING_DB_PATH,
            'exists': cls.ARGUMENT_MINING_DB_PATH.exists(),
            'is_git_repo': (cls.ARGUMENT_MINING_DB_PATH / '.git').exists()
        }
        
        return status


# Global configuration instance
config = Config() 