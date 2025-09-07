"""
DeBERTa implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import DeBERTa components
try:
    from argmining.implementations.encoder_model_loader import NonTrainedEncoderModelLoader, MODEL_CONFIGS
    DEBERTA_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    DEBERTA_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Try to provide more helpful error message
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class DeBERTaImplementation(BaseImplementation):
    """DeBERTa implementation for argument mining."""
    
    def __init__(self):
        super().__init__("deberta")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize DeBERTa implementation."""
        if not DEBERTA_AVAILABLE:
            log_initialization(self.logger, "DeBERTa", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Get DeBERTa configuration
            deberta_config = MODEL_CONFIGS.get('deberta')
            if not deberta_config:
                log_initialization(self.logger, "DeBERTa", "failed", "No configuration found")
                return False
            
            # Extract model paths from params
            model_paths = deberta_config['params'].get('model_paths')
            if not model_paths:
                log_initialization(self.logger, "DeBERTa", "failed", "No model paths found")
                return False
            
            # Initialize components
            self.adu_classifier = NonTrainedEncoderModelLoader(model_paths=model_paths)
            # DeBERTa doesn't have linking capability
            self.linker = None
            
            log_initialization(self.logger, "DeBERTa", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "DeBERTa", "failed", f"Initialization error: {e}")
            self.logger.error(f"DeBERTa initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if DeBERTa implementation is available."""
        return DEBERTA_AVAILABLE and 'deberta' in MODEL_CONFIGS
