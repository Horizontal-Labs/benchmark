"""
DeBERTa implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation

# Import DeBERTa components
try:
    from argmining.implementations.encoder_model_loader import NonTrainedEncoderModelLoader, MODEL_CONFIGS
    DEBERTA_AVAILABLE = True
except ImportError:
    DEBERTA_AVAILABLE = False


class DeBERTaImplementation(BaseImplementation):
    """DeBERTa implementation for argument mining."""
    
    def __init__(self):
        super().__init__("deberta")
    
    def initialize(self) -> bool:
        """Initialize DeBERTa implementation."""
        if not DEBERTA_AVAILABLE:
            return False
        
        try:
            # Get DeBERTa configuration
            deberta_config = MODEL_CONFIGS.get('deberta')
            if not deberta_config:
                return False
            
            # Extract model paths from params
            model_paths = deberta_config['params'].get('model_paths')
            if not model_paths:
                return False
            
            # Initialize components
            self.adu_classifier = NonTrainedEncoderModelLoader(model_paths=model_paths)
            # DeBERTa doesn't have linking capability
            self.linker = None
            
            return True
        except Exception as e:
            print(f"Failed to initialize DeBERTa implementation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if DeBERTa implementation is available."""
        return DEBERTA_AVAILABLE and 'deberta' in MODEL_CONFIGS
