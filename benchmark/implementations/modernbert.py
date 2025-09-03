"""
ModernBERT implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation

# Import ModernBERT components
try:
    from argmining.implementations.encoder_model_loader import PeftEncoderModelLoader, MODEL_CONFIGS
    MODERNBERT_AVAILABLE = True
except ImportError:
    MODERNBERT_AVAILABLE = False


class ModernBERTImplementation(BaseImplementation):
    """ModernBERT implementation for argument mining."""
    
    def __init__(self):
        super().__init__("modernbert")
    
    def initialize(self) -> bool:
        """Initialize ModernBERT implementation."""
        if not MODERNBERT_AVAILABLE:
            return False
        
        try:
            # Get ModernBERT configuration
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if not modernbert_config:
                return False
            
            # Initialize components
            self.adu_classifier = PeftEncoderModelLoader(**modernbert_config['params'])
            # ModernBERT doesn't have linking capability
            self.linker = None
            
            return True
        except Exception as e:
            print(f"Failed to initialize ModernBERT implementation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if ModernBERT implementation is available."""
        return MODERNBERT_AVAILABLE and 'modernbert' in MODEL_CONFIGS
