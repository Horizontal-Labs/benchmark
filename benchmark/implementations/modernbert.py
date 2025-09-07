"""
ModernBERT implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

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
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize ModernBERT implementation."""
        if not MODERNBERT_AVAILABLE:
            log_initialization(self.logger, "ModernBERT", "failed", "ModernBERT components not available")
            return False
        
        try:
            # Get ModernBERT configuration
            modernbert_config = MODEL_CONFIGS.get('modernbert')
            if not modernbert_config:
                log_initialization(self.logger, "ModernBERT", "failed", "No configuration found")
                return False
            
            # Initialize components
            self.adu_classifier = PeftEncoderModelLoader(**modernbert_config['params'])
            # ModernBERT doesn't have linking capability
            self.linker = None
            
            log_initialization(self.logger, "ModernBERT", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "ModernBERT", "failed", f"Initialization error: {e}")
            self.logger.error(f"ModernBERT initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if ModernBERT implementation is available."""
        return MODERNBERT_AVAILABLE and 'modernbert' in MODEL_CONFIGS
