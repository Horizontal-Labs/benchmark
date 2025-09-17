"""
ModernBERT Base implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import ModernBERT components
try:
    from argmining.implementations.encoder_model_loader import PeftEncoderModelLoader, MODEL_CONFIGS
    MODERNBERT_BASE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODERNBERT_BASE_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Try to provide more helpful error message
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class ModernBERTBaseImplementation(BaseImplementation):
    """
    ModernBERT Base implementation for argument mining.
    
    This implementation uses the base ModernBERT model without fine-tuning.
    It may produce poorer results compared to the fine-tuned version but is useful
    for comparison purposes.
    """
    
    def __init__(self):
        super().__init__("modernbert-base")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize ModernBERT Base implementation."""
        if not MODERNBERT_BASE_AVAILABLE:
            log_initialization(self.logger, "ModernBERT Base", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components using base ModernBERT model without adapters
            # Use PeftEncoderModelLoader but with empty adapter paths to use only the base model
            self.adu_classifier = PeftEncoderModelLoader(
                base_model_path="answerdotai/ModernBERT-base",
                adapter_paths={}  # Empty adapter paths means no fine-tuning, just base model
            )
            # ModernBERT doesn't have linking capability
            self.linker = None
            
            log_initialization(self.logger, "ModernBERT Base", "success", "Using base model without fine-tuning")
            return True
        except Exception as e:
            log_initialization(self.logger, "ModernBERT Base", "failed", f"Initialization error: {e}")
            self.logger.error(f"ModernBERT Base initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if ModernBERT Base implementation is available."""
        return MODERNBERT_BASE_AVAILABLE
