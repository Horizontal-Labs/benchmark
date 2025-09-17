"""
TinyLlama Base implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import TinyLlama components
try:
    from argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    TINYLLAMA_BASE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    TINYLLAMA_BASE_AVAILABLE = False
    IMPORT_ERROR = str(e)
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class TinyLlamaBaseImplementation(BaseImplementation):
    """
    TinyLlama Base implementation for argument mining.
    
    This implementation explicitly uses the base TinyLlama model without fine-tuning.
    It may produce poorer results compared to the fine-tuned version but is useful
    for comparison purposes.
    """
    
    def __init__(self):
        super().__init__("tinyllama-base")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize TinyLlama Base implementation."""
        if not TINYLLAMA_BASE_AVAILABLE:
            log_initialization(self.logger, "TinyLlama Base", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components with explicit base model (no adapter)
            self.adu_classifier = TinyLLamaLLMClassifier(use_adapter=False)
            # TinyLlama doesn't have linking capability
            self.linker = None
            
            # Check if adapter was successfully disabled
            if hasattr(self.adu_classifier, 'use_adapter') and not self.adu_classifier.use_adapter:
                log_initialization(self.logger, "TinyLlama Base", "success", "Using base model without fine-tuning")
            else:
                log_initialization(self.logger, "TinyLlama Base", "warning", "Expected base model, but adapter might be active or failed to configure")
            return True
        except Exception as e:
            log_initialization(self.logger, "TinyLlama Base", "failed", f"Initialization error: {e}")
            self.logger.error(f"TinyLlama Base initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if TinyLlama Base implementation is available."""
        return TINYLLAMA_BASE_AVAILABLE
