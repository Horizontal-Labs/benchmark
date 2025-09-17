"""
TinyLlama Fine-tuned implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import TinyLlama components
try:
    from argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    TINYLLAMA_FINETUNED_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    TINYLLAMA_FINETUNED_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Try to provide more helpful error message
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class TinyLlamaFinetunedImplementation(BaseImplementation):
    """
    TinyLlama Fine-tuned implementation for argument mining.
    
    This implementation explicitly uses the fine-tuned PEFT adapter for better performance
    on argument mining tasks. It will fall back to the base model if the adapter fails to load.
    """
    
    def __init__(self):
        super().__init__("tinyllama-finetuned")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize TinyLlama Fine-tuned implementation."""
        if not TINYLLAMA_FINETUNED_AVAILABLE:
            log_initialization(self.logger, "TinyLlama Fine-tuned", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components with explicit fine-tuned adapter
            self.adu_classifier = TinyLLamaLLMClassifier(use_adapter=True)
            # TinyLlama doesn't have linking capability
            self.linker = None
            
            # Check if adapter was successfully loaded
            if hasattr(self.adu_classifier, 'use_adapter') and self.adu_classifier.use_adapter:
                log_initialization(self.logger, "TinyLlama Fine-tuned", "success", "Using fine-tuned adapter")
            else:
                log_initialization(self.logger, "TinyLlama Fine-tuned", "warning", "Using base model only - adapter failed to load")
            return True
        except Exception as e:
            log_initialization(self.logger, "TinyLlama Fine-tuned", "failed", f"Initialization error: {e}")
            self.logger.error(f"TinyLlama Fine-tuned initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if TinyLlama Fine-tuned implementation is available."""
        return TINYLLAMA_FINETUNED_AVAILABLE
