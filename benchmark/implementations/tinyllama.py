"""
TinyLlama implementation for argument mining.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation

# Import TinyLlama components
try:
    from argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier
    TINYLLAMA_AVAILABLE = True
except ImportError:
    TINYLLAMA_AVAILABLE = False


class TinyLlamaImplementation(BaseImplementation):
    """TinyLlama implementation for argument mining."""
    
    def __init__(self):
        super().__init__("tinyllama")
    
    def initialize(self) -> bool:
        """Initialize TinyLlama implementation."""
        if not TINYLLAMA_AVAILABLE:
            return False
        
        try:
            # Initialize components
            self.adu_classifier = TinyLLamaLLMClassifier()
            # TinyLlama doesn't have linking capability
            self.linker = None
            
            return True
        except Exception as e:
            print(f"Failed to initialize TinyLlama implementation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if TinyLlama implementation is available."""
        return TINYLLAMA_AVAILABLE
