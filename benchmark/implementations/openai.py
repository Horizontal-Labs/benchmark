"""
OpenAI implementation for argument mining.
"""

import os
import traceback
from typing import Dict, Any

from .base import BaseImplementation

# Import OpenAI components
try:
    from argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
    from argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIImplementation(BaseImplementation):
    """OpenAI implementation for argument mining."""
    
    def __init__(self):
        super().__init__("openai")
    
    def initialize(self) -> bool:
        """Initialize OpenAI implementation."""
        if not OPENAI_AVAILABLE:
            return False
        
        try:
            # Check if API key is available
            if not os.getenv("OPEN_AI_KEY") and not os.getenv("OPENAI_API_KEY"):
                return False
            
            # Initialize components
            self.adu_classifier = OpenAILLMClassifier()
            self.linker = OpenAIClaimPremiseLinker()
            
            return True
        except Exception as e:
            print(f"Failed to initialize OpenAI implementation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if OpenAI implementation is available."""
        return OPENAI_AVAILABLE and bool(os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY"))
