"""
OpenAI implementation for argument mining.
"""

import os
import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

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
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize OpenAI implementation."""
        if not OPENAI_AVAILABLE:
            log_initialization(self.logger, "OpenAI", "failed", "OpenAI components not available")
            return False
        
        try:
            # Check if API key is available
            if not os.getenv("OPEN_AI_KEY") and not os.getenv("OPENAI_API_KEY"):
                log_initialization(self.logger, "OpenAI", "failed", "No API key found")
                return False
            
            # Initialize components
            self.adu_classifier = OpenAILLMClassifier()
            self.linker = OpenAIClaimPremiseLinker()
            
            log_initialization(self.logger, "OpenAI", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "OpenAI", "failed", f"Initialization error: {e}")
            self.logger.error(f"OpenAI initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if OpenAI implementation is available."""
        return OPENAI_AVAILABLE and bool(os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY"))
