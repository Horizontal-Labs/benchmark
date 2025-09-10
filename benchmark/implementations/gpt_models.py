"""
OpenAI GPT models implementation for argument mining benchmark.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import OpenAI components
try:
    from argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
    OPENAI_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    OPENAI_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Try to provide more helpful error message
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class GPT41Implementation(BaseImplementation):
    """GPT-4.1 implementation for argument mining."""
    
    def __init__(self):
        super().__init__("gpt-4.1")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize GPT-4.1 implementation."""
        if not OPENAI_AVAILABLE:
            log_initialization(self.logger, "GPT-4.1", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components
            self.adu_classifier = OpenAILLMClassifier(model_name="gpt-4.1")
            self.linker = None  # OpenAI models don't have separate linking capability
            
            log_initialization(self.logger, "GPT-4.1", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "GPT-4.1", "failed", f"Initialization error: {e}")
            self.logger.error(f"GPT-4.1 initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPT-4.1 implementation is available."""
        return OPENAI_AVAILABLE


class GPT5Implementation(BaseImplementation):
    """GPT-5 implementation for argument mining."""
    
    def __init__(self):
        super().__init__("gpt-5")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize GPT-5 implementation."""
        if not OPENAI_AVAILABLE:
            log_initialization(self.logger, "GPT-5", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components
            self.adu_classifier = OpenAILLMClassifier(model_name="gpt-5")
            self.linker = None  # OpenAI models don't have separate linking capability
            
            log_initialization(self.logger, "GPT-5", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "GPT-5", "failed", f"Initialization error: {e}")
            self.logger.error(f"GPT-5 initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPT-5 implementation is available."""
        return OPENAI_AVAILABLE


class GPT5MiniImplementation(BaseImplementation):
    """GPT-5 Mini implementation for argument mining."""
    
    def __init__(self):
        super().__init__("gpt-5-mini")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize GPT-5 Mini implementation."""
        if not OPENAI_AVAILABLE:
            log_initialization(self.logger, "GPT-5 Mini", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components
            self.adu_classifier = OpenAILLMClassifier(model_name="gpt-5-mini")
            self.linker = None  # OpenAI models don't have separate linking capability
            
            log_initialization(self.logger, "GPT-5 Mini", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "GPT-5 Mini", "failed", f"Initialization error: {e}")
            self.logger.error(f"GPT-5 Mini initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPT-5 Mini implementation is available."""
        return OPENAI_AVAILABLE

