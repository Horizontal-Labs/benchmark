"""
HuggingFace Chat LLM models implementation for argument mining benchmark.
"""

import traceback
from typing import Dict, Any

from .base import BaseImplementation
from ..utils.logging_utils import get_logger, log_initialization

# Import HuggingFace Chat LLM components
try:
    from argmining.implementations.hf_chat_llm_classifier import HFChatLLMClassifier
    HF_CHAT_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    HF_CHAT_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Try to provide more helpful error message
    if "relative import" in str(e).lower():
        IMPORT_ERROR = f"Relative import issue: {e}. This is likely due to the external API's import structure."
    else:
        IMPORT_ERROR = f"Import failed: {e}"


class Llama33BImplementation(BaseImplementation):
    """Llama 3.2 3B Instruct implementation for argument mining."""
    
    def __init__(self):
        super().__init__("llama3-3b")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize Llama 3.2 3B Instruct implementation."""
        if not HF_CHAT_AVAILABLE:
            log_initialization(self.logger, "Llama 3.2 3B Instruct", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components
            self.adu_classifier = HFChatLLMClassifier(
                base_model_id="meta-llama/Llama-3.2-3B-Instruct",
                name="Llama3.2-3B-Instruct"
            )
            self.linker = None  # HuggingFace chat models don't have separate linking capability
            
            log_initialization(self.logger, "Llama 3.2 3B Instruct", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "Llama 3.2 3B Instruct", "failed", f"Initialization error: {e}")
            self.logger.error(f"Llama 3.2 3B Instruct initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if Llama 3.2 3B Instruct implementation is available."""
        return HF_CHAT_AVAILABLE


class Qwen25BImplementation(BaseImplementation):
    """Qwen 2.5 1.5B Instruct implementation for argument mining."""
    
    def __init__(self):
        super().__init__("qwen2.5-1.5b")
        self.logger = get_logger()
    
    def initialize(self) -> bool:
        """Initialize Qwen 2.5 1.5B Instruct implementation."""
        if not HF_CHAT_AVAILABLE:
            log_initialization(self.logger, "Qwen 2.5 1.5B Instruct", "failed", f"Import error: {IMPORT_ERROR}")
            return False
        
        try:
            # Initialize components
            self.adu_classifier = HFChatLLMClassifier(
                base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
                name="Qwen2.5-1.5B-Instruct"
            )
            self.linker = None  # HuggingFace chat models don't have separate linking capability
            
            log_initialization(self.logger, "Qwen 2.5 1.5B Instruct", "success")
            return True
        except Exception as e:
            log_initialization(self.logger, "Qwen 2.5 1.5B Instruct", "failed", f"Initialization error: {e}")
            self.logger.error(f"Qwen 2.5 1.5B Instruct initialization traceback: {traceback.format_exc()}")
            return False
    
    def is_available(self) -> bool:
        """Check if Qwen 2.5 1.5B Instruct implementation is available."""
        return HF_CHAT_AVAILABLE

