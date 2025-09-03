"""
Base implementation interface for argument mining implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseImplementation(ABC):
    """Abstract base class for argument mining implementations."""
    
    def __init__(self, name: str):
        self.name = name
        self.adu_classifier = None
        self.linker = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the implementation. Returns True if successful."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the implementation is available and ready to use."""
        pass
    
    def get_adu_classifier(self):
        """Get the ADU classifier instance."""
        return self.adu_classifier
    
    def get_linker(self):
        """Get the claim-premise linker instance."""
        return self.linker
    
    def supports_task(self, task_name: str) -> bool:
        """Check if this implementation supports a specific task."""
        if task_name == 'adu_extraction':
            return self.adu_classifier is not None
        elif task_name == 'stance_classification':
            return self.adu_classifier is not None
        elif task_name == 'claim_premise_linking':
            return self.linker is not None
        return False
