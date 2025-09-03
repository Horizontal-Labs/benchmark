"""
Task-specific benchmark implementations.
"""

from .base import BaseTask
from .adu_extraction import ADUExtractionTask
from .stance_classification import StanceClassificationTask
from .claim_premise_linking import ClaimPremiseLinkingTask

__all__ = [
    "BaseTask",
    "ADUExtractionTask",
    "StanceClassificationTask",
    "ClaimPremiseLinkingTask",
]
