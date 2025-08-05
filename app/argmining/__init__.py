"""
Argument Mining API package for the benchmark.
This integrates the argument-mining-api repository functionality.
"""

from .interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from .interfaces.claim_premise_linker import ClaimPremiseLinker
from .models.argument_units import (
    ArgumentUnit, 
    UnlinkedArgumentUnits, 
    LinkedArgumentUnits, 
    LinkedArgumentUnitsWithStance,
    StanceRelation,
    ClaimPremiseRelationship
)

__all__ = [
    'AduAndStanceClassifier',
    'ClaimPremiseLinker',
    'ArgumentUnit',
    'UnlinkedArgumentUnits',
    'LinkedArgumentUnits',
    'LinkedArgumentUnitsWithStance',
    'StanceRelation',
    'ClaimPremiseRelationship'
]
