#!/usr/bin/env python3
"""
Argument Mining Module Bridge

This module imports and re-exports the necessary classes and functions
from the external argument-mining-api submodule to provide a clean
interface for the benchmark application.
"""

import sys
from pathlib import Path

# Add external API to path if not already there
external_api = Path(__file__).parent.parent.parent / "external" / "api"
if external_api.exists() and str(external_api) not in sys.path:
    sys.path.insert(0, str(external_api))

# Import all necessary modules from external API using importlib
try:
    import importlib
    
    # Import interfaces
    adu_classifier_module = importlib.import_module('app.argmining.interfaces.adu_and_stance_classifier')
    claim_linker_module = importlib.import_module('app.argmining.interfaces.claim_premise_linker')
    
    # Import models
    models_module = importlib.import_module('app.argmining.models.argument_units')
    
    # Import implementations
    openai_classifier_module = importlib.import_module('app.argmining.implementations.openai_llm_classifier')
    tinyllama_classifier_module = importlib.import_module('app.argmining.implementations.tinyllama_llm_classifier')
    encoder_loader_module = importlib.import_module('app.argmining.implementations.encoder_model_loader')
    openai_linker_module = importlib.import_module('app.argmining.implementations.openai_claim_premise_linker')
    
    # Get the classes from the modules
    AduAndStanceClassifier = adu_classifier_module.AduAndStanceClassifier
    ClaimPremiseLinker = claim_linker_module.ClaimPremiseLinker
    ArgumentUnit = models_module.ArgumentUnit
    UnlinkedArgumentUnits = models_module.UnlinkedArgumentUnits
    LinkedArgumentUnits = models_module.LinkedArgumentUnits
    LinkedArgumentUnitsWithStance = models_module.LinkedArgumentUnitsWithStance
    StanceRelation = models_module.StanceRelation
    ClaimPremiseRelationship = models_module.ClaimPremiseRelationship
    OpenAILLMClassifier = openai_classifier_module.OpenAILLMClassifier
    TinyLLamaLLMClassifier = tinyllama_classifier_module.TinyLLamaLLMClassifier
    PeftEncoderModelLoader = encoder_loader_module.PeftEncoderModelLoader
    NonTrainedEncoderModelLoader = encoder_loader_module.NonTrainedEncoderModelLoader
    MODEL_CONFIGS = encoder_loader_module.MODEL_CONFIGS
    OpenAIClaimPremiseLinker = openai_linker_module.OpenAIClaimPremiseLinker
    
    # Define what should be available when importing from this module
    __all__ = [
        # Interfaces
        'AduAndStanceClassifier',
        'ClaimPremiseLinker',
        
        # Models
        'ArgumentUnit',
        'UnlinkedArgumentUnits', 
        'LinkedArgumentUnits',
        'LinkedArgumentUnitsWithStance',
        'StanceRelation',
        'ClaimPremiseRelationship',
        
        # Implementations
        'OpenAILLMClassifier',
        'TinyLLamaLLMClassifier',
        'PeftEncoderModelLoader',
        'NonTrainedEncoderModelLoader',
        'OpenAIClaimPremiseLinker',
        'MODEL_CONFIGS'
    ]
    
except ImportError as e:
    # If imports fail, provide helpful error message
    print(f"Error importing from external API: {e}")
    print(f"Make sure the external API submodule is properly initialized at: {external_api}")
    raise
