"""
Claim-premise linking benchmark task.
"""

import time
from typing import Dict, Any, List
from ..core.results import BenchmarkResult
from ..utils.metrics_calculator import MetricsCalculator
from .base import BaseTask


class ClaimPremiseLinkingTask(BaseTask):
    """Claim-premise linking benchmark task."""
    
    def __init__(self):
        super().__init__("claim_premise_linking")
    
    def prepare_data(self, raw_data: tuple) -> List[Dict[str, Any]]:
        """Prepare data specifically for claim-premise linking task."""
        claims, premises, topics = raw_data
        data = []
        
        for i in range(len(claims)):
            claim_text = claims[i].text if hasattr(claims[i], 'text') else str(claims[i])
            
            # Create ground truth relationships
            relationships = []
            if i < len(premises):
                premise_text = premises[i].text if hasattr(premises[i], 'text') else str(premises[i])
                relationships.append({
                    'claim': claim_text,
                    'premise': premise_text,
                    'relationship': 'supports'  # Default relationship
                })
            
            sample = {
                'text': claim_text,
                'ground_truth': {
                    'relationships': relationships
                }
            }
            data.append(sample)
        
        return data
    
    def run_benchmark(self, implementation, data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Run the claim-premise linking benchmark."""
        results = []
        linker = implementation.get_linker()
        
        if not linker:
            return results
        
        for i, sample in enumerate(data):
            try:
                start_time = time.time()
                
                # Create the data structure expected by the linker
                from argmining.models.argument_units import UnlinkedArgumentUnits, ArgumentUnit
                from uuid import uuid4
                
                # Create argument units from the sample data
                claims = [ArgumentUnit(
                    type='claim',
                    uuid=uuid4(),
                    text=sample['text'],
                    start_pos=0,
                    end_pos=len(sample['text']),
                    confidence=1.0
                )]
                
                premises = []
                if 'ground_truth' in sample and 'relationships' in sample['ground_truth']:
                    for rel in sample['ground_truth']['relationships']:
                        if 'premise' in rel:
                            premises.append(ArgumentUnit(
                                type='premise',
                                uuid=uuid4(),
                                text=rel['premise'],
                                start_pos=0,
                                end_pos=len(rel['premise']),
                                confidence=1.0
                            ))
                
                # Create unlinked argument units
                unlinked_units = UnlinkedArgumentUnits(claims=claims, premises=premises)
                
                # Run claim-premise linking
                predictions = linker.link_claims_to_premises(unlinked_units)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Calculate comprehensive metrics
                metrics = self.calculate_metrics(predictions, sample['ground_truth']['relationships'])
                
                # Create result
                result = BenchmarkResult(
                    task_name=self.name,
                    implementation_name=implementation.name,
                    sample_id=str(i),
                    execution_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metrics=metrics,
                    performance={'inference_time': inference_time},
                    predictions=predictions,
                    ground_truth=sample['ground_truth']['relationships']
                )
                results.append(result)
                
            except Exception as e:
                # Create error result
                result = BenchmarkResult(
                    task_name=self.name,
                    implementation_name=implementation.name,
                    sample_id=str(i),
                    execution_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metrics={},
                    performance={},
                    predictions=None,
                    ground_truth=sample['ground_truth']['relationships'],
                    error_message=str(e),
                    success=False
                )
                results.append(result)
        
        return results
    
    def calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive metrics for claim-premise linking."""
        # Use the comprehensive metrics calculator
        return MetricsCalculator.calculate_claim_premise_linking_metrics(predictions, ground_truth)
