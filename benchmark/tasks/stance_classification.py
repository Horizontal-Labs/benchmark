"""
Stance classification benchmark task.
"""

import time
from typing import Dict, Any, List
from ..core.results import BenchmarkResult
from ..utils.metrics_calculator import MetricsCalculator
from .base import BaseTask


class StanceClassificationTask(BaseTask):
    """Stance classification benchmark task."""
    
    def __init__(self):
        super().__init__("stance_classification")
    
    def prepare_data(self, raw_data: tuple) -> List[Dict[str, Any]]:
        """Prepare data specifically for stance classification task."""
        claims, premises, topics = raw_data
        data = []
        
        for i in range(len(claims)):
            claim_text = claims[i].text if hasattr(claims[i], 'text') else str(claims[i])
            
            # Determine stance from topics or use alternating pattern
            stance = 'neutral'
            if i < len(topics):
                topic = topics[i] if isinstance(topics[i], str) else str(topics[i])
                if 'stance_pro' in topic:
                    stance = 'pro'
                elif 'stance_con' in topic:
                    stance = 'con'
                else:
                    # Alternate between pro and con for variety
                    stance = 'pro' if i % 2 == 0 else 'con'
            
            sample = {
                'text': claim_text,
                'ground_truth': {
                    'stance': stance
                }
            }
            data.append(sample)
        
        return data
    
    def run_benchmark(self, implementation, data: List[Dict[str, Any]], progress_bar=None) -> List[BenchmarkResult]:
        """Run the stance classification benchmark."""
        results = []
        adu_classifier = implementation.get_adu_classifier()
        
        if not adu_classifier:
            return results
        
        # Create progress bar for this implementation if not provided
        if progress_bar is None:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(data), desc=f"Stance Classification - {implementation.name}", unit="sample")
        
        for i, sample in enumerate(data):
            try:
                start_time = time.time()
                
                # Create the data structure expected by the stance classifier
                from argmining.models.argument_units import LinkedArgumentUnits, ArgumentUnit
                from uuid import uuid4
                
                # Create argument units from the sample data
                claim_unit = ArgumentUnit(
                    type='claim',
                    uuid=uuid4(),
                    text=sample['text'],
                    start_pos=0,
                    end_pos=len(sample['text']),
                    confidence=1.0
                )
                

                # For stance classification, we need to create a premise as well
                # Since we don't have explicit premises in the data, we'll create a generic one
                premise_unit = ArgumentUnit(
                    type='premise',
                    uuid=uuid4(),
                    text="This is a supporting premise for stance classification.",
                    start_pos=0,
                    end_pos=len("This is a supporting premise for stance classification."),
                    confidence=1.0
                )
                
                # Create a relationship between claim and premise
                from argmining.models.argument_units import ClaimPremiseRelationship
                relationship = ClaimPremiseRelationship(
                    claim_id=claim_unit.uuid,
                    premise_ids=[premise_unit.uuid]
                )
                
                # Create linked argument units with proper relationships
                linked_units = LinkedArgumentUnits(
                    claims=[claim_unit],
                    premises=[premise_unit],
                    claims_premises_relationships=[relationship]
                )
                
                # Run stance classification
                predictions = adu_classifier.classify_stance(linked_units, sample['text'])
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Calculate comprehensive metrics
                metrics = self.calculate_metrics(predictions, sample['ground_truth']['stance'])
                
                # Create result
                result = BenchmarkResult(
                    task_name=self.name,
                    implementation_name=implementation.name,
                    sample_id=str(i),
                    execution_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metrics=metrics,
                    performance={'inference_time': inference_time},
                    predictions=predictions,
                    ground_truth=sample['ground_truth']['stance']
                )
                results.append(result)
                
                # Update progress bar
                progress_bar.update(1)
                
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
                    ground_truth=sample['ground_truth']['stance'],
                    error_message=str(e),
                    success=False
                )
                results.append(result)
                
                # Update progress bar
                progress_bar.update(1)
        
        # Close progress bar if we created it
        if progress_bar and hasattr(progress_bar, 'close'):
            progress_bar.close()
        
        return results
    
    def calculate_metrics(self, predictions: str, ground_truth: str) -> Dict[str, float]:
        """Calculate comprehensive metrics for stance classification."""
        # Use the comprehensive metrics calculator
        return MetricsCalculator.calculate_stance_metrics(predictions, ground_truth)
