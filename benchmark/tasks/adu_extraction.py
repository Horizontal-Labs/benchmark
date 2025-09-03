"""
ADU extraction benchmark task.
"""

import time
from typing import Dict, Any, List
from ..core.results import BenchmarkResult
from ..utils.metrics_calculator import MetricsCalculator
from .base import BaseTask


class ADUExtractionTask(BaseTask):
    """ADU extraction benchmark task."""
    
    def __init__(self):
        super().__init__("adu_extraction")
    
    def prepare_data(self, raw_data: tuple) -> List[Dict[str, Any]]:
        """Prepare data specifically for ADU extraction task."""
        claims, premises, topics = raw_data
        data = []
        
        for i in range(len(claims)):
            # For ADU extraction, we need the full text and ground truth ADUs
            claim_text = claims[i].text if hasattr(claims[i], 'text') else str(claims[i])
            
            # Create ground truth ADUs from claims and premises
            ground_truth_adus = [claim_text]
            if i < len(premises):
                premise_text = premises[i].text if hasattr(premises[i], 'text') else str(premises[i])
                ground_truth_adus.append(premise_text)
            
            sample = {
                'text': claim_text,
                'ground_truth': {
                    'adus': ground_truth_adus
                }
            }
            data.append(sample)
        
        return data
    
    def run_benchmark(self, implementation, data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Run the ADU extraction benchmark."""
        results = []
        adu_classifier = implementation.get_adu_classifier()
        
        if not adu_classifier:
            return results
        
        for i, sample in enumerate(data):
            try:
                start_time = time.time()
                
                # Run ADU extraction
                predictions = adu_classifier.classify_adus(sample['text'])
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Calculate comprehensive metrics
                metrics = self.calculate_metrics(predictions, sample['ground_truth']['adus'])
                
                # Create result
                result = BenchmarkResult(
                    task_name=self.name,
                    implementation_name=implementation.name,
                    sample_id=str(i),
                    execution_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metrics=metrics,
                    performance={'inference_time': inference_time},
                    predictions=predictions,
                    ground_truth=sample['ground_truth']['adus']
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
                    ground_truth=sample['ground_truth']['adus'],
                    error_message=str(e),
                    success=False
                )
                results.append(result)
        
        return results
    
    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Calculate comprehensive metrics for ADU extraction."""
        # Use the comprehensive metrics calculator
        return MetricsCalculator.calculate_adu_extraction_metrics(predictions, ground_truth)
