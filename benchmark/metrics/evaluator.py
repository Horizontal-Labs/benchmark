"""
Metrics evaluation utilities.
"""

from typing import Dict, Any, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class MetricsEvaluator:
    """Evaluator for various benchmark metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
        """Calculate classification metrics (precision, recall, F1, accuracy)."""
        if not y_true or not y_pred:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy)
            }
        except Exception:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
    
    @staticmethod
    def calculate_token_level_metrics(pred_tokens: List[str], gt_tokens: List[str]) -> Dict[str, float]:
        """Calculate token-level metrics for sequence tasks."""
        if not pred_tokens or not gt_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert to sets for token-level comparison
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)
        
        if not gt_set:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        intersection = pred_set.intersection(gt_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(gt_set)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
