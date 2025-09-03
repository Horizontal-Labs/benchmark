"""
Utility module for calculating comprehensive benchmark metrics.
"""

from typing import Dict, Any, List, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MetricsCalculator:
    """Utility class for calculating comprehensive benchmark metrics."""
    
    @staticmethod
    def calculate_classification_metrics(predictions: Union[List, np.ndarray], 
                                      ground_truth: Union[List, np.ndarray],
                                      labels: List[str] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics including confusion matrix.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            labels: List of label names (optional)
            
        Returns:
            Dictionary containing accuracy, precision, recall, f1, and confusion matrix metrics
        """
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
        
        # Convert to numpy arrays if needed
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.array(ground_truth)
        
        # Calculate basic metrics
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Handle binary classification
        if len(np.unique(ground_truth)) == 2:
            # Binary classification metrics
            precision = precision_score(ground_truth, predictions, average='binary', zero_division=0)
            recall = recall_score(ground_truth, predictions, average='binary', zero_division=0)
            f1 = f1_score(ground_truth, predictions, average='binary', zero_division=0)
            
            # Confusion matrix for binary classification
            cm = confusion_matrix(ground_truth, predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases
                tn, fp, fn, tp = 0, 0, 0, 0
        else:
            # Multi-class classification
            precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
            recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
            f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
            
            # For multi-class, we'll calculate confusion matrix but focus on overall metrics
            cm = confusion_matrix(ground_truth, predictions)
            # For multi-class, we'll set tp, tn, fp, fn to 0 as they're not directly applicable
            tn, fp, fn, tp = 0, 0, 0, 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    @staticmethod
    def calculate_stance_metrics(predictions: List[str], 
                               ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate metrics specifically for stance classification.
        
        Args:
            predictions: List of predicted stances
            ground_truth: List of ground truth stances
            
        Returns:
            Dictionary containing stance classification metrics
        """
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
        
        # Convert to lowercase for comparison
        pred_lower = [p.lower() if p else '' for p in predictions]
        gt_lower = [g.lower() if g else '' for g in ground_truth]
        
        # Calculate accuracy
        correct = sum(1 for p, g in zip(pred_lower, gt_lower) if p == g)
        accuracy = correct / len(gt_lower) if gt_lower else 0.0
        
        # For stance classification, we'll treat it as a multi-class problem
        # and calculate weighted metrics
        try:
            precision = precision_score(gt_lower, pred_lower, average='weighted', zero_division=0)
            recall = recall_score(gt_lower, pred_lower, average='weighted', zero_division=0)
            f1 = f1_score(gt_lower, pred_lower, average='weighted', zero_division=0)
        except:
            # Fallback if sklearn fails
            precision = accuracy
            recall = accuracy
            f1 = accuracy
        
        # Calculate confusion matrix metrics for stance classification
        # We'll treat this as a binary classification problem for confusion matrix
        # where we count correct predictions as true positives
        correct_predictions = sum(1 for p, g in zip(pred_lower, gt_lower) if p == g)
        incorrect_predictions = len(gt_lower) - correct_predictions if gt_lower else 0
        
        # For stance classification, we'll use a simplified approach:
        # tp = correct predictions, fp = incorrect predictions, tn = 0, fn = 0
        tp = correct_predictions
        tn = 0  # No true negatives in this context
        fp = incorrect_predictions
        fn = 0  # No false negatives in this context
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    @staticmethod
    def calculate_adu_extraction_metrics(predictions, 
                                       ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate metrics for ADU extraction.
        
        Args:
            predictions: List of predicted ADUs
            ground_truth: List of ground truth ADUs
            
        Returns:
            Dictionary containing ADU extraction metrics
        """
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
        
        # This is a simplified metric calculation for ADU extraction
        # In practice, you might want more sophisticated metrics like IoU or exact match
        
        # Extract the number of ADUs from the UnlinkedArgumentUnits object
        try:
            if hasattr(predictions, 'claims'):
                pred_claims = len(predictions.claims)
            else:
                pred_claims = 0
                
            if hasattr(predictions, 'premises'):
                pred_premises = len(predictions.premises)
            else:
                pred_premises = 0
                
            pred_count = pred_claims + pred_premises
        except:
            pred_count = 0
            
        gt_count = len(ground_truth) if ground_truth else 0
        
        if gt_count == 0:
            accuracy = 1.0 if pred_count == 0 else 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
        else:
            accuracy = 1.0 if pred_count == gt_count else 0.0
            # Calculate confusion matrix metrics
            if pred_count == gt_count:
                # Perfect match
                tp = gt_count
                tn = 0
                fp = 0
                fn = 0
            elif pred_count > gt_count:
                # Over-prediction
                tp = gt_count
                tn = 0
                fp = pred_count - gt_count
                fn = 0
            else:
                # Under-prediction
                tp = pred_count
                tn = 0
                fp = 0
                fn = gt_count - pred_count
        
        # Set other metrics to accuracy for now
        precision = accuracy
        recall = accuracy
        f1 = accuracy
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    @staticmethod
    def calculate_claim_premise_linking_metrics(predictions, 
                                              ground_truth: List[Dict]) -> Dict[str, float]:
        """
        Calculate metrics for claim-premise linking.
        
        Args:
            predictions: LinkedArgumentUnits object with predicted links
            ground_truth: List of ground truth links
            
        Returns:
            Dictionary containing claim-premise linking metrics
        """
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0
            }
        
        # This is a simplified metric calculation for claim-premise linking
        # In practice, you might want more sophisticated metrics
        
        # Extract the number of relationships from the LinkedArgumentUnits object
        try:
            if hasattr(predictions, 'claims_premises_relationships'):
                pred_count = len(predictions.claims_premises_relationships)
            else:
                pred_count = 0
        except:
            pred_count = 0
            
        gt_count = len(ground_truth) if ground_truth else 0
        
        if gt_count == 0:
            accuracy = 1.0 if pred_count == 0 else 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
        else:
            accuracy = 1.0 if pred_count == gt_count else 0.0
            # Calculate confusion matrix metrics
            if pred_count == gt_count:
                # Perfect match
                tp = gt_count
                tn = 0
                fp = 0
                fn = 0
            elif pred_count > gt_count:
                # Over-prediction
                tp = gt_count
                tn = 0
                fp = pred_count - gt_count
                fn = 0
            else:
                # Under-prediction
                tp = pred_count
                tn = 0
                fp = 0
                fn = gt_count - pred_count
        
        # Set other metrics to accuracy for now
        precision = accuracy
        recall = accuracy
        f1 = accuracy
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
