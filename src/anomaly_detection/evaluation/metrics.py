"""
Evaluation Metrics Module - Computing various metrics for anomaly detection
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score,
                            confusion_matrix, classification_report)
from scipy import stats

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AnomalyMetrics:
    """
    Metrics computation for anomaly detection
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_all(self, scores: np.ndarray, X: np.ndarray = None, 
                    y_true: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all metrics
        
        Args:
            scores: Anomaly scores (0-1)
            X: Feature matrix (optional)
            y_true: True labels (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['mean_score'] = float(np.mean(scores))
        metrics['std_score'] = float(np.std(scores))
        metrics['min_score'] = float(np.min(scores))
        metrics['max_score'] = float(np.max(scores))
        metrics['median_score'] = float(np.median(scores))
        
        # Percentiles
        for p in [25, 50, 75, 90, 95, 99]:
            metrics[f'percentile_{p}'] = float(np.percentile(scores, p))
        
        # Distribution metrics
        metrics['skewness'] = float(stats.skew(scores))
        metrics['kurtosis'] = float(stats.kurtosis(scores))
        
        # If labels are provided, compute classification metrics
        if y_true is not None:
            # Convert scores to binary predictions using threshold optimization
            threshold = self._find_optimal_threshold(scores, y_true)
            y_pred = (scores >= threshold).astype(int)
            
            metrics['threshold'] = float(threshold)
            metrics['precision'] = float(precision_score(y_true, y_pred))
            metrics['recall'] = float(recall_score(y_true, y_pred))
            metrics['f1'] = float(f1_score(y_true, y_pred))
            
            # AUC metrics
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, scores))
                metrics['pr_auc'] = float(average_precision_score(y_true, scores))
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['false_positive_rate'] = float(fp / (fp + tn) if (fp + tn) > 0 else 0)
            metrics['true_negative_rate'] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
            metrics['false_negative_rate'] = float(fn / (fn + tp) if (fn + tp) > 0 else 0)
            metrics['true_positive_rate'] = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
            
            # Additional metrics
            metrics['specificity'] = metrics['true_negative_rate']
            metrics['sensitivity'] = metrics['true_positive_rate']
            metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _find_optimal_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        """
        Find optimal threshold using F1 score
        """
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.linspace(0.1, 0.9, 50):
            y_pred = (scores >= threshold).astype(int)
            try:
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
        
        return best_threshold
    
    def compute_detection_delay(self, timestamps: np.ndarray, 
                                anomaly_indices: np.ndarray) -> float:
        """
        Compute average detection delay for anomalies
        """
        if len(anomaly_indices) == 0:
            return 0.0
        
        delays = []
        for i in range(1, len(anomaly_indices)):
            delay = timestamps[anomaly_indices[i]] - timestamps[anomaly_indices[i-1]]
            delays.append(delay)
        
        return float(np.mean(delays)) if delays else 0.0
    
    def compute_performance_metrics(self, detection_times: List[float]) -> Dict[str, float]:
        """
        Compute performance metrics
        """
        if not detection_times:
            return {}
        
        metrics = {
            'mean_detection_time': float(np.mean(detection_times)),
            'median_detection_time': float(np.median(detection_times)),
            'std_detection_time': float(np.std(detection_times)),
            'min_detection_time': float(np.min(detection_times)),
            'max_detection_time': float(np.max(detection_times)),
            'throughput': float(len(detection_times) / np.sum(detection_times))
        }
        
        # Percentiles
        for p in [75, 90, 95]:
            metrics[f'detection_time_p{p}'] = float(np.percentile(detection_times, p))
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        """
        if not self.metrics_history:
            return {'message': 'No metrics available'}
        
        # Compute average metrics
        avg_metrics = {}
        for metric in self.metrics_history[0].keys():
            values = [m.get(metric, 0) for m in self.metrics_history if m.get(metric) is not None]
            if values:
                avg_metrics[f'avg_{metric}'] = float(np.mean(values))
                avg_metrics[f'std_{metric}'] = float(np.std(values))
        
        return avg_metrics