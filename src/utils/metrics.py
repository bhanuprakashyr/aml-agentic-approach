"""
Metrics computation for fraud detection evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Tuple, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        y_pred: Predicted labels
        threshold: Classification threshold (for info only)
        
    Returns:
        Dictionary of metrics
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5
        }
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob)
    }
    
    return metrics


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """
    Compute detailed classification metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels
        
    Returns:
        Dictionary with detailed metrics including confusion matrix
    """
    basic_metrics = compute_metrics(y_true, y_prob, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # Average precision (area under PR curve)
    if len(np.unique(y_true)) >= 2:
        avg_precision = average_precision_score(y_true, y_prob)
    else:
        avg_precision = 0.0
    
    detailed = {
        **basic_metrics,
        'specificity': specificity,
        'npv': npv,
        'avg_precision': avg_precision,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return detailed


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced data.
    
    Uses inverse frequency weighting.
    
    Args:
        labels: Array of labels (0 or 1)
        
    Returns:
        Array of weights [weight_class_0, weight_class_1]
    """
    # Only consider labeled nodes (exclude -1)
    labeled_mask = labels != -1
    labels_subset = labels[labeled_mask]
    
    # Count classes
    n_class_0 = (labels_subset == 0).sum()
    n_class_1 = (labels_subset == 1).sum()
    n_total = n_class_0 + n_class_1
    
    # Inverse frequency weighting
    weight_0 = n_total / (2 * n_class_0) if n_class_0 > 0 else 1.0
    weight_1 = n_total / (2 * n_class_1) if n_class_1 > 0 else 1.0
    
    return np.array([weight_0, weight_1], dtype=np.float32)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_metric = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_metric:
            best_metric = score
            best_threshold = thresh
    
    return best_threshold, best_metric


def precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int
) -> float:
    """
    Compute precision at K.
    
    Precision among top-K predictions ranked by probability.
    Useful for fraud detection where we want to prioritize
    the most suspicious transactions.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Precision at K
    """
    # Get indices of top-k predictions
    top_k_idx = np.argsort(y_prob)[-k:]
    
    # Precision among top-k
    top_k_labels = y_true[top_k_idx]
    precision_k = top_k_labels.sum() / k
    
    return precision_k


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = ['licit', 'illicit']
):
    """Print sklearn classification report."""
    print("\nClassification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)
    
    # Generate dummy data
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.8, 0.9, 0.7, 0.2, 0.1])
    y_pred = (y_prob >= 0.5).astype(int)
    
    print("Testing metrics computation")
    print("="*50)
    
    # Basic metrics
    metrics = compute_metrics(y_true, y_prob, y_pred)
    print("\nBasic metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Detailed metrics
    detailed = compute_detailed_metrics(y_true, y_prob, y_pred)
    print("\nDetailed metrics:")
    print(f"  Confusion matrix: {detailed['confusion_matrix']}")
    
    # Optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob, 'f1')
    print(f"\nOptimal threshold: {opt_thresh:.2f} (F1: {opt_f1:.4f})")
    
    # Precision at K
    p_at_3 = precision_at_k(y_true, y_prob, 3)
    print(f"Precision@3: {p_at_3:.4f}")
