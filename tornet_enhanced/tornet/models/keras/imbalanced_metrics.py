"""
Advanced metrics for imbalanced data evaluation.
"""
import keras
from keras import ops
import numpy as np


def balanced_accuracy(y_true, y_pred, threshold=0.5):
    """
    Balanced accuracy for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Balanced accuracy value
    """
    y_true = ops.cast(y_true, 'float32')
    y_pred = ops.cast(y_pred, 'float32')
    
    # Convert predictions to binary
    y_pred_binary = ops.cast(y_pred > threshold, 'float32')
    
    # Calculate per-class accuracy
    # True positives and false negatives for positive class
    tp = ops.sum(y_true * y_pred_binary)
    fn = ops.sum(y_true * (1 - y_pred_binary))
    
    # True negatives and false positives for negative class
    tn = ops.sum((1 - y_true) * (1 - y_pred_binary))
    fp = ops.sum((1 - y_true) * y_pred_binary)
    
    # Calculate sensitivity (recall for positive class)
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Calculate specificity (recall for negative class)
    specificity = tn / (tn + fp + 1e-8)
    
    # Balanced accuracy is the average of sensitivity and specificity
    return (sensitivity + specificity) / 2


def g_mean(y_true, y_pred, threshold=0.5):
    """
    Geometric mean for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Geometric mean value
    """
    y_true = ops.cast(y_true, 'float32')
    y_pred = ops.cast(y_pred, 'float32')
    
    # Convert predictions to binary
    y_pred_binary = ops.cast(y_pred > threshold, 'float32')
    
    # Calculate per-class metrics
    tp = ops.sum(y_true * y_pred_binary)
    fn = ops.sum(y_true * (1 - y_pred_binary))
    tn = ops.sum((1 - y_true) * (1 - y_pred_binary))
    fp = ops.sum((1 - y_true) * y_pred_binary)
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Geometric mean
    return ops.sqrt(sensitivity * specificity)


def matthews_correlation_coefficient(y_true, y_pred, threshold=0.5):
    """
    Matthews Correlation Coefficient for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        MCC value
    """
    y_true = ops.cast(y_true, 'float32')
    y_pred = ops.cast(y_pred, 'float32')
    
    # Convert predictions to binary
    y_pred_binary = ops.cast(y_pred > threshold, 'float32')
    
    # Calculate confusion matrix elements
    tp = ops.sum(y_true * y_pred_binary)
    tn = ops.sum((1 - y_true) * (1 - y_pred_binary))
    fp = ops.sum((1 - y_true) * y_pred_binary)
    fn = ops.sum(y_true * (1 - y_pred_binary))
    
    # Calculate MCC
    numerator = tp * tn - fp * fn
    denominator = ops.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    
    return numerator / denominator


def cohen_kappa(y_true, y_pred, threshold=0.5):
    """
    Cohen's Kappa for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Cohen's Kappa value
    """
    y_true = ops.cast(y_true, 'float32')
    y_pred = ops.cast(y_pred, 'float32')
    
    # Convert predictions to binary
    y_pred_binary = ops.cast(y_pred > threshold, 'float32')
    
    # Calculate confusion matrix elements
    tp = ops.sum(y_true * y_pred_binary)
    tn = ops.sum((1 - y_true) * (1 - y_pred_binary))
    fp = ops.sum((1 - y_true) * y_pred_binary)
    fn = ops.sum(y_true * (1 - y_pred_binary))
    
    # Calculate total
    total = tp + tn + fp + fn
    
    # Calculate observed agreement
    po = (tp + tn) / (total + 1e-8)
    
    # Calculate expected agreement
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total * total + 1e-8)
    
    # Calculate Kappa
    kappa = (po - pe) / (1 - pe + 1e-8)
    
    return kappa


def optimal_threshold_finder(y_true, y_pred, metric='f1'):
    """
    Find optimal threshold for imbalanced data.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'g_mean', 'mcc')
    
    Returns:
        Optimal threshold value
    """
    # Convert to numpy for threshold search
    if hasattr(y_true, 'numpy'):
        y_true_np = y_true.numpy()
    else:
        y_true_np = y_true
    
    if hasattr(y_pred, 'numpy'):
        y_pred_np = y_pred.numpy()
    else:
        y_pred_np = y_pred
    
    # Flatten if needed
    y_true_np = y_true_np.flatten()
    y_pred_np = y_pred_np.flatten()
    
    # Test different thresholds
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_np > threshold).astype(int)
        
        # Calculate confusion matrix
        tp = np.sum((y_true_np == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_np == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_np == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_np == 1) & (y_pred_binary == 0))
        
        if metric == 'f1':
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            score = 2 * precision * recall / (precision + recall + 1e-8)
        elif metric == 'balanced_accuracy':
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            score = (sensitivity + specificity) / 2
        elif metric == 'g_mean':
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            score = np.sqrt(sensitivity * specificity)
        elif metric == 'mcc':
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
            score = numerator / denominator
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


class ImbalancedMetrics:
    """
    Custom metrics class for imbalanced data.
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def balanced_accuracy(self, y_true, y_pred):
        return balanced_accuracy(y_true, y_pred, self.threshold)
    
    def g_mean(self, y_true, y_pred):
        return g_mean(y_true, y_pred, self.threshold)
    
    def mcc(self, y_true, y_pred):
        return matthews_correlation_coefficient(y_true, y_pred, self.threshold)
    
    def kappa(self, y_true, y_pred):
        return cohen_kappa(y_true, y_pred, self.threshold)
