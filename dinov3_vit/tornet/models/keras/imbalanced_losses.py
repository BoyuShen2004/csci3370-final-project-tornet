"""
Advanced loss functions specifically designed for severely imbalanced data.
"""
import keras
from keras import ops
import numpy as np


def _prep(y_true, y_pred):
    """Prepare inputs for loss calculation."""
    y_true = ops.cast(y_true, 'float32')
    y_pred = ops.cast(y_pred, 'float32')
    return y_true, y_pred


def focal_loss_imbalanced(y_true, y_pred, alpha=0.75, gamma=3.0):
    """
    Enhanced Focal Loss for severely imbalanced data.
    
    Args:
        y_true: Binary class labels (0 or 1)
        y_pred: Model predictions (0 to 1)
        alpha: Weighting factor for rare class (default 0.75 for severe imbalance)
        gamma: Focusing parameter (default 3.0 for hard examples)
    
    Returns:
        Focal loss value
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Calculate cross entropy
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    # Calculate p_t
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    
    # Calculate alpha_t
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    
    # Calculate focal weight
    focal_weight = alpha_t * ops.power(1 - p_t, gamma)
    
    # Calculate focal loss
    focal_loss = focal_weight * ce
    
    return ops.mean(focal_loss)


def dice_loss_imbalanced(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss for imbalanced data.
    
    Args:
        y_true: Binary class labels
        y_pred: Model predictions
        smooth: Smoothing factor
    
    Returns:
        Dice loss value
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Flatten
    y_true_f = ops.flatten(y_true)
    y_pred_f = ops.flatten(y_pred)
    
    # Calculate intersection
    intersection = ops.sum(y_true_f * y_pred_f)
    
    # Calculate dice coefficient
    dice = (2. * intersection + smooth) / (ops.sum(y_true_f) + ops.sum(y_pred_f) + smooth)
    
    # Return dice loss
    return 1 - dice


def tversky_loss_imbalanced(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky Loss for imbalanced data with emphasis on recall.
    
    Args:
        y_true: Binary class labels
        y_pred: Model predictions
        alpha: Weight for false negatives (higher = more emphasis on recall)
        beta: Weight for false positives
        smooth: Smoothing factor
    
    Returns:
        Tversky loss value
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Flatten
    y_true_f = ops.flatten(y_true)
    y_pred_f = ops.flatten(y_pred)
    
    # Calculate true positives, false negatives, false positives
    tp = ops.sum(y_true_f * y_pred_f)
    fn = ops.sum(y_true_f * (1 - y_pred_f))
    fp = ops.sum((1 - y_true_f) * y_pred_f)
    
    # Calculate Tversky index
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Return Tversky loss
    return 1 - tversky


def combined_imbalanced_loss(y_true, y_pred, focal_weight=0.6, dice_weight=0.4):
    """
    Combined loss function for imbalanced data.
    
    Args:
        y_true: Binary class labels
        y_pred: Model predictions
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
    
    Returns:
        Combined loss value
    """
    focal = focal_loss_imbalanced(y_true, y_pred)
    dice = dice_loss_imbalanced(y_true, y_pred)
    
    return focal_weight * focal + dice_weight * dice


def class_balanced_loss(y_true, y_pred, beta=0.9999):
    """
    Class-Balanced Loss for imbalanced data.
    
    Args:
        y_true: Binary class labels
        y_pred: Model predictions
        beta: Effective number of samples parameter
    
    Returns:
        Class-balanced loss value
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Calculate class frequencies (approximate)
    n_0 = ops.sum(1 - y_true)
    n_1 = ops.sum(y_true)
    n_total = n_0 + n_1
    
    # Calculate effective number of samples
    effective_num_0 = (1 - ops.power(beta, n_0)) / (1 - beta)
    effective_num_1 = (1 - ops.power(beta, n_1)) / (1 - beta)
    
    # Calculate class weights
    weight_0 = (1 - beta) / effective_num_0
    weight_1 = (1 - beta) / effective_num_1
    
    # Calculate weighted cross entropy
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=False)
    weights = weight_0 * (1 - y_true) + weight_1 * y_true
    
    return ops.mean(weights * ce)


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0):
    """
    Weighted binary crossentropy for imbalanced data.
    
    Args:
        y_true: Binary class labels
        y_pred: Model predictions
        pos_weight: Weight for positive class (default 10.0 for 1% positive class)
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Calculate weighted cross entropy
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    # Apply class weights
    weights = y_true * pos_weight + (1 - y_true) * 1.0
    
    return ops.mean(weights * ce)


def adaptive_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, adaptive_gamma=True):
    """
    Adaptive Focal Loss that adjusts gamma based on training progress.
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Calculate cross entropy
    ce = ops.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    # Calculate p_t
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    
    # Calculate alpha_t
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    
    # Adaptive gamma based on prediction confidence
    if adaptive_gamma:
        # Increase gamma for very confident wrong predictions
        confidence = ops.abs(p_t - 0.5) * 2  # 0 to 1
        adaptive_gamma = gamma * (1 + confidence)
    else:
        adaptive_gamma = gamma
    
    # Calculate focal weight
    focal_weight = alpha_t * ops.power(1 - p_t, adaptive_gamma)
    
    return ops.mean(focal_weight * ce)


def improved_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, label_smoothing=0.1):
    """
    Improved Focal Loss with label smoothing for better generalization.
    """
    y_true, y_pred = _prep(y_true, y_pred)
    
    # Apply label smoothing
    y_true_smooth = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    
    # Calculate cross entropy
    ce = ops.binary_crossentropy(y_true_smooth, y_pred, from_logits=False)
    
    # Calculate p_t
    p_t = y_pred * y_true_smooth + (1 - y_pred) * (1 - y_true_smooth)
    
    # Calculate alpha_t
    alpha_t = alpha * y_true_smooth + (1 - alpha) * (1 - y_true_smooth)
    
    # Calculate focal weight
    focal_weight = alpha_t * ops.power(1 - p_t, gamma)
    
    return ops.mean(focal_weight * ce)
