"""
Metrics for segmentation evaluation.
"""
import numpy as np
import torch
from typing import Dict


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    dice = (2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute IoU (Intersection over Union)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum((pred_flat + target_flat) > 0)
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'dice': dice_coefficient(pred, target),
        'iou': iou_score(pred, target),
    }


def compute_metrics_batch(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute metrics for a batch."""
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    dice_scores = []
    iou_scores = []
    
    for pred, target in zip(preds_np, targets_np):
        dice = dice_coefficient(pred, target)
        iou = iou_score(pred, target)
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'dice_std': np.std(dice_scores),
        'iou_std': np.std(iou_scores),
    }

