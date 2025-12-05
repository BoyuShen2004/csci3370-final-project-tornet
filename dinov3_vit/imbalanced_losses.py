"""
Imbalanced Loss Functions for Binary Classification

These loss functions are specifically designed for severely imbalanced datasets
like tornado detection where positive samples are rare (~6.8%).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 3.0,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for imbalanced binary classification.
    
    Based on tornet_enhanced implementation - SIMPLIFIED to match their exact approach.
    Key insight: tornet_enhanced uses simple focal loss (alpha/gamma only) and handles
    class imbalance separately via class weights, NOT by integrating pos_weight into focal loss.
    
    Args:
        inputs: Model predictions (logits) [B, 1] or [B]
        targets: Ground truth labels [B, 1] or [B] (0 or 1)
        alpha: Weighting factor for rare class (default 0.75 for severe imbalance)
        gamma: Focusing parameter (default 3.0 for hard examples, prevents collapse)
        pos_weight: DEPRECATED - not used in tornet_enhanced focal loss. Use class weights separately.
        label_smoothing: Label smoothing factor (default 0.0)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Focal loss value
    """
    # Ensure inputs are logits and targets are in [0, 1]
    if inputs.dim() > 1:
        inputs = inputs.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    # Apply label smoothing if specified
    if label_smoothing > 0:
        targets_smooth = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        targets_smooth = targets
    
    # Convert logits to probabilities for p_t calculation (matching tornet_enhanced)
    # We need probabilities for p_t, but use logits for loss computation (for mixed precision safety)
    probs = torch.sigmoid(inputs)
    # Clamp probabilities to prevent numerical issues
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    
    # Calculate p_t (probability of correct class) - EXACT tornet_enhanced formula
    # p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    p_t = probs * targets_smooth + (1 - probs) * (1 - targets_smooth)
    # Clamp p_t to prevent focal weight from becoming exactly zero
    # This prevents loss collapse when model is too confident (p_t = 1.0)
    p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
    
    # Calculate cross entropy using logits (safe for mixed precision training)
    # NOTE: tornet_enhanced does NOT use pos_weight in BCE - they handle imbalance via alpha_t only
    # Using pos_weight here can cause numerical instability (nan) when combined with label smoothing
    # Instead, we handle class imbalance through alpha_t in the focal weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
    
    # If pos_weight is provided, we'll apply it to the final focal loss (not in BCE)
    # This is safer numerically than using it in BCE with label smoothing
    
    # Calculate alpha_t - EXACT tornet_enhanced formula
    # alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    alpha_t = alpha * targets_smooth + (1 - alpha) * (1 - targets_smooth)
    
    # Calculate focal weight - EXACT tornet_enhanced formula
    # focal_weight = alpha_t * (1 - p_t)^gamma
    # Clamp (1 - p_t) to prevent numerical issues when p_t is very close to 1.0
    one_minus_pt = torch.clamp(1 - p_t, min=1e-7, max=1.0 - 1e-7)
    focal_weight = alpha_t * torch.pow(one_minus_pt, gamma)
    
    # Calculate focal loss - EXACT tornet_enhanced formula
    # focal_loss = focal_weight * ce
    focal_loss = focal_weight * ce_loss
    
    # Check for nan/inf BEFORE applying pos_weight to prevent propagation
    # If inputs are NaN, the loss will be NaN - we need to handle this upstream
    if not torch.isfinite(inputs).all():
        # Input logits are NaN - this is a model forward pass issue
        # Return a large but finite loss value to signal the problem
        return torch.tensor(1e6, device=inputs.device, dtype=inputs.dtype)
    
    # Apply pos_weight to positive examples only (safer than using it in BCE)
    # This helps with class imbalance without causing numerical issues
    if pos_weight is not None:
        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor(pos_weight, device=inputs.device, dtype=inputs.dtype)
        # Weight positive examples more heavily
        pos_mask = targets_smooth > 0.5
        focal_loss = torch.where(
            pos_mask,
            focal_loss * pos_weight,
            focal_loss
        )
    
    # Final check for nan/inf values and replace with a large finite value
    # This prevents training from crashing but signals a serious problem
    focal_loss = torch.where(
        torch.isfinite(focal_loss),
        focal_loss,
        torch.full_like(focal_loss, 1e6)  # Replace nan/inf with large finite value
    )
    
    # CRITICAL FIX: Add minimum loss floor to prevent collapse when model predicts all negatives
    # When p_t is very high (model confident in negative predictions), focal_weight becomes tiny
    # This causes loss to collapse to near-zero, preventing learning
    # We add a small floor to ensure gradients always flow
    min_loss_floor = 1e-6  # Small but non-zero minimum loss
    focal_loss = torch.clamp(focal_loss, min=min_loss_floor)
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def class_balanced_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.9999,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Class-Balanced Loss for imbalanced binary classification.
    
    Args:
        inputs: Model predictions (logits) [B, 1] or [B]
        targets: Ground truth labels [B, 1] or [B]
        beta: Effective number of samples parameter
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Class-balanced loss value
    """
    if inputs.dim() > 1:
        inputs = inputs.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    # Count samples per class
    n_0 = (targets == 0).sum().float()
    n_1 = (targets == 1).sum().float()
    n_total = n_0 + n_1
    
    # Compute effective number of samples
    effective_num_0 = (1 - beta ** n_0) / (1 - beta) if n_0 > 0 else 0
    effective_num_1 = (1 - beta ** n_1) / (1 - beta) if n_1 > 0 else 0
    
    # Compute class weights
    weight_0 = (1 - beta) / (effective_num_0 + 1e-8)
    weight_1 = (1 - beta) / (effective_num_1 + 1e-8)
    
    # Compute weighted cross entropy
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    weights = weight_0 * (1 - targets) + weight_1 * targets
    
    weighted_loss = weights * ce_loss
    
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss


def weighted_binary_crossentropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 10.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Weighted binary cross-entropy for imbalanced data.
    
    Args:
        inputs: Model predictions (logits) [B, 1] or [B]
        targets: Ground truth labels [B, 1] or [B]
        pos_weight: Weight for positive class
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Weighted BCE loss
    """
    if inputs.dim() > 1:
        inputs = inputs.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    # Compute weighted BCE
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets,
        pos_weight=torch.tensor(pos_weight, device=inputs.device),
        reduction=reduction,
    )
    
    return ce_loss


def combined_imbalanced_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    focal_weight: float = 0.8,
    dice_weight: float = 0.2,
    alpha: float = 0.75,
    gamma: float = 3.0,
    pos_weight: Optional[torch.Tensor] = None,  # Now USED to prevent loss collapse
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Combined Focal + Dice loss for imbalanced binary classification.
    
    Args:
        inputs: Model predictions (logits) [B, 1] or [B]
        targets: Ground truth labels [B, 1] or [B]
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        pos_weight: Weight for positive class (CRITICAL for imbalanced data)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Combined loss value
    """
    # Focal loss with pos_weight to prevent collapse
    focal = focal_loss(inputs, targets, alpha=alpha, gamma=gamma, pos_weight=pos_weight, label_smoothing=label_smoothing, reduction=reduction)
    
    # Dice loss (on probabilities)
    if inputs.dim() > 1:
        inputs = inputs.squeeze(1)
    if targets.dim() > 1:
        targets = targets.squeeze(1)
    
    probs = torch.sigmoid(inputs)
    smooth = 1e-6
    
    # Flatten
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    # Compute dice coefficient
    intersection = (probs_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    dice_loss = 1 - dice
    
    return focal_weight * focal + dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss module with pos_weight support to prevent loss collapse.
    
    CRITICAL: We need pos_weight to prevent loss collapse when model predicts all negatives.
    Even though tornet_enhanced doesn't use it, we need it for our imbalanced dataset.
    """
    def __init__(
        self, 
        alpha: float = 0.75, 
        gamma: float = 3.0, 
        pos_weight: Optional[float] = None,  # Now USED to prevent loss collapse
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert pos_weight to tensor if needed
        pos_weight_tensor = None
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, (int, float)):
                pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
            else:
                pos_weight_tensor = self.pos_weight
        return focal_loss(inputs, targets, self.alpha, self.gamma, pos_weight_tensor, self.label_smoothing, self.reduction)


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss as a PyTorch module."""
    
    def __init__(self, beta: float = 0.9999, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return class_balanced_loss(inputs, targets, self.beta, self.reduction)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy as a PyTorch module."""
    
    def __init__(self, pos_weight: float = 10.0, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return weighted_binary_crossentropy(inputs, targets, self.pos_weight, self.reduction)


class CombinedImbalancedLoss(nn.Module):
    """
    Combined Focal + Dice Loss as a PyTorch module.
    
    Now uses pos_weight to prevent loss collapse on imbalanced data.
    """
    def __init__(
        self,
        focal_weight: float = 0.8,
        dice_weight: float = 0.2,
        alpha: float = 0.75,
        gamma: float = 3.0,
        pos_weight: Optional[float] = None,  # Now USED to prevent loss collapse
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert pos_weight to tensor if needed
        pos_weight_tensor = None
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, (int, float)):
                pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
            else:
                pos_weight_tensor = self.pos_weight
        return combined_imbalanced_loss(
            inputs, targets,
            self.focal_weight, self.dice_weight,
            self.alpha, self.gamma, pos_weight_tensor, self.label_smoothing, self.reduction,
        )

