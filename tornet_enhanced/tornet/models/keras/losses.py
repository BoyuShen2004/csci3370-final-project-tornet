"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import keras
from keras import ops

def _prep(class_labels, logits):
    y_true = ops.cast(class_labels, dtype=logits.dtype)
    y_pred = ops.sigmoid(logits) # p=1 means class label=1
    return y_true,y_pred


def mae_loss( class_labels, logits, sample_weights=None ):
    """
    class_labels represents tensor of known binary classes (1,0)
    logits are output of final classification layer that has not yet been run through a sigmoid (from_logits)
    """
    y_true,y_pred=_prep(class_labels, logits)
    if sample_weights is not None:
        denom=ops.sum(sample_weights)
        return ops.sum( sample_weights*ops.absolute(y_true-y_pred) ) / denom
    else:
        return ops.mean( ops.absolute(y_true-y_pred) )


def jaccard_loss(class_labels, logits):
    """
    class_labels represents tensor of known binary classes (1,0)
    logits are output of final classification layer that has not yet been run through a sigmoid (from_logits)
    """
    y_true,y_pred=_prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred - intersection
    # Calculate the Jaccard similarity coefficient (IoU)
    iou = intersection / (union + keras.config.epsilon())  # Adding a small epsilon to prevent division by zero
    # Jaccard loss is the complement of the Jaccard similarity
    return ops.mean(1 - iou)


def dice_loss(class_labels, logits):
    """
    y_true:   [Batch, 1]
    y_pred:   [Batch, 1]
    
    """
    y_true,y_pred=_prep(class_labels, logits)
    intersection = y_true * y_pred
    union = y_true + y_pred
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return ops.mean(1.0-dice)


def focal_loss(class_labels, logits, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        class_labels: Binary class labels (0 or 1)
        logits: Raw model outputs (before sigmoid)
        alpha: Weighting factor for rare class (default 0.25)
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Focal loss value
    """
    y_true, y_pred = _prep(class_labels, logits)
    
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


def tversky_loss(class_labels, logits, alpha=0.3, beta=0.7):
    """
    Tversky Loss for better precision-recall balance.
    
    Args:
        class_labels: Binary class labels (0 or 1)
        logits: Raw model outputs (before sigmoid)
        alpha: Weight for false positives (default 0.3)
        beta: Weight for false negatives (default 0.7)
    
    Returns:
        Tversky loss value
    """
    y_true, y_pred = _prep(class_labels, logits)
    
    # Calculate true positives, false positives, false negatives
    tp = ops.sum(y_true * y_pred)
    fp = ops.sum((1 - y_true) * y_pred)
    fn = ops.sum(y_true * (1 - y_pred))
    
    # Calculate Tversky index
    tversky = (tp + 1e-7) / (tp + alpha * fp + beta * fn + 1e-7)
    
    return 1 - tversky


def combined_loss(class_labels, logits, focal_weight=0.7, dice_weight=0.3):
    """
    Combined Focal Loss and Dice Loss for better performance on imbalanced data.
    
    Args:
        class_labels: Binary class labels (0 or 1)
        logits: Raw model outputs (before sigmoid)
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
    
    Returns:
        Combined loss value
    """
    focal = focal_loss(class_labels, logits)
    dice = dice_loss(class_labels, logits)
    
    return focal_weight * focal + dice_weight * dice
