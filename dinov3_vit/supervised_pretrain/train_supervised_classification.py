"""
Supervised Classification Training Script

This script performs supervised training on training split (J(te) mod 20 < 17)
for binary classification (tornado vs no tornado) using Julian Day Modulo partitioning.
Matches tornet_enhanced approach with validation monitoring.
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
import json
from datetime import datetime
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from classification_model import ClassificationModel
from data_loader_tornet import create_tornet_dataloader, TorNetClassificationDataset
from imbalanced_losses import FocalLoss, ClassBalancedLoss, CombinedImbalancedLoss
from supervised_pretrain.utils import setup_logging, load_config, save_checkpoint, load_checkpoint


def oversample_positive_in_batch(images, labels, oversample_ratio=1.5, min_positives=2):
    """
    Oversample positive examples in a batch to ensure model sees them frequently.
    
    CRITICAL FIX: If batch has zero positives, we MUST add some from a positive pool.
    This prevents model collapse to all-negative predictions.
    
    Args:
        images: Batch of images [B, C, H, W] (torch.Tensor)
        labels: Batch of labels [B] (torch.Tensor, 0 or 1)
        oversample_ratio: Ratio for oversampling positive class (e.g., 1.5 means
                         50% more positive examples)
        min_positives: Minimum number of positive examples required in batch
    
    Returns:
        Oversampled images and labels (torch.Tensor)
    """
    # Convert to numpy for easier manipulation
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    positive_mask = labels_np == 1
    negative_mask = labels_np == 0
    
    n_positive = np.sum(positive_mask)
    n_negative = np.sum(negative_mask)
    
    # CRITICAL: If no positives in batch, we need to add some
    # This should rarely happen with WeightedRandomSampler, but we handle it anyway
    if n_positive == 0:
        # This is a serious problem - batch has no positives
        # In practice, this shouldn't happen with proper WeightedRandomSampler
        # But we log a warning and return the batch as-is (will be handled by loss)
        import warnings
        warnings.warn(f"Batch has zero positive examples! This indicates WeightedRandomSampler is not working correctly.")
        return images, labels
    
    # Only oversample if we have both classes and need more positives
    if n_positive > 0 and n_negative > 0 and n_positive < n_negative * oversample_ratio:
        pos_indices = np.where(positive_mask)[0]
        neg_indices = np.where(negative_mask)[0]
        
        # Calculate how many positive examples to add
        target_positives = max(min_positives, int(n_negative * oversample_ratio))
        n_oversample = max(0, target_positives - n_positive)
        
        if n_oversample > 0:
            # Randomly sample positive examples with replacement
            oversample_indices = np.random.choice(pos_indices, n_oversample, replace=True)
            
            # Combine original and oversampled
            balanced_indices = np.concatenate([pos_indices, oversample_indices, neg_indices])
            
            # Shuffle to mix positive and negative examples
            np.random.shuffle(balanced_indices)
            
            # Convert back to torch tensors if needed
            if isinstance(images, torch.Tensor):
                return images[balanced_indices], labels[balanced_indices]
            else:
                return images[balanced_indices], labels[balanced_indices]
    
    # No oversampling needed or can't oversample
    return images, labels

# Note: We removed WeightedRandomSampler - class imbalance is handled via pos_weight in loss function
from tornet.data.loader import read_file


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics."""
    with torch.no_grad():
        # Ensure shapes are correct
        pred = pred.squeeze()  # [B] or [B, 1] -> [B]
        target = target.squeeze()  # [B] or [B, 1] -> [B]
        
        # Handle case where pred or target might be 0-d
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        probs = torch.sigmoid(pred)
        pred_binary = (probs > 0.5).float()
        
        # Accuracy
        correct = (pred_binary == target).float()
        accuracy = correct.mean().item()
        
        # Confusion matrix components
        tp = ((pred_binary == 1) & (target == 1)).float().sum().item()
        fp = ((pred_binary == 1) & (target == 0)).float().sum().item()
        fn = ((pred_binary == 0) & (target == 1)).float().sum().item()
        tn = ((pred_binary == 0) & (target == 0)).float().sum().item()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # AUC and AUCPR using sklearn (matching tornet_enhanced)
        probs_np = probs.cpu().numpy()
        target_np = target.cpu().numpy().astype(int)
        
        if len(np.unique(target_np)) > 1:  # Both classes present
            try:
                auc = roc_auc_score(target_np, probs_np)
            except ValueError:
                auc = 0.5
            try:
                aucpr = average_precision_score(target_np, probs_np)
            except ValueError:
                aucpr = 0.0
        else:
            auc = 0.5
            aucpr = 0.0
        
        # Additional metrics for imbalanced datasets
        # Balanced accuracy (average of sensitivity and specificity)
        sensitivity = recall  # TP / (TP + FN)
        specificity = tn / (tn + fp + 1e-8)  # TN / (TN + FP)
        balanced_acc = (sensitivity + specificity) / 2
        
        # Positive class ratio in batch (for debugging)
        pos_ratio = (target == 1).float().mean().item()
        
        return {
            'BinaryAccuracy': accuracy,  # Match tornet_enhanced naming
            'AUC': auc,
            'AUCPR': aucpr,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'TruePositives': tp,
            'FalsePositives': fp,
            'FalseNegatives': fn,
            'TrueNegatives': tn,
            'balanced_accuracy': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'pos_ratio': pos_ratio,
            'mean_prob': probs.mean().item(),
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_freq: int = 100,
    use_mixed_precision: bool = False,
    scaler: torch.amp.GradScaler = None,
    logger=None,
    config=None,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_metrics = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Optional per-batch composition logging (disabled to keep logs compact)
        
        # No oversampling needed - tornet_enhanced doesn't use it
        # Natural distribution with Focal Loss is sufficient
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Check batch composition (for monitoring)
        # With 7.4% positive ratio and batch_size=64, ~1% of batches will have 0 positives
        # This is normal with natural distribution - Focal Loss with pos_weight handles it
        n_positives = (labels > 0.5).sum().item()
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_mixed_precision and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                
                # Check if logits are NaN/inf BEFORE computing loss
                if not torch.isfinite(logits).all():
                    if logger is not None:
                        logger.error(f"Batch {batch_idx}: Model outputs NaN/inf logits! This indicates a problem in the forward pass.")
                        logger.error(f"  Logits: min={logits.min().item() if torch.isfinite(logits).any() else 'all_nan'}, max={logits.max().item() if torch.isfinite(logits).any() else 'all_nan'}, mean={logits.mean().item() if torch.isfinite(logits).any() else 'all_nan'}")
                        logger.error(f"  Images: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}, has_nan={torch.isnan(images).any().item()}")
                    # Replace NaN logits with zeros to prevent crash, but skip this batch
                    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
                    # Skip this batch - don't backpropagate through NaN
                    optimizer.zero_grad()
                    continue
                
                loss = criterion(logits, labels)
                
                # Check for nan/inf loss values (critical for debugging)
                if not torch.isfinite(loss):
                    if logger is not None:
                        with torch.no_grad():
                            probs_debug = torch.sigmoid(logits.squeeze())
                            logger.error(f"Batch {batch_idx}: Loss is nan/inf! Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                            logger.error(f"  Probs: min={probs_debug.min().item():.6f}, max={probs_debug.max().item():.6f}, mean={probs_debug.mean().item():.6f}")
                            logger.error(f"  Labels: {labels.sum().item()} positives, {(labels < 0.5).sum().item()} negatives")
                    # Skip this batch - don't backpropagate through NaN loss
                    optimizer.zero_grad()
                    continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for nan/inf gradients before stepping
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    if logger is not None and batch_idx < 10:  # Only log first few batches
                        logger.warning(f"Batch {batch_idx}: NaN/Inf gradients detected in {name}, skipping update")
                    has_nan_grad = True
                    break
            
            if not has_nan_grad:
                scaler.step(optimizer)
                scaler.update()
            else:
                # Skip update if gradients are invalid
                # CRITICAL: Must call scaler.update() even when skipping to reset scaler state
                # Otherwise, next batch's unscale_() will fail
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision
            logits = model(images)
            
            # Check if logits are NaN/inf BEFORE computing loss
            if not torch.isfinite(logits).all():
                if logger is not None:
                    logger.error(f"Batch {batch_idx}: Model outputs NaN/inf logits! This indicates a problem in the forward pass.")
                    logger.error(f"  Logits: min={logits.min().item() if torch.isfinite(logits).any() else 'all_nan'}, max={logits.max().item() if torch.isfinite(logits).any() else 'all_nan'}, mean={logits.mean().item() if torch.isfinite(logits).any() else 'all_nan'}")
                    logger.error(f"  Images: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}, has_nan={torch.isnan(images).any().item()}")
                # Skip this batch - don't backpropagate through NaN
                optimizer.zero_grad()
                continue
            
            loss = criterion(logits, labels)
            
            # Check for nan/inf loss values (critical for debugging)
            if not torch.isfinite(loss):
                if logger is not None:
                    with torch.no_grad():
                        probs_debug = torch.sigmoid(logits.squeeze())
                        logger.error(f"Batch {batch_idx}: Loss is nan/inf! Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        logger.error(f"  Probs: min={probs_debug.min().item():.6f}, max={probs_debug.max().item():.6f}, mean={probs_debug.mean().item():.6f}")
                        logger.error(f"  Labels: {labels.sum().item()} positives, {(labels < 0.5).sum().item()} negatives")
                # Skip this batch - don't backpropagate through NaN loss
                optimizer.zero_grad()
                continue
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for nan/inf gradients before stepping
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    if logger is not None and batch_idx < 10:  # Only log first few batches
                        logger.warning(f"Batch {batch_idx}: NaN/Inf gradients detected in {name}, skipping update")
                    has_nan_grad = True
                    break
            
            if not has_nan_grad:
                optimizer.step()
            else:
                # Skip update if gradients are invalid
                optimizer.zero_grad()
        
        # Update scheduler (only if we actually did an update)
        if scheduler is not None and torch.isfinite(loss):
            scheduler.step()
        
        # Compute metrics (only if logits are valid)
        if torch.isfinite(logits).all():
            metrics = compute_metrics(logits, labels)
            all_metrics.append(metrics)
        else:
            # Skip metrics for NaN batches, but add dummy metrics to maintain batch count
            # This prevents index errors but signals the problem
            all_metrics.append({
                'BinaryAccuracy': 0.0,
                'AUC': 0.5,
                'AUCPR': 0.0,
                'F1': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'balanced_accuracy': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'TruePositives': 0,
                'FalsePositives': 0,
                'FalseNegatives': int(labels.sum().item()) if torch.isfinite(labels).all() else 0,
                'TrueNegatives': int((labels < 0.5).sum().item()) if torch.isfinite(labels).all() else 0,
            })
        
        # Debug: Check if loss is actually zero (shouldn't happen)
        loss_value = loss.item()
        if loss_value < 1e-6 and batch_idx < 10 and logger is not None:  # Only log first few batches
            with torch.no_grad():
                probs_debug = torch.sigmoid(logits.squeeze())
                mean_prob = probs_debug.mean().item()
                min_prob = probs_debug.min().item()
                max_prob = probs_debug.max().item()
                mean_logit = logits.squeeze().mean().item()
                logger.warning(f"Batch {batch_idx}: Loss={loss_value:.8f}, MeanProb={mean_prob:.6f}, "
                             f"ProbRange=[{min_prob:.6f}, {max_prob:.6f}], MeanLogit={mean_logit:.4f}")
        
        total_loss += loss_value
        num_batches += 1
        
        # Print metrics every log_freq batches (default 20) to reduce log size for long training runs
        # This is more practical for day-long training runs
        batch_size = labels.shape[0] if isinstance(labels, torch.Tensor) else len(labels)
        
        # Output running averages every log_freq batches (default 20)
        if (batch_idx + 1) % log_freq == 0 or (batch_idx + 1) == len(dataloader) or batch_idx == 0:
            # Compute running averages up to current batch
            avg_loss = total_loss / num_batches
            recent_metrics = all_metrics  # Use all metrics so far for running average
            avg_acc = np.mean([m['BinaryAccuracy'] for m in recent_metrics])
            avg_f1 = np.mean([m['F1'] for m in recent_metrics])
            avg_recall = np.mean([m['Recall'] for m in recent_metrics])
            avg_precision = np.mean([m['Precision'] for m in recent_metrics])
            avg_auc = np.mean([m['AUC'] for m in recent_metrics])
            avg_aucpr = np.mean([m['AUCPR'] for m in recent_metrics])
            
            # Sum confusion matrix components (running totals)
            total_tp = sum([m['TruePositives'] for m in recent_metrics])
            total_fp = sum([m['FalsePositives'] for m in recent_metrics])
            total_fn = sum([m['FalseNegatives'] for m in recent_metrics])
            total_tn = sum([m['TrueNegatives'] for m in recent_metrics])
            
            # Output to stdout (.out file) - clear, readable format
            # Print every log_freq batches (default 20) - running averages over last log_freq batches
            avg_pos_ratio = sum([m.get('pos_ratio', 0) for m in recent_metrics]) / len(recent_metrics) if recent_metrics else 0
            msg = (f"Epoch {epoch} | Batch {batch_idx + 1:5d}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | "
                  f"AUC: {avg_auc:.5f} | "
                  f"AUCPR: {avg_aucpr:.4f} | "
                  f"F1: {avg_f1:.4f} | "
                  f"Prec: {avg_precision:.4f} | "
                  f"Rec: {avg_recall:.4f} | "
                  f"TP: {total_tp:.0f} FP: {total_fp:.0f} FN: {total_fn:.0f} TN: {total_tn:.0f} | "
                  f"Pos: {avg_pos_ratio*batch_size:.1f}/{batch_size} (avg over last {min(log_freq, batch_idx+1)} batches)")
            print(msg)
            if logger is not None:
                logger.info(msg)
    
    avg_loss = total_loss / num_batches
    # Aggregate metrics across all batches
    avg_metrics = {
        'BinaryAccuracy': np.mean([m['BinaryAccuracy'] for m in all_metrics]),
        'AUC': np.mean([m['AUC'] for m in all_metrics]),
        'AUCPR': np.mean([m['AUCPR'] for m in all_metrics]),
        'F1': np.mean([m['F1'] for m in all_metrics]),
        'Precision': np.mean([m['Precision'] for m in all_metrics]),
        'Recall': np.mean([m['Recall'] for m in all_metrics]),
        'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        # Sum confusion matrix components across all batches
        'TruePositives': sum([m['TruePositives'] for m in all_metrics]),
        'FalsePositives': sum([m['FalsePositives'] for m in all_metrics]),
        'FalseNegatives': sum([m['FalseNegatives'] for m in all_metrics]),
        'TrueNegatives': sum([m['TrueNegatives'] for m in all_metrics]),
    }
    
    return avg_loss, avg_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = False,
):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_metrics = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
            
            # Compute metrics
            metrics = compute_metrics(logits, labels)
            all_metrics.append(metrics)
            
            # Collect predictions for overall AUC calculation
            probs = torch.sigmoid(logits.squeeze())
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.squeeze().cpu().numpy().astype(int))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Calculate overall AUC and AUCPR on full validation set
    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    
    if len(np.unique(all_targets_np)) > 1:
        try:
            overall_auc = roc_auc_score(all_targets_np, all_preds_np)
        except ValueError:
            overall_auc = 0.5
        try:
            overall_aucpr = average_precision_score(all_targets_np, all_preds_np)
        except ValueError:
            overall_aucpr = 0.0
    else:
        overall_auc = 0.5
        overall_aucpr = 0.0
    
    # Find optimal threshold that maximizes F1 (like tornet_enhanced)
    optimal_threshold = 0.5
    optimal_f1 = 0.0
    optimal_precision = 0.0
    optimal_recall = 0.0
    optimal_tp = 0
    optimal_fp = 0
    optimal_fn = 0
    optimal_tn = 0
    
    if len(np.unique(all_targets_np)) > 1:
        # Try thresholds from 0.05 to 0.95 in steps of 0.01
        thresholds = np.arange(0.05, 0.95, 0.01)
        for thresh in thresholds:
            pred_binary = (all_preds_np > thresh).astype(int)
            try:
                f1 = f1_score(all_targets_np, pred_binary, zero_division=0)
                if f1 > optimal_f1:
                    optimal_f1 = f1
                    optimal_threshold = thresh
                    optimal_precision = precision_score(all_targets_np, pred_binary, zero_division=0)
                    optimal_recall = recall_score(all_targets_np, pred_binary, zero_division=0)
                    # Calculate confusion matrix at optimal threshold
                    optimal_tp = ((pred_binary == 1) & (all_targets_np == 1)).sum()
                    optimal_fp = ((pred_binary == 1) & (all_targets_np == 0)).sum()
                    optimal_fn = ((pred_binary == 0) & (all_targets_np == 1)).sum()
                    optimal_tn = ((pred_binary == 0) & (all_targets_np == 0)).sum()
            except:
                continue
    
    # Aggregate metrics at fixed threshold 0.5 (standard)
    avg_metrics = {
        'BinaryAccuracy': np.mean([m['BinaryAccuracy'] for m in all_metrics]),
        'AUC': overall_auc,  # Use overall AUC calculated on full validation set
        'AUCPR': overall_aucpr,  # Use overall AUCPR calculated on full validation set
        'F1': np.mean([m['F1'] for m in all_metrics]),
        'Precision': np.mean([m['Precision'] for m in all_metrics]),
        'Recall': np.mean([m['Recall'] for m in all_metrics]),
        'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        # Sum confusion matrix components at threshold 0.5
        'TruePositives': sum([m['TruePositives'] for m in all_metrics]),
        'FalsePositives': sum([m['FalsePositives'] for m in all_metrics]),
        'FalseNegatives': sum([m['FalseNegatives'] for m in all_metrics]),
        'TrueNegatives': sum([m['TrueNegatives'] for m in all_metrics]),
        # Optimal threshold metrics (like tornet_enhanced)
        'optimal_threshold': optimal_threshold,
        'optimal_F1': optimal_f1,
        'optimal_Precision': optimal_precision,
        'optimal_Recall': optimal_recall,
        'optimal_TP': int(optimal_tp),
        'optimal_FP': int(optimal_fp),
        'optimal_FN': int(optimal_fn),
        'optimal_TN': int(optimal_tn),
    }
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Supervised classification training on all data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    # Allow env override for pretrained encoder for portability
    env_pretrained = os.environ.get('DINOV3_PRETRAINED')
    if env_pretrained:
        config['model']['pretrained_checkpoint'] = env_pretrained
    
    # Create output folder structure similar to tornet (timestamp-jobid-None format)
    # Get job ID from SLURM environment or generate one
    job_id = os.environ.get('SLURM_JOB_ID', 'None')
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = f"dino_supervised_{timestamp}-{job_id}-None"
    
    # Create main output directory in supervised_pretrain/output/
    # Ensure output folder is at: /projects/weilab/shenb/csci3370/dino/supervised_pretrain/output
    checkpoint_dir_config = config['output']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir_config):
        # Relative path: resolve relative to project root
        # project_root = /projects/weilab/shenb/csci3370/dino
        # checkpoint_dir_config = supervised_pretrain/output
        # Result: /projects/weilab/shenb/csci3370/dino/supervised_pretrain/output
        base_output_dir = project_root / checkpoint_dir_config
    else:
        # Absolute path: use as-is
        base_output_dir = Path(checkpoint_dir_config)
    
    # Create output folder: supervised_pretrain/output/dino_supervised_YYYYMMDDHHMMSS-{JOB_ID}-None/
    # Final path: /projects/weilab/shenb/csci3370/dino/supervised_pretrain/output/dino_supervised_YYYYMMDDHHMMSS-{JOB_ID}-None/
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify the path is correct (before logger is created)
    expected_base = project_root / 'supervised_pretrain' / 'output'
    if base_output_dir.resolve() != expected_base.resolve():
        print(f"WARNING: Output directory path may be unexpected: {base_output_dir.resolve()}")
        print(f"Expected: {expected_base.resolve()}")
    
    # Create checkpoints subdirectory
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config to use new checkpoint directory
    config['output']['checkpoint_dir'] = str(checkpoint_dir)
    
    # Setup logging (create logs subdirectory in output folder)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir))
    logger.info("Starting supervised classification training")
    logger.info(f"Project root: {project_root.resolve()}")
    logger.info(f"Base output directory: {base_output_dir.resolve()}")
    logger.info(f"Output directory: {output_dir.resolve()}")
    logger.info(f"Checkpoint directory: {checkpoint_dir.resolve()}")
    logger.info(f"Config: {config}")
    
    # Save config and metadata files (similar to tornet)
    # Save config.yaml
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to: {config_path}")
    
    # Save params.json (training parameters)
    params = {
        'model_name': model_name,
        'job_id': job_id,
        'timestamp': timestamp,
        'training': config['training'],
        'model': config['model'],
        'data': {
            'train_years': config['data']['train_years'],
            'val_years': config['data']['val_years'],
            'variables': config['data']['variables'],
        }
    }
    params_path = output_dir / 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved params to: {params_path}")
    
    # Save data.json (dataset info - will be updated after dataset creation)
    data_info = {
        'model_name': model_name,
        'job_id': job_id,
        'timestamp': timestamp,
    }
    data_path = output_dir / 'data.json'
    with open(data_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    
    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get data root from environment
    data_root = os.environ.get('TORNET_ROOT', config['data'].get('data_root', '/projects/weilab/shenb/csci3370/data'))
    logger.info(f"TORNET_ROOT: {data_root}")
    
    # Use catalog-based partitioning (tornet_enhanced style)
    use_catalog_type = config['data'].get('use_catalog_type', False)  # Default False for backward compatibility
    use_temporal_val = config['data'].get('use_temporal_val', False)  # Default False for backward compatibility
    
    # Get train and validation years
    train_years = config['data'].get('train_years', config['data'].get('years', list(range(2013, 2023))))
    val_years = config['data'].get('val_years', list(range(2021, 2023)))  # NEW: Recent years for validation
    
    if use_temporal_val:
        logger.info("Using temporal validation split (tornet_enhanced style)")
        logger.info(f"  Training years: {train_years}")
        logger.info(f"  Validation years: {val_years}")
        
        # Create training dataset from train_years
        logger.info(f"Creating training dataset from years {train_years}...")
        train_dataset = TorNetClassificationDataset(
            data_root=data_root,
            data_type="train",
            years=train_years,
            julian_modulo=config['data'].get('julian_modulo', 20),
            training_threshold=config['data'].get('training_threshold', 17),
            img_size=config['model']['img_size'],
            variables=config['data'].get('variables', None),
            random_state=config['data'].get('random_state', 1234),
            use_augmentation=config['training'].get('use_augmentation', True),
            use_catalog_type=use_catalog_type,
        )
        
        # Create validation dataset from val_years (temporal split)
        logger.info(f"Creating validation dataset from years {val_years}...")
        val_dataset = TorNetClassificationDataset(
            data_root=data_root,
            data_type="train",  # Still use "train" type, but different years
            years=val_years,
            julian_modulo=config['data'].get('julian_modulo', 20),
            training_threshold=config['data'].get('training_threshold', 17),
            img_size=config['model']['img_size'],
            variables=config['data'].get('variables', None),
            random_state=config['data'].get('random_state', 1234),
            use_augmentation=False,  # No augmentation for validation
            use_catalog_type=use_catalog_type,
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"  Class distribution: Positive={train_dataset.pos_ratio:.3f}, Negative={train_dataset.neg_ratio:.3f}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"  Class distribution: Positive={val_dataset.pos_ratio:.3f}, Negative={val_dataset.neg_ratio:.3f}")
        
        # Update data.json with dataset information
        data_info.update({
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'train_pos_ratio': float(train_dataset.pos_ratio),
            'val_pos_ratio': float(val_dataset.pos_ratio) if hasattr(val_dataset, 'pos_ratio') else 0.0,
        })
        with open(data_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        logger.info(f"Updated data.json with dataset information")
        
        # Verify both splits have positive examples
        if train_dataset.pos_ratio == 0.0:
            raise RuntimeError("CRITICAL: Training set has NO positive examples!")
        if val_dataset.pos_ratio == 0.0:
            raise RuntimeError("CRITICAL: Validation set has NO positive examples!")
        
    else:
        # Original stratified random split approach
        logger.info("Using stratified random validation split (original DINO approach)")
        logger.info(f"Data partitioning: Julian Day Modulo (julian_modulo={config['data'].get('julian_modulo', 20)}, training_threshold={config['data'].get('training_threshold', 17)})")
        logger.info(f"Training split: J(te) mod {config['data'].get('julian_modulo', 20)} < {config['data'].get('training_threshold', 17)}")
        
        # Create full training dataset (J mod 20 < 17) - no augmentation for splitting
        logger.info("Creating training dataset (J mod 20 < 17)...")
        full_train_dataset = TorNetClassificationDataset(
            data_root=data_root,
            data_type="train",  # Use training split only (J mod 20 < 17)
            years=config['data'].get('years', list(range(2013, 2023))),
            julian_modulo=config['data'].get('julian_modulo', 20),
            training_threshold=config['data'].get('training_threshold', 17),
            img_size=config['model']['img_size'],
            variables=config['data'].get('variables', None),
            random_state=config['data'].get('random_state', 1234),
            use_augmentation=False,  # No augmentation for base dataset
            use_catalog_type=use_catalog_type,
        )
        
        logger.info(f"Full training dataset size: {len(full_train_dataset)}")
        logger.info(f"Estimated class distribution: Positive={full_train_dataset.pos_ratio:.3f}, Negative={full_train_dataset.neg_ratio:.3f}")
        
        # CRITICAL: Use stratified split to ensure both train and val have positive examples
        # Random split can accidentally put all positives in one split, causing model collapse
        labels = None
        valid_indices = None
        
        # Try to load labels from pre-computed sample weights file (if it exists from old runs)
        weights_path = Path(config['output']['checkpoint_dir']) / "sample_weights.pkl"
        if weights_path.exists():
            try:
                import pickle
                logger.info(f"Loading labels from pre-computed sample weights: {weights_path}")
                with open(weights_path, 'rb') as f:
                    weights_data = pickle.load(f)
                
                # Check if file_list matches and labels are available
                if (weights_data.get('file_list') and 
                    len(weights_data['file_list']) == len(full_train_dataset.file_list) and
                    weights_data.get('labels') is not None):
                    # Verify file list matches (check first few)
                    if all(weights_data['file_list'][i] == full_train_dataset.file_list[i] for i in range(min(10, len(full_train_dataset)))):
                        labels = weights_data['labels']
                        # Only use indices where we have valid labels (same length as file_list)
                        valid_indices = list(range(len(labels)))
                        logger.info(f"Successfully loaded labels from pre-computed weights ({len(labels)} labels)")
                    else:
                        logger.warning("File list mismatch in pre-computed weights - will read labels from files")
                else:
                    logger.warning("Pre-computed weights missing labels or file_list - will read labels from files")
            except Exception as e:
                logger.warning(f"Failed to load labels from pre-computed weights: {e}. Will read labels from files.")
        
        # If labels not loaded from pre-computed weights, read them from files
        if labels is None:
            logger.info("Reading labels from files for stratified train/val split (this may take a few minutes)...")
            try:
                from tornet.data.loader import read_file
            except ImportError:
                import sys
                sys.path.insert(0, '/projects/weilab/shenb/csci3370/tornet_enhanced')
                from tornet.data.loader import read_file
            
            # Read labels for all files to enable stratified splitting
            labels = []
            valid_indices = []
            total_files = len(full_train_dataset)
            log_interval = max(1, total_files // 20)  # Log every 5%
            
            for i in range(total_files):
                if i % log_interval == 0:
                    logger.info(f"  Reading labels: {i}/{total_files} ({100*i/total_files:.1f}%)")
                
                try:
                    data = read_file(full_train_dataset.file_list[i], variables=full_train_dataset.variables, n_frames=1, tilt_last=True)
                    label_arr = data['label']
                    if label_arr.ndim == 0:
                        label = label_arr.item()
                    else:
                        label = label_arr[-1] if len(label_arr) > 0 else 0
                    labels.append(1 if label > 0 else 0)
                    valid_indices.append(i)
                except Exception as e:
                    # Skip files that can't be read
                    if i % 1000 == 0:
                        logger.warning(f"Error reading file {i} for stratified split: {e}")
                    continue
            
            if len(valid_indices) == 0:
                raise RuntimeError("No valid files found for stratified split!")
            
            logger.info(f"Successfully read labels for {len(valid_indices)}/{len(full_train_dataset)} files")
            
            # Save labels to weights file for future use (to avoid re-reading)
            try:
                import pickle
                weights_data = {
                    'labels': labels,
                    'file_list': [full_train_dataset.file_list[i] for i in valid_indices],
                    'valid_indices': valid_indices,
                }
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                with open(weights_path, 'wb') as f:
                    pickle.dump(weights_data, f)
                logger.info(f"Saved labels to {weights_path} for future use (will skip reading labels next time)")
            except Exception as e:
                logger.warning(f"Failed to save labels: {e}. Will re-read labels next time.")
        
        # Use stratified split to ensure both splits have positive examples
        val_split = config['data'].get('val_split', 0.1)
        train_indices_list, val_indices_list = train_test_split(
            valid_indices,
            test_size=val_split,
            stratify=labels,
            random_state=config['data'].get('random_state', 1234),
            shuffle=True
        )
        
        # Convert to torch random_split compatible format
        class IndexDataset:
            def __init__(self, indices):
                self.indices = indices
            def __len__(self):
                return len(self.indices)
        
        train_indices = IndexDataset(train_indices_list)
        val_indices = IndexDataset(val_indices_list)
        
        # Verify class distribution in each split
        train_labels = [labels[valid_indices.index(i)] for i in train_indices_list]
        val_labels = [labels[valid_indices.index(i)] for i in val_indices_list]
        train_pos_ratio = sum(train_labels) / len(train_labels) if len(train_labels) > 0 else 0.0
        val_pos_ratio = sum(val_labels) / len(val_labels) if len(val_labels) > 0 else 0.0
        
        logger.info(f"Train dataset size: {len(train_indices)} ({len(train_indices)/len(valid_indices)*100:.1f}%)")
        logger.info(f"  Train class distribution: Positive={train_pos_ratio:.4f} ({sum(train_labels)}/{len(train_labels)}), Negative={1-train_pos_ratio:.4f}")
        logger.info(f"Validation dataset size: {len(val_indices)} ({len(val_indices)/len(valid_indices)*100:.1f}%)")
        logger.info(f"  Val class distribution: Positive={val_pos_ratio:.4f} ({sum(val_labels)}/{len(val_labels)}), Negative={1-val_pos_ratio:.4f}")
        
        # Update data.json with dataset information (for stratified split)
        data_info.update({
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_pos_ratio': float(train_pos_ratio),
            'val_pos_ratio': float(val_pos_ratio),
        })
        with open(data_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        logger.info(f"Updated data.json with dataset information")
        
        if train_pos_ratio == 0.0:
            raise RuntimeError("CRITICAL: Training set has NO positive examples! Stratified split failed.")
        if val_pos_ratio == 0.0:
            raise RuntimeError("CRITICAL: Validation set has NO positive examples! Stratified split failed.")
        
        # Create train dataset with augmentation
        train_dataset = TorNetClassificationDataset(
            data_root=data_root,
            data_type="train",
            years=config['data'].get('years', list(range(2013, 2023))),
            julian_modulo=config['data'].get('julian_modulo', 20),
            training_threshold=config['data'].get('training_threshold', 17),
            img_size=config['model']['img_size'],
            variables=config['data'].get('variables', None),
            random_state=config['data'].get('random_state', 1234),
            use_augmentation=config['training'].get('use_augmentation', True),
            use_catalog_type=use_catalog_type,
        )
        # Filter to only train indices
        train_dataset.file_list = [full_train_dataset.file_list[i] for i in train_indices_list]
        # Shuffle file_list (like tornet_enhanced shuffles catalog) - DataLoader will shuffle again each epoch
        np.random.seed(config['data'].get('random_state', 1234))
        np.random.shuffle(train_dataset.file_list)
        # Update pos_ratio and neg_ratio to reflect actual training split distribution
        train_dataset.pos_ratio = train_pos_ratio
        train_dataset.neg_ratio = 1.0 - train_pos_ratio
        
        # Create validation dataset without augmentation
        val_dataset = TorNetClassificationDataset(
            data_root=data_root,
            data_type="train",  # Still from training split, but used for validation
            years=config['data'].get('years', list(range(2013, 2023))),
            julian_modulo=config['data'].get('julian_modulo', 20),
            training_threshold=config['data'].get('training_threshold', 17),
            img_size=config['model']['img_size'],
            variables=config['data'].get('variables', None),
            random_state=config['data'].get('random_state', 1234),
            use_augmentation=False,  # No augmentation for validation
            use_catalog_type=use_catalog_type,
        )
        # Filter to only validation indices
        val_dataset.file_list = [full_train_dataset.file_list[i] for i in val_indices_list]
        # No need to shuffle validation dataset
        # Update pos_ratio and neg_ratio to reflect actual validation split distribution
        val_dataset.pos_ratio = val_pos_ratio
        val_dataset.neg_ratio = 1.0 - val_pos_ratio
    
    # Use natural data distribution (like tornet_enhanced) - no custom sampler
    # tornet_enhanced works by:
    # 1. Shuffling catalog once at initialization
    # 2. Using natural sequential access to shuffled file_list
    # 3. Relying on Focal Loss with pos_weight to handle imbalance
    # With 7.4% positive ratio and batch_size=64, we expect ~4-5 positives per batch on average
    # This is sufficient for learning (tornet_enhanced proves this works)
    logger.info("=" * 80)
    logger.info("Using natural data distribution (tornet_enhanced approach)")
    logger.info("  - Dataset shuffles file_list at initialization (like tornet_enhanced shuffles catalog)")
    logger.info("  - DataLoader shuffles each epoch for additional randomness")
    logger.info("  - Focal Loss with pos_weight handles class imbalance")
    logger.info(f"  - With {train_dataset.pos_ratio:.1%} positives and batch_size={config['training']['batch_size']}, expect ~{int(config['training']['batch_size'] * train_dataset.pos_ratio)} positives per batch on average")
    logger.info("=" * 80)
    
    # CRITICAL: Use num_workers=0 to avoid multiprocessing issues with file reading
    # tornet_enhanced uses generators which don't have this issue
    # With num_workers > 0, file reads can fail silently, causing all labels to default to 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,  # Shuffle each epoch (dataset already shuffled at init, but this adds epoch-level randomness)
        num_workers=0,  # CRITICAL: Use 0 to avoid multiprocessing file read errors (tornet_enhanced approach)
        pin_memory=config['training']['pin_memory'],
        drop_last=True,  # Drop last incomplete batch for training
    )
    
    # Create validation dataloader (no augmentation, no class balancing)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing file read errors
        pin_memory=config['training']['pin_memory'],
    )
    
    # Create model
    logger.info("Creating classification model...")
    
    # Resolve pretrained checkpoint path relative to project root
    pretrained_checkpoint = config['model'].get('pretrained_checkpoint')
    if pretrained_checkpoint:
        # If path is relative, resolve it relative to project root
        checkpoint_path = Path(pretrained_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        pretrained_checkpoint = str(checkpoint_path)
        logger.info(f"Pretrained checkpoint path: {pretrained_checkpoint}")
        if not Path(pretrained_checkpoint).exists():
            logger.warning(f"Pretrained checkpoint not found at {pretrained_checkpoint}, using randomly initialized encoder")
        else:
            logger.info(f"Found pretrained checkpoint at {pretrained_checkpoint}")
    
    model = ClassificationModel(
        encoder_checkpoint=pretrained_checkpoint,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model'].get('hidden_dim', 256),
        dropout=config['model'].get('dropout', 0.5),
        use_cls_token=config['model'].get('use_cls_token', True),
        pos_ratio=train_dataset.pos_ratio,  # Pass actual class distribution for bias initialization
    ).to(device)
    
    # Classification head bias is initialized to match class distribution
    # This prevents model from starting at 50% positives and collapsing to 0% negatives
    # Formula: bias = log(pos_ratio / (1 - pos_ratio)) so sigmoid(bias) ≈ pos_ratio
    initial_bias = np.log(train_dataset.pos_ratio / (1 - train_dataset.pos_ratio))
    logger.info(f"Classification head bias initialized to {initial_bias:.4f} (matching class distribution)")
    logger.info(f"  Class distribution: pos_ratio={train_dataset.pos_ratio:.4f}, neg_ratio={train_dataset.neg_ratio:.4f}")
    logger.info(f"  Model will start predicting ~{train_dataset.pos_ratio*100:.1f}% positives (prevents collapse to all negatives)")
    
    # Freeze encoder initially if specified (train only new layers for first few epochs)
    freeze_encoder_epochs = config['training'].get('freeze_encoder_epochs', 0)
    if freeze_encoder_epochs > 0:
        logger.info(f"Freezing encoder for first {freeze_encoder_epochs} epochs (training only new layers)")
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        logger.info("Encoder frozen. Only new layers (channel_proj, feature_combiner, classification_head) will be trained.")
    
    # Performance optimization: torch.compile (PyTorch 2.0+)
    if config['training'].get('use_torch_compile', False):
        try:
            logger.info("Compiling model with torch.compile for faster training...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (may not be available): {e}. Continuing without compilation.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function - Match tornet_enhanced exactly (NO pos_weight in focal loss)
    # tornet_enhanced handles class imbalance via alpha in focal loss, NOT via pos_weight
    # Using pos_weight causes model to over-predict positives (high recall, low precision)
    loss_type = config['training'].get('loss_type', 'focal')
    
    if loss_type == 'focal':
        # Match tornet_enhanced parameters EXACTLY: alpha=0.25, gamma=2.0, label_smoothing=0.1
        # NO pos_weight - tornet_enhanced doesn't use it, and it causes over-prediction of positives
        criterion = FocalLoss(
            alpha=config['training'].get('focal_alpha', 0.25),  # tornet_enhanced: 0.25
            gamma=config['training'].get('focal_gamma', 2.0),  # tornet_enhanced: 2.0
            pos_weight=None,  # CRITICAL: tornet_enhanced doesn't use pos_weight - it causes over-prediction
            label_smoothing=config['training'].get('label_smoothing', 0.1),  # tornet_enhanced: 0.1
        )
        logger.info(f"Using FocalLoss (tornet_enhanced style): alpha={config['training'].get('focal_alpha', 0.25)}, gamma={config['training'].get('focal_gamma', 2.0)}")
        logger.info(f"  NO pos_weight (tornet_enhanced approach - alpha handles class imbalance)")
        logger.info(f"  label_smoothing={config['training'].get('label_smoothing', 0.1)} (prevents overconfidence)")
        logger.info(f"  Natural distribution with Focal Loss (no oversampling needed)")
    elif loss_type == 'class_balanced':
        criterion = ClassBalancedLoss(
            beta=config['training'].get('class_balanced_beta', 0.9999),
        )
    elif loss_type == 'combined':
        # Calculate pos_weight for combined loss (only used if needed)
        pos_weight_value = train_dataset.neg_ratio / (train_dataset.pos_ratio + 1e-8)
        criterion = CombinedImbalancedLoss(
            focal_weight=config['training'].get('focal_weight', 0.8),
            dice_weight=config['training'].get('dice_weight', 0.2),
            alpha=config['training'].get('focal_alpha', 0.75),
            gamma=config['training'].get('focal_gamma', 3.0),
            pos_weight=None,  # Don't use pos_weight in combined loss either (match tornet_enhanced)
            label_smoothing=config['training'].get('label_smoothing', 0.0),
        )
        logger.info(f"Using CombinedImbalancedLoss: focal_weight={config['training'].get('focal_weight', 0.8)}, dice_weight={config['training'].get('dice_weight', 0.2)}")
        logger.info(f"  NO pos_weight (tornet_enhanced approach)")
    else:
        # For BCE, we can optionally use pos_weight (but tornet_enhanced doesn't use BCE)
        # Calculate pos_weight for reference, but we'll use it only if explicitly needed
        pos_weight_value = train_dataset.neg_ratio / (train_dataset.pos_ratio + 1e-8)
        # Use pos_weight for BCE to handle class imbalance (standard PyTorch approach)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
        logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_value:.4f} (standard PyTorch approach for BCE)")
    
    logger.info(f"Loss type: {loss_type}")
    
    # Optimizer with differential learning rates
    # Use higher LR for classification head and channel projection (new layers)
    # Use lower LR for pretrained encoder (fine-tuning)
    learning_rate = config['training']['learning_rate']
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    # Separate parameters: encoder (pretrained) vs new layers (classification head, channel_proj, feature_combiner)
    encoder_params = []
    new_layer_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            new_layer_params.append(param)
    
    # Use lower LR for encoder (fine-tuning), higher LR for new layers
    encoder_lr = learning_rate * 0.1  # 10x lower for pretrained encoder
    new_layer_lr = learning_rate * 5.0  # 5x higher for new layers (they need to learn from scratch, more aggressive)
    
    optimizer = optim.AdamW(
        [
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': config['training']['weight_decay']},
            {'params': new_layer_params, 'lr': new_layer_lr, 'weight_decay': config['training']['weight_decay']},
        ],
    )
    
    logger.info(f"Using differential learning rates:")
    logger.info(f"  Encoder (pretrained): {encoder_lr:.6f} (fine-tuning)")
    logger.info(f"  New layers (head, channel_proj, feature_combiner): {new_layer_lr:.6f} (learning from scratch)")
    logger.info(f"  Encoder params: {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"  New layer params: {sum(p.numel() for p in new_layer_params):,}")
    
    # Learning rate scheduler
    num_steps = len(train_dataloader) * config['training']['num_epochs']
    warmup_steps = len(train_dataloader) * config['training']['warmup_epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (num_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    checkpoint_dir = config['output']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Starting training...")
    logger.info(f"Total epochs: {config['training']['num_epochs']}")
    logger.info(f"Batches per epoch (train): {len(train_dataloader)}")
    logger.info(f"Batches per epoch (val): {len(val_dataloader)}")
    log_freq = config['output'].get('log_freq', 20)
    logger.info(f"Progress will be printed every {log_freq} batches (running averages over last {log_freq} batches)")
    logger.info(f"  This reduces log size for long training runs (e.g., day-long training)")
    
    # Print header to stdout for easy monitoring in SLURM logs
    print("\n" + "="*150)
    print(f"TRAINING PROGRESS (printed every {log_freq} batches)")
    print("="*150)
    print(f"Epoch | Batch    | Loss    | Acc     | AUC     | AUCPR   | F1      | Prec    | Rec     | TP  FP  FN  TN  | Pos (avg)")
    print("-"*150)
    
    start_epoch = 1
    best_val_auc = 0.0  # Track val_AUC like tornet_enhanced
    global_step = 0
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        resume_path = args.resume
    else:
        resume_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    
    if os.path.exists(resume_path):
        try:
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            start_epoch, best_metric, global_step = load_checkpoint(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            start_epoch += 1
            logger.info(f"Resumed from epoch {start_epoch - 1}, best_val_AUC {best_val_auc:.5f}")
        except (RuntimeError, ValueError, Exception) as e:
            logger.warning(f"Failed to load checkpoint from {resume_path}: {e}")
            logger.warning("Starting training from scratch. Renaming corrupted checkpoint to avoid future issues.")
            # Rename corrupted checkpoint to avoid trying to load it again
            import shutil
            corrupted_path = resume_path + '.corrupted'
            try:
                shutil.move(resume_path, corrupted_path)
                logger.info(f"Renamed corrupted checkpoint to: {corrupted_path}")
            except Exception as rename_error:
                logger.warning(f"Could not rename corrupted checkpoint: {rename_error}")
            start_epoch = 1
            best_val_auc = 0.0
            global_step = 0
    else:
        logger.info("No checkpoint found, starting training from scratch")
    
    # Setup mixed precision training if enabled
    use_mixed_precision = config['training'].get('use_mixed_precision', False)
    scaler = None
    if use_mixed_precision:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Mixed precision training enabled (FP16/BF16)")
    
    # Validation frequency (validate every N epochs)
    val_freq = config['training'].get('val_freq', 1)
    logger.info(f"Validation frequency: every {val_freq} epoch(s)")
    
    # Track if encoder has been unfrozen
    encoder_unfrozen = False
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        # Unfreeze encoder after freeze period
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1 and not encoder_unfrozen:
            logger.info(f"Unfreezing encoder at epoch {epoch} (after {freeze_encoder_epochs} epochs of training new layers)")
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = True
            encoder_unfrozen = True
            
            # Recreate optimizer with unfrozen encoder parameters
            encoder_params = []
            new_layer_params = []
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    new_layer_params.append(param)
            
            optimizer = optim.AdamW(
                [
                    {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': config['training']['weight_decay']},
                    {'params': new_layer_params, 'lr': new_layer_lr, 'weight_decay': config['training']['weight_decay']},
                ],
            )
            
            # Recreate scheduler with new optimizer
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info("Optimizer and scheduler recreated with unfrozen encoder parameters")
        
        # Print epoch header (matching tornet_enhanced format)
        print(f"\n{'='*120}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']} - Training")
        print(f"{'='*120}")
        
        # Training
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_freq=config['output']['log_freq'],
            use_mixed_precision=use_mixed_precision,
            scaler=scaler,
            logger=logger,
            config=config['training'],
        )
        
        # Validation (only if val_freq matches)
        if epoch % val_freq == 0 or epoch == config['training']['num_epochs']:
            print(f"\n{'='*120}")
            print(f"Epoch {epoch}/{config['training']['num_epochs']} - Validation")
            print(f"{'='*120}")
            val_loss, val_metrics = validate_epoch(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                use_mixed_precision=use_mixed_precision,
            )
        else:
            # Skip validation, use previous metrics
            logger.info(f"Skipping validation at epoch {epoch} (val_freq={val_freq})")
            # Update global_step (number of batches processed)
            global_step += len(train_dataloader)
            
            # Print epoch summary (without validation metrics)
            print(f"\n{'='*120}")
            print(f"Epoch {epoch}/{config['training']['num_epochs']} COMPLETED (validation skipped)")
            print(f"{'='*120}")
            print(f"Train metrics:")
            print(f"  Loss: {train_loss:.4f} | Acc: {train_metrics['BinaryAccuracy']:.4f} | "
                  f"AUC: {train_metrics['AUC']:.5f} | AUCPR: {train_metrics['AUCPR']:.4f} | "
                  f"F1: {train_metrics['F1']:.4f} | Prec: {train_metrics['Precision']:.4f} | "
                  f"Rec: {train_metrics['Recall']:.4f}")
            logger.info(f"EPOCH {epoch}/{config['training']['num_epochs']} COMPLETED (validation skipped)")
            logger.info(f"Train - Loss: {train_loss:.4f}, BinaryAccuracy: {train_metrics['BinaryAccuracy']:.4f}, "
                       f"AUC: {train_metrics['AUC']:.4f}, AUCPR: {train_metrics['AUCPR']:.4f}, "
                       f"F1: {train_metrics['F1']:.4f}")
            
            # Save checkpoint (latest only, not best)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=False,
                global_step=global_step,
            )
            continue
        
        current_lr = optimizer.param_groups[0]['lr']
        val_auc = val_metrics['AUC']
        
        # Check if val_AUC improved (matching tornet_enhanced format exactly)
        is_best = val_auc > best_val_auc
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        
        # Print epoch summary with clear formatting
        print(f"\n{'='*120}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']} COMPLETED")
        print(f"{'='*120}")
        
        if is_best:
            if best_val_auc == 0.0:
                print(f"✓ BEST MODEL! val_AUC improved from None to {val_auc:.5f}")
            else:
                print(f"✓ BEST MODEL! val_AUC improved from {best_val_auc:.5f} to {val_auc:.5f}")
            print(f"  Saving best model to: {checkpoint_path}")
            best_val_auc = val_auc
        else:
            print(f"  val_AUC: {val_auc:.5f} (best: {best_val_auc:.5f})")
        
        # Print detailed metrics
        print(f"\nTrain metrics:")
        print(f"  Loss: {train_loss:.4f} | Acc: {train_metrics['BinaryAccuracy']:.4f} | "
              f"AUC: {train_metrics['AUC']:.5f} | AUCPR: {train_metrics['AUCPR']:.4f} | "
              f"F1: {train_metrics['F1']:.4f} | Prec: {train_metrics['Precision']:.4f} | "
              f"Rec: {train_metrics['Recall']:.4f}")
        print(f"\nVal metrics (threshold=0.5):")
        print(f"  Loss: {val_loss:.4f} | Acc: {val_metrics['BinaryAccuracy']:.4f} | "
              f"AUC: {val_auc:.5f} | AUCPR: {val_metrics['AUCPR']:.4f} | "
              f"F1: {val_metrics['F1']:.4f} | Prec: {val_metrics['Precision']:.4f} | "
              f"Rec: {val_metrics['Recall']:.4f}")
        print(f"\nVal Confusion Matrix (threshold=0.5):")
        print(f"  TP: {val_metrics['TruePositives']:.0f} | TN: {val_metrics['TrueNegatives']:.0f} | "
              f"FP: {val_metrics['FalsePositives']:.0f} | FN: {val_metrics['FalseNegatives']:.0f}")
        
        # Print optimal threshold metrics (like tornet_enhanced)
        print(f"\nVal metrics (optimal threshold={val_metrics['optimal_threshold']:.3f}):")
        print(f"  F1: {val_metrics['optimal_F1']:.4f} | Prec: {val_metrics['optimal_Precision']:.4f} | "
              f"Rec: {val_metrics['optimal_Recall']:.4f}")
        print(f"\nVal Confusion Matrix (optimal threshold={val_metrics['optimal_threshold']:.3f}):")
        print(f"  TP: {val_metrics['optimal_TP']} | TN: {val_metrics['optimal_TN']} | "
              f"FP: {val_metrics['optimal_FP']} | FN: {val_metrics['optimal_FN']}")
        print(f"{'='*120}\n")
        
        logger.info(f"EPOCH {epoch}/{config['training']['num_epochs']} COMPLETED")
        logger.info(f"Train - Loss: {train_loss:.4f}, BinaryAccuracy: {train_metrics['BinaryAccuracy']:.4f}, "
                   f"AUC: {train_metrics['AUC']:.4f}, AUCPR: {train_metrics['AUCPR']:.4f}, "
                   f"F1: {train_metrics['F1']:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, BinaryAccuracy: {val_metrics['BinaryAccuracy']:.4f}, "
                   f"AUC: {val_auc:.5f}, AUCPR: {val_metrics['AUCPR']:.4f}, "
                   f"F1: {val_metrics['F1']:.4f} (threshold=0.5)")
        logger.info(f"Val Confusion Matrix (threshold=0.5) - TP: {val_metrics['TruePositives']:.0f}, FP: {val_metrics['FalsePositives']:.0f}, "
                   f"FN: {val_metrics['FalseNegatives']:.0f}, TN: {val_metrics['TrueNegatives']:.0f}")
        logger.info(f"Val Optimal Threshold - threshold: {val_metrics['optimal_threshold']:.3f}, "
                   f"F1: {val_metrics['optimal_F1']:.4f}, Prec: {val_metrics['optimal_Precision']:.4f}, "
                   f"Rec: {val_metrics['optimal_Recall']:.4f}")
        logger.info(f"Val Confusion Matrix (optimal threshold={val_metrics['optimal_threshold']:.3f}) - "
                   f"TP: {val_metrics['optimal_TP']}, FP: {val_metrics['optimal_FP']}, "
                   f"FN: {val_metrics['optimal_FN']}, TN: {val_metrics['optimal_TN']}")
        
        # Update global_step (number of batches processed: train + val)
        global_step += len(train_dataloader) + len(val_dataloader)
        
        # Save checkpoint based on val_AUC (matching tornet_enhanced)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,  # Use validation loss for checkpoint
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
            scheduler=scheduler,
            global_step=global_step,
        )
        
        if epoch % config['output']['checkpoint_freq'] == 0 or is_best:
            logger.info(f"Saved checkpoint at epoch {epoch} (val_AUC: {val_auc:.5f})")
    
    logger.info("Training completed!")
    logger.info(f"Best validation AUC: {best_val_auc:.5f}")
    logger.info(f"Final model saved to: {checkpoint_dir}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    # Save final summary
    summary = {
        'model_name': model_name,
        'job_id': job_id,
        'timestamp': timestamp,
        'best_val_auc': float(best_val_auc),
        'total_epochs': config['training']['num_epochs'],
        'final_epoch': epoch,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to: {summary_path}")


if __name__ == '__main__':
    main()

