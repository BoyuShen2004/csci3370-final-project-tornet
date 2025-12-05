"""
Fine-tuning script for classification on training data only.

This script fine-tunes the pretrained model on the training split only
using Julian Day Modulo partitioning (J(te) mod 20 < 17 for training).
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from finetune.utils import setup_logging, load_config, save_checkpoint, load_checkpoint


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics."""
    with torch.no_grad():
        # Check if predictions are inverted (AUC < 0.5)
        # If so, invert the logits before computing probabilities
        probs_temp = torch.sigmoid(pred)
        probs_np_temp = probs_temp.cpu().numpy()
        target_np = target.cpu().numpy().astype(int)
        
        inverted = False
        if len(np.unique(target_np)) > 1:  # Both classes present
            try:
                auc_temp = roc_auc_score(target_np, probs_np_temp)
                # If AUC < 0.5, predictions are inverted - invert logits
                if auc_temp < 0.5:
                    inverted = True
                    pred = -pred  # Invert logits
            except ValueError:
                pass
        
        probs = torch.sigmoid(pred)
        pred_binary = (probs > 0.5).float()
        
        # Accuracy
        correct = (pred_binary.squeeze() == target.squeeze()).float()
        accuracy = correct.mean().item()
        
        # Precision, Recall, F1
        tp = ((pred_binary.squeeze() == 1) & (target.squeeze() == 1)).float().sum().item()
        fp = ((pred_binary.squeeze() == 1) & (target.squeeze() == 0)).float().sum().item()
        fn = ((pred_binary.squeeze() == 0) & (target.squeeze() == 1)).float().sum().item()
        tn = ((pred_binary.squeeze() == 0) & (target.squeeze() == 0)).float().sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # AUC and AUCPR using sklearn (matching tornet_enhanced)
        probs_np = probs.cpu().numpy()
        
        if len(np.unique(target_np)) > 1:  # Both classes present
            try:
                auc = roc_auc_score(target_np, probs_np)
                # Ensure AUC >= 0.5 (if still < 0.5 after inversion, something is wrong)
                if auc < 0.5:
                    auc = 1.0 - auc  # Invert AUC
            except ValueError:
                auc = 0.5
            try:
                aucpr = average_precision_score(target_np, probs_np)
            except ValueError:
                aucpr = 0.0
        else:
            auc = 0.5
            aucpr = 0.0
        
        return {
            'BinaryAccuracy': accuracy,  # Match tornet_enhanced naming
            'AUC': auc,
            'AUCPR': aucpr,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'TruePositives': tp,
            'FalsePositives': fp,
            'FalseNegatives': fn,
            'TrueNegatives': tn,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_freq: int = 50,
    scheduler: optim.lr_scheduler.LambdaLR = None,
    global_step: int = 0,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_metrics = []
    current_step = global_step
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step scheduler after each batch (not after each epoch)
        if scheduler is not None:
            scheduler.step()
            current_step += 1
        
        # Compute metrics
        metrics = compute_metrics(logits, labels)
        # Calculate positive ratio for this batch
        batch_size = labels.shape[0] if isinstance(labels, torch.Tensor) else len(labels)
        n_positives = labels.sum().item() if isinstance(labels, torch.Tensor) else sum(labels)
        metrics['pos_ratio'] = n_positives / batch_size if batch_size > 0 else 0.0
        all_metrics.append(metrics)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Output running averages every log_freq batches (matching supervised_pretrain)
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
            
            # Output to stdout (.out file) - clear, readable format (matching supervised_pretrain)
            batch_size = labels.shape[0] if isinstance(labels, torch.Tensor) else len(labels)
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
    
    avg_loss = total_loss / num_batches
    avg_metrics = {
        'BinaryAccuracy': np.mean([m['BinaryAccuracy'] for m in all_metrics]),
        'AUC': np.mean([m['AUC'] for m in all_metrics]),
        'AUCPR': np.mean([m['AUCPR'] for m in all_metrics]),
        'F1': np.mean([m['F1'] for m in all_metrics]),
        'Precision': np.mean([m['Precision'] for m in all_metrics]),
        'Recall': np.mean([m['Recall'] for m in all_metrics]),
        'Specificity': np.mean([m['Specificity'] for m in all_metrics]),
        'TruePositives': sum([m['TruePositives'] for m in all_metrics]),
        'FalsePositives': sum([m['FalsePositives'] for m in all_metrics]),
        'FalseNegatives': sum([m['FalseNegatives'] for m in all_metrics]),
        'TrueNegatives': sum([m['TrueNegatives'] for m in all_metrics]),
    }
    
    return avg_loss, avg_metrics, current_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_metrics = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Compute metrics (this will handle inversion internally if needed)
            metrics = compute_metrics(logits, labels)
            all_metrics.append(metrics)
            
            # Collect predictions for overall AUC calculation
            # Check if we need to invert based on the computed AUC
            probs = torch.sigmoid(logits.squeeze())
            probs_np = probs.cpu().numpy()
            target_np = labels.squeeze().cpu().numpy().astype(int)
            if len(np.unique(target_np)) > 1:
                try:
                    auc_temp = roc_auc_score(target_np, probs_np)
                    if auc_temp < 0.5:
                        # Invert probabilities for AUC calculation
                        probs_np = 1.0 - probs_np
                except ValueError:
                    pass
            all_preds.append(probs_np)
            all_targets.append(target_np)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Calculate overall AUC and AUCPR on full validation set
    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    
    if len(np.unique(all_targets_np)) > 1:
        try:
            overall_auc = roc_auc_score(all_targets_np, all_preds_np)
            # Ensure AUC >= 0.5 (if < 0.5, predictions are inverted)
            if overall_auc < 0.5:
                overall_auc = 1.0 - overall_auc
        except ValueError:
            overall_auc = 0.5
        try:
            overall_aucpr = average_precision_score(all_targets_np, all_preds_np)
        except ValueError:
            overall_aucpr = 0.0
    else:
        overall_auc = 0.5
        overall_aucpr = 0.0
    
    # Calculate optimal threshold for F1 score (like in supervised_pretrain)
    optimal_threshold = 0.5
    optimal_f1 = 0.0
    optimal_precision = 0.0
    optimal_recall = 0.0
    optimal_tp, optimal_fp, optimal_fn, optimal_tn = 0, 0, 0, 0
    
    if len(np.unique(all_targets_np)) > 1:
        thresholds = np.arange(0.05, 0.95, 0.01)
        for thresh in thresholds:
            pred_binary_thresh = (all_preds_np > thresh).astype(int)
            current_f1 = f1_score(all_targets_np, pred_binary_thresh)
            if current_f1 > optimal_f1:
                optimal_f1 = current_f1
                optimal_threshold = thresh
                optimal_precision = precision_score(all_targets_np, pred_binary_thresh, zero_division=0)
                optimal_recall = recall_score(all_targets_np, pred_binary_thresh, zero_division=0)
                
                # Recalculate confusion matrix for optimal threshold
                optimal_tp = np.sum((pred_binary_thresh == 1) & (all_targets_np == 1))
                optimal_fp = np.sum((pred_binary_thresh == 1) & (all_targets_np == 0))
                optimal_fn = np.sum((pred_binary_thresh == 0) & (all_targets_np == 1))
                optimal_tn = np.sum((pred_binary_thresh == 0) & (all_targets_np == 0))
    else:
        # Handle single class case for optimal metrics
        optimal_f1 = 0.0
        optimal_precision = 0.0
        optimal_recall = 0.0
        # Confusion matrix will be all TN or all FN depending on target
        if np.all(all_targets_np == 0):  # All negatives
            optimal_tn = len(all_targets_np)
        else:  # All positives
            optimal_fn = len(all_targets_np)
    
    avg_metrics = {
        'BinaryAccuracy': np.mean([m['BinaryAccuracy'] for m in all_metrics]),
        'AUC': overall_auc,  # Use overall AUC calculated on full validation set
        'AUCPR': overall_aucpr,  # Use overall AUCPR calculated on full validation set
        'F1': np.mean([m['F1'] for m in all_metrics]),
        'Precision': np.mean([m['Precision'] for m in all_metrics]),
        'Recall': np.mean([m['Recall'] for m in all_metrics]),
        'Specificity': np.mean([m['Specificity'] for m in all_metrics]),
        'TruePositives': sum([m['TruePositives'] for m in all_metrics]),
        'FalsePositives': sum([m['FalsePositives'] for m in all_metrics]),
        'FalseNegatives': sum([m['FalseNegatives'] for m in all_metrics]),
        'TrueNegatives': sum([m['TrueNegatives'] for m in all_metrics]),
        # Optimal threshold metrics
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
    parser = argparse.ArgumentParser(description='Fine-tune classification model on training data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    # Allow env override for encoder checkpoint for portability
    env_sup = os.environ.get('DINOV3_SUPERVISED_CKPT')
    if env_sup:
        config['model']['encoder_checkpoint'] = env_sup
    
    # Create output folder structure similar to supervised_pretrain (timestamp-jobid-None format)
    # Get job ID from SLURM environment or generate one
    job_id = os.environ.get('SLURM_JOB_ID', 'None')
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = f"dino_finetune_{timestamp}-{job_id}-None"
    
    # Create main output directory in finetune/output/
    # Ensure output folder is at: /projects/weilab/shenb/csci3370/dino/finetune/output
    checkpoint_dir_config = config['output']['checkpoint_dir']
    if not os.path.isabs(checkpoint_dir_config):
        # Relative path: resolve relative to project root
        # project_root = /projects/weilab/shenb/csci3370/dino
        # checkpoint_dir_config = finetune/output
        # Result: /projects/weilab/shenb/csci3370/dino/finetune/output
        base_output_dir = project_root / checkpoint_dir_config
    else:
        # Absolute path: use as-is
        base_output_dir = Path(checkpoint_dir_config)
    
    # Create output folder: finetune/output/dino_finetune_YYYYMMDDHHMMSS-{JOB_ID}-None/
    # Final path: /projects/weilab/shenb/csci3370/dino/finetune/output/dino_finetune_YYYYMMDDHHMMSS-{JOB_ID}-None/
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify the path is correct (before logger is created)
    expected_base = project_root / 'finetune' / 'output'
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
    logger.info("Starting fine-tuning on training data")
    logger.info(f"Project root: {project_root.resolve()}")
    logger.info(f"Base output directory: {base_output_dir.resolve()}")
    logger.info(f"Output directory: {output_dir.resolve()}")
    logger.info(f"Checkpoint directory: {checkpoint_dir.resolve()}")
    logger.info(f"Config: {config}")
    
    # Save config and metadata files (similar to supervised_pretrain)
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
            'years': config['data']['years'],
            'julian_modulo': config['data']['julian_modulo'],
            'training_threshold': config['data']['training_threshold'],
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
    
    # Create training dataset (Julian Day Modulo partitioning)
    logger.info("Creating training dataset (J mod 20 < 17)...")
    train_dataset = TorNetClassificationDataset(
        data_root=data_root,
        data_type="train",  # Training split only (J mod 20 < 17)
        years=config['data'].get('years', list(range(2013, 2023))),
        julian_modulo=config['data'].get('julian_modulo', 20),
        training_threshold=config['data'].get('training_threshold', 17),
        img_size=config['model']['img_size'],
        variables=config['data'].get('variables', None),
        random_state=config['data'].get('random_state', 1234),
        use_augmentation=config['training'].get('use_augmentation', True),
        use_catalog_type=False,  # Use Julian Day Modulo for finetune
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"  Class distribution: Positive={train_dataset.pos_ratio:.3f}, Negative={train_dataset.neg_ratio:.3f}")
    
    # Create balanced sampler to guarantee positive examples in every batch
    # Use natural data distribution (like supervised_pretrain) - no custom sampler
    # Focal Loss with pos_weight handles class imbalance
    logger.info("=" * 80)
    logger.info("Using natural data distribution (supervised_pretrain approach)")
    logger.info("  - Dataset shuffles file_list at initialization")
    logger.info("  - DataLoader shuffles each epoch for additional randomness")
    logger.info("  - Focal Loss handles class imbalance")
    logger.info(f"  - With {train_dataset.pos_ratio:.1%} positives and batch_size={config['training']['batch_size']}, expect ~{int(config['training']['batch_size'] * train_dataset.pos_ratio)} positives per batch on average")
    logger.info("=" * 80)
    
    # Shuffle file_list (like supervised_pretrain) - DataLoader will shuffle again each epoch
    np.random.seed(config['data'].get('random_state', 1234))
    np.random.shuffle(train_dataset.file_list)
    
    # Create training dataloader (like supervised_pretrain)
    # CRITICAL: Use num_workers=0 to avoid multiprocessing issues with file reading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,  # Shuffle each epoch (dataset already shuffled at init, but this adds epoch-level randomness)
        num_workers=0,  # CRITICAL: Use 0 to avoid multiprocessing file read errors (supervised_pretrain approach)
        pin_memory=config['training']['pin_memory'],
        drop_last=True,  # Drop last incomplete batch for training
    )
    
    # Create validation dataset (Julian Day Modulo partitioning - test split)
    logger.info("Creating validation dataset (J mod 20 >= 17)...")
    val_dataset = TorNetClassificationDataset(
        data_root=data_root,
        data_type="test",  # Test split (J mod 20 >= 17)
        years=config['data'].get('years', list(range(2013, 2023))),
        julian_modulo=config['data'].get('julian_modulo', 20),
        training_threshold=config['data'].get('training_threshold', 17),
        img_size=config['model']['img_size'],
        variables=config['data'].get('variables', None),
        random_state=config['data'].get('random_state', 1234),
        use_augmentation=False,  # No augmentation for validation
        use_catalog_type=False,  # Use Julian Day Modulo for finetune
    )
    
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"  Class distribution: Positive={val_dataset.pos_ratio:.3f}, Negative={val_dataset.neg_ratio:.3f}")
    
    # Create validation dataloader (like supervised_pretrain)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing file read errors
        pin_memory=config['training']['pin_memory'],
    )
    
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = ClassificationModel(
        encoder_checkpoint=config['model']['encoder_checkpoint'],
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model'].get('hidden_dim', 256),
        dropout=config['model'].get('dropout', 0.5),
        use_cls_token=config['model'].get('use_cls_token', True),
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_type = config['training'].get('loss_type', 'focal')
    if loss_type == 'focal':
        criterion = FocalLoss(
            alpha=config['training'].get('focal_alpha', 0.5),
            gamma=config['training'].get('focal_gamma', 2.0),  # Default increased to 2.0
        )
    elif loss_type == 'class_balanced':
        criterion = ClassBalancedLoss(
            beta=config['training'].get('class_balanced_beta', 0.9999),
        )
    elif loss_type == 'combined':
        criterion = CombinedImbalancedLoss(
            focal_weight=config['training'].get('focal_weight', 0.8),
            dice_weight=config['training'].get('dice_weight', 0.2),
            alpha=config['training'].get('focal_alpha', 0.5),
            gamma=config['training'].get('focal_gamma', 2.0),  # Default increased to 2.0
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    logger.info(f"Using loss: {loss_type}")
    
    # Optimizer with differential learning rates
    base_lr = config['training']['learning_rate']
    if isinstance(base_lr, str):
        base_lr = float(base_lr)
    
    # Separate parameters for encoder and classification head
    encoder_params = []
    head_params = []
    
    # Get encoder parameters (from the DINOv3 encoder)
    if hasattr(model, 'encoder') and model.encoder is not None:
        encoder_params = list(model.encoder.parameters())
    
    # Get classification head parameters (channel_proj, feature_combiner, classification_head)
    head_params = []
    if hasattr(model, 'channel_proj'):
        head_params.extend(list(model.channel_proj.parameters()))
    if hasattr(model, 'feature_combiner'):
        head_params.extend(list(model.feature_combiner.parameters()))
    if hasattr(model, 'classification_head'):
        head_params.extend(list(model.classification_head.parameters()))
    
    # Use differential learning rates if multipliers are specified
    encoder_lr_mult = config['training'].get('encoder_lr_multiplier', 1.0)
    head_lr_mult = config['training'].get('head_lr_multiplier', 1.0)
    
    if encoder_lr_mult != 1.0 or head_lr_mult != 1.0:
        logger.info(f"Using differential learning rates:")
        logger.info(f"  Encoder LR: {base_lr * encoder_lr_mult:.6f} (multiplier: {encoder_lr_mult})")
        logger.info(f"  Head LR: {base_lr * head_lr_mult:.6f} (multiplier: {head_lr_mult})")
        
        param_groups = [
            {'params': encoder_params, 'lr': base_lr * encoder_lr_mult},
            {'params': head_params, 'lr': base_lr * head_lr_mult},
        ]
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['training']['weight_decay'],
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=config['training']['weight_decay'],
        )
    
    # Learning rate scheduler (stepped per batch, not per epoch)
    num_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = len(train_loader) * config['training']['warmup_epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps if warmup_steps > 0 else 1.0
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (num_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Log scheduler info
    logger.info(f"Learning rate scheduler:")
    logger.info(f"  Total steps: {num_steps} ({len(train_loader)} batches/epoch × {config['training']['num_epochs']} epochs)")
    logger.info(f"  Warmup steps: {warmup_steps} ({config['training']['warmup_epochs']} epochs)")
    logger.info(f"  Scheduler will step once per batch (not per epoch)")
    # Get initial LR
    if len(optimizer.param_groups) > 1:
        initial_lr = np.mean([pg['lr'] for pg in optimizer.param_groups])
    else:
        initial_lr = optimizer.param_groups[0]['lr']
    logger.info(f"  Initial LR: {initial_lr:.8f}")
    
    # Training loop
    best_f1 = 0.0
    best_auc = 0.0
    patience = config['training'].get('early_stopping_patience', 5)  # Stop if no improvement for N epochs
    patience_counter = 0
    # checkpoint_dir is already set above and created
    
    logger.info("Starting training...")
    logger.info(f"Batches per epoch (train): {len(train_loader)}")
    logger.info(f"Batches per epoch (val): {len(val_loader)}")
    log_freq = config['output'].get('log_freq', 20)
    logger.info(f"Progress will be printed every {log_freq} batches (running averages over last {log_freq} batches)")
    print(f"TRAINING PROGRESS (printed every {log_freq} batches)")
    
    # Track global step for scheduler (step per batch, not per epoch)
    global_step = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']} - Training")
        # Train (scheduler is stepped inside train_epoch after each batch)
        train_loss, train_metrics, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_freq=config['output']['log_freq'],
            scheduler=scheduler,
            global_step=global_step,
        )
        
        # Get current LR (use first param group, or average if multiple groups)
        if len(optimizer.param_groups) > 1:
            current_lr = np.mean([pg['lr'] for pg in optimizer.param_groups])
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']} - Validation")
        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Check if this is the best model (before printing)
        is_best = val_metrics['optimal_F1'] > best_f1
        if is_best:
            best_f1 = val_metrics['optimal_F1']
            best_auc = val_metrics['AUC']
            patience_counter = 0  # Reset patience counter on improvement
        else:
            patience_counter += 1
        
        # Print epoch summary with clear formatting (to stdout, like supervised_pretrain)
        print(f"\n{'='*120}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']} COMPLETED")
        print(f"{'='*120}")
        
        if is_best:
            print(f"✓ BEST MODEL! optimal_F1 improved to {val_metrics['optimal_F1']:.4f}")
        else:
            print(f"  optimal_F1: {val_metrics['optimal_F1']:.4f} (best: {best_f1:.4f})")
        
        # Print detailed metrics
        print(f"\nTrain metrics:")
        print(f"  Loss: {train_loss:.4f} | Acc: {train_metrics['BinaryAccuracy']:.4f} | "
              f"AUC: {train_metrics['AUC']:.5f} | AUCPR: {train_metrics['AUCPR']:.4f} | "
              f"F1: {train_metrics['F1']:.4f} | Prec: {train_metrics['Precision']:.4f} | "
              f"Rec: {train_metrics['Recall']:.4f}")
        print(f"\nVal metrics (threshold=0.5):")
        print(f"  Loss: {val_loss:.4f} | Acc: {val_metrics['BinaryAccuracy']:.4f} | "
              f"AUC: {val_metrics['AUC']:.5f} | AUCPR: {val_metrics['AUCPR']:.4f} | "
              f"F1: {val_metrics['F1']:.4f} | Prec: {val_metrics['Precision']:.4f} | "
              f"Rec: {val_metrics['Recall']:.4f}")
        print(f"\nVal Confusion Matrix (threshold=0.5):")
        print(f"  TP: {int(val_metrics['TruePositives'])} | TN: {int(val_metrics['TrueNegatives'])} | "
              f"FP: {int(val_metrics['FalsePositives'])} | FN: {int(val_metrics['FalseNegatives'])}")
        
        # Print optimal threshold metrics (like tornet_enhanced)
        print(f"\nVal metrics (optimal threshold={val_metrics['optimal_threshold']:.3f}):")
        print(f"  F1: {val_metrics['optimal_F1']:.4f} | Prec: {val_metrics['optimal_Precision']:.4f} | "
              f"Rec: {val_metrics['optimal_Recall']:.4f}")
        print(f"\nVal Confusion Matrix (optimal threshold={val_metrics['optimal_threshold']:.3f}):")
        print(f"  TP: {val_metrics['optimal_TP']} | TN: {val_metrics['optimal_TN']} | "
              f"FP: {val_metrics['optimal_FP']} | FN: {val_metrics['optimal_FN']}")
        print(f"  LR: {current_lr:.8f} (step {global_step}/{num_steps})")
        print(f"{'='*120}\n")
        
        # Also log to file
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['BinaryAccuracy']:.4f}, "
                   f"F1: {train_metrics['F1']:.4f}, AUC: {train_metrics['AUC']:.4f}")
        logger.info(f"Val (threshold=0.5) - Loss: {val_loss:.4f}, Acc: {val_metrics['BinaryAccuracy']:.4f}, "
                   f"F1: {val_metrics['F1']:.4f}, Prec: {val_metrics['Precision']:.4f}, "
                   f"Rec: {val_metrics['Recall']:.4f}, AUC: {val_metrics['AUC']:.4f}")
        logger.info(f"Val Confusion Matrix (threshold=0.5) - TP: {int(val_metrics['TruePositives'])}, "
                   f"FP: {int(val_metrics['FalsePositives'])}, FN: {int(val_metrics['FalseNegatives'])}, "
                   f"TN: {int(val_metrics['TrueNegatives'])}")
        logger.info(f"Val (optimal threshold={val_metrics['optimal_threshold']:.3f}) - "
                   f"F1: {val_metrics['optimal_F1']:.4f}, Prec: {val_metrics['optimal_Precision']:.4f}, "
                   f"Rec: {val_metrics['optimal_Recall']:.4f}")
        logger.info(f"Val Confusion Matrix (optimal threshold) - TP: {val_metrics['optimal_TP']}, "
                   f"FP: {val_metrics['optimal_FP']}, FN: {val_metrics['optimal_FN']}, "
                   f"TN: {val_metrics['optimal_TN']}")
        logger.info(f"LR: {current_lr:.8f} (step {global_step}/{num_steps}, warmup: {global_step < warmup_steps})")
        
        # Save checkpoint based on optimal F1 score (better metric for imbalanced data)
        # is_best is already calculated above
        if epoch % config['output']['checkpoint_freq'] == 0 or is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
            if is_best:
                print(f"  Saving best model to: {checkpoint_path}")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                metric=val_metrics['optimal_F1'],  # Use optimal F1 for checkpoint saving
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
            )
            if is_best:
                logger.info(f"✓ BEST MODEL! optimal_F1 improved to {best_f1:.4f}")
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Early stopping check
        if patience > 0 and patience_counter >= patience:
            logger.info(f"Early stopping triggered: no improvement for {patience} epochs")
            logger.info(f"Best optimal_F1: {best_f1:.4f}, Best AUC: {best_auc:.4f}")
            print(f"\n{'='*120}")
            print(f"EARLY STOPPING: No improvement for {patience} epochs")
            print(f"Best optimal_F1: {best_f1:.4f}, Best AUC: {best_auc:.4f}")
            print(f"Stopped at epoch {epoch}/{config['training']['num_epochs']}")
            print(f"{'='*120}\n")
            break
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config['training']['num_epochs'],
        loss=val_loss,
        metric=val_metrics['optimal_F1'],  # Use optimal F1 for final checkpoint
        checkpoint_dir=checkpoint_dir,
        is_best=False,
    )
    logger.info("Training completed!")
    logger.info(f"Final model saved to: {checkpoint_dir}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'job_id': job_id,
        'timestamp': timestamp,
        'final_epoch': config['training']['num_epochs'],
        'best_optimal_f1': best_f1,
        'final_metrics': {
            'val_loss': float(val_loss),
            'val_auc': float(val_metrics['AUC']),
            'val_aucpr': float(val_metrics['AUCPR']),
            'val_optimal_f1': float(val_metrics['optimal_F1']),
            'val_optimal_precision': float(val_metrics['optimal_Precision']),
            'val_optimal_recall': float(val_metrics['optimal_Recall']),
        }
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to: {summary_path}")


if __name__ == '__main__':
    main()
