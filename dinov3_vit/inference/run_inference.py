"""
Inference script for classification on test/validation data.

This script evaluates the fine-tuned model on the test/validation split
using Julian Day Modulo partitioning (J(te) mod 20 >= 17 for testing).
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
from typing import Dict
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from classification_model import ClassificationModel
from data_loader_tornet import create_tornet_dataloader


def compute_classification_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        pred_binary = (probs > 0.5).float()
        
        # Basic metrics
        correct = (pred_binary.squeeze() == target.squeeze()).float()
        accuracy = correct.mean().item()
        
        # Confusion matrix
        tp = ((pred_binary.squeeze() == 1) & (target.squeeze() == 1)).float().sum().item()
        fp = ((pred_binary.squeeze() == 1) & (target.squeeze() == 0)).float().sum().item()
        fn = ((pred_binary.squeeze() == 0) & (target.squeeze() == 1)).float().sum().item()
        tn = ((pred_binary.squeeze() == 0) & (target.squeeze() == 0)).float().sum().item()
        
        # Derived metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # CSI (Critical Success Index)
        csi = tp / (tp + fp + fn + 1e-8)
        
        # FAR (False Alarm Rate)
        far = fp / (tp + fp + 1e-8)
        
        # HSS (Heidke Skill Score)
        n = tp + tn + fp + fn
        hss = ((tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-8))
        
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
        
        return {
            'BinaryAccuracy': accuracy,  # Match tornet_enhanced naming
            'AUC': auc,
            'AUCPR': aucpr,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'CSI': csi,  # Critical Success Index
            'FAR': far,  # False Alarm Rate
            'HSS': hss,  # Heidke Skill Score
            'TruePositives': int(tp),
            'FalsePositives': int(fp),
            'FalseNegatives': int(fn),
            'TrueNegatives': int(tn),
        }


def run_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> Dict[str, float]:
    """Run evaluation on dataset."""
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Compute metrics
            metrics = compute_classification_metrics(logits, labels)
            all_metrics.append(metrics)
            
            # Store predictions
            all_predictions.extend((probs > 0.5).cpu().numpy().flatten().tolist())
            all_targets.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Calculate overall AUC and AUCPR on full test set (matching tornet_enhanced)
    all_probs_np = np.array(all_probs)
    all_targets_np = np.array(all_targets).astype(int)
    
    if len(np.unique(all_targets_np)) > 1:
        try:
            overall_auc = roc_auc_score(all_targets_np, all_probs_np)
        except ValueError:
            overall_auc = 0.5
        try:
            overall_aucpr = average_precision_score(all_targets_np, all_probs_np)
        except ValueError:
            overall_aucpr = 0.0
    else:
        overall_auc = 0.5
        overall_aucpr = 0.0
    
    # Aggregate metrics (sum confusion matrix, average others)
    total_tp = int(np.sum([m['TruePositives'] for m in all_metrics]))
    total_fp = int(np.sum([m['FalsePositives'] for m in all_metrics]))
    total_fn = int(np.sum([m['FalseNegatives'] for m in all_metrics]))
    total_tn = int(np.sum([m['TrueNegatives'] for m in all_metrics]))
    
    # Recalculate metrics from aggregated confusion matrix
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    specificity = total_tn / (total_tn + total_fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    csi = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    far = total_fp / (total_tp + total_fp + 1e-8)
    n = total_tp + total_tn + total_fp + total_fn
    hss = ((total_tp * total_tn - total_fp * total_fn) / 
           ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn) + 1e-8))
    accuracy = (total_tp + total_tn) / (n + 1e-8)
    
    avg_metrics = {
        'BinaryAccuracy': accuracy,
        'AUC': overall_auc,  # Use overall AUC calculated on full test set
        'AUCPR': overall_aucpr,  # Use overall AUCPR calculated on full test set
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'CSI': csi,
        'FAR': far,
        'HSS': hss,
        'TruePositives': total_tp,
        'FalsePositives': total_fp,
        'FalseNegatives': total_fn,
        'TrueNegatives': total_tn,
    }
    
    # Print results (matching tornet_enhanced format)
    print("\n" + "="*60)
    print("CLASSIFICATION EVALUATION RESULTS (Test Split)")
    print("="*60)
    print(f"Data partitioning: Julian Day Modulo (J(te) mod 20 >= 17)")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {total_tp}")
    print(f"  True Negatives:  {total_tn}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"\nClassification Metrics:")
    print(f"  BinaryAccuracy:  {accuracy:.4f}")
    print(f"  AUC:             {overall_auc:.5f}")
    print(f"  AUCPR:           {overall_aucpr:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"\nWeather Metrics:")
    print(f"  CSI (Critical Success Index): {csi:.4f}")
    print(f"  FAR (False Alarm Rate):       {far:.4f}")
    print(f"  HSS (Heidke Skill Score):     {hss:.4f}")
    print("="*60)
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate classification model on test data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Allow env override for finetuned checkpoint
    env_ckpt = os.environ.get('DINOV3_FINETUNE_CKPT')
    if env_ckpt:
        config['model']['checkpoint'] = env_ckpt
    
    # Create unique output directory (similar to finetune)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    model_name = f"dino_inference_{timestamp}-{job_id}-None"
    
    # Create output folder structure similar to finetune
    base_output_dir = project_root / 'inference' / 'output'
    output_dir = base_output_dir / model_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with unique output directory
    config['data']['output_dir'] = str(output_dir)
    
    print(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data root from environment
    data_root = os.environ.get('TORNET_ROOT', config['data'].get('data_root', '/projects/weilab/shenb/csci3370/data'))
    print(f"TORNET_ROOT: {data_root}")
    
    # Create dataset (test split using Julian Day Modulo partitioning)
    print("Creating test dataset (J mod 20 >= 17)...")
    from data_loader_tornet import TorNetClassificationDataset
    from torch.utils.data import DataLoader
    
    test_dataset = TorNetClassificationDataset(
        data_root=data_root,
        data_type="test",  # Test split (J mod 20 >= 17)
        years=config['data'].get('years', list(range(2013, 2023))),
        julian_modulo=config['data'].get('julian_modulo', 20),
        training_threshold=config['data'].get('training_threshold', 17),
        img_size=config['model']['img_size'],
        variables=config['data'].get('variables', None),
        random_state=config['data'].get('random_state', 1234),
        use_augmentation=False,
        use_catalog_type=False,  # Use Julian Day Modulo for inference
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"  Class distribution: Positive={test_dataset.pos_ratio:.3f}, Negative={test_dataset.neg_ratio:.3f}")
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['inference']['num_workers'],
        pin_memory=True,
    )
    print(f"Test dataset size: {len(dataloader.dataset)}")
    
    # Load model
    print("Loading model...")
    model = ClassificationModel(
        encoder_checkpoint=None,  # Not needed for inference
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model'].get('hidden_dim', 256),
        dropout=config['model'].get('dropout', 0.5),
        use_cls_token=config['model'].get('use_cls_token', True),
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = config['model']['checkpoint']
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
    else:
        # PyTorch 2.6+ changed default weights_only=True, but our checkpoints contain numpy scalars
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Save config to output directory
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to: {config_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'job_id': job_id,
        'checkpoint_path': checkpoint_path,
        'test_dataset_size': len(test_dataset),
        'test_pos_ratio': test_dataset.pos_ratio,
        'test_neg_ratio': test_dataset.neg_ratio,
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    # Run evaluation
    print("Running evaluation...")
    metrics = run_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=str(output_dir),
    )
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'job_id': job_id,
        'checkpoint_path': checkpoint_path,
        'metrics': metrics,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nEvaluation completed!")
    return metrics


if __name__ == '__main__':
    main()
