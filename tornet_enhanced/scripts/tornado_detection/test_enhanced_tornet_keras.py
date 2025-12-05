"""
Enhanced TorNet Evaluation Script with Julian Day Modulo Partitioning

This script provides comprehensive evaluation of the enhanced model using the same
data partitioning methodology as the paper:
- Julian Day Modulo 20 partitioning (J(t_e) mod 20 < 17 for training, >= 17 for testing)
- 30-minute and 0.25-degree overlap removal
- Training: 171,666 samples (84.5%), Testing: 31,467 samples (15.5%)

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import sys
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import keras
from keras.models import load_model

from tornet.data.loader import get_dataloader
from tornet.data.constants import ALL_VARIABLES
from tornet.data.tornet_partitioning import load_tornet_partitioning
from tornet.metrics.keras import metrics as tfm

# Import custom layers to register them with Keras for deserialization
from tornet.models.keras.layers import CoordConv2D, FillNaNs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration for enhanced TorNet model
ENHANCED_TORNET_CONFIG = {
    'input_variables': ALL_VARIABLES,
    'test_years': list(range(2013, 2023)),  # All years for Julian Day Modulo partitioning
    'batch_size': 64,
    'dataloader_kwargs': {
        'select_keys': ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'range_folded_mask', 'coordinates']
    }
}


def load_enhanced_model(model_path: str):
    """
    Load the enhanced model with proper custom layer registration.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded Keras model
    """
    try:
        model = load_model(model_path, compile=False)
        logger.info(f"Successfully loaded enhanced model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model_performance(model, test_loader, config: Dict, partitioning_info: Dict) -> Dict:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        test_loader: Test data loader
        config: Configuration dictionary
        partitioning_info: TorNet partitioning information
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Starting enhanced TorNet model evaluation...")
    logger.info(f"Partitioning info: {partitioning_info['training_samples']} training, {partitioning_info['testing_samples']} testing")
    
    # Optionally compile for progress-rich evaluate (shows per-step metrics)
    try:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [
            keras.metrics.AUC(from_logits=True, name='AUC', num_thresholds=2000),
            keras.metrics.AUC(from_logits=True, curve='PR', name='AUCPR', num_thresholds=2000),
            tfm.BinaryAccuracy(True, name='BinaryAccuracy'),
            tfm.Precision(True, name='Precision'),
            tfm.Recall(True, name='Recall'),
            tfm.F1Score(from_logits=True, name='F1'),
        ]
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        logger.info("Running Keras evaluate() to show per-step progress...")
        _ = model.evaluate(test_loader, verbose=1, return_dict=True)
    except Exception as e:
        logger.warning(f"Evaluate with progress failed or skipped: {e}")

    # First pass: collect true labels (quiet)
    y_true_list = []
    total_batches = 0
    for _, batch_y in test_loader:
        y_true_list.extend(batch_y.flatten())
        total_batches += 1

    # Second pass: run prediction with verbose=1 to show per-step progress
    logger.info("Running model.predict() with progress to show per-step output...")
    y_pred_proba = model.predict(test_loader, verbose=1).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = np.array(y_true_list)
    
    logger.info(f"Evaluation completed. Processed {total_batches} batches.")
    logger.info(f"Total samples: {len(y_true)}")
    logger.info(f"Positive samples: {np.sum(y_true)}")
    logger.info(f"Negative samples: {len(y_true) - np.sum(y_true)}")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
    # Persist raw arrays for downstream plotting/saving
    try:
        metrics['y_true'] = y_true.astype(int).tolist()
        metrics['y_pred'] = y_pred.astype(int).tolist()
        metrics['y_pred_proba'] = y_pred_proba.astype(float).tolist()
    except Exception:
        pass
    
    # Add partitioning information to metrics
    metrics['partitioning_info'] = partitioning_info
    
    return metrics


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary containing all metrics
    """
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Critical Success Index (CSI)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Heidke Skill Score (HSS)
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
    
    # Threat Score (TS)
    ts = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # False Alarm Rate (FAR)
    far = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    # Probability of Detection (POD)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Probability of False Detection (POFD)
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'csi': csi,
        'hss': hss,
        'ts': ts,
        'far': far,
        'pod': pod,
        'pofd': pofd,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return metrics


def plot_evaluation_curves(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str):
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plots
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation curves saved to {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tornado', 'Tornado'],
                yticklabels=['No Tornado', 'Tornado'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def save_evaluation_results(metrics: Dict, save_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        save_path: Path to save results
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to {save_path}")


def print_evaluation_summary(metrics: Dict):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "="*70)
    print("ENHANCED TORNET MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Print partitioning information
    if 'partitioning_info' in metrics:
        partitioning = metrics['partitioning_info']
        print(f"\n📊 DATA PARTITIONING:")
        print(f"  Method: {partitioning.get('method', 'julian_day_modulo')}")
        print(f"  Training samples: {partitioning.get('training_samples', 'N/A')} ({partitioning.get('training_percentage', 0):.1f}%)")
        print(f"  Testing samples: {partitioning.get('testing_samples', 'N/A')} ({partitioning.get('testing_percentage', 0):.1f}%)")
        print(f"  Overlap removal: {partitioning.get('overlap_removal', False)}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"\n📈 AUC METRICS:")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:      {metrics['pr_auc']:.4f}")
    
    print(f"\n🌪️  TORNADO DETECTION METRICS:")
    print(f"  CSI:         {metrics['csi']:.4f}")
    print(f"  HSS:         {metrics['hss']:.4f}")
    print(f"  Threat Score: {metrics['ts']:.4f}")
    print(f"  POD:         {metrics['pod']:.4f}")
    print(f"  FAR:         {metrics['far']:.4f}")
    
    print(f"\n📋 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"  True Negatives:  {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")
    print(f"  True Positives:  {cm['tp']}")
    
    print("\n" + "="*70)


def main():
    """
    Main evaluation function.
    """
    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use the latest enhanced model
        model_path = "/projects/weilab/shenb/csci3370/tornet/tornado_enhanced/final_model.keras"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please provide a valid model path as an argument")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading enhanced model from: {model_path}")
    model = load_enhanced_model(model_path)
    
    # Get data root
    data_root = os.environ['TORNET_ROOT']
    logger.info(f"TORNET_ROOT: {data_root}")
    
    # Load partitioning information
    partitioning_path = os.path.join(os.path.dirname(model_path), 'partitioning_info.json')
    if os.path.exists(partitioning_path):
        partitioning_info = load_tornet_partitioning(partitioning_path)
        logger.info(f"Loaded partitioning info: {partitioning_info['training_samples']} training, {partitioning_info['testing_samples']} testing")
    else:
        logger.warning("Partitioning information not found. Using default configuration.")
        partitioning_info = {
            'method': 'julian_day_modulo',
            'training_samples': 171666,
            'testing_samples': 31467,
            'training_percentage': 84.5,
            'testing_percentage': 15.5
        }
    
    # Create test data loader using the same interface as training
    # dataloader: one of ["keras", "tensorflow", "torch", ...]
    dataloader_name = "keras"
    test_loader = get_dataloader(
        dataloader=dataloader_name,
        data_root=data_root,
        years=ENHANCED_TORNET_CONFIG['test_years'],
        data_type="test",
        batch_size=ENHANCED_TORNET_CONFIG['batch_size'],
        weights=None,
        **ENHANCED_TORNET_CONFIG['dataloader_kwargs']
    )
    
    # Evaluate model
    metrics = evaluate_model_performance(model, test_loader, ENHANCED_TORNET_CONFIG, partitioning_info)
    
    # Print results
    print_evaluation_summary(metrics)
    
    # Save results
    results_dir = os.path.dirname(model_path)
    save_evaluation_results(metrics, os.path.join(results_dir, 'enhanced_tornet_evaluation_results.json'))
    
    # Generate plots (if matplotlib is available)
    try:
        plot_evaluation_curves(
            np.array(metrics.get('y_true', [])),
            np.array(metrics.get('y_pred_proba', [])),
            os.path.join(results_dir, 'enhanced_tornet_evaluation_curves.png')
        )
        
        plot_confusion_matrix(
            np.array(metrics.get('y_true', [])),
            np.array(metrics.get('y_pred', [])),
            os.path.join(results_dir, 'enhanced_tornet_confusion_matrix.png')
        )
    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    
    logger.info("Enhanced TorNet evaluation completed!")


if __name__ == '__main__':
    main()
