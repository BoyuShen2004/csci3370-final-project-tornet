"""
Utility functions for fine-tuning.
"""
import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any
import logging


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metric: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metric': metric,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_classification_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Saved model to {final_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda',
):
    """Load training checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except (RuntimeError, EOFError, IOError) as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}. The checkpoint file may be corrupted. Please delete it and start training from scratch.")
    
    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint file {checkpoint_path} does not contain a dictionary. It may be corrupted.")
    
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"Checkpoint file {checkpoint_path} does not contain 'model_state_dict'. It may be corrupted.")
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Failed to load model state dict from {checkpoint_path}: {e}. The checkpoint may be incompatible with the current model.")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}. Continuing without optimizer state.")
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    metric = checkpoint.get('metric', 0.0)
    
    return epoch, loss, metric

