"""
Utility functions for SSL pretraining.
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
    checkpoint_dir: str,
    is_best: bool = False,
    scheduler=None,
    global_step: int = 0,
):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'global_step': global_step,
    }
    
    # Save scheduler state if provided
    if scheduler is not None:
        if hasattr(scheduler, 'last_epoch'):
            checkpoint['scheduler_state'] = {
                'last_epoch': scheduler.last_epoch,
            }
        else:
            # For custom scheduler, save what we can
            checkpoint['scheduler_state'] = {
                'last_epoch': getattr(scheduler, 'last_epoch', global_step - 1),
            }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
    
    # Save student encoder only for fine-tuning
    encoder_path = os.path.join(checkpoint_dir, 'encoder_pretrained.pth')
    encoder_state_dict = {}
    for k, v in model.state_dict().items():
        if 'student_backbone' in k:
            # Remove 'student_backbone.' prefix
            new_key = k.replace('student_backbone.', '')
            encoder_state_dict[new_key] = v
    
    torch.save(encoder_state_dict, encoder_path)
    print(f"Saved encoder checkpoint to {encoder_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = 'cuda',
):
    """Load training checkpoint."""
    try:
        # PyTorch 2.6+ changed default weights_only=True, but our checkpoints contain numpy scalars
        # Since these are our own checkpoint files, we trust them and set weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state' in checkpoint:
        try:
            scheduler_state = checkpoint['scheduler_state']
            if hasattr(scheduler, 'last_epoch'):
                scheduler.last_epoch = scheduler_state.get('last_epoch', checkpoint.get('global_step', 0) - 1)
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}. Continuing without scheduler state.")
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    global_step = checkpoint.get('global_step', 0)
    
    return epoch, loss, global_step

