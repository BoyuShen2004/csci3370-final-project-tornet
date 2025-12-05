"""
Visualization utilities for segmentation results.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create overlay visualization of mask on image.
    Args:
        image: Input image [H, W, 3] or [H, W]
        mask: Binary mask [H, W]
        alpha: Transparency of overlay
    Returns:
        Overlay image [H, W, 3]
    """
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)
    
    # Create colored mask (red overlay)
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = 255  # Red channel
    
    # Apply mask
    mask_bool = mask > 0.5
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * mask_colored[mask_bool]
    
    return overlay.astype(np.uint8)


def save_prediction_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray = None,
    output_path: str = None,
    title: str = "Segmentation Result",
):
    """
    Save visualization with image, prediction, and optionally ground truth.
    """
    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    
    # Original image
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    overlay = create_overlay(image, prediction)
    axes[1].imshow(overlay)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth (if provided)
    if ground_truth is not None:
        overlay_gt = create_overlay(image, ground_truth)
        axes[2].imshow(overlay_gt)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
):
    """Save a single overlay image."""
    overlay = create_overlay(image, mask, alpha)
    Image.fromarray(overlay).save(output_path)


def create_comparison_grid(
    images: list,
    predictions: list,
    ground_truths: list = None,
    output_path: str = None,
    n_cols: int = 3,
):
    """Create a grid comparison of multiple predictions."""
    n_samples = len(images)
    n_rows = (n_samples + n_cols - 1) // n_cols
    n_cols_plot = 3 if ground_truths is not None else 2
    
    fig, axes = plt.subplots(n_rows, n_cols * n_cols_plot, figsize=(5 * n_cols * n_cols_plot, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols
        
        # Original image
        ax = axes[row, col * n_cols_plot]
        if len(images[idx].shape) == 2:
            ax.imshow(images[idx], cmap='gray')
        else:
            ax.imshow(images[idx])
        ax.set_title(f'Sample {idx+1} - Original')
        ax.axis('off')
        
        # Prediction
        ax = axes[row, col * n_cols_plot + 1]
        overlay = create_overlay(images[idx], predictions[idx])
        ax.imshow(overlay)
        ax.set_title(f'Sample {idx+1} - Prediction')
        ax.axis('off')
        
        # Ground truth (if provided)
        if ground_truths is not None:
            ax = axes[row, col * n_cols_plot + 2]
            overlay_gt = create_overlay(images[idx], ground_truths[idx])
            ax.imshow(overlay_gt)
            ax.set_title(f'Sample {idx+1} - Ground Truth')
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        for c in range(n_cols_plot):
            axes[row, col * n_cols_plot + c].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

