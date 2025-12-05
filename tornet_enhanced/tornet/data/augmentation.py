"""
Advanced data augmentation techniques specifically designed for radar data.

This module provides various augmentation strategies that are appropriate
for weather radar data, including geometric transformations, noise addition,
and weather-specific augmentations.
"""

import numpy as np
import keras
from keras import layers
from typing import Dict, List, Tuple, Optional
import random


class RadarDataAugmentation:
    """
    Advanced data augmentation for radar data.
    
    This class provides various augmentation techniques that are appropriate
    for weather radar data, including geometric transformations, noise addition,
    and weather-specific augmentations.
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 translation_range: float = 0.1,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.01,
                 brightness_range: float = 0.1,
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 flip_probability: float = 0.5,
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            translation_range: Maximum translation as fraction of image size
            scale_range: Range for scaling (min, max)
            noise_std: Standard deviation for Gaussian noise
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            flip_probability: Probability of horizontal flip
            mixup_alpha: Alpha parameter for Mixup augmentation
            cutmix_alpha: Alpha parameter for CutMix augmentation
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.flip_probability = flip_probability
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def random_rotation(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random rotation to radar data.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            # Apply rotation to each channel
            x_rotated = np.zeros_like(x)
            for i in range(x.shape[-1]):
                x_rotated[:, :, i] = self._rotate_channel(x[:, :, i], angle)
            return x_rotated, y
        return x, y
    
    def _rotate_channel(self, channel: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a single channel by the given angle."""
        from scipy.ndimage import rotate
        return rotate(channel, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
    
    def random_translation(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random translation to radar data.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < 0.5:
            h_shift = random.uniform(-self.translation_range, self.translation_range) * x.shape[0]
            w_shift = random.uniform(-self.translation_range, self.translation_range) * x.shape[1]
            
            # Apply translation to each channel
            x_translated = np.zeros_like(x)
            for i in range(x.shape[-1]):
                x_translated[:, :, i] = self._translate_channel(x[:, :, i], h_shift, w_shift)
            return x_translated, y
        return x, y
    
    def _translate_channel(self, channel: np.ndarray, h_shift: float, w_shift: float) -> np.ndarray:
        """Translate a single channel by the given shifts."""
        from scipy.ndimage import shift
        return shift(channel, (h_shift, w_shift), order=1, mode='nearest')
    
    def random_scaling(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random scaling to radar data.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < 0.5:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            # Apply scaling to each channel
            x_scaled = np.zeros_like(x)
            for i in range(x.shape[-1]):
                x_scaled[:, :, i] = self._scale_channel(x[:, :, i], scale)
            return x_scaled, y
        return x, y
    
    def _scale_channel(self, channel: np.ndarray, scale: float) -> np.ndarray:
        """Scale a single channel by the given factor."""
        from scipy.ndimage import zoom
        return zoom(channel, scale, order=1, mode='nearest')
    
    def random_noise(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random Gaussian noise to radar data.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, x.shape)
            x_noisy = x + noise
            return x_noisy, y
        return x, y
    
    def random_brightness_contrast(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random brightness and contrast adjustments.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < 0.5:
            # Brightness adjustment
            brightness = random.uniform(-self.brightness_range, self.brightness_range)
            x = x + brightness
            
            # Contrast adjustment
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            x = x * contrast
            
            # Clip values to valid range
            x = np.clip(x, 0, 1)
            return x, y
        return x, y
    
    def random_flip(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random horizontal flip.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        if random.random() < self.flip_probability:
            x_flipped = np.flip(x, axis=1)
            return x_flipped, y
        return x, y
    
    def mixup(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Mixup augmentation.
        
        Args:
            x1, y1: First sample
            x2, y2: Second sample
            
        Returns:
            Mixed sample and labels
        """
        if random.random() < 0.5:
            alpha = random.betavariate(self.mixup_alpha, self.mixup_alpha)
            x_mixed = alpha * x1 + (1 - alpha) * x2
            y_mixed = alpha * y1 + (1 - alpha) * y2
            return x_mixed, y_mixed
        return x1, y1
    
    def cutmix(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CutMix augmentation.
        
        Args:
            x1, y1: First sample
            x2, y2: Second sample
            
        Returns:
            Mixed sample and labels
        """
        if random.random() < 0.5:
            alpha = random.betavariate(self.cutmix_alpha, self.cutmix_alpha)
            lam = alpha
            
            # Generate random bounding box
            h, w = x1.shape[:2]
            cut_rat = np.sqrt(1. - lam)
            cut_h = int(h * cut_rat)
            cut_w = int(w * cut_rat)
            
            # Random center
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            # Bounding box coordinates
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Apply CutMix
            x1_cutmix = x1.copy()
            x1_cutmix[bby1:bby2, bbx1:bbx2] = x2[bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            y_cutmix = lam * y1 + (1 - lam) * y2
            
            return x1_cutmix, y_cutmix
        return x1, y1
    
    def apply_augmentation(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations in sequence.
        
        Args:
            x: Input radar data
            y: Labels
            
        Returns:
            Augmented data and labels
        """
        # Apply augmentations in random order
        augmentations = [
            self.random_rotation,
            self.random_translation,
            self.random_scaling,
            self.random_noise,
            self.random_brightness_contrast,
            self.random_flip
        ]
        
        random.shuffle(augmentations)
        
        for aug_func in augmentations:
            x, y = aug_func(x, y)
        
        return x, y


def create_augmented_dataloader(dataloader, augmentation: RadarDataAugmentation):
    """
    Create an augmented version of the dataloader.
    
    Args:
        dataloader: Original dataloader
        augmentation: Augmentation instance
        
    Returns:
        Augmented dataloader
    """
    def augmented_generator():
        for batch_x, batch_y in dataloader:
            augmented_x = []
            augmented_y = []
            
            for i in range(len(batch_x)):
                x_aug, y_aug = augmentation.apply_augmentation(batch_x[i], batch_y[i])
                augmented_x.append(x_aug)
                augmented_y.append(y_aug)
            
            yield np.array(augmented_x), np.array(augmented_y)
    
    return augmented_generator


class ClassBalancedSampler:
    """
    Class-balanced sampling to address class imbalance.
    """
    
    def __init__(self, labels: np.ndarray, alpha: float = 0.5):
        """
        Initialize balanced sampler.
        
        Args:
            labels: Array of labels
            alpha: Sampling ratio for minority class
        """
        self.labels = labels
        self.alpha = alpha
        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]
        
    def sample_batch(self, batch_size: int) -> np.ndarray:
        """
        Sample a balanced batch.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Indices for balanced batch
        """
        n_positive = int(batch_size * self.alpha)
        n_negative = batch_size - n_positive
        
        pos_indices = np.random.choice(self.positive_indices, n_positive, replace=True)
        neg_indices = np.random.choice(self.negative_indices, n_negative, replace=True)
        
        return np.concatenate([pos_indices, neg_indices])


class AdvancedLearningRateScheduler:
    """
    Advanced learning rate scheduling strategies for imbalanced data.
    """
    
    @staticmethod
    def cosine_annealing_with_warmup(epoch, total_epochs, warmup_epochs=5, 
                                   base_lr=1e-3, min_lr=1e-6):
        """
        Cosine annealing with warmup for better convergence.
        """
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    @staticmethod
    def one_cycle_policy(epoch, total_epochs, max_lr=1e-2, div_factor=25, 
                        pct_start=0.3, final_div_factor=1e4):
        """
        One cycle learning rate policy for faster convergence.
        """
        if epoch < total_epochs * pct_start:
            # Increasing phase
            progress = epoch / (total_epochs * pct_start)
            return max_lr / div_factor + (max_lr - max_lr / div_factor) * progress
        else:
            # Decreasing phase
            progress = (epoch - total_epochs * pct_start) / (total_epochs * (1 - pct_start))
            return max_lr / final_div_factor + (max_lr - max_lr / final_div_factor) * (1 - progress)
    
    @staticmethod
    def adaptive_lr_for_imbalanced(epoch, total_epochs, base_lr=1e-3, 
                                 imbalance_factor=10.0, patience=5):
        """
        Adaptive learning rate that adjusts based on class imbalance.
        """
        # Start with higher learning rate for imbalanced data
        if epoch < total_epochs * 0.3:
            return base_lr * imbalance_factor
        elif epoch < total_epochs * 0.7:
            return base_lr * (imbalance_factor * 0.5)
        else:
            return base_lr


def create_advanced_callbacks(config):
    """
    Create advanced callbacks for training with imbalanced data.
    
    Args:
        config: Training configuration
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping with AUC monitoring
    if config.get('early_stopping_patience', 0) > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_AUC',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ))
    
    # Learning rate reduction on plateau
    if config.get('reduce_lr_patience', 0) > 0:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_AUC',
            factor=config.get('reduce_lr_factor', 0.5),
            patience=config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ))
    
    # Model checkpointing
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=f"{config['exp_dir']}/checkpoints/tornadoDetector_{{epoch:03d}}.keras",
        monitor='val_AUC',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='max'
    ))
    
    # Learning rate scheduler
    if config.get('use_cosine_annealing', False):
        def lr_schedule(epoch):
            return AdvancedLearningRateScheduler.cosine_annealing_with_warmup(
                epoch, config['epochs'], 
                warmup_epochs=config.get('warmup_epochs', 5),
                base_lr=config['learning_rate'],
                min_lr=config['learning_rate'] * 0.01
            )
        
        callbacks.append(keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
    
    # Custom callback for imbalanced data monitoring
    class ImbalancedDataMonitor(keras.callbacks.Callback):
        def __init__(self, validation_data, patience=3):
            super().__init__()
            self.validation_data = validation_data
            self.patience = patience
            self.wait = 0
            self.best_auc = 0
            
        def on_epoch_end(self, epoch, logs=None):
            # Monitor precision-recall balance
            val_precision = logs.get('val_Precision', 0)
            val_recall = logs.get('val_Recall', 0)
            val_auc = logs.get('val_AUC', 0)
            
            # Check for precision-recall imbalance
            if val_precision > 0 and val_recall > 0:
                pr_ratio = val_precision / val_recall
                if pr_ratio > 10 or pr_ratio < 0.1:  # Severe imbalance
                    print(f"Warning: Severe precision-recall imbalance (ratio: {pr_ratio:.2f})")
            
            # Monitor AUC improvement
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.wait = 0
            else:
                self.wait += 1
                
            if self.wait >= self.patience:
                print(f"Warning: AUC not improving for {self.patience} epochs")
    
    # Add custom monitoring callback
    callbacks.append(ImbalancedDataMonitor(None, patience=3))
    
    return callbacks


def create_optimized_optimizer(config):
    """
    Create optimized optimizer for imbalanced data.
    
    Args:
        config: Training configuration
    
    Returns:
        Optimized optimizer
    """
    # Use AdamW with weight decay for better generalization
    optimizer = keras.optimizers.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    return optimizer


def create_balanced_data_generator(data_loader, class_weights=None, oversample_ratio=2.0):
    """
    Create a balanced data generator for imbalanced data.
    
    Args:
        data_loader: Original data loader
        class_weights: Class weights dictionary
        oversample_ratio: Ratio for oversampling minority class
    
    Returns:
        Balanced data generator
    """
    def balanced_generator():
        for batch_x, batch_y in data_loader:
            # Calculate class distribution
            positive_mask = batch_y == 1
            negative_mask = batch_y == 0
            
            n_positive = np.sum(positive_mask)
            n_negative = np.sum(negative_mask)
            
            if n_positive > 0 and n_negative > 0:
                # Oversample positive class
                if n_positive < n_negative * oversample_ratio:
                    # Sample more positive examples
                    pos_indices = np.where(positive_mask)[0]
                    neg_indices = np.where(negative_mask)[0]
                    
                    # Oversample positive class
                    n_oversample = int(n_negative * oversample_ratio) - n_positive
                    oversample_indices = np.random.choice(pos_indices, n_oversample, replace=True)
                    
                    # Combine original and oversampled
                    balanced_indices = np.concatenate([pos_indices, oversample_indices, neg_indices])
                    
                    # Shuffle
                    np.random.shuffle(balanced_indices)
                    
                    yield batch_x[balanced_indices], batch_y[balanced_indices]
                else:
                    yield batch_x, batch_y
            else:
                yield batch_x, batch_y
    
    return balanced_generator
