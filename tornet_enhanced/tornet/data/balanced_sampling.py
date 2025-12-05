"""
Balanced sampling and data augmentation for imbalanced tornado detection.

This module implements advanced sampling strategies and data augmentation
techniques specifically designed for severely imbalanced radar data.
"""

import numpy as np
import keras
from keras.utils import to_categorical
import random

# Try to import sklearn, fall back to numpy if not available
try:
    from sklearn.utils import resample
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using numpy fallbacks")


def balanced_sampling_strategy(X, y, strategy='smote', ratio=0.5):
    """
    Implement balanced sampling strategies for imbalanced data.
    
    Args:
        X: Input features
        y: Binary labels (0 or 1)
        strategy: Sampling strategy ('smote', 'adasyn', 'random_oversample', 'smote_tomek')
        ratio: Target ratio of positive to negative samples
    
    Returns:
        Resampled X, y
    """
    if strategy == 'random_oversample':
        return random_oversample(X, y, ratio)
    elif strategy == 'smote':
        return smote_oversample(X, y, ratio)
    elif strategy == 'adasyn':
        return adasyn_oversample(X, y, ratio)
    elif strategy == 'smote_tomek':
        return smote_tomek_oversample(X, y, ratio)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def random_oversample(X, y, ratio=0.5):
    """Random oversampling of minority class."""
    # Separate majority and minority classes
    majority_class = X[y == 0]
    minority_class = X[y == 1]
    majority_labels = y[y == 0]
    minority_labels = y[y == 1]
    
    # Calculate target number of minority samples
    n_majority = len(majority_class)
    n_minority_target = int(n_majority * ratio)
    n_minority_current = len(minority_class)
    
    if n_minority_current < n_minority_target:
        # Oversample minority class
        if SKLEARN_AVAILABLE:
            minority_oversampled, minority_labels_oversampled = resample(
                minority_class, minority_labels,
                replace=True,
                n_samples=n_minority_target,
                random_state=42
            )
        else:
            # Fallback to numpy
            indices = np.random.choice(len(minority_class), size=n_minority_target, replace=True)
            minority_oversampled = minority_class[indices]
            minority_labels_oversampled = minority_labels[indices]
        
        # Combine majority and oversampled minority
        X_balanced = np.concatenate([majority_class, minority_oversampled])
        y_balanced = np.concatenate([majority_labels, minority_labels_oversampled])
    else:
        # Undersample minority class if needed
        if SKLEARN_AVAILABLE:
            minority_undersampled, minority_labels_undersampled = resample(
                minority_class, minority_labels,
                replace=False,
                n_samples=n_minority_target,
                random_state=42
            )
        else:
            # Fallback to numpy
            indices = np.random.choice(len(minority_class), size=n_minority_target, replace=False)
            minority_undersampled = minority_class[indices]
            minority_labels_undersampled = minority_labels[indices]
        
        X_balanced = np.concatenate([majority_class, minority_undersampled])
        y_balanced = np.concatenate([majority_labels, minority_labels_undersampled])
    
    # Shuffle the data
    indices = np.random.permutation(len(X_balanced))
    return X_balanced[indices], y_balanced[indices]


def smote_oversample(X, y, ratio=0.5):
    """
    SMOTE (Synthetic Minority Oversampling Technique) implementation.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        print("SMOTE not available, falling back to random oversampling")
        return random_oversample(X, y, ratio)


def adasyn_oversample(X, y, ratio=0.5):
    """
    ADASYN (Adaptive Synthetic Sampling) implementation.
    """
    try:
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        print("ADASYN not available, falling back to random oversampling")
        return random_oversample(X, y, ratio)


def smote_tomek_oversample(X, y, ratio=0.5):
    """
    SMOTE + Tomek Links cleaning implementation.
    """
    try:
        from imblearn.combine import SMOTETomek
        smote_tomek = SMOTETomek(sampling_strategy=ratio, random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        print("SMOTETomek not available, falling back to random oversampling")
        return random_oversample(X, y, ratio)


def positive_class_augmentation(X, y, augmentation_factor=2.0):
    """
    Augment positive class samples using various techniques.
    
    Args:
        X: Input features
        y: Binary labels
        augmentation_factor: How much to augment positive samples
    
    Returns:
        Augmented X, y
    """
    # Separate positive and negative samples
    positive_mask = y == 1
    negative_mask = y == 0
    
    X_positive = X[positive_mask]
    y_positive = y[positive_mask]
    X_negative = X[negative_mask]
    y_negative = y[negative_mask]
    
    # Calculate number of augmented samples needed
    n_positive_current = len(X_positive)
    n_positive_target = int(n_positive_current * augmentation_factor)
    n_positive_augment = n_positive_target - n_positive_current
    
    if n_positive_augment > 0:
        # Generate augmented positive samples
        X_positive_augmented = []
        y_positive_augmented = []
        
        for _ in range(n_positive_augment):
            # Randomly select a positive sample
            idx = random.randint(0, n_positive_current - 1)
            sample = X_positive[idx].copy()
            
            # Apply augmentation techniques
            augmented_sample = apply_radar_augmentation(sample)
            X_positive_augmented.append(augmented_sample)
            y_positive_augmented.append(1)
        
        # Combine original and augmented data
        X_positive_augmented = np.array(X_positive_augmented)
        y_positive_augmented = np.array(y_positive_augmented)
        
        X_positive_combined = np.concatenate([X_positive, X_positive_augmented])
        y_positive_combined = np.concatenate([y_positive, y_positive_augmented])
    else:
        X_positive_combined = X_positive
        y_positive_combined = y_positive
    
    # Combine with negative samples
    X_balanced = np.concatenate([X_negative, X_positive_combined])
    y_balanced = np.concatenate([y_negative, y_positive_combined])
    
    # Shuffle the data
    indices = np.random.permutation(len(X_balanced))
    return X_balanced[indices], y_balanced[indices]


def apply_radar_augmentation(sample):
    """
    Apply radar-specific augmentation techniques.
    
    Args:
        sample: Radar data sample
    
    Returns:
        Augmented sample
    """
    augmented = sample.copy()
    
    # Random noise injection
    if random.random() < 0.3:
        noise_std = 0.01
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented = augmented + noise
    
    # Random intensity scaling
    if random.random() < 0.4:
        scale_factor = random.uniform(0.8, 1.2)
        augmented = augmented * scale_factor
    
    # Random spatial rotation (90, 180, 270 degrees)
    if random.random() < 0.3:
        rotation = random.choice([1, 2, 3])  # 90, 180, 270 degrees
        augmented = np.rot90(augmented, k=rotation, axes=(0, 1))
    
    # Random horizontal flip
    if random.random() < 0.3:
        augmented = np.fliplr(augmented)
    
    # Random vertical flip
    if random.random() < 0.3:
        augmented = np.flipud(augmented)
    
    return augmented


def create_balanced_data_generator(X, y, batch_size=32, augmentation=True, sampling_strategy='smote'):
    """
    Create a balanced data generator for training.
    
    Args:
        X: Input features
        y: Binary labels
        batch_size: Batch size
        augmentation: Whether to apply augmentation
        sampling_strategy: Sampling strategy to use
    
    Returns:
        Balanced data generator
    """
    def balanced_generator():
        while True:
            # Apply balanced sampling
            X_balanced, y_balanced = balanced_sampling_strategy(X, y, strategy=sampling_strategy)
            
            # Apply positive class augmentation if requested
            if augmentation:
                X_balanced, y_balanced = positive_class_augmentation(X_balanced, y_balanced)
            
            # Create batches
            n_samples = len(X_balanced)
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_balanced[batch_indices]
                y_batch = y_balanced[batch_indices]
                
                yield X_batch, y_batch
    
    return balanced_generator


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced data.
    
    Args:
        y: Binary labels
    
    Returns:
        Dictionary of class weights
    """
    if SKLEARN_AVAILABLE:
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return {i: weight for i, weight in enumerate(class_weights)}
    else:
        # Fallback calculation
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        n_classes = len(classes)
        
        class_weights = {}
        for i, count in enumerate(counts):
            class_weights[i] = total_samples / (n_classes * count)
        
        return class_weights