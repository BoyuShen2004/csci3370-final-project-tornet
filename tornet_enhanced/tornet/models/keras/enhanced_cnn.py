"""
Enhanced CNN architecture with ResNet-style blocks, attention mechanisms,
and advanced techniques for improved tornado detection performance.

This module provides an improved CNN architecture that builds upon the baseline
with the following enhancements:
- ResNet-style residual connections
- Spatial and channel attention mechanisms
- Multi-scale feature extraction
- Advanced regularization techniques
"""

from typing import Dict, List, Tuple
import numpy as np
import keras
from keras import layers, ops
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


def spatial_attention_block(x, name_prefix=""):
    """
    Spatial attention mechanism to focus on important regions.
    
    Args:
        x: Input tensor
        name_prefix: Prefix for layer names
    
    Returns:
        Attention-weighted tensor
    """
    # Global average pooling and max pooling
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_spatial_avg")(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True, name=f"{name_prefix}_spatial_max")(x)
    
    # Concatenate and apply 1x1 convolution
    concat = layers.Concatenate(axis=-1, name=f"{name_prefix}_spatial_concat")([avg_pool, max_pool])
    attention = layers.Conv2D(1, 1, activation='sigmoid', name=f"{name_prefix}_spatial_attention")(concat)
    
    # Apply attention
    return layers.Multiply(name=f"{name_prefix}_spatial_apply")([x, attention])


def channel_attention_block(x, reduction_ratio=16, name_prefix=""):
    """
    Channel attention mechanism to focus on important features.
    
    Args:
        x: Input tensor
        reduction_ratio: Reduction ratio for dense layers
        name_prefix: Prefix for layer names
    
    Returns:
        Attention-weighted tensor
    """
    # Global average pooling and max pooling
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name_prefix}_channel_avg")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name_prefix}_channel_max")(x)
    
    # Shared MLP
    def shared_mlp(inputs, name_suffix):
        dense1 = layers.Dense(
            inputs.shape[-1] // reduction_ratio,
            activation='relu',
            name=f"{name_prefix}_channel_dense1_{name_suffix}"
        )(inputs)
        dense2 = layers.Dense(
            inputs.shape[-1],
            name=f"{name_prefix}_channel_dense2_{name_suffix}"
        )(dense1)
        return dense2
    
    avg_mlp = shared_mlp(avg_pool, "avg")
    max_mlp = shared_mlp(max_pool, "max")
    
    # Add and apply sigmoid
    attention = layers.Add(name=f"{name_prefix}_channel_add")([avg_mlp, max_mlp])
    attention = layers.Activation('sigmoid', name=f"{name_prefix}_channel_sigmoid")(attention)
    
    # Reshape and apply attention
    attention = layers.Reshape((1, 1, -1), name=f"{name_prefix}_channel_reshape")(attention)
    return layers.Multiply(name=f"{name_prefix}_channel_apply")([x, attention])


def residual_block(x, filters, kernel_size=3, stride=1, name_prefix=""):
    """
    ResNet-style residual block with batch normalization and activation.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolution
        stride: Stride for convolution
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor
    """
    # Store input for residual connection
    shortcut = x
    
    # First convolution
    conv1 = layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same',
        name=f"{name_prefix}_conv1"
    )(x)
    bn1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")(conv1)
    relu1 = layers.Activation('relu', name=f"{name_prefix}_relu1")(bn1)
    
    # Second convolution
    conv2 = layers.Conv2D(
        filters, kernel_size, padding='same',
        name=f"{name_prefix}_conv2"
    )(relu1)
    bn2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")(conv2)
    
    # Handle dimension mismatch for residual connection
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding='same',
            name=f"{name_prefix}_shortcut"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)
    
    # Add residual connection
    add = layers.Add(name=f"{name_prefix}_add")([bn2, shortcut])
    return layers.Activation('relu', name=f"{name_prefix}_relu2")(add)


def multi_scale_block(x, filters, name_prefix=""):
    """
    Multi-scale feature extraction block using different kernel sizes.
    
    Args:
        x: Input tensor
        filters: Number of filters
        name_prefix: Prefix for layer names
    
    Returns:
        Concatenated multi-scale features
    """
    # Different kernel sizes for multi-scale processing
    conv1x1 = layers.Conv2D(filters//4, 1, padding='same', name=f"{name_prefix}_conv1x1")(x)
    conv3x3 = layers.Conv2D(filters//4, 3, padding='same', name=f"{name_prefix}_conv3x3")(x)
    conv5x5 = layers.Conv2D(filters//4, 5, padding='same', name=f"{name_prefix}_conv5x5")(x)
    conv7x7 = layers.Conv2D(filters//4, 7, padding='same', name=f"{name_prefix}_conv7x7")(x)
    
    # Concatenate multi-scale features
    return layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([conv1x1, conv3x3, conv5x5, conv7x7])


def normalize(inputs, variable):
    """
    Normalize inputs based on variable-specific min/max values.
    """
    min_val, max_val = CHANNEL_MIN_MAX[variable]
    return layers.Lambda(lambda x: (x - min_val) / (max_val - min_val))(inputs)


def build_enhanced_model(shape: Tuple[int] = (120, 240, 2),
                         c_shape: Tuple[int] = (120, 240, 2),
                         input_variables: List[str] = ALL_VARIABLES,
                         start_filters: int = 64,
                         l2_reg: float = 0.001,
                         background_flag: float = -3.0,
                         include_range_folded: bool = True,
                         head: str = 'maxpool',
                         use_attention: bool = True,
                         use_residual: bool = True,
                         use_multiscale: bool = True):
    """
    Build enhanced CNN model with advanced techniques.
    
    Args:
        shape: Input shape (height, width, sweeps)
        c_shape: Coordinate shape
        input_variables: List of input variables
        start_filters: Number of starting filters
        l2_reg: L2 regularization strength
        background_flag: Background value for NaN replacement
        include_range_folded: Whether to include range folded mask
        head: Type of head (maxpool, global_avg, attention)
        use_attention: Whether to use attention mechanisms
        use_residual: Whether to use residual connections
        use_multiscale: Whether to use multi-scale processing
    
    Returns:
        Compiled Keras model
    """
    # Create input layers for each input_variables
    inputs = {}
    for v in input_variables:
        inputs[v] = keras.Input(shape, name=v)
    n_sweeps = shape[2]
    
    # Normalize inputs and concatenate along channel dim
    normalized_inputs = layers.Concatenate(axis=-1, name='Concatenate1')(
        [normalize(inputs[v], v) for v in input_variables]
    )
    
    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)
    
    # Add channel for range folded gates
    if include_range_folded:
        range_folded = keras.Input(shape, name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = layers.Concatenate(axis=-1, name='Concatenate2')(
            [normalized_inputs, range_folded]
        )
    
    # Input coordinate information
    coords = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = coords
    
    # Keep data and coordinates separate for CoordConv2D
    x = normalized_inputs
    
    # Squeeze time dimension if present (from (batch, time, height, width, channels) to (batch, height, width, channels))
    if len(x.shape) == 5:  # (batch, time, height, width, channels)
        x = layers.Lambda(lambda t: ops.squeeze(t, axis=1), name='squeeze_time')(x)
    if len(coords.shape) == 5:  # (batch, time, height, width, channels)
        coords = layers.Lambda(lambda t: ops.squeeze(t, axis=1), name='squeeze_coords_time')(coords)
    
    # Initial convolution with coordinate-aware convolution
    x, coords = CoordConv2D(start_filters, 7, kernel_regularizer=keras.regularizers.l2(l2_reg), activation='relu', padding='same', name='coord_conv_initial')([x, coords])
    x = layers.BatchNormalization(name='bn_initial')(x)
    
    # Enhanced feature extraction blocks - build on baseline success
    # Use same structure as baseline but with residual connections as enhancement
    block_filters = [start_filters, start_filters * 2, start_filters * 4]  # 3 blocks like baseline
    
    for i, filters in enumerate(block_filters):
        block_name = f"block_{i+1}"
        
        # Store input for residual connection
        residual_input = x
        
        # First conv in block (same as baseline)
        x = layers.Conv2D(filters, 3, strides=2 if i > 0 else 1, padding='same', 
                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                         name=f"{block_name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
        x = layers.Activation('relu', name=f"{block_name}_relu1")(x)
        
        # Second conv in block (same as baseline)
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                         name=f"{block_name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
        
        # Add residual connection if enabled (this is the key enhancement)
        if use_residual:
            # Match dimensions if needed
            if residual_input.shape[-1] != x.shape[-1] or (i > 0 and residual_input.shape[1] != x.shape[1]):
                residual_input = layers.Conv2D(filters, 1, strides=2 if i > 0 else 1, padding='same',
                                             kernel_regularizer=keras.regularizers.l2(l2_reg),
                                             name=f"{block_name}_residual_conv")(residual_input)
            x = layers.Add(name=f"{block_name}_add")([x, residual_input])
        
        x = layers.Activation('relu', name=f"{block_name}_relu2")(x)
        
        # Update coordinates with pooling to match spatial dimensions
        if i > 0:  # Apply pooling to coordinates after first block
            coords = layers.MaxPool2D(pool_size=2, strides=2, padding='same', name=f"{block_name}_coords_pool")(coords)
        
        # Attention mechanisms (only if enabled - keep conservative)
        if use_attention:
            # Channel attention
            x = channel_attention_block(x, name_prefix=f"{block_name}_channel")
            # Spatial attention
            x = spatial_attention_block(x, name_prefix=f"{block_name}_spatial")
        
        # Dropout for regularization (same as baseline)
        x = layers.Dropout(0.1, name=f"{block_name}_dropout")(x)
    
    # Global feature aggregation
    if head == 'maxpool':
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    elif head == 'global_avg':
        x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    elif head == 'attention':
        # Attention-based pooling
        attention_weights = layers.Dense(1, activation='sigmoid', name='attention_weights')(x)
        x = layers.Multiply(name='attention_pool')([x, attention_weights])
        x = layers.GlobalAveragePooling2D(name='attention_global_avg')(x)
    else:
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    
    # Final classification layers (similar to baseline)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout_final2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='enhanced_tornado_cnn')
    
    return model
