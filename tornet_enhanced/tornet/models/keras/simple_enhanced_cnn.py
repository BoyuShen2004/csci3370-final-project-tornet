"""
Simple Enhanced CNN architecture that builds on the baseline VGG model.

This is essentially the baseline VGG model with just residual connections added.
This should be much more stable and actually learn properly, unlike the complex
enhanced model that was having learning issues.

The key enhancement is just adding residual connections to the baseline VGG blocks.
"""

from typing import Dict, List, Tuple
import numpy as np
import keras
from keras import layers, ops
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


def build_simple_enhanced_model(shape, c_shape, start_filters=48, l2_reg=1e-5, input_variables=None, head='maxpool', use_attention=False, use_residual=True, use_multiscale=False, background_flag=-3.0):
    """
    Build simple enhanced CNN model that's essentially baseline VGG + residual connections.
    
    This should be much more stable and actually learn properly.
    
    Args:
        shape: Input shape for radar data
        c_shape: Input shape for coordinates
        start_filters: Number of starting filters
        l2_reg: L2 regularization strength
        input_variables: List of input variables
        head: Type of head (maxpool, global_avg, attention)
        use_attention: Whether to use attention mechanisms (ignored in simple model)
        use_residual: Whether to use residual connections
        use_multiscale: Whether to use multi-scale processing (ignored in simple model)
    
    Returns:
        Compiled Keras model
    """
    if input_variables is None:
        input_variables = ALL_VARIABLES
    
    # Input preprocessing
    inputs = {}
    normalized_inputs = None
    
    # Process each input variable
    for var in input_variables:
        input_layer = keras.Input(shape, name=var)
        inputs[var] = input_layer
        
        # Normalize to [0, 1]
        min_val, max_val = CHANNEL_MIN_MAX[var]
        normalized = (input_layer - min_val) / (max_val - min_val)
        
        if normalized_inputs is None:
            normalized_inputs = normalized
        else:
            normalized_inputs = layers.Concatenate(axis=-1, name=f'Concatenate_{var}')([normalized_inputs, normalized])
    
    # Replace nan pixel with background flag (same as baseline)
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)
    
    # Add channel for range folded gates (always include if in input_variables)
    if 'range_folded_mask' in input_variables:
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
    
    # Initial convolution with coordinate-aware convolution (same as baseline)
    x, coords = CoordConv2D(start_filters, 7, kernel_regularizer=keras.regularizers.l2(l2_reg), activation='relu', padding='same', name='coord_conv_initial')([x, coords])
    x = layers.BatchNormalization(name='bn_initial')(x)
    
    # Simple VGG-style blocks with optional residual connections
    # This is much simpler than the previous complex architecture
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
        
        # Add residual connection if enabled (this is the ONLY enhancement)
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
        
        # Simple dropout for regularization (same as baseline)
        x = layers.Dropout(0.1, name=f"{block_name}_dropout")(x)
    
    # Global feature aggregation (same as baseline)
    if head == 'maxpool':
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    elif head == 'global_avg':
        x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    else:
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    
    # Final classification layers (same as baseline)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout_final')(x)
    x = layers.Dense(128, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout_final2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='simple_enhanced_tornado_cnn')
    
    return model
