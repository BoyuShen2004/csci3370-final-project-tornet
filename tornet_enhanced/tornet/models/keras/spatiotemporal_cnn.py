"""
Spatio-temporal CNN models for tornado detection with enhanced temporal modeling.

This module implements 3D CNN and ConvLSTM architectures to better capture
spatio-temporal patterns in radar data.
"""

from typing import Dict, List, Tuple
import numpy as np
import keras
from keras import layers, ops
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


def build_3d_cnn_model(shape: Tuple[int] = (120, 240, 2),
                      c_shape: Tuple[int] = (120, 240, 2),
                      input_variables: List[str] = ALL_VARIABLES,
                      start_filters: int = 32,
                      l2_reg: float = 1e-5,
                      background_flag: float = -3.0,
                      include_range_folded: bool = True,
                      head: str = 'maxpool'):
    """
    Build 3D CNN model for spatio-temporal tornado detection.
    
    Args:
        shape: Input shape for radar data (height, width, time_steps)
        c_shape: Input shape for coordinates
        input_variables: List of input variables
        start_filters: Number of starting filters
        l2_reg: L2 regularization strength
        background_flag: Background value for NaN pixels
        include_range_folded: Whether to include range folded mask
        head: Type of head (maxpool, global_avg, attention)
    
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
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = layers.Concatenate(axis=-1, name='Concatenate2')(
            [normalized_inputs, range_folded])
        
    # Input coordinate information
    coords = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = coords
    
    # Reshape for 3D convolution: (batch, height, width, time, channels)
    x = layers.Reshape((shape[0], shape[1], shape[2], -1))(normalized_inputs)
    
    # 3D CNN blocks
    x = conv3d_block(x, start_filters, l2_reg, name='conv3d_1')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_1')(x)
    
    x = conv3d_block(x, start_filters * 2, l2_reg, name='conv3d_2')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_2')(x)
    
    x = conv3d_block(x, start_filters * 4, l2_reg, name='conv3d_3')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_3')(x)
    
    x = conv3d_block(x, start_filters * 8, l2_reg, name='conv3d_4')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_4')(x)
    
    # Global temporal pooling
    x = layers.GlobalAveragePooling3D(name='global_avg_pool3d')(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), name='dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), name='dense_2')(x)
    x = layers.Dropout(0.5, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='tornado_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='3d_cnn_tornado_detector')
    return model


def build_convlstm_model(shape: Tuple[int] = (120, 240, 2),
                        c_shape: Tuple[int] = (120, 240, 2),
                        input_variables: List[str] = ALL_VARIABLES,
                        start_filters: int = 32,
                        l2_reg: float = 1e-5,
                        background_flag: float = -3.0,
                        include_range_folded: bool = True,
                        head: str = 'maxpool'):
    """
    Build ConvLSTM model for spatio-temporal tornado detection.
    
    Args:
        shape: Input shape for radar data (height, width, time_steps)
        c_shape: Input shape for coordinates
        input_variables: List of input variables
        start_filters: Number of starting filters
        l2_reg: L2 regularization strength
        background_flag: Background value for NaN pixels
        include_range_folded: Whether to include range folded mask
        head: Type of head (maxpool, global_avg, attention)
    
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
        range_folded = keras.Input(shape[:2] + (n_sweeps,), name='range_folded_mask')
        inputs['range_folded_mask'] = range_folded
        normalized_inputs = layers.Concatenate(axis=-1, name='Concatenate2')(
            [normalized_inputs, range_folded])
        
    # Input coordinate information
    coords = keras.Input(c_shape, name='coordinates')
    inputs['coordinates'] = coords
    
    # Reshape for ConvLSTM: (batch, time, height, width, channels)
    x = layers.Reshape((shape[2], shape[0], shape[1], -1))(normalized_inputs)
    
    # ConvLSTM blocks
    x = convlstm_block(x, start_filters, l2_reg, return_sequences=True, name='convlstm_1')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_1')(x)
    
    x = convlstm_block(x, start_filters * 2, l2_reg, return_sequences=True, name='convlstm_2')
    x = layers.MaxPooling3D((1, 2, 2), name='maxpool3d_2')(x)
    
    x = convlstm_block(x, start_filters * 4, l2_reg, return_sequences=False, name='convlstm_3')
    x = layers.MaxPooling2D((2, 2), name='maxpool2d_1')(x)
    
    # Global spatial pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool2d')(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), name='dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg), name='dense_2')(x)
    x = layers.Dropout(0.5, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='tornado_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='convlstm_tornado_detector')
    return model


def conv3d_block(x, filters, l2_reg, name):
    """3D Convolutional block with batch normalization and activation."""
    x = layers.Conv3D(filters, (3, 3, 3), padding='same', 
                      kernel_regularizer=keras.regularizers.l2(l2_reg), name=f'{name}_conv3d')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('relu', name=f'{name}_relu')(x)
    return x


def convlstm_block(x, filters, l2_reg, return_sequences=True, name='convlstm'):
    """ConvLSTM block with batch normalization and activation."""
    x = layers.ConvLSTM2D(filters, (3, 3), padding='same', 
                          kernel_regularizer=keras.regularizers.l2(l2_reg),
                          return_sequences=return_sequences, name=f'{name}_convlstm2d')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('relu', name=f'{name}_relu')(x)
    return x


def normalize(x, var_name):
    """Normalize input variable using min-max scaling."""
    min_val, max_val = CHANNEL_MIN_MAX[var_name]
    return (x - min_val) / (max_val - min_val)
