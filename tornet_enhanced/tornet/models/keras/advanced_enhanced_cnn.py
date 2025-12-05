"""
Advanced Enhanced CNN with sophisticated attention mechanisms and feature extraction.

This model incorporates:
1. Multi-scale feature extraction
2. Spatial and channel attention mechanisms
3. Residual connections with squeeze-and-excitation
4. Advanced pooling strategies
5. Better feature fusion
"""

import keras
from keras import layers, ops
from tornet.models.keras.layers import CoordConv2D, FillNaNs
from tornet.data.constants import CHANNEL_MIN_MAX, ALL_VARIABLES


def channel_attention_block(x, reduction_ratio=16, name_prefix="channel_att"):
    """
    Channel Attention Block (Squeeze-and-Excitation).
    
    Args:
        x: Input tensor
        reduction_ratio: Reduction ratio for the bottleneck
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor with channel attention applied
    """
    channels = x.shape[-1]
    
    # Global average pooling
    gap = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    
    # Global max pooling
    gmp = layers.GlobalMaxPooling2D(name=f"{name_prefix}_gmp")(x)
    
    # Shared MLP for both paths
    shared_dense1 = layers.Dense(channels // reduction_ratio, activation='relu', name=f"{name_prefix}_shared_dense1")
    shared_dense2 = layers.Dense(channels, activation='sigmoid', name=f"{name_prefix}_shared_dense2")
    
    # Process both paths
    gap_out = shared_dense2(shared_dense1(gap))
    gmp_out = shared_dense2(shared_dense1(gmp))
    
    # Combine and reshape
    attention_weights = layers.Add(name=f"{name_prefix}_add")([gap_out, gmp_out])
    attention_weights = layers.Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(attention_weights)
    
    # Apply attention
    return layers.Multiply(name=f"{name_prefix}_multiply")([x, attention_weights])


def spatial_attention_block(x, name_prefix="spatial_att"):
    """
    Spatial Attention Block.
    
    Args:
        x: Input tensor
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor with spatial attention applied
    """
    # Channel-wise average and max pooling
    avg_pool = layers.Lambda(lambda t: ops.mean(t, axis=-1, keepdims=True), name=f"{name_prefix}_avg_pool")(x)
    max_pool = layers.Lambda(lambda t: ops.max(t, axis=-1, keepdims=True), name=f"{name_prefix}_max_pool")(x)
    
    # Concatenate
    concat = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])
    
    # Convolution to generate spatial attention map
    attention_map = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=f"{name_prefix}_conv")(concat)
    
    # Apply attention
    return layers.Multiply(name=f"{name_prefix}_multiply")([x, attention_map])


def multi_scale_feature_extraction(x, filters, name_prefix="multiscale"):
    """
    Multi-scale feature extraction using different kernel sizes.
    
    Args:
        x: Input tensor
        filters: Number of filters
        name_prefix: Prefix for layer names
    
    Returns:
        Concatenated multi-scale features
    """
    # Different kernel sizes for multi-scale processing
    conv1x1 = layers.Conv2D(filters, 1, padding='same', activation='relu', name=f"{name_prefix}_1x1")(x)
    conv3x3 = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name_prefix}_3x3")(x)
    conv5x5 = layers.Conv2D(filters, 5, padding='same', activation='relu', name=f"{name_prefix}_5x5")(x)
    
    # Concatenate multi-scale features
    return layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([conv1x1, conv3x3, conv5x5])


def residual_attention_block(x, filters, strides=1, use_attention=True, name_prefix="res_att"):
    """
    Residual block with attention mechanisms.
    
    Args:
        x: Input tensor
        filters: Number of filters
        strides: Stride for convolution
        use_attention: Whether to use attention mechanisms
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor
    """
    residual = x
    
    # First convolution
    x = layers.Conv2D(filters, 3, strides=strides, padding='same', name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{name_prefix}_relu1")(x)
    
    # Second convolution
    x = layers.Conv2D(filters, 3, padding='same', name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # Channel attention
    if use_attention:
        x = channel_attention_block(x, name_prefix=f"{name_prefix}_channel")
    
    # Residual connection
    if strides > 1 or residual.shape[-1] != filters:
        residual = layers.Conv2D(filters, 1, strides=strides, padding='same', name=f"{name_prefix}_residual_conv")(residual)
        residual = layers.BatchNormalization(name=f"{name_prefix}_residual_bn")(residual)
    
    x = layers.Add(name=f"{name_prefix}_add")([x, residual])
    x = layers.Activation('relu', name=f"{name_prefix}_relu2")(x)
    
    # Spatial attention
    if use_attention:
        x = spatial_attention_block(x, name_prefix=f"{name_prefix}_spatial")
    
    return x


def feature_fusion_block(features_list, name_prefix="fusion"):
    """
    Feature fusion block that combines features from different scales.
    
    Args:
        features_list: List of feature tensors
        name_prefix: Prefix for layer names
    
    Returns:
        Fused features
    """
    # Ensure all features have the same spatial dimensions
    target_shape = features_list[0].shape[1:3]  # Height, width
    
    aligned_features = []
    for i, feat in enumerate(features_list):
        if feat.shape[1:3] != target_shape:
            # Resize to target shape
            feat = layers.Lambda(
                lambda t: ops.image.resize(t, target_shape, interpolation='bilinear'),
                name=f"{name_prefix}_resize_{i}"
            )(feat)
        aligned_features.append(feat)
    
    # Concatenate features
    fused = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")(aligned_features)
    
    # Reduce channels
    fused = layers.Conv2D(256, 1, activation='relu', name=f"{name_prefix}_reduce")(fused)
    fused = layers.BatchNormalization(name=f"{name_prefix}_bn")(fused)
    
    return fused


def build_advanced_enhanced_model(shape, c_shape, start_filters=48, l2_reg=1e-5, 
                                input_variables=None, head='attention', 
                                use_attention=True, use_residual=True, 
                                use_multiscale=True, background_flag=-3.0):
    """
    Build advanced enhanced CNN model with sophisticated attention mechanisms.
    
    Args:
        shape: Input shape for radar data
        c_shape: Input shape for coordinates
        start_filters: Number of starting filters
        l2_reg: L2 regularization strength
        input_variables: List of input variables
        head: Type of head (attention, maxpool, global_avg)
        use_attention: Whether to use attention mechanisms
        use_residual: Whether to use residual connections
        use_multiscale: Whether to use multi-scale processing
        background_flag: Value to fill NaNs with
    
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
        normalized = layers.Lambda(lambda x: (x - min_val) / (max_val - min_val), name=f'normalize_{var}')(input_layer)
        
        if normalized_inputs is None:
            normalized_inputs = normalized
        else:
            normalized_inputs = layers.Concatenate(axis=-1, name=f'Concatenate_{var}')([normalized_inputs, normalized])
    
    # Replace nan pixel with background flag
    normalized_inputs = FillNaNs(background_flag)(normalized_inputs)
    
    # Add channel for range folded gates
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
    
    # Squeeze time dimension if present
    if len(x.shape) == 5:
        x = layers.Lambda(lambda t: ops.squeeze(t, axis=1), name='squeeze_time')(x)
    if len(coords.shape) == 5:
        coords = layers.Lambda(lambda t: ops.squeeze(t, axis=1), name='squeeze_coords_time')(coords)
    
    # Initial convolution with coordinate-aware convolution
    x, coords = CoordConv2D(start_filters, 7, kernel_regularizer=keras.regularizers.l2(l2_reg), 
                           activation='relu', padding='same', name='coord_conv_initial')([x, coords])
    x = layers.BatchNormalization(name='bn_initial')(x)
    
    # Store features for multi-scale fusion
    feature_maps = []
    
    # Enhanced feature extraction blocks
    block_filters = [start_filters, start_filters * 2, start_filters * 4, start_filters * 8]
    
    for i, filters in enumerate(block_filters):
        block_name = f"block_{i+1}"
        
        # Multi-scale feature extraction
        if use_multiscale and i > 0:  # Skip first block to avoid too much complexity
            x = multi_scale_feature_extraction(x, filters // 3, name_prefix=f"{block_name}_multiscale")
        
        # Residual attention block
        if use_residual:
            x = residual_attention_block(x, filters, strides=2 if i > 0 else 1, 
                                       use_attention=use_attention, name_prefix=f"{block_name}_res_att")
        else:
            # Standard convolution blocks
            x = layers.Conv2D(filters, 3, strides=2 if i > 0 else 1, padding='same', 
                             kernel_regularizer=keras.regularizers.l2(l2_reg),
                             name=f"{block_name}_conv1")(x)
            x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
            x = layers.Activation('relu', name=f"{block_name}_relu1")(x)
            
            x = layers.Conv2D(filters, 3, padding='same', 
                             kernel_regularizer=keras.regularizers.l2(l2_reg),
                             name=f"{block_name}_conv2")(x)
            x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
            x = layers.Activation('relu', name=f"{block_name}_relu2")(x)
        
        # Store feature maps for fusion
        feature_maps.append(x)
        
        # Update coordinates with pooling
        if i > 0:
            coords = layers.MaxPool2D(pool_size=2, strides=2, padding='same', name=f"{block_name}_coords_pool")(coords)
        
        # Dropout for regularization
        x = layers.Dropout(0.1, name=f"{block_name}_dropout")(x)
    
    # Feature fusion from different scales
    if use_multiscale and len(feature_maps) > 1:
        x = feature_fusion_block(feature_maps[-3:], name_prefix="feature_fusion")  # Use last 3 scales
    
    # Global feature aggregation
    if head == 'attention':
        # Attention-based pooling
        # Channel attention
        x = channel_attention_block(x, name_prefix="final_channel_att")
        
        # Spatial attention
        x = spatial_attention_block(x, name_prefix="final_spatial_att")
        
        # Global average pooling with attention
        attention_weights = layers.Dense(1, activation='sigmoid', name='attention_weights')(x)
        x = layers.Multiply(name='attention_pool')([x, attention_weights])
        x = layers.GlobalAveragePooling2D(name='attention_global_avg')(x)
        
    elif head == 'maxpool':
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    elif head == 'global_avg':
        x = layers.GlobalAveragePooling2D(name='global_avgpool')(x)
    else:
        x = layers.GlobalMaxPooling2D(name='global_maxpool')(x)
    
    # Final classification layers with residual connections
    dense_input = x
    
    # First dense layer
    x = layers.Dense(512, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='dense1_bn')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    
    # Second dense layer with residual connection
    residual_dense = layers.Dense(256, name='residual_dense')(dense_input)
    x = layers.Dense(256, activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='dense2_bn')(x)
    x = layers.Add(name='dense_residual')([x, residual_dense])
    x = layers.Activation('relu', name='dense2_relu')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    
    # Third dense layer
    x = layers.Dense(128, activation='relu', name='dense3')(x)
    x = layers.BatchNormalization(name='dense3_bn')(x)
    x = layers.Dropout(0.2, name='dropout3')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='advanced_enhanced_tornado_cnn')
    
    return model
