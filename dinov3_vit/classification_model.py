"""
Classification Model with DINOv3 Encoder for Binary Classification

This model uses DINOv3 as a feature extractor and adds a classification head
for binary classification (tornado vs no tornado).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import sys
from pathlib import Path
import numpy as np

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent / "dinov3"))
from dinov3.hub.backbones import dinov3_vits16


class ClassificationHead(nn.Module):
    """Classification head for binary classification."""
    
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        pos_ratio: float = 0.063,  # Default positive class ratio for bias initialization
    ):
        super().__init__()
        # Deeper classification head for better expressiveness
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)  # Increased capacity
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize final layer bias to match class distribution
        # CRITICAL: Initialize bias so model starts predicting ~pos_ratio% positives
        # This prevents model from starting at 50% and collapsing to 0% negatives
        # Formula: sigmoid(bias) = pos_ratio, so bias = log(pos_ratio / (1 - pos_ratio))
        with torch.no_grad():
            if pos_ratio > 0 and pos_ratio < 1:
                initial_bias = np.log(pos_ratio / (1 - pos_ratio))
                self.fc4.bias.fill_(initial_bias)
            else:
                self.fc4.bias.fill_(0.0)  # Fallback to neutral if pos_ratio is invalid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, embed_dim]
        Returns:
            Logits [B, 1]
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x


class ClassificationModel(nn.Module):
    """
    Classification model with DINOv3 encoder for binary classification.
    
    This model:
    1. Uses DINOv3 ViT-S/16 as feature extractor
    2. Extracts CLS token or global average pooling of patch tokens
    3. Passes through classification head for binary prediction
    """
    
    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        img_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        use_cls_token: bool = True,
        pos_ratio: float = 0.063,  # Positive class ratio for bias initialization
    ):
        """
        Args:
            encoder_checkpoint: Path to pretrained encoder checkpoint
            img_size: Input image size
            patch_size: Patch size for ViT
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout rate
            use_cls_token: If True, use CLS token; else use global average pooling
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Load DINOv3 encoder
        print(f"Loading DINOv3 encoder...")
        self.encoder = dinov3_vits16(pretrained=False, weights=None)
        
        # Load encoder weights if provided
        if encoder_checkpoint and Path(encoder_checkpoint).exists():
            try:
                print(f"Loading encoder weights from: {encoder_checkpoint}")
                # PyTorch 2.6+ changed default weights_only=True, but our checkpoints contain numpy scalars
                checkpoint = torch.load(encoder_checkpoint, map_location='cpu', weights_only=False)
                
                # DINOv3 checkpoints can have different structures
                if isinstance(checkpoint, dict):
                    # Try different key formats (check most specific first)
                    if 'model_state_dict' in checkpoint:
                        # Supervised pretrain checkpoint format
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'encoder' in checkpoint:
                        state_dict = checkpoint['encoder']
                    else:
                        # Assume the dict itself is the state_dict
                        state_dict = checkpoint
                else:
                    # Direct state_dict
                    state_dict = checkpoint
                
                # Filter encoder weights (exclude decoder, classification_head, channel_proj, feature_combiner, etc.)
                encoder_state_dict = {}
                for k, v in state_dict.items():
                    # Check if this is an encoder key
                    # Supervised pretrain checkpoints have keys like 'encoder.blocks.0.attn.qkv.weight'
                    # Direct encoder checkpoints have keys like 'blocks.0.attn.qkv.weight'
                    if k.startswith('encoder.'):
                        # Remove 'encoder.' prefix for loading into self.encoder
                        new_key = k.replace('encoder.', '', 1)
                        # Exclude decoder, classification_head, channel_proj, feature_combiner
                        if (not new_key.startswith('decoder') and 
                            not new_key.startswith('classification_head') and 
                            not new_key.startswith('head') and
                            not new_key.startswith('channel_proj') and
                            not new_key.startswith('feature_combiner')):
                            encoder_state_dict[new_key] = v
                    elif (not k.startswith('decoder') and 
                          not k.startswith('classification_head') and 
                          not k.startswith('head') and
                          not k.startswith('channel_proj') and
                          not k.startswith('feature_combiner')):
                        # Direct encoder keys (no prefix) - keep as is
                        # DINOv3 encoder keys typically start with 'blocks', 'norm', 'patch_embed', etc.
                        encoder_state_dict[k] = v
                
                if len(encoder_state_dict) == 0:
                    print("Warning: No encoder weights found in checkpoint, using randomly initialized encoder")
                else:
                    missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                    if missing_keys:
                        print(f"Warning: Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
                    if unexpected_keys:
                        print(f"Warning: Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
                    print(f"Successfully loaded {len(encoder_state_dict)} encoder weight keys")
            except Exception as e:
                import traceback
                print(f"Error loading encoder checkpoint: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Using randomly initialized encoder")
        else:
            if encoder_checkpoint:
                print(f"Encoder checkpoint not found at: {encoder_checkpoint}, using randomly initialized encoder")
            else:
                print("No encoder checkpoint provided, using randomly initialized encoder")
        
        # Improved channel projection: Multi-layer conv to better preserve radar information
        # Instead of simple 1x1 conv, use a small CNN to learn better channel combination
        self.channel_proj = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=3, padding=1),  # 6 -> 12 channels
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 6, kernel_size=3, padding=1),  # 12 -> 6 channels
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 3, kernel_size=1),  # 6 -> 3 channels (final projection)
        )
        
        # Feature combiner: Combines CLS token with spatial pooling features
        # Only used when use_cls_token=True
        if use_cls_token:
            self.feature_combiner = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.feature_combiner = None
        
        # Classification head
        # Initialize bias to match class distribution to prevent collapse to all negatives
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pos_ratio=pos_ratio,  # Use provided pos_ratio for bias initialization
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W] where C is number of radar channels
        Returns:
            Classification logits [B, 1]
        """
        # Convert multi-channel radar data to RGB-like format for DINOv3
        # DINOv3 expects 3 channels, so we need to adapt
        # Use improved multi-layer channel projection to preserve radar information
        if x.shape[1] != 3:
            x = self.channel_proj(x)
        
        # Encode with DINOv3
        # Prepare tokens (includes CLS and storage tokens)
        x_tokens, (H, W) = self.encoder.prepare_tokens_with_masks(x, None)  # [B, num_tokens, embed_dim]
        
        # Pass through encoder blocks
        for blk in self.encoder.blocks:
            if self.encoder.rope_embed is not None:
                rope_sincos = self.encoder.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x_tokens = blk(x_tokens, rope_sincos)
        
        # Apply norm
        x_encoded = self.encoder.norm(x_tokens)  # [B, num_tokens, embed_dim]
        
        # Extract features: Combine CLS token with spatial pooling for richer representation
        # This helps capture both global context (CLS) and spatial patterns (pooling)
        n_extra_tokens = 1 + self.encoder.n_storage_tokens  # CLS + storage tokens
        cls_token = x_encoded[:, 0]  # [B, embed_dim]
        patch_tokens = x_encoded[:, n_extra_tokens:]  # [B, num_patches, embed_dim]
        
        if self.use_cls_token:
            # Combine CLS token with global average pooling for richer features
            spatial_features = patch_tokens.mean(dim=1)  # [B, embed_dim]
            # Concatenate and project back to embed_dim
            combined = torch.cat([cls_token, spatial_features], dim=1)  # [B, 2*embed_dim]
            # Use feature combiner to learn optimal combination
            features = self.feature_combiner(combined)  # [B, embed_dim]
        else:
            # Use only global average pooling of patch tokens
            features = patch_tokens.mean(dim=1)  # [B, embed_dim]
        
        # Classification
        logits = self.classification_head(features)  # [B, 1]
        
        return logits

