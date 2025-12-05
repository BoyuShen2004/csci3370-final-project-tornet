"""
Segmentation model with DINOv3 encoder and UNet-like decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
from pathlib import Path

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dinov3"))
from dinov3.hub.backbones import dinov3_vits16


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SegmentationDecoder(nn.Module):
    """UNet-like decoder for segmentation."""
    
    def __init__(
        self,
        embed_dim: int = 384,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 2,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size
        
        # Initial projection
        self.proj = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Final classification head
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder features [B, num_patches, embed_dim]
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, N, D = x.shape
        
        # Reshape to spatial format
        H = W = self.num_patches_per_side
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
        
        # Initial projection
        x = self.proj(x)
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final upsampling to original resolution
        x = F.interpolate(x, size=(self.num_patches_per_side * self.patch_size, 
                                   self.num_patches_per_side * self.patch_size),
                         mode='bilinear', align_corners=False)
        
        # Classification head
        x = self.final_conv(x)
        
        return x


class SegmentationModel(nn.Module):
    """Segmentation model with DINOv3 encoder and UNet decoder."""
    
    def __init__(
        self,
        encoder_checkpoint: str,
        img_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 384,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 2,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Load DINOv3 encoder
        print(f"Loading encoder from {encoder_checkpoint}")
        self.encoder = dinov3_vits16(pretrained=False, weights=None)
        
        # Load encoder weights
        if encoder_checkpoint and Path(encoder_checkpoint).exists():
            try:
                checkpoint = torch.load(encoder_checkpoint, map_location='cpu')
                if isinstance(checkpoint, dict):
                    # Try different key formats
                    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                else:
                    state_dict = checkpoint
                
                # Filter encoder weights
                encoder_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith('decoder') and not k.startswith('mask_token'):
                        encoder_state_dict[k] = v
                
                missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
                print("Successfully loaded encoder weights")
            except Exception as e:
                print(f"Error loading encoder checkpoint: {e}")
                print("Using randomly initialized encoder")
        else:
            print("Encoder checkpoint not found, using randomly initialized encoder")
        
        # Segmentation decoder
        self.decoder = SegmentationDecoder(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
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
        
        # Apply norm and extract patch tokens (skip CLS and storage tokens)
        x_encoded = self.encoder.norm(x_tokens)  # [B, num_tokens, embed_dim]
        # Remove CLS token and storage tokens, keep only patch tokens
        n_extra_tokens = 1 + self.encoder.n_storage_tokens  # CLS + storage tokens
        x_encoded = x_encoded[:, n_extra_tokens:]  # [B, num_patches, embed_dim]
        
        # Decode to segmentation
        logits = self.decoder(x_encoded)  # [B, num_classes, H, W]
        
        return logits

