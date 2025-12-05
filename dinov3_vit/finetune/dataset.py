"""
Dataset for supervised segmentation fine-tuning.
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import tifffile


class SegmentationDataset(Dataset):
    """Dataset for image-mask pairs."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 384,
        is_training: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.is_training = is_training
        
        # Collect matching image-mask pairs
        image_paths = sorted(list(self.image_dir.glob('*.tif')) + list(self.image_dir.glob('*.tiff')) +
                            list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        
        self.pairs = []
        for img_path in image_paths:
            # Try to find corresponding mask
            mask_name = img_path.stem + '.tif'
            mask_path = self.mask_dir / mask_name
            if not mask_path.exists():
                mask_path = self.mask_dir / (img_path.stem + '.tiff')
            if not mask_path.exists():
                mask_path = self.mask_dir / (img_path.stem + '.png')
            
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))
        
        print(f"Found {len(self.pairs)} image-mask pairs")
        
        # Data augmentation
        if is_training:
            self.image_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        
        # Load image
        try:
            if img_path.suffix.lower() in ['.tif', '.tiff']:
                img = tifffile.imread(str(img_path))
                if img.dtype != np.uint8:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype(np.uint8)
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                img = Image.fromarray(img)
            else:
                img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        
        # Load mask
        try:
            if mask_path.suffix.lower() in ['.tif', '.tiff']:
                mask = tifffile.imread(str(mask_path))
            else:
                mask = np.array(Image.open(mask_path))
            
            # Convert to binary if needed
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask, mode='L')
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            mask = Image.new('L', (self.img_size, self.img_size), color=0)
        
        # Apply transforms
        img_tensor = self.image_transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).long().squeeze(0)  # [H, W]
        
        return img_tensor, mask_tensor


def create_dataloaders(
    image_dir: str,
    mask_dir: str,
    img_size: int = 384,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    # Create full dataset
    full_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_size=img_size,
        is_training=True,
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update training flag for val dataset
    val_dataset.dataset.is_training = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader

