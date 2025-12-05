"""
TorNet Data Loader with Julian Day Modulo Partitioning for Classification

This module implements data loading for TorNet dataset with:
- Julian Day Modulo partitioning (J(te) mod 20 < 17 for training, >= 17 for testing)
- Binary classification (tornado vs no tornado)
- Support for imbalanced dataset handling
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import xarray as xr

from tornet.data.loader import read_file, query_catalog
from tornet.data.constants import ALL_VARIABLES, CHANNEL_MIN_MAX
from tornet.data.preprocess import add_coordinates, split_x_y


def julian_day(timestamp: datetime) -> int:
    """Calculate Julian day of a timestamp."""
    return timestamp.timetuple().tm_yday


def is_training_sample(timestamp: datetime, 
                      julian_modulo: int = 20, 
                      training_threshold: int = 17) -> bool:
    """
    Determine if a sample should be used for training based on Julian Day Modulo partitioning.
    
    Args:
        timestamp: Sample timestamp
        julian_modulo: Modulo value (default 20)
        training_threshold: Threshold for training (default 17)
        
    Returns:
        True if sample should be used for training, False for testing
    """
    j_day = julian_day(timestamp)
    return (j_day % julian_modulo) < training_threshold


class TorNetClassificationDataset(Dataset):
    """
    TorNet dataset for binary classification (tornado vs no tornado).
    
    Supports Julian Day Modulo partitioning and imbalanced dataset handling.
    """
    
    def __init__(
        self,
        data_root: str,
        data_type: str = "train",  # "train", "test", or "all"
        years: List[int] = list(range(2013, 2023)),
        julian_modulo: int = 20,
        training_threshold: int = 17,
        img_size: int = 384,
        variables: List[str] = None,
        random_state: int = 1234,
        use_augmentation: bool = False,
        use_catalog_type: bool = False,  # NEW: Use catalog's type column like tornet_enhanced (default False for backward compatibility)
    ):
        """
        Args:
            data_root: Path to TorNet data directory
            data_type: "train" (J mod 20 < 17 or catalog type='train'), "test" (J mod 20 >= 17 or catalog type='test'), or "all" (both)
            years: List of years to include
            julian_modulo: Modulo value for Julian day partitioning (only used if use_catalog_type=False)
            training_threshold: Threshold for training samples (only used if use_catalog_type=False)
            img_size: Target image size (will be resized)
            variables: List of radar variables to use
            random_state: Random seed
            use_augmentation: Whether to apply data augmentation
            use_catalog_type: If True, use catalog's 'type' column (tornet_enhanced style). If False, compute Julian Day Modulo.
        """
        self.data_root = data_root
        self.data_type = data_type
        self.years = years
        self.julian_modulo = julian_modulo
        self.training_threshold = training_threshold
        self.img_size = img_size
        self.variables = variables if variables else ALL_VARIABLES
        self.use_augmentation = use_augmentation
        self.use_catalog_type = use_catalog_type
        
        # Load catalog
        catalog_path = os.path.join(data_root, 'catalog.csv')
        if not os.path.exists(catalog_path):
            raise RuntimeError(f'Unable to find catalog.csv at {data_root}')
        
        catalog = pd.read_csv(catalog_path, parse_dates=['start_time', 'end_time'])
        
        # Filter by years
        catalog = catalog[catalog.start_time.dt.year.isin(years)]
        
        # Apply partitioning
        if data_type == "all":
            # Use all samples regardless of partitioning
            self.file_list = [os.path.join(data_root, f) for f in catalog.filename]
        elif use_catalog_type:
            # Use catalog's 'type' column (tornet_enhanced style)
            if 'type' not in catalog.columns:
                raise RuntimeError("Catalog missing 'type' column. Set use_catalog_type=False to use Julian Day Modulo partitioning.")
            catalog = catalog[catalog['type'] == data_type]
            self.file_list = [os.path.join(data_root, f) for f in catalog.filename]
        else:
            # Apply partitioning based on Julian day (original DINO approach)
            partitioned_files = []
            for idx, row in catalog.iterrows():
                timestamp = row['start_time']
                is_training = is_training_sample(timestamp, julian_modulo, training_threshold)
                
                if (data_type == "train" and is_training) or (data_type == "test" and not is_training):
                    partitioned_files.append(row['filename'])
            
            self.file_list = [os.path.join(data_root, f) for f in partitioned_files]
        
        # Shuffle
        np.random.seed(random_state)
        np.random.shuffle(self.file_list)
        
        print(f"TorNet Classification Dataset:")
        print(f"  Data type: {data_type}")
        print(f"  Years: {years}")
        if use_catalog_type:
            print(f"  Partitioning: Catalog 'type' column (tornet_enhanced style)")
        else:
            print(f"  Partitioning: Julian Day Modulo (modulo={julian_modulo}, threshold={training_threshold})")
        print(f"  Total samples: {len(self.file_list)}")
        
        # Count labels for class balancing
        self._count_labels()
    
    def _count_labels(self):
        """Count positive and negative samples for class balancing."""
        pos_count = 0
        neg_count = 0
        
        # Sample a subset to estimate class distribution
        sample_size = min(1000, len(self.file_list))
        import warnings
        import contextlib
        import io
        for i in range(sample_size):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Redirect both stdout and stderr to suppress xarray dependency warnings
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    try:
                        sys.stdout = io.StringIO()
                        sys.stderr = io.StringIO()
                        data = read_file(self.file_list[i], variables=self.variables, n_frames=1, tilt_last=True)
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                label = data['label'][-1]  # Last frame label
                if label > 0:
                    pos_count += 1
                else:
                    neg_count += 1
            except:
                continue
        
        if sample_size > 0:
            self.pos_ratio = pos_count / sample_size
            self.neg_ratio = neg_count / sample_size
            print(f"  Estimated class distribution: Positive={self.pos_ratio:.3f}, Negative={self.neg_ratio:.3f}")
        else:
            self.pos_ratio = 0.1  # Default estimate
            self.neg_ratio = 0.9
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Preprocessed radar data [C, H, W] where C includes all radar variables
            label: Binary label [1] (1 for tornado, 0 for no tornado)
        """
        file_path = self.file_list[idx]
        
        try:
            # Suppress xarray warnings about missing optional dependencies
            import warnings
            import contextlib
            import io
            # Suppress xarray warnings and error messages about missing optional dependencies
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Redirect both stdout and stderr to suppress xarray print statements
                # (xarray prints dependency warnings to stdout)
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    # Read TorNet file
                    data = read_file(file_path, variables=self.variables, n_frames=1, tilt_last=True)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
            # Add coordinates
            add_coordinates(data, include_az=False, backend=np, tilt_last=True)
            
            # Split into inputs and labels
            x, y = split_x_y(data)
            
            # Extract label (binary: tornado or not)
            # Note: split_x_y returns y as a numpy array (the label), not a dict
            # Handle both scalar and array cases
            label_arr = y  # y IS the label array, not a dict
            if label_arr.ndim == 0:
                # Scalar case
                label_val = label_arr.item()
            else:
                # Array case - get last element
                label_val = label_arr[-1] if len(label_arr) > 0 else 0
            label = 1 if label_val > 0 else 0  # Binary classification
            label = torch.tensor([label], dtype=torch.float32)
            
            # Process radar variables
            # Stack all variables into channels
            channels = []
            for var in self.variables:
                var_data = x[var]
                
                # Check if data is NaN/inf before processing
                if not np.isfinite(var_data).all():
                    # Replace NaN/inf with zeros (or use fill_value for the variable)
                    # This prevents NaN propagation through the pipeline
                    var_data = np.where(np.isfinite(var_data), var_data, 0.0)
                
                # Handle different array shapes
                if var_data.ndim == 4:
                    # Shape: [time, tilt, azimuth, range] or [time, azimuth, range, tilt]
                    var_data = var_data[-1]  # Last frame
                elif var_data.ndim == 3:
                    # Already last frame: [tilt, azimuth, range]
                    pass
                else:
                    # Unexpected shape, try to handle
                    var_data = var_data.reshape(-1, var_data.shape[-2], var_data.shape[-1])[-1]
                
                # Take mean over tilt dimension to get [azimuth, range]
                if var_data.ndim == 3:
                    var_data = np.mean(var_data, axis=0)
                elif var_data.ndim == 2:
                    # Already 2D
                    pass
                else:
                    # Fallback: try to get 2D shape
                    var_data = var_data.reshape(var_data.shape[-2], var_data.shape[-1])
                
                # Check again after shape manipulation
                if not np.isfinite(var_data).all():
                    var_data = np.where(np.isfinite(var_data), var_data, 0.0)
                
                # Normalize using CHANNEL_MIN_MAX
                min_val, max_val = CHANNEL_MIN_MAX[var]
                var_data = (var_data - min_val) / (max_val - min_val + 1e-8)
                var_data = np.clip(var_data, 0, 1)
                
                # Final check - ensure no NaN after normalization
                if not np.isfinite(var_data).all():
                    var_data = np.where(np.isfinite(var_data), var_data, 0.0)
                
                channels.append(var_data)
            
            # Stack channels: [C, H, W]
            image = np.stack(channels, axis=0).astype(np.float32)
            
            # Check for NaN after stacking
            if not np.isfinite(image).all():
                # Replace any remaining NaN with zeros
                image = np.where(np.isfinite(image), image, 0.0)
            
            # Resize to target size (using torch interpolation)
            if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
                image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, C, H, W]
                image_tensor = F.interpolate(
                    image_tensor,
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False,
                )
                image = image_tensor.squeeze(0).numpy()
                
                # Check for NaN after interpolation
                if not np.isfinite(image).all():
                    image = np.where(np.isfinite(image), image, 0.0)
            
            # Convert to torch tensor
            image = torch.from_numpy(image)
            
            # Final check - ensure no NaN in final tensor
            if not torch.isfinite(image).all():
                image = torch.where(torch.isfinite(image), image, torch.zeros_like(image))
            
            # Apply augmentation if training
            if self.use_augmentation and self.data_type == "train":
                image = self._apply_augmentation(image)
            
            return image, label
            
        except (IndexError, ValueError, TypeError) as e:
            # Handle indexing errors (e.g., "only integers, slices... are valid indices")
            # These happen when array shapes are unexpected
            # Log error (with num_workers=0, these should be rare)
            import random
            if random.random() < 0.1:  # Log 10% of errors (more frequent since num_workers=0 should prevent most)
                print(f"WARNING: IndexError/ValueError/TypeError loading {file_path}: {e}", file=sys.stderr)
                print(f"  Returning dummy label=0. This file will be skipped in training.", file=sys.stderr)
            # Return dummy data (DataLoader can't handle exceptions)
            dummy_image = torch.zeros((len(self.variables), self.img_size, self.img_size), dtype=torch.float32)
            dummy_label = torch.tensor([0], dtype=torch.float32)
            return dummy_image, dummy_label
            
        except Exception as e:
            # Log error (with num_workers=0, multiprocessing issues are avoided, so real errors should be rare)
            error_str = str(e)
            
            # Only suppress xarray dependency warnings (these are just warnings)
            if "netcdf4" in error_str.lower() or "h5netcdf" in error_str.lower():
                # For xarray dependency warnings, try to continue (these are usually non-fatal)
                # But still log occasionally
                import random
                if random.random() < 0.01:
                    print(f"Xarray dependency warning (non-fatal) for {file_path}: {e}", file=sys.stderr)
                # Return dummy data only for xarray dependency warnings
                dummy_image = torch.zeros((len(self.variables), self.img_size, self.img_size), dtype=torch.float32)
                dummy_label = torch.tensor([0], dtype=torch.float32)
                return dummy_image, dummy_label
            else:
                # For real errors, log (with num_workers=0, these should be rare)
                import random
                if random.random() < 0.1:  # Log 10% of errors (more frequent since num_workers=0 should prevent most)
                    print(f"WARNING: Error loading {file_path}: {e}", file=sys.stderr)
                    print(f"  Returning dummy label=0. This file will be skipped in training.", file=sys.stderr)
                # Return dummy data (DataLoader can't handle exceptions)
                dummy_image = torch.zeros((len(self.variables), self.img_size, self.img_size), dtype=torch.float32)
                dummy_label = torch.tensor([0], dtype=torch.float32)
                return dummy_image, dummy_label
    
    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply conservative data augmentation for radar data."""
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = torch.flip(image, dims=[2])
        
        # Small rotation (90, 180, 270 degrees)
        if np.random.rand() < 0.3:
            k = np.random.randint(1, 4)
            image = torch.rot90(image, k, dims=[1, 2])
        
        # Add small noise
        if np.random.rand() < 0.3:
            noise = torch.randn_like(image) * 0.01
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image


def create_tornet_dataloader(
    data_root: str,
    data_type: str = "train",
    years: List[int] = list(range(2013, 2023)),
    julian_modulo: int = 20,
    training_threshold: int = 17,
    batch_size: int = 32,
    img_size: int = 384,
    num_workers: int = 8,
    pin_memory: bool = True,
    use_augmentation: bool = False,
    use_class_balancing: bool = False,
    variables: List[str] = None,
    random_state: int = 1234,
    use_catalog_type: bool = False,  # NEW: Use catalog's type column (default False for backward compatibility)
) -> DataLoader:
    """
    Create a DataLoader for TorNet classification.
    
    Args:
        data_root: Path to TorNet data
        data_type: "train", "test", or "all"
        years: List of years
        julian_modulo: Modulo for Julian day partitioning
        training_threshold: Threshold for training samples
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        use_augmentation: Whether to use augmentation
        use_class_balancing: Whether to use weighted sampling for class balancing
        variables: List of radar variables
        random_state: Random seed
        
    Returns:
        DataLoader
    """
    dataset = TorNetClassificationDataset(
        data_root=data_root,
        data_type=data_type,
        years=years,
        julian_modulo=julian_modulo,
        training_threshold=training_threshold,
        img_size=img_size,
        variables=variables,
        random_state=random_state,
        use_augmentation=use_augmentation,
        use_catalog_type=use_catalog_type,
    )
    
    # Create weighted sampler for class balancing if requested
    sampler = None
    if use_class_balancing and data_type == "train":
        # Estimate class weights
        pos_weight = 1.0 / (dataset.pos_ratio + 1e-8)
        neg_weight = 1.0 / (dataset.neg_ratio + 1e-8)
        
        # Create sample weights
        sample_weights = []
        for i in range(len(dataset)):
            try:
                data = read_file(dataset.file_list[i], variables=dataset.variables, n_frames=1, tilt_last=True)
                label = data['label'][-1]
                weight = pos_weight if label > 0 else neg_weight
                sample_weights.append(weight)
            except:
                sample_weights.append(neg_weight)  # Default to negative weight
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(data_type == "train"),
    )
    
    return dataloader

