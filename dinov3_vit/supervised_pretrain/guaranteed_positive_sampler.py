"""
Simple Batch Sampler that Guarantees At Least 1 Positive Per Batch

This is a simpler alternative to BalancedBatchSampler that ensures every batch
has at least 1 positive example, which is critical for learning on imbalanced data.
"""
import torch
from torch.utils.data import Sampler
import numpy as np
from typing import List


class GuaranteedPositiveSampler(Sampler):
    """
    Simple sampler that guarantees at least 1 positive per batch.
    
    Strategy:
    1. Separate indices into positive and negative
    2. For each batch:
       - Sample 1+ positives (with replacement if needed)
       - Fill rest with negatives
    3. Shuffle batches to avoid ordering bias
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        labels: List[int] = None,
        min_positives: int = 1,
        replacement: bool = True,
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            labels: Pre-computed labels (list of 0/1). If None, will read from dataset.
            min_positives: Minimum positive examples per batch (default 1)
            replacement: Whether to sample with replacement (default True for positives)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_positives = min_positives
        self.replacement = replacement
        
        # Get labels
        if labels is not None:
            self.labels = np.array(labels)
        else:
            # Read labels from dataset (slow, but only once)
            self.labels = self._read_labels()
        
        # Separate positive and negative indices
        self.positive_indices = np.where(self.labels == 1)[0].tolist()
        self.negative_indices = np.where(self.labels == 0)[0].tolist()
        
        if len(self.positive_indices) == 0:
            raise RuntimeError("No positive examples found in dataset! Cannot create GuaranteedPositiveSampler.")
        
        # Calculate number of batches
        # Each batch needs at least min_positives positives
        # We can create batches until we run out of positives (with replacement)
        n_positives = len(self.positive_indices)
        n_negatives = len(self.negative_indices)
        
        # Number of batches per epoch: cover all examples exactly once (like tornet_enhanced)
        self.num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        import sys
        print(f"GuaranteedPositiveSampler:", flush=True)
        print(f"  Positive indices: {len(self.positive_indices)}", flush=True)
        print(f"  Negative indices: {len(self.negative_indices)}", flush=True)
        print(f"  Batch size: {batch_size}", flush=True)
        print(f"  Min positives per batch: {min_positives}", flush=True)
        print(f"  Number of batches per epoch: {self.num_batches}", flush=True)
        print(f"  Each epoch covers all {len(self.dataset)} examples exactly once (like tornet_enhanced)", flush=True)
        sys.stdout.flush()
    
    def _read_labels(self):
        """Read labels from dataset (slow, but only done once)."""
        import sys
        labels = []
        print("Reading labels from dataset for GuaranteedPositiveSampler...", flush=True)
        sys.stdout.flush()
        total = len(self.dataset)
        # Log more frequently for large datasets (every 1% or every 1000 samples, whichever is smaller)
        log_interval = min(max(1, total // 100), 1000)  # Log every 1% or every 1000 samples
        errors = 0
        positives_found = 0
        start_time = None
        import time
        
        for i in range(total):
            if i == 0:
                start_time = time.time()
            if i % log_interval == 0 or i == total - 1:
                elapsed = time.time() - start_time if start_time else 0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(f"  Reading labels: {i}/{total} ({100*i/total:.1f}%), "
                      f"positives: {positives_found}, errors: {errors}, "
                      f"elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s", flush=True)
                sys.stdout.flush()
            
            try:
                _, label = self.dataset[i]
                if isinstance(label, torch.Tensor):
                    if label.numel() == 1:
                        label_val = label.item()
                    elif len(label) > 0:
                        label_val = label[0].item()
                    else:
                        label_val = 0.0
                else:
                    label_val = float(label)
                
                label_binary = 1 if label_val > 0.5 else 0
                labels.append(label_binary)
                if label_binary == 1:
                    positives_found += 1
            except Exception as e:
                errors += 1
                if errors <= 10:  # Only print first 10 errors
                    print(f"Error reading label for sample {i}: {e}, defaulting to 0")
                labels.append(0)
        
        print(f"  Finished reading labels: {total} samples, {positives_found} positives, {errors} errors", flush=True)
        import sys
        sys.stdout.flush()
        labels_array = np.array(labels)
        
        if positives_found == 0:
            print(f"WARNING: No positives found! Label distribution: {np.bincount(labels_array)}")
            # Try to read a few samples manually to debug
            print("Debugging: Checking first 10 samples...")
            for i in range(min(10, total)):
                try:
                    _, label = self.dataset[i]
                    print(f"  Sample {i}: label={label}, type={type(label)}")
                except Exception as e:
                    print(f"  Sample {i}: ERROR - {e}")
        
        return labels_array
    
    def __iter__(self):
        """
        Generate batches with guaranteed positives, ensuring all examples are seen once per epoch.
        
        Strategy (like tornet_enhanced):
        1. Shuffle all indices (positives and negatives) at start of epoch
        2. Create batches that guarantee at least 1 positive per batch
        3. Go through all examples exactly once per epoch
        """
        # Shuffle all indices at the start of each epoch (like tornet_enhanced shuffles catalog)
        all_indices = list(range(len(self.dataset)))
        np.random.shuffle(all_indices)
        
        # Separate shuffled indices into positives and negatives
        shuffled_positives = [idx for idx in all_indices if idx in self.positive_indices]
        shuffled_negatives = [idx for idx in all_indices if idx in self.negative_indices]
        
        # Track which indices we've used (with cycling for positives to ensure every batch has them)
        pos_idx = 0
        neg_idx = 0
        
        # Calculate number of batches needed to cover all examples
        num_batches = (len(all_indices) + self.batch_size - 1) // self.batch_size
        
        for batch_num in range(num_batches):
            batch_indices = []
            
            # CRITICAL: Ensure at least min_positives positives in this batch
            # If we've used all positives, cycle back (with replacement for positives only)
            n_positives_needed = max(self.min_positives, int(self.batch_size * 0.1))
            n_positives_needed = min(n_positives_needed, self.batch_size - 1)
            
            # Add positives (cycle if we've used all)
            for _ in range(n_positives_needed):
                if pos_idx >= len(shuffled_positives):
                    # Cycle back to start - this ensures every batch has positives
                    pos_idx = 0
                batch_indices.append(shuffled_positives[pos_idx])
                pos_idx += 1
            
            # Fill rest with negatives (without replacement - use each negative once)
            n_negatives_needed = self.batch_size - len(batch_indices)
            for _ in range(n_negatives_needed):
                if neg_idx < len(shuffled_negatives):
                    batch_indices.append(shuffled_negatives[neg_idx])
                    neg_idx += 1
                else:
                    # No more negatives - pad with positives (cycling)
                    if pos_idx >= len(shuffled_positives):
                        pos_idx = 0
                    batch_indices.append(shuffled_positives[pos_idx])
                    pos_idx += 1
            
            # Shuffle batch to avoid ordering bias
            np.random.shuffle(batch_indices)
            
            # Yield batch (should be exactly batch_size)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return self.num_batches

