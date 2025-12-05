"""
BalancedBatchSampler for finetune compatibility.

This is a wrapper around GuaranteedPositiveSampler that provides
the interface expected by the finetune script (min_positives, pos_ratio).
"""
from .guaranteed_positive_sampler import GuaranteedPositiveSampler


class BalancedBatchSampler(GuaranteedPositiveSampler):
    """
    BalancedBatchSampler that guarantees positive examples in every batch.
    
    This is a compatibility wrapper around GuaranteedPositiveSampler
    that matches the interface expected by finetune scripts.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        min_positives: int = 2,
        pos_ratio: float = 0.3,
        replacement: bool = True,
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            min_positives: Minimum positive examples per batch (default 2)
            pos_ratio: Target positive ratio per batch (default 0.3, used to calculate min_positives)
            replacement: Whether to sample with replacement (default True)
        """
        # Calculate min_positives based on pos_ratio if not explicitly set
        # Ensure at least min_positives, but also try to match pos_ratio
        target_min_pos = max(min_positives, int(batch_size * pos_ratio))
        target_min_pos = min(target_min_pos, batch_size - 1)  # Don't exceed batch_size
        
        # Initialize parent class with the calculated min_positives
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            labels=None,  # Will be read from dataset
            min_positives=target_min_pos,
            replacement=replacement,
        )

