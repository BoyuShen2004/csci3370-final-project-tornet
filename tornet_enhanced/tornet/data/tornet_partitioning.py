"""
TorNet Data Partitioning Implementation

This module implements the exact data partitioning methodology described in the paper:
- Julian Day Modulo 20 partitioning (J(t_e) mod 20 < 17 for training, >= 17 for testing)
- 30-minute and 0.25-degree overlap removal
- Training: 171,666 samples (84.5%), Testing: 31,467 samples (15.5%)

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import json
from pathlib import Path


def julian_day(timestamp: datetime) -> int:
    """
    Calculate Julian day of a timestamp.
    
    Args:
        timestamp: datetime object
        
    Returns:
        Julian day (1-366)
    """
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


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in degrees.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in degrees
    """
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def remove_overlapping_samples(training_samples: List[Dict], 
                              testing_samples: List[Dict],
                              overlap_time_minutes: int = 30,
                              overlap_distance_degrees: float = 0.25) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove training samples that overlap with testing samples in time and space.
    
    Args:
        training_samples: List of training sample dictionaries
        testing_samples: List of testing sample dictionaries
        overlap_time_minutes: Time overlap threshold in minutes
        overlap_distance_degrees: Distance overlap threshold in degrees
        
    Returns:
        Tuple of (filtered_training_samples, testing_samples)
    """
    print(f"Removing overlapping samples...")
    print(f"Original training samples: {len(training_samples)}")
    print(f"Original testing samples: {len(testing_samples)}")
    
    # Convert to DataFrames for easier processing
    train_df = pd.DataFrame(training_samples)
    test_df = pd.DataFrame(testing_samples)
    
    # Convert timestamps to datetime if they're strings
    if 'timestamp' in train_df.columns:
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    if 'timestamp' in test_df.columns:
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Find overlapping samples
    overlapping_indices = []
    
    for i, train_sample in train_df.iterrows():
        train_time = train_sample['timestamp']
        train_lat = train_sample.get('latitude', 0)
        train_lon = train_sample.get('longitude', 0)
        
        for j, test_sample in test_df.iterrows():
            test_time = test_sample['timestamp']
            test_lat = test_sample.get('latitude', 0)
            test_lon = test_sample.get('longitude', 0)
            
            # Check time overlap
            time_diff = abs((train_time - test_time).total_seconds() / 60)  # minutes
            if time_diff <= overlap_time_minutes:
                # Check distance overlap
                distance = calculate_distance(train_lat, train_lon, test_lat, test_lon)
                if distance <= overlap_distance_degrees:
                    overlapping_indices.append(i)
                    break
    
    # Remove overlapping samples from training set
    filtered_train_df = train_df.drop(overlapping_indices)
    
    print(f"Removed {len(overlapping_indices)} overlapping training samples")
    print(f"Final training samples: {len(filtered_train_df)}")
    print(f"Final testing samples: {len(test_df)}")
    
    return filtered_train_df.to_dict('records'), test_df.to_dict('records')


def create_tornet_partitioning(dataset_path: str, 
                              output_path: str,
                              julian_modulo: int = 20,
                              training_threshold: int = 17,
                              overlap_removal: bool = True,
                              overlap_time_minutes: int = 30,
                              overlap_distance_degrees: float = 0.25) -> Dict:
    """
    Create TorNet data partitioning using Julian Day Modulo methodology.
    
    Args:
        dataset_path: Path to the TorNet dataset
        output_path: Path to save partitioning information
        julian_modulo: Modulo value for Julian day (default 20)
        training_threshold: Threshold for training samples (default 17)
        overlap_removal: Whether to remove overlapping samples
        overlap_time_minutes: Time overlap threshold in minutes
        overlap_distance_degrees: Distance overlap threshold in degrees
        
    Returns:
        Dictionary containing partitioning information
    """
    print("Creating TorNet data partitioning...")
    print(f"Julian modulo: {julian_modulo}")
    print(f"Training threshold: {training_threshold}")
    print(f"Overlap removal: {overlap_removal}")
    
    # Load dataset metadata (assuming it contains timestamp and location info)
    # This is a placeholder - you'll need to adapt this to your actual data structure
    samples = []
    
    # For now, create a sample structure - you'll need to replace this with actual data loading
    # This is just to demonstrate the partitioning logic
    sample_data = {
        'sample_id': [],
        'timestamp': [],
        'latitude': [],
        'longitude': [],
        'label': [],
        'file_path': []
    }
    
    # Load actual data here - this is a placeholder
    # You'll need to implement the actual data loading logic
    print("Loading dataset...")
    # TODO: Implement actual data loading from dataset_path
    
    # Apply Julian Day Modulo partitioning
    training_samples = []
    testing_samples = []
    
    for i, sample in enumerate(samples):
        timestamp = sample['timestamp']
        is_training = is_training_sample(timestamp, julian_modulo, training_threshold)
        
        if is_training:
            training_samples.append(sample)
        else:
            testing_samples.append(sample)
    
    print(f"Initial partitioning:")
    print(f"  Training samples: {len(training_samples)}")
    print(f"  Testing samples: {len(testing_samples)}")
    
    # Remove overlapping samples if requested
    if overlap_removal:
        training_samples, testing_samples = remove_overlapping_samples(
            training_samples, testing_samples,
            overlap_time_minutes, overlap_distance_degrees
        )
    
    # Create partitioning information
    partitioning_info = {
        'method': 'julian_day_modulo',
        'julian_modulo': julian_modulo,
        'training_threshold': training_threshold,
        'overlap_removal': overlap_removal,
        'overlap_time_minutes': overlap_time_minutes,
        'overlap_distance_degrees': overlap_distance_degrees,
        'training_samples': len(training_samples),
        'testing_samples': len(testing_samples),
        'total_samples': len(training_samples) + len(testing_samples),
        'training_percentage': len(training_samples) / (len(training_samples) + len(testing_samples)) * 100,
        'testing_percentage': len(testing_samples) / (len(training_samples) + len(testing_samples)) * 100,
        'training_sample_ids': [s['sample_id'] for s in training_samples],
        'testing_sample_ids': [s['sample_id'] for s in testing_samples]
    }
    
    # Save partitioning information
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(partitioning_info, f, indent=2)
    
    print(f"Partitioning information saved to: {output_path}")
    print(f"Final results:")
    print(f"  Training samples: {partitioning_info['training_samples']} ({partitioning_info['training_percentage']:.1f}%)")
    print(f"  Testing samples: {partitioning_info['testing_samples']} ({partitioning_info['testing_percentage']:.1f}%)")
    
    return partitioning_info


def load_tornet_partitioning(partitioning_path: str) -> Dict:
    """
    Load TorNet partitioning information from file.
    
    Args:
        partitioning_path: Path to partitioning JSON file
        
    Returns:
        Dictionary containing partitioning information
    """
    with open(partitioning_path, 'r') as f:
        return json.load(f)


def get_training_samples(partitioning_info: Dict) -> List[str]:
    """
    Get list of training sample IDs.
    
    Args:
        partitioning_info: Partitioning information dictionary
        
    Returns:
        List of training sample IDs
    """
    return partitioning_info['training_sample_ids']


def get_testing_samples(partitioning_info: Dict) -> List[str]:
    """
    Get list of testing sample IDs.
    
    Args:
        partitioning_info: Partitioning information dictionary
        
    Returns:
        List of testing sample IDs
    """
    return partitioning_info['testing_sample_ids']


def validate_partitioning(partitioning_info: Dict) -> bool:
    """
    Validate that the partitioning meets the expected criteria.
    
    Args:
        partitioning_info: Partitioning information dictionary
        
    Returns:
        True if partitioning is valid, False otherwise
    """
    # Check if we have the expected number of samples (approximately)
    expected_training = 171666  # 84.5%
    expected_testing = 31467    # 15.5%
    expected_total = expected_training + expected_testing
    
    actual_training = partitioning_info['training_samples']
    actual_testing = partitioning_info['testing_samples']
    actual_total = actual_training + actual_testing
    
    # Allow for some tolerance
    tolerance = 0.1  # 10% tolerance
    
    training_valid = abs(actual_training - expected_training) / expected_training < tolerance
    testing_valid = abs(actual_testing - expected_testing) / expected_testing < tolerance
    
    print(f"Partitioning validation:")
    print(f"  Expected training: {expected_training}, Actual: {actual_training}, Valid: {training_valid}")
    print(f"  Expected testing: {expected_testing}, Actual: {actual_testing}, Valid: {testing_valid}")
    
    return training_valid and testing_valid


if __name__ == "__main__":
    # Example usage
    dataset_path = "/projects/weilab/shenb/csci3370/data"
    output_path = "/projects/weilab/shenb/csci3370/tornet/partitioning_info.json"
    
    partitioning_info = create_tornet_partitioning(
        dataset_path=dataset_path,
        output_path=output_path,
        julian_modulo=20,
        training_threshold=17,
        overlap_removal=True,
        overlap_time_minutes=30,
        overlap_distance_degrees=0.25
    )
    
    # Validate partitioning
    is_valid = validate_partitioning(partitioning_info)
    print(f"Partitioning is valid: {is_valid}")
