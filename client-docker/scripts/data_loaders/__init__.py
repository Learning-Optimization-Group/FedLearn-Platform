"""
Data loaders package for federated learning.
Contains data loading utilities for different datasets.
"""

from .ecg_loader import (
    ECGDataset,
    dirichlet_split,
    get_or_create_split,
    get_ecg_loaders,
    get_test_loader
)

__all__ = [
    'ECGDataset',
    'dirichlet_split',
    'get_or_create_split',
    'get_ecg_loaders',
    'get_test_loader'
]