# examples/decomfl/config.py
"""
Configuration for DeComFL Federated Learning - ECG Classification
"""

import torch
from pathlib import Path


class Config:
    """Configuration for DeComFL federated learning"""

    # =========================================================================
    # Data Configuration
    # =========================================================================
    DATA_PATH = "ecg_data/ecg.csv"
    DATA_FRACTION = 1.0  # Use 100% of data
    TEST_SIZE = 0.2  # 20% for test set
    INPUT_DIM = 140  # Will be updated based on actual data

    # =========================================================================
    # Federated Learning Configuration
    # =========================================================================
    NUM_CLIENTS = 5  # Total number of clients
    CLIENTS_PER_ROUND = 2  # Number of clients sampled per round
    MIN_FIT_CLIENTS = 2  # Minimum clients for aggregation
    ALPHA = 0.5  # Dirichlet parameter for non-IID data split

    # =========================================================================
    # DeComFL Algorithm Parameters (from Algorithm 3 & 4)
    # =========================================================================
    NUM_ROUNDS = 100  # R: Number of communication rounds
    NUM_LOCAL_STEPS = 1  # K: Number of local SGD steps per round
    NUM_PERTURBATIONS = 10  # P: Number of random perturbations
    LEARNING_RATE = 0.001  # η: Learning rate
    SMOOTHING_PARAM = 0.001  # μ: Smoothing parameter for zeroth-order gradient

    # =========================================================================
    # Model Configuration
    # =========================================================================
    HIDDEN_DIM = 64  # Hidden layer dimension
    NUM_CLASSES = 2  # Binary classification (Normal vs Abnormal)

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_TEST = 256
    NUM_WORKERS = 0  # DataLoader workers (0 for Windows compatibility)
    SEED = 42

    # =========================================================================
    # Server Configuration
    # =========================================================================
    SERVER_ADDRESS = "localhost:50051"

    # =========================================================================
    # Paths
    # =========================================================================
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    LOGS_DIR = Path("logs")

    # Create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # =========================================================================
    # Device Configuration
    # =========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 70)
        print("DeComFL CONFIGURATION")
        print("=" * 70)
        print("\nData Configuration:")
        print(f"  Data path: {cls.DATA_PATH}")
        print(f"  Data fraction: {cls.DATA_FRACTION * 100}%")
        print(f"  Test size: {cls.TEST_SIZE * 100}%")

        print("\nFederated Learning:")
        print(f"  Total number of clients: {cls.NUM_CLIENTS}")
        print(f"  Clients per round: {cls.CLIENTS_PER_ROUND}")
        print(f"  Minimum clients for aggregation: {cls.MIN_FIT_CLIENTS}")
        print(f"  Alpha (Dirichlet): {cls.ALPHA}")

        print("\nDeComFL Parameters:")
        print(f"  Communication rounds (R): {cls.NUM_ROUNDS}")
        print(f"  Local steps (K): {cls.NUM_LOCAL_STEPS}")
        print(f"  Perturbations (P): {cls.NUM_PERTURBATIONS}")
        print(f"  Learning rate (η): {cls.LEARNING_RATE}")
        print(f"  Smoothing (μ): {cls.SMOOTHING_PARAM}")

        print("\nModel Configuration:")
        print(f"  Hidden dimension: {cls.HIDDEN_DIM}")
        print(f"  Number of classes: {cls.NUM_CLASSES}")

        print("\nTraining Configuration:")
        print(f"  Train batch size: {cls.BATCH_SIZE_TRAIN}")
        print(f"  Test batch size: {cls.BATCH_SIZE_TEST}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Random seed: {cls.SEED}")

        print("\nServer Configuration:")
        print(f"  Server address: {cls.SERVER_ADDRESS}")
        print("=" * 70 + "\n")