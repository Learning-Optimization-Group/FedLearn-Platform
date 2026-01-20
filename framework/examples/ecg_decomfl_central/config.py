# =============================================================================
# config.py - Configuration for DeComFL Training
# =============================================================================

import torch
from pathlib import Path


class Config:
    """Configuration for DeComFL federated learning"""

    # =========================================================================
    # Data Configuration
    # =========================================================================
    DATA_PATH = "ecg_data/ecg.csv"
    DATA_FRACTION = 1.0  # Use 100% of data (change to 0.1 for 10%, etc.)
    TEST_SIZE = 0.2  # 20% for test set

    # =========================================================================
    # Federated Learning Configuration
    # =========================================================================
    NUM_CLIENTS = 1  # For centralized training, use 1 client
    ALPHA = 0.5  # Dirichlet parameter (not used for 1 client)

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
    NUM_WORKERS = 4  # DataLoader workers (set to 0 if issues)
    SEED = 42

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
        print(f"  Number of clients: {cls.NUM_CLIENTS}")
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
        print("=" * 70 + "\n")

    @classmethod
    def get_model_save_path(cls, round_idx: int = None):
        """Get path for saving model checkpoint"""
        if round_idx is not None:
            return cls.CHECKPOINT_DIR / f"model_round_{round_idx}.pth"
        return cls.CHECKPOINT_DIR / "model_final.pth"

    @classmethod
    def get_results_save_path(cls):
        """Get path for saving training results"""
        return cls.RESULTS_DIR / "training_history.npz"