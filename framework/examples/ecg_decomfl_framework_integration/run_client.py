# examples/decomfl/run_client.py
"""
DeComFL Client Example - ECG Classification
Run federated learning client using DeComFL (zeroth-order optimization).
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.fedlearn.client.decomfl_client import DeComFLClient
from src.fedlearn.client.decomfl_start import start_decomfl_client
from config import Config
from model import create_model
from data import get_ecg_loaders


def main():
    """Main client function."""
    parser = argparse.ArgumentParser(description='DeComFL Client')
    parser.add_argument('--client-id', type=str, required=True, help='Unique client identifier')
    parser.add_argument('--server', type=str, default=Config.SERVER_ADDRESS, help='Server address')
    args = parser.parse_args()

    client_id = args.client_id
    server_address = args.server

    print("\n" + "=" * 70)
    print(f"DeComFL CLIENT: {client_id}")
    print("=" * 70)
    print(f"Server: {server_address}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 70 + "\n")

    # Load data
    print(f"[{client_id}] Loading ECG data...")
    data_path = Config.DATA_PATH

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # Update input dimension
    Config.INPUT_DIM = X.shape[1]

    # Parse client ID to get index (e.g., "client_0" -> 0)
    try:
        client_idx = int(client_id.split('_')[-1])
    except:
        print(f"[{client_id}] Warning: Could not parse client index. Using hash-based assignment.")
        client_idx = hash(client_id) % Config.NUM_CLIENTS

    # Get client's data partition
    print(f"[{client_id}] Loading data partition for client {client_idx}...")
    train_loader, _, data_info = get_ecg_loaders(
        X=X,
        y=y,
        client_id=client_idx,
        num_clients=Config.NUM_CLIENTS,
        batch_size_train=Config.BATCH_SIZE_TRAIN,
        batch_size_test=Config.BATCH_SIZE_TEST,
        data_fraction=Config.DATA_FRACTION,
        alpha=Config.ALPHA,
        test_size=Config.TEST_SIZE,
        num_workers=0,
        seed=Config.SEED
    )

    print(f"[{client_id}] Data loaded:")
    print(f"  Training samples: {data_info['train_samples']}")
    print(f"  Normal: {data_info['train_normal']}, Abnormal: {data_info['train_abnormal']}")

    # Create local model
    print(f"[{client_id}] Creating local model...")
    model = create_model(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE
    )

    # Create DeComFL client
    print(f"[{client_id}] Creating DeComFL client...")
    client = DeComFLClient(
        model=model,
        train_loader=train_loader,
        smoothing_param=Config.SMOOTHING_PARAM,
        device=Config.DEVICE
    )

    # Start client
    print(f"[{client_id}] Connecting to server...")
    print("=" * 70 + "\n")

    start_decomfl_client(
        server_address=server_address,
        client=client,
        client_id=client_id
    )


if __name__ == "__main__":
    main()