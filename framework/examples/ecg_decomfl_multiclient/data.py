# =============================================================================
# data.py - ECG Data Loading for Federated Learning
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG signals"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Split data indices using Dirichlet distribution for non-IID partitioning.

    Args:
        labels: Array of labels (0 or 1 for binary classification)
        num_clients: Number of clients to split data across
        alpha: Dirichlet concentration parameter
               - Lower alpha (e.g., 0.1) = more non-IID (heterogeneous)
               - Higher alpha (e.g., 10.0) = more IID (homogeneous)
        seed: Random seed for reproducibility

    Returns:
        client_indices: List of indices for each client
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))

    # Dirichlet distribution for each class
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        # Get indices for class k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Split according to Dirichlet proportions
        proportions = label_distribution[k]
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_splits = np.split(idx_k, splits)

        # Assign to clients
        for i, idx in enumerate(idx_splits):
            client_indices[i].extend(idx)

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def get_or_create_split(
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int,
        alpha: float,
        data_fraction: float,
        test_size: float = 0.2,
        seed: int = 42
):
    """
    Get existing data split or create a new one.
    This ensures all clients use the SAME split.

    Args:
        X: Feature array (samples, features)
        y: Label array (samples,)
        num_clients: Total number of clients
        alpha: Dirichlet parameter for non-IID split
        data_fraction: Fraction of data to use (1.0 = full dataset)
        test_size: Fraction of data for test set
        seed: Random seed

    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
        client_indices_list: List of indices for each client
    """
    # Create cache directory
    cache_dir = Path("./data_splits")
    cache_dir.mkdir(exist_ok=True)

    # Create unique filename based on configuration
    split_file = cache_dir / f"ecg_clients{num_clients}_alpha{alpha}_frac{data_fraction}_seed{seed}.pkl"

    # Convert y to integer if needed
    y = y.astype(int)

    if split_file.exists():
        # Load existing split
        print(f"Loading existing data split from {split_file}")
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            X_train = split_data['X_train']
            X_test = split_data['X_test']
            y_train = split_data['y_train']
            y_test = split_data['y_test']
            client_indices_list = split_data['client_indices']
    else:
        # Create new split
        print(f"Creating new data split and saving to {split_file}")

        # 1. First split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # 2. Use only fraction of training data if specified
        if data_fraction < 1.0:
            num_samples = int(len(X_train) * data_fraction)
            indices = np.random.RandomState(seed).permutation(len(X_train))[:num_samples]
            X_train = X_train[indices]
            y_train = y_train[indices]
            print(f"Using {data_fraction * 100}% of training data: {num_samples} samples")

        # 3. Split training data across clients using Dirichlet
        client_indices_list = dirichlet_split(y_train, num_clients, alpha, seed)

        # 4. Save the split
        split_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'client_indices': client_indices_list
        }
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)

        # Print distribution
        print(f"\nData distribution across {num_clients} clients:")
        for i, indices in enumerate(client_indices_list):
            client_labels = y_train[indices]
            print(f"  Client {i}: {len(indices)} samples "
                  f"(Normal: {np.sum(client_labels == 0)}, "
                  f"Abnormal: {np.sum(client_labels == 1)})")

        print(f"\nTest set: {len(X_test)} samples "
              f"(Normal: {np.sum(y_test == 0)}, "
              f"Abnormal: {np.sum(y_test == 1)})")

    return X_train, X_test, y_train, y_test, client_indices_list


def get_ecg_loaders(
        X: np.ndarray,
        y: np.ndarray,
        client_id: int,
        num_clients: int,
        batch_size_train: int = 128,
        batch_size_test: int = 128,
        data_fraction: float = 1.0,
        alpha: float = 0.5,
        test_size: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
):
    """
    Prepares DataLoaders for a specific client in federated learning.

    Args:
        X: Full ECG feature array (n_samples, n_features)
        y: Full label array (n_samples,)
        client_id: Client ID (0 to num_clients-1)
        num_clients: Total number of clients
        batch_size_train: Batch size for training
        batch_size_test: Batch size for testing
        data_fraction: Fraction of data to use (1.0 = full dataset)
        alpha: Dirichlet parameter (lower = more non-IID)
        test_size: Fraction for test set
        num_workers: Number of workers for data loading
        seed: Random seed

    Returns:
        train_loader: DataLoader for client's training data
        test_loader: DataLoader for test data (shared across all clients)
        data_info: Dictionary with data statistics
    """
    # Get or create consistent split
    X_train, X_test, y_train, y_test, client_indices_list = get_or_create_split(
        X=X,
        y=y,
        num_clients=num_clients,
        alpha=alpha,
        data_fraction=data_fraction,
        test_size=test_size,
        seed=seed
    )

    # Get this client's training data
    client_indices = client_indices_list[client_id]
    X_client = X_train[client_indices]
    y_client = y_train[client_indices]

    # Create datasets
    train_dataset = ECGDataset(X_client, y_client)
    test_dataset = ECGDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Collect data info
    data_info = {
        'client_id': client_id,
        'num_clients': num_clients,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'train_normal': int(np.sum(y_client == 0)),
        'train_abnormal': int(np.sum(y_client == 1)),
        'test_normal': int(np.sum(y_test == 0)),
        'test_abnormal': int(np.sum(y_test == 1)),
        'input_dim': X.shape[1],
        'alpha': alpha,
        'data_fraction': data_fraction
    }

    print(f"\n{'=' * 60}")
    print(f"CLIENT {client_id} DATA LOADED")
    print(f"{'=' * 60}")
    print(f"Training samples: {data_info['train_samples']}")
    print(f"  - Normal: {data_info['train_normal']}")
    print(f"  - Abnormal: {data_info['train_abnormal']}")
    print(f"Test samples: {data_info['test_samples']}")
    print(f"  - Normal: {data_info['test_normal']}")
    print(f"  - Abnormal: {data_info['test_abnormal']}")
    print(f"Input dimension: {data_info['input_dim']}")
    print(f"{'=' * 60}\n")

    return train_loader, test_loader, data_info


def get_test_loader(
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int,
        batch_size: int = 128,
        alpha: float = 0.5,
        data_fraction: float = 1.0,
        test_size: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
):
    """
    Get test DataLoader without client partitioning.
    Used by the server for global evaluation.

    Args:
        X: Full ECG feature array
        y: Full label array
        num_clients: Number of clients (needed for cache consistency)
        batch_size: Batch size for testing
        alpha: Dirichlet parameter (for cache consistency)
        data_fraction: Data fraction (for cache consistency)
        test_size: Fraction for test set
        num_workers: Number of workers
        seed: Random seed

    Returns:
        test_loader: DataLoader for test data
        data_info: Dictionary with test data statistics
    """
    # Get split (will use cached version if available)
    X_train, X_test, y_train, y_test, _ = get_or_create_split(
        X=X,
        y=y,
        num_clients=num_clients,
        alpha=alpha,
        data_fraction=data_fraction,
        test_size=test_size,
        seed=seed
    )

    # Create test dataset
    test_dataset = ECGDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    data_info = {
        'test_samples': len(test_dataset),
        'test_normal': int(np.sum(y_test == 0)),
        'test_abnormal': int(np.sum(y_test == 1)),
        'input_dim': X.shape[1]
    }

    return test_loader, data_info


# Example usage and testing
if __name__ == "__main__":
    print("Testing ECG Data Loader...")

    # Create dummy ECG data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 140

    X_dummy = np.random.randn(n_samples, n_features).astype(np.float32)
    y_dummy = np.random.randint(0, 2, n_samples).astype(np.int64)

    print(f"Dummy data shape: X={X_dummy.shape}, y={y_dummy.shape}")
    print(f"Label distribution: {np.bincount(y_dummy)}\n")

    # Test with 3 clients
    num_clients = 3

    for client_id in range(num_clients):
        train_loader, test_loader, info = get_ecg_loaders(
            X=X_dummy,
            y=y_dummy,
            client_id=client_id,
            num_clients=num_clients,
            batch_size_train=32,
            batch_size_test=32,
            alpha=0.5,
            num_workers=0  # Set to 0 for testing
        )

        # Test iteration
        for batch_X, batch_y in train_loader:
            print(f"Client {client_id} - Batch shape: X={batch_X.shape}, y={batch_y.shape}")
            break

    print("\n✓ Data loader test completed successfully!")