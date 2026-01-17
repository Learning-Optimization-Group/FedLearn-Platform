# =============================================================================
# data.py - Configurable data loading for CB and SST-2
# =============================================================================

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import pickle
import os
from pathlib import Path
from config import DATASET_CONFIGS, MODEL_NAME


def dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float = 1.0, seed: int = 42):
    """
    Split data indices using Dirichlet distribution for non-IID partitioning.

    Args:
        labels: Array of labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)  # Set seed for reproducibility
    num_classes = len(np.unique(labels))

    # Dirichlet distribution
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = label_distribution[k]
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_splits = np.split(idx_k, splits)

        for i, idx in enumerate(idx_splits):
            client_indices[i].extend(idx)

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def get_or_create_split(dataset_name: str, num_clients: int, alpha: float, data_fraction: float, labels: np.ndarray):
    """
    Get existing data split or create a new one.
    This ensures all clients use the SAME split.

    Args:
        dataset_name: Name of dataset
        num_clients: Total number of clients
        alpha: Dirichlet parameter
        data_fraction: Fraction of data used
        labels: Label array

    Returns:
        client_indices_list: List of indices for each client
    """
    # Create cache directory
    cache_dir = Path("./data_splits")
    cache_dir.mkdir(exist_ok=True)

    # Create unique filename based on configuration
    split_file = cache_dir / f"{dataset_name}_clients{num_clients}_alpha{alpha}_frac{data_fraction}.pkl"

    if split_file.exists():
        # Load existing split
        print(f"Loading existing data split from {split_file}")
        with open(split_file, 'rb') as f:
            client_indices_list = pickle.load(f)
    else:
        # Create new split
        print(f"Creating new data split and saving to {split_file}")
        client_indices_list = dirichlet_split(labels, num_clients, alpha, seed=42)

        # Save the split
        with open(split_file, 'wb') as f:
            pickle.dump(client_indices_list, f)

        # Print distribution
        print(f"\nData distribution across {num_clients} clients:")
        for i, indices in enumerate(client_indices_list):
            print(f"  Client {i}: {len(indices)} samples")

    return client_indices_list


def get_llm_loaders(
        dataset_name: str,
        client_id: int,
        num_clients: int,
        data_fraction: float = 1.0,
        data_path: str = "./data",
):
    """
    Prepares tokenized dataset partition for a given client.

    Args:
        dataset_name: "cb" or "sst2"
        client_id: Client ID (0 to num_clients-1)
        num_clients: Total number of clients
        data_fraction: Fraction of data to use (1.0 = full dataset, 0.1 = 10%)
        data_path: Path to cache datasets

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        config: Dataset configuration object
    """
    # Get configuration
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the dataset
    if config.dataset_config:
        raw_train = load_dataset(config.dataset_name, config.dataset_config, split=config.train_split)
        raw_test = load_dataset(config.dataset_name, config.dataset_config, split=config.test_split)
    else:
        raw_train = load_dataset(config.dataset_name, split=config.train_split)
        raw_test = load_dataset(config.dataset_name, split=config.test_split)

    # 3. Use fraction of data if specified
    if data_fraction < 1.0:
        num_samples = int(len(raw_train) * data_fraction)
        raw_train = raw_train.select(range(num_samples))
        print(f"Using {data_fraction * 100}% of training data: {num_samples} samples")

    # 4. Define tokenization function based on text column type
    def tokenize_function(examples):
        text_col = config.text_column

        if isinstance(text_col, tuple):
            # For datasets like CB with premise and hypothesis
            tokenized = tokenizer(
                examples[text_col[0]],
                examples[text_col[1]],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        else:
            # For datasets like SST-2 with single text column
            tokenized = tokenizer(
                examples[text_col],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )


        tokenized['labels'] = examples[config.label_column]
        return tokenized

    # 5. Tokenize datasets
    tokenized_train = raw_train.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_train.column_names
    )
    tokenized_train.set_format("torch")

    # 6. Get or create consistent data split (FIXED!)
    # Extract labels for splitting
    labels = np.array(raw_train[config.label_column])

    # Get the SAME split for all clients
    client_indices_list = get_or_create_split(
        dataset_name=dataset_name,
        num_clients=num_clients,
        alpha=config.alpha,
        data_fraction=data_fraction,
        labels=labels
    )

    # Get this client's indices from the consistent split
    client_indices = client_indices_list[client_id]
    client_dataset = Subset(tokenized_train, client_indices)

    print(f"Client {client_id}: {len(client_dataset)} samples")

    # 7. Create training DataLoader
    train_loader = DataLoader(
        client_dataset,
        batch_size=config.batch_size_train,
        shuffle=True
    )

    # 8. Create test DataLoader (full test set for server evaluation)
    tokenized_test = raw_test.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_test.column_names
    )
    tokenized_test.set_format("torch")

    test_loader = DataLoader(
        tokenized_test,
        batch_size=config.batch_size_test,
        shuffle=False
    )

    return train_loader, test_loader, config


def get_test_loader(dataset_name: str, data_fraction: float = 1.0):
    """Load test set without any client partitioning."""
    config = DATASET_CONFIGS[dataset_name]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test set only
    if config.dataset_config:
        raw_test = load_dataset(config.dataset_name, config.dataset_config, split=config.test_split)
    else:
        raw_test = load_dataset(config.dataset_name, split=config.test_split)

    # Tokenize
    def tokenize_function(examples):
        text_col = config.text_column
        if isinstance(text_col, tuple):
            tokenized = tokenizer(
                examples[text_col[0]], examples[text_col[1]],
                truncation=True, max_length=config.max_length, padding="max_length"
            )
        else:
            tokenized = tokenizer(
                examples[text_col],
                truncation=True, max_length=config.max_length, padding="max_length"
            )
        tokenized['labels'] = examples[config.label_column]
        return tokenized

    tokenized_test = raw_test.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_test.column_names
    )
    tokenized_test.set_format("torch")

    test_loader = DataLoader(
        tokenized_test,
        batch_size=config.batch_size_test,
        shuffle=False
    )

    return test_loader, config