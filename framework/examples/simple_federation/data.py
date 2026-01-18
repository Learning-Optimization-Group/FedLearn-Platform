import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loader(client_id: int, num_clients: int, data_path: str = "./data") -> DataLoader:
    """Loads a non-IID partition of the MNIST dataset for a specific client."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)

    # Simple non-IID partitioning: give each client a slice of the dataset
    total_size = len(full_dataset)
    partition_size = total_size // num_clients
    start_idx = client_id * partition_size
    end_idx = start_idx + partition_size

    indices = list(range(start_idx, end_idx))
    client_dataset = Subset(full_dataset, indices)

    return DataLoader(client_dataset, batch_size=32, shuffle=True)


def get_test_loader(data_path: str = "./data") -> DataLoader:
    """Loads the entire MNIST test set for server-side evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=128)