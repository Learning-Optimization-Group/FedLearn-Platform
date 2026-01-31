from dataclasses import dataclass
from typing import Literal
from typing import Literal, Optional, Tuple, Union

@dataclass
class DatasetConfig:
    """Configuration for different datasets"""

    name: str
    dataset_name: str
    dataset_config: Optional[str]
    text_column: Union[str, Tuple[str, str]]
    label_column: str
    num_classes: int
    max_length: int
    learning_rate: float
    num_rounds: int
    batch_size_train: int
    batch_size_test: int
    train_split: str
    test_split: str
    num_clients: int
    clients_per_round: int
    alpha: float
    local_epochs: int


@dataclass
class ECGDatasetConfig:
    """Configuration specifically for ECG dataset"""

    name: str
    num_classes: int
    input_dim: int
    hidden_dim: int
    learning_rate: float
    num_rounds: int
    batch_size_train: int
    batch_size_test: int
    test_size: float  # Train/test split ratio
    num_clients: int
    clients_per_round: int
    alpha: float  # Dirichlet concentration parameter
    local_epochs: int
    data_fraction: float  # Fraction of data to use
    seed: int


@dataclass
class DeComFLConfig:
    """Configuration for DeComFL algorithm hyperparameters"""

    num_local_steps: int  # K - number of local gradient steps
    num_perturbations: int  # P - number of random perturbations
    learning_rate: float  # η - server learning rate
    smoothing_param: float  # μ - smoothing parameter for ZO estimation
    seed: int  # Random seed for reproducibility


DATASET_CONFIGS = {
    "cb": DatasetConfig(
        name="cb",
        dataset_name="super_glue",
        dataset_config="cb",
        text_column=("premise", "hypothesis"),  # CB has premise and hypothesis
        label_column="label",
        num_classes=3,  # entailment, contradiction, neutral
        max_length=256,  # CB may need longer sequences
        learning_rate=2e-6,
        num_rounds=1000,  # From Table 5
        batch_size_train=8,  # Small batch for small dataset
        batch_size_test=8,
        train_split="train",
        test_split="validation",
        num_clients=1,
        clients_per_round=1,
        alpha=1.0,
        local_epochs=1,  # K=1
    ),
    "sst2": DatasetConfig(
        name="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        text_column="sentence",
        label_column="label",
        num_classes=2,  # binary sentiment classification
        max_length=128,
        learning_rate=2e-6,
        num_rounds=50,  # From Table 5 (can reduce to 500-1000 for quick experiments)
        batch_size_train=32,  # From appendix C.1
        batch_size_test=64,
        train_split="train",
        test_split="validation",  # SST-2 uses validation as test
        num_clients=2,
        clients_per_round=2,
        alpha=1.0,
        local_epochs=1,  # K=1
    ),
    "ecg": ECGDatasetConfig(
        name="ecg",
        num_classes=2,  # Binary classification (Normal/Abnormal)
        input_dim=140,  # ECG feature dimension
        hidden_dim=64,  # Hidden layer dimension
        learning_rate=0.001,  # Higher LR for zeroth-order optimization
        num_rounds=100,  # Number of federated rounds
        batch_size_train=128,
        batch_size_test=128,
        test_size=0.2,  # 20% test split
        num_clients=5,  # Default number of clients
        clients_per_round=2,  # Clients participating per round
        alpha=1.0,  # Dirichlet parameter for non-IID split
        local_epochs=1,  # K=1 for DeComFL
        data_fraction=1.0,  # Use full dataset
        seed=42
    ),
}


# DeComFL Algorithm Configurations
DECOMFL_CONFIGS = {
    "default": DeComFLConfig(
        num_local_steps=1,  # K - local gradient steps
        num_perturbations=10,  # P - random perturbations
        learning_rate=0.001,  # η - server learning rate
        smoothing_param=0.001,  # μ - smoothing parameter
        seed=42
    ),
    "ecg": DeComFLConfig(
        num_local_steps=1,  # K - optimized for ECG
        num_perturbations=10,  # P - sufficient for 140-dim input
        learning_rate=0.001,  # η - works well with ECG
        smoothing_param=0.001,  # μ - balanced exploration
        seed=42
    ),
}


# Model configurations
MODEL_NAME = "facebook/opt-125m"
SMOOTH_PARAMETER_MU = 1e-3


# Helper function to get config
def get_dataset_config(dataset_name: str):
    """
    Get dataset configuration by name.

    Args:
        dataset_name: Name of the dataset (cb, sst2, ecg)

    Returns:
        Dataset configuration object

    Raises:
        ValueError: If dataset name is not found
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_decomfl_config(config_name: str = "default"):
    """
    Get DeComFL configuration by name.

    Args:
        config_name: Name of the configuration (default, ecg)

    Returns:
        DeComFL configuration object

    Raises:
        ValueError: If config name is not found
    """
    if config_name not in DECOMFL_CONFIGS:
        raise ValueError(
            f"Unknown DeComFL config: {config_name}. "
            f"Available configs: {list(DECOMFL_CONFIGS.keys())}"
        )
    return DECOMFL_CONFIGS[config_name]