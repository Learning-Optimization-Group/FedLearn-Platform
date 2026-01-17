from dataclasses import dataclass
from typing import Literal

@dataclass
class DatasetConfig:
    """Configuration for different datasets"""

    name:str
    dataset_name:str
    dataset_config:str|None
    text_column:str|tuple[str,str]
    label_column:str
    num_classes:int
    max_length:int
    learning_rate:float
    num_rounds:int
    batch_size_train:int
    batch_size_test:int
    train_split:str
    test_split:str
    num_clients:int
    clients_per_round:int
    alpha:float # Dirichlet concentration
    local_epochs:int

DATASET_CONFIGS = {
    "cb": DatasetConfig(
        name="cb",
        dataset_name="super_glue",
        dataset_config="cb",
        text_column=("premise", "hypothesis"),  # CB has premise and hypothesis
        label_column="label",
        num_classes=3,  # entailment, contradiction, neutral
        max_length=256,  # CB may need longer sequences
        learning_rate=2e-6,  # From Table 5
        num_rounds=100,  # From Table 5
        batch_size_train=8,  # Small batch for small dataset
        batch_size_test=8,
        train_split="train",
        test_split="validation",
        num_clients=8,
        clients_per_round=2,
        alpha=1.0,
        local_epochs=1,  # K=1
    ),
    "sst2": DatasetConfig(
        name="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        text_column="sentence",
        label_column="label",
        num_classes=2,
        max_length=128,

        # Hyperparameters (same as successful CB run)
        learning_rate=2e-6,
        local_epochs=1,
        num_rounds=500,  # More than CB due to larger dataset

        # Batch configuration
        batch_size_train=32,  # Larger than CB
        batch_size_test=64,

        # Federated settings
        num_clients=2,
        clients_per_round=2,
        alpha=1.0,

        # Dataset splits
        train_split="train",
        test_split="validation",
    ),
}

MODEL_NAME = "facebook/opt-125m"
SMOOTH_PARAMETER_MU = 1e-3
