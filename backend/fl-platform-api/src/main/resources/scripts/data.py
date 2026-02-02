from config import DATASET_CONFIGS
import sys
import io

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torchvision.transforms as transforms
from config import DATASET_CONFIGS

# ==============================================================================
# --- SERVER-SIDE EVALUATION HELPERS ---
# ==============================================================================
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torchvision import datasets, transforms
from config import DATASET_CONFIGS, get_dataset_config

def load_server_test_data(is_llm: bool, dataset_name: str = None):
    """Load test data for server-side evaluation."""
    if is_llm:
        if dataset_name is None:
            raise ValueError("dataset_name is required for LLM")

        # Get config object (it's a DatasetConfig dataclass, not a dict)
        config = get_dataset_config(dataset_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset - use attribute access (config.dataset_name, not config["dataset_name"])
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        test_dataset = dataset["validation"]

        # Tokenize based on dataset type
        if dataset_name == "cb":
            def tokenize_function(examples):
                return tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=128
                )
        elif dataset_name == "sst2":
            def tokenize_function(examples):
                return tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=128
                )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Apply tokenization
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        # CRITICAL: Rename 'label' to 'labels' for HuggingFace models
        tokenized_test = tokenized_test.rename_column("label", "labels")

        # Set format to include labels
        tokenized_test.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        # Create DataLoader with proper collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        return DataLoader(
            tokenized_test,
            batch_size=config.eval_batch_size,  # Use attribute access
            collate_fn=data_collator,
            shuffle=False
        )
    else:
        # CNN: Load CIFAR-10 test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        return DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False
        )