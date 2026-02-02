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

        # Get config object
        config = get_dataset_config(dataset_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        test_dataset = dataset[config.test_split]

        # Tokenize based on dataset type
        if dataset_name == "cb":
            def tokenize_function(examples):
                return tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",  # CHANGED: Use max_length like working version
                    max_length=config.max_length
                )
        elif dataset_name == "sst2":
            def tokenize_function(examples):
                return tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",  # CHANGED: Use max_length like working version
                    max_length=config.max_length
                )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Apply tokenization
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        # Rename label to labels
        tokenized_test = tokenized_test.rename_column("label", "labels")

        # Set format to torch WITHOUT DataCollatorWithPadding
        tokenized_test.set_format("torch")


        return DataLoader(
            tokenized_test,
            batch_size=config.batch_size_test,
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