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

        config = get_dataset_config(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(config.dataset_name, config.dataset_config)
        test_dataset = dataset[config.test_split]

        # Tokenize
        if dataset_name == "cb":
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=config.max_length
                )
                tokenized['labels'] = examples['label']  # Add labels here
                return tokenized
        elif dataset_name == "sst2":
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=config.max_length
                )
                tokenized['labels'] = examples['label']  # Add labels here
                return tokenized
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Apply tokenization and remove original columns
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )

        # Set format - only return these tensor columns
        tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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