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
def load_server_test_data(is_llm: bool, dataset_name: str = "sst2"):
    """
    Load test data for server-side evaluation.

    Args:
        is_llm: Whether using LLM or CNN
        dataset_name: "cb" or "sst2" for LLM
    """
    if is_llm:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]

        # Load tokenizer
        from transformers import AutoTokenizer, DataCollatorWithPadding
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_dataset(config["dataset_name"], config.get("dataset_config"))
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

        return torch.utils.data.DataLoader(
            tokenized_test,
            batch_size=config["eval_batch_size"],
            collate_fn=data_collator,
            shuffle=False
        )
    else:
        # CNN: CIFAR-10
        testset = load_dataset("cifar10", split="test")
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        testset.set_transform(apply_transforms)
        return DataLoader(testset, batch_size=64, num_workers=0)