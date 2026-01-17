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
        config = DATASET_CONFIGS[dataset_name]
        MODEL_NAME = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load test split
        if dataset_name == "cb":
            raw_dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.test_split)
        else:  # sst2
            raw_dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.test_split)

        print(f"Loaded {dataset_name} test set: {len(raw_dataset)} samples")

        def tokenize_function(examples):
            """Tokenize based on dataset type (single or pair)."""
            # Check if text_column is a tuple (sentence pair) or string (single sentence)
            if isinstance(config.text_column, tuple):  # CB: premise + hypothesis
                return tokenizer(
                    examples[config.text_column[0]],  # premise
                    examples[config.text_column[1]],  # hypothesis
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_length
                )
            else:  # SST-2: single sentence
                return tokenizer(
                    examples[config.text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_length
                )

        # Tokenize and preserve labels
        tokenized_testset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_dataset.column_names
        ).with_format("torch")
        #     remove_columns=[col for col in raw_dataset.column_names if col != config.label_column]
        # ).rename_column(config.label_column, "labels").with_format("torch")

        return DataLoader(tokenized_testset, batch_size=config.batch_size_test, num_workers=0)
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