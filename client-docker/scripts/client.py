from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class ClientLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        client_id = self.extra.get('client_id', 'BOOTING')
        return f"[ClientID: {client_id}] - {msg}", kwargs


base_logger = logging.getLogger("FedLearn-EdgeClient")
logger = ClientLogAdapter(base_logger, {'client_id': 'BOOTING'})

import time as _time

logger.info("=" * 60)
logger.info("FedLearn Client — Starting up...")
logger.info(f"Python: {sys.version}")
logger.info("=" * 60)

_t0 = _time.time()

logger.info("[BOOT] Importing argparse, os...")
import argparse
import os

# sklearn MUST be imported before torch on ARM64 to resolve libgomp static-TLS
# allocation issues. Also used transitively via data_loaders.ecg_loader.
logger.info("[BOOT] Pre-loading sklearn (ARM64 libgomp TLS fix)...")
import sklearn  # noqa: F401
logger.info(f"[BOOT] ✓ sklearn {sklearn.__version__}")

logger.info(f"[BOOT] Importing torch... (this can take 1-3 min on Jetson)")
import torch
logger.info(f"[BOOT] ✓ torch {torch.__version__} loaded in {_time.time()-_t0:.1f}s | CUDA: {torch.cuda.is_available()}")

import time
import threading

logger.info("[BOOT] Importing numpy, pandas...")
import numpy as np
import pandas as pd
logger.info(f"[BOOT] ✓ numpy {np.__version__}, pandas {pd.__version__}")

from collections import OrderedDict
from typing import List, Tuple
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

logger.info("[BOOT] Importing psutil...")
import psutil

logger.info("[BOOT] Importing fedlearn...")
import fedlearn as fl
from fedlearn.client import DeComFLClient
logger.info("[BOOT] ✓ fedlearn + DeComFLClient loaded")

logger.info("[BOOT] Importing flwr_datasets...")
from flwr_datasets import FederatedDataset
logger.info("[BOOT] ✓ flwr_datasets loaded")

logger.info("[BOOT] Importing torchvision...")
import torchvision.transforms as transforms
logger.info("[BOOT] ✓ torchvision loaded")

logger.info("[BOOT] Importing HuggingFace datasets...")
from datasets import load_dataset
logger.info("[BOOT] ✓ datasets loaded")

logger.info("[BOOT] Importing transformers (slowest on ARM)...")
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
logger.info(f"[BOOT] ✓ transformers loaded")

logger.info("[BOOT] Importing models...")
from models import CnnNet
logger.info(f"[BOOT] ✓ All imports complete in {_time.time()-_t0:.1f}s")


try:
    import pynvml
    logger.info(f'[BOOT] ✓ pynvml {pynvml.__version__}')
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    PYNVML_AVAILABLE = False


utilization_log = []

# These are overridden at runtime by --model-type CLI arg inside main().
# Default to CNN so the early boot log line is accurate for the common case.
USE_LLM = False
USE_MLP = False

# --- Configuration ---
NUM_PARTITIONS = 10
BATCH_SIZE = 32 if not USE_LLM else 1
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
MODEL_NAME = "facebook/opt-125m"

# Dataset configurations (matching DeComFL paper)
DATASET_CONFIGS = {
    "cb": {
        "num_classes": 3,
        "learning_rate": 2e-6,
        "local_epochs": 1,
        "max_length": 256,
        "dataset_key": "super_glue",
        "dataset_name": "cb",
        "text_column": "premise",
        "text2_column": "hypothesis",
        "label_column": "label",
    },
    "sst2": {
        "num_classes": 2,
        "learning_rate": 5e-5,
        "local_epochs": 1,
        "max_length": 128,
        "dataset_key": "glue",
        "dataset_name": "sst2",
        "text_column": "sentence",
        "text2_column": None,
        "label_column": "label",
    }
}

# Training hyperparameters
LLM_WEIGHT_DECAY = 0.01
LLM_MAX_GRAD_NORM = 1.0
LLM_WARMUP_RATIO = 0.1
CNN_LEARNING_RATE = 1e-3
MLP_LEARNING_RATE = 1e-3

DATASET_NAME = "sst2"

logger.info(f"Client operating on {DEVICE}")
logger.info(
    f"--- RUNNING EXPERIMENT: "
    f"{'LLM (OPT-125M)' if USE_LLM else 'MLP (ECG)' if USE_MLP else 'CNN (CIFAR-10)'} ---"
)


# ==============================================================================
# --- Memory Logging ---
# ==============================================================================
def log_processing_usage(step_tag=""):
    process = psutil.Process()
    cpu_ram = process.memory_info().rss / 1024**2

    timestamp = time.time()

    entry = {
        "timestamp": timestamp,
        "step": step_tag,
        "cpu_ram_mb": cpu_ram,
        "gpu_alloc_mb": None,
        "gpu_reserved_mb": None,
        "gpu_util_percent": None,
    }

    if torch.cuda.is_available():
        entry["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1024**2
        entry["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2

        if PYNVML_AVAILABLE:
            util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
            entry["gpu_util_percent"] = util.gpu

    elif DEVICE == "mps":
        try:
            entry["gpu_alloc_mb"] = torch.mps.current_allocated_memory() / 1024**2
            # MPS does not expose reserved memory or utilization %
            entry["gpu_reserved_mb"] = None
            entry["gpu_util_percent"] = None
        except Exception:
            pass  # MPS memory API may not exist on older PyTorch builds

    utilization_log.append(entry)

    logger.info(
        f"[Usage] {step_tag} CPU RAM {cpu_ram:.2f} MB "
        f"GPU alloc {entry['gpu_alloc_mb']} MB "
        f"GPU util {entry['gpu_util_percent']}"
    )


# ==============================================================================
# --- ECG Data Loading ---
# ==============================================================================
def load_ecg_data_from_csv(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ECG data from CSV; last column is label, rest are features."""
    logger.info(f"Loading ECG data from {dataset_path}")
    df = pd.read_csv(dataset_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    logger.info(f"Loaded ECG data: X shape={X.shape}, y shape={y.shape}")
    return X, y


# ==============================================================================
# --- Data and Training Logic ---
# ==============================================================================
def dirichlet_split(labels, num_clients, alpha=1.0, seed=42):
    """Split data using Dirichlet distribution for non-IID data."""
    np.random.seed(seed)
    num_classes = len(np.unique(labels))

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

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def load_data(partition_id: int, dataset_name: str, dataset_path: str = None, num_clients: int = 10):
    """Load data with Dirichlet split (LLM/CNN path; ECG uses get_ecg_loaders)."""
    if USE_LLM:
        from pathlib import Path
        import pickle
        from torch.utils.data import Subset

        config = DATASET_CONFIGS[dataset_name]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if dataset_name == "cb":
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")
        else:
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")

        logger.info(f"Dataset {dataset_name} loaded: {len(raw_dataset)} samples")

        def tokenize_function(examples):
            if config["text2_column"]:
                tokenized = tokenizer(
                    examples[config["text_column"]],
                    examples[config["text2_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"],
                )
            else:
                tokenized = tokenizer(
                    examples[config["text_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"],
                )
            tokenized['labels'] = examples[config["label_column"]]
            return tokenized

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_dataset.column_names,
        ).with_format("torch")

        labels = np.array(raw_dataset[config["label_column"]])

        cache_dir = Path("./data_splits")
        cache_dir.mkdir(exist_ok=True)

        alpha = 1.0
        split_file = cache_dir / f"{dataset_name}_clients{num_clients}_alpha{alpha}.pkl"

        if split_file.exists():
            logger.info(f"Loading existing split from {split_file}")
            with open(split_file, 'rb') as f:
                client_indices_list = pickle.load(f)
        else:
            logger.info(f"Creating new Dirichlet split (alpha={alpha})")
            client_indices_list = dirichlet_split(labels, num_clients, alpha, seed=42)
            with open(split_file, 'wb') as f:
                pickle.dump(client_indices_list, f)

            for i, indices in enumerate(client_indices_list):
                client_labels = labels[indices]
                dist = np.bincount(client_labels, minlength=config["num_classes"])
                logger.info(f"  Client {i}: {len(indices)} samples - Class dist: {dist}")

        client_indices = client_indices_list[partition_id]
        client_dataset = Subset(tokenized_dataset, client_indices)

        client_labels = labels[client_indices]
        dist = np.bincount(client_labels, minlength=config["num_classes"])
        logger.info(f"Client {partition_id} label distribution: {dist}")

        train_loader = DataLoader(
            client_dataset,
            batch_size=config.get("batch_size_train", 8),
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            client_dataset,
            batch_size=config.get("batch_size_test", 8),
            shuffle=False,
            num_workers=0,
        )
        return train_loader, test_loader
    else:
        # CNN: CIFAR-10
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_PARTITIONS})
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        return (
            DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, num_workers=0),
        )


def train(net, trainloader, epochs: int, dataset_name: str, progress_callback=None):
    """Train the model with dataset-specific hyperparameters."""
    if USE_LLM:
        config = DATASET_CONFIGS[dataset_name]
        learning_rate = config["learning_rate"]
    elif USE_MLP:
        from config import get_dataset_config
        config = get_dataset_config("ecg")
        learning_rate = config.learning_rate
    else:
        learning_rate = CNN_LEARNING_RATE

    logger.info(f"[Training] Dataset: {dataset_name} | LR: {learning_rate} | Epochs: {epochs}")
    logger.info(f"[Training] Batches: {len(trainloader)} | Batch size: {trainloader.batch_size}")
    logger.info(f"[Training] Device: {DEVICE} | Model: {'LLM' if USE_LLM else 'MLP' if USE_MLP else 'CNN'}")

    if USE_LLM:
        if DEVICE == "cpu":
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
            logger.info(f"[Training] AdamW (CPU mode) LR={learning_rate:.2e}")
        else:
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=LLM_WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        logger.info("[Training] Optimizer: Adam")

    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    total_steps = len(trainloader) * epochs
    current_step = 0

    logger.info(f"[Training] Starting {total_steps} steps for {epochs} epoch(s)")

    first_epoch_params = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        if epoch == 0:
            first_epoch_params = {name: param.clone().detach() for name, param in net.named_parameters()}

        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()

            if USE_LLM:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = net(**batch)
                loss = outputs.loss
            elif USE_MLP:
                features, labels = batch
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = net(features)
                loss = criterion(outputs, labels)
            else:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("[Training] Invalid loss (NaN/Inf), skipping batch")
                continue
            if loss.item() > 100.0:
                logger.warning(f"[Training] Extremely high loss ({loss.item():.2f}), skipping batch")
                continue

            loss.backward()

            if current_step % 10 == 0:
                log_processing_usage(f"batch {current_step}")

            optimizer.step()

            current_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if progress_callback:
                progress_callback(current_step, total_steps)

            if current_step % 10 == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"[Training] Epoch {epoch+1}/{epochs}, "
                    f"Step {current_step}/{total_steps}: "
                    f"Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}, LR = {current_lr:.2e}"
                )

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        logger.info(f"[Training] Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f} ({epoch_steps} batches)")

        if epoch == epochs - 1 and first_epoch_params is not None:
            param_changes = []
            for name, param in net.named_parameters():
                if name in first_epoch_params:
                    diff = (param - first_epoch_params[name]).abs().mean().item()
                    param_changes.append(diff)
            if param_changes:
                avg_change = sum(param_changes) / len(param_changes)
                max_change = max(param_changes)
                logger.info(f"[Training] Param change avg={avg_change:.6e}, max={max_change:.6e}")
                if avg_change < 1e-10:
                    logger.warning("[Training] Parameters barely changed!")

    log_processing_usage("after training finished")


# ==============================================================================
# --- Custom Client Class for FedLearn with Heartbeat Support ---
# ==============================================================================
class ZOSLClient(fl.Client):
    def __init__(
        self,
        partition_id: int,
        dataset_name: str = "sst2",
        dataset_path: str = None,
        num_clients: int = 10,
    ):
        self.partition_id = partition_id
        self.dataset_name = dataset_name
        self.grpc_client = None

        if USE_MLP:
            from models.ecg_mlp import ECGModel
            from config import get_dataset_config

            config = get_dataset_config("ecg")
            self.net = ECGModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes,
            ).to(DEVICE)
            logger.info(f"Loaded ECG MLP model ({config.num_classes} classes)")

            # Load ECG data via get_ecg_loaders
            from data_loaders.ecg_loader import get_ecg_loaders
            X, y = load_ecg_data_from_csv(dataset_path)
            self.trainloader, self.valloader, _ = get_ecg_loaders(
                X=X,
                y=y,
                client_id=partition_id,
                num_clients=num_clients,
                batch_size_train=config.batch_size_train,
                batch_size_test=config.batch_size_test,
                data_fraction=config.data_fraction,
                alpha=config.alpha,
                test_size=config.test_size,
                num_workers=0,
                seed=config.seed,
            )
        elif USE_LLM:
            config = DATASET_CONFIGS[dataset_name]
            self.net = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=config["num_classes"],
                use_safetensors=True,
            )
            self.net.to(DEVICE)
            logger.info(f"Loaded {MODEL_NAME} for {dataset_name} ({config['num_classes']} classes)")

            self.trainloader, self.valloader = load_data(
                partition_id=self.partition_id,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                num_clients=num_clients,
            )
        else:
            self.net = CnnNet().to(DEVICE)
            logger.info("Loaded CNN for CIFAR-10")

            self.trainloader, self.valloader = load_data(
                partition_id=self.partition_id,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                num_clients=num_clients,
            )

        log_processing_usage("after model init")
        logger.info(f"Data loaded successfully for client {partition_id}.")

    def set_grpc_client(self, grpc_client):
        self.grpc_client = grpc_client
        logger.info("[Client] gRPC client configured for heartbeat updates.")

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.net.state_dict()

    def fit(
        self,
        parameters: OrderedDict[str, torch.Tensor],
        config: dict,
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        parameters = OrderedDict({k: v.to(DEVICE) for k, v in parameters.items()})
        self.net.load_state_dict(parameters)

        if USE_LLM:
            local_epochs = config.get("local_epochs", DATASET_CONFIGS[self.dataset_name]["local_epochs"])
            logger.info(f"Dataset: {self.dataset_name} | Local epochs: {local_epochs}")
        elif USE_MLP:
            from config import get_dataset_config
            ecg_config = get_dataset_config("ecg")
            local_epochs = config.get("local_epochs", ecg_config.local_epochs)
        else:
            local_epochs = config.get("local_epochs", 1)

        def progress_callback(current_step, total_steps):
            if self.grpc_client:
                self.grpc_client.update_status("training", current_step, total_steps)

        import gc
        gc.collect()

        train(
            self.net,
            self.trainloader,
            epochs=local_epochs,
            dataset_name=self.dataset_name,
            progress_callback=progress_callback,
        )

        log_processing_usage("after training finished")
        return self.net.state_dict(), len(self.trainloader.dataset)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_DATASET_PATH = os.path.join(SCRIPT_DIR, "ecg_data", "ecg.csv")
ECG_NUM_CLIENTS = 5
ECG_STRATEGY = "DeComFL"


def create_decomfl_compatible_loader(original_loader, is_llm=False):
    """Wrap a DataLoader to make it compatible with DeComFLClient (LLM path only)."""
    if not is_llm:
        return original_loader

    class LLMBatchWrapper:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                labels = batch.pop('labels')
                inputs = batch
                yield inputs, labels

        def __len__(self):
            return len(self.loader)

        @property
        def dataset(self):
            return self.loader.dataset

    return LLMBatchWrapper(original_loader)


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
def main():
    global USE_LLM, USE_MLP, DATASET_NAME, BATCH_SIZE, MODEL_NAME, logger

    parser = argparse.ArgumentParser(description="FedLearn gRPC Client with Heartbeat")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--server-address", type=str, required=True, help="gRPC server address")
    parser.add_argument("--partition-id", type=int, required=True,
                        choices=range(0, NUM_PARTITIONS), help="Client partition ID")
    parser.add_argument("--model-type", type=str, choices=["CNN", "TRANSFORMER", "MLP"], help="Model type")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, default="cb",
                        choices=["cb", "sst2", "ecg"], help="Dataset")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="FL strategy (FedAvg or DeComFL)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM (deprecated, use --model-type TRANSFORMER)")

    args = parser.parse_args()

    if args.model_name:
        MODEL_NAME = args.model_name

    # Rebind logger with per-client ID
    logger = ClientLogAdapter(
        base_logger,
        {'client_id': f'project_{args.project_id}_client_{args.partition_id}'},
    )

    # Determine model type
    if args.model_type:
        USE_LLM = (args.model_type.upper() == "TRANSFORMER")
        USE_MLP = (args.model_type.upper() == "MLP")
    elif args.use_llm:
        USE_LLM = True
        USE_MLP = False
    else:
        USE_LLM = False
        USE_MLP = False

    # === HARDCODED ECG/MLP OVERRIDE ===
    if USE_MLP:
        args.dataset = "ecg"
        dataset_path = ECG_DATASET_PATH
        num_clients = ECG_NUM_CLIENTS
        args.strategy = ECG_STRATEGY
        logger.info(f"MLP detected - dataset={args.dataset} path={dataset_path} "
                    f"num_clients={num_clients} strategy={args.strategy}")
    else:
        dataset_path = None
        num_clients = 2

    DATASET_NAME = args.dataset

    if USE_LLM:
        config = DATASET_CONFIGS[args.dataset]
        BATCH_SIZE = config.get("batch_size_train", 8)
    elif USE_MLP:
        from config import get_dataset_config
        config = get_dataset_config("ecg")
        BATCH_SIZE = config.batch_size_train
    else:
        BATCH_SIZE = 32

    logger.info("=" * 60)
    logger.info("Starting FedLearn Client")
    logger.info("=" * 60)
    logger.info(f"  Project ID: {args.project_id}")
    logger.info(f"  Partition ID: {args.partition_id}")
    logger.info(f"  Model Type: {args.model_type or ('LLM' if USE_LLM else 'CNN')}")
    logger.info(f"  Model Name: {args.model_name or MODEL_NAME}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Dataset: {args.dataset.upper()}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    if USE_LLM:
        logger.info(f"  Num classes: {DATASET_CONFIGS[args.dataset]['num_classes']}")
        logger.info(f"  Learning rate: {DATASET_CONFIGS[args.dataset]['learning_rate']}")
        logger.info(f"  Local epochs: {DATASET_CONFIGS[args.dataset]['local_epochs']}")
    elif USE_MLP:
        from config import get_dataset_config
        ecg_cfg = get_dataset_config("ecg")
        logger.info(f"  Num classes: {ecg_cfg.num_classes}")
        logger.info(f"  Input dim: {ecg_cfg.input_dim}")
        logger.info(f"  Learning rate: {ecg_cfg.learning_rate}")
        logger.info(f"  Local epochs: {ecg_cfg.local_epochs}")
        logger.info(f"  Dataset path: {dataset_path}")
        logger.info(f"  Total clients: {num_clients}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Server: {args.server_address}")
    logger.info("=" * 60)

    client_id = f"project_{args.project_id}_client_{args.partition_id}"

    if args.strategy.lower() == 'decomfl':
        logger.info("Using DeComFL client from framework")
        from config import get_decomfl_config

        if USE_MLP:
            from models.ecg_mlp import ECGModel
            from config import get_dataset_config

            ecg_config = get_dataset_config("ecg")
            decomfl_config = get_decomfl_config("ecg")

            net = ECGModel(
                input_dim=ecg_config.input_dim,
                hidden_dim=ecg_config.hidden_dim,
                num_classes=ecg_config.num_classes,
            ).to(DEVICE)

            X, y = load_ecg_data_from_csv(dataset_path)
            from data_loaders.ecg_loader import get_ecg_loaders
            trainloader, _, _ = get_ecg_loaders(
                X=X,
                y=y,
                client_id=args.partition_id,
                num_clients=num_clients,
                batch_size_train=ecg_config.batch_size_train,
                batch_size_test=ecg_config.batch_size_test,
                data_fraction=ecg_config.data_fraction,
                alpha=ecg_config.alpha,
                test_size=ecg_config.test_size,
                num_workers=0,
                seed=ecg_config.seed,
            )
        elif USE_LLM:
            config = DATASET_CONFIGS[args.dataset]
            decomfl_config = get_decomfl_config("default")

            net = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=config["num_classes"],
                use_safetensors=True,
            ).to(DEVICE)

            trainloader_original, _ = load_data(
                partition_id=args.partition_id,
                dataset_name=args.dataset,
                dataset_path=None,
                num_clients=num_clients,
            )
            trainloader = create_decomfl_compatible_loader(trainloader_original, is_llm=True)
        else:
            decomfl_config = get_decomfl_config("default")
            net = CnnNet().to(DEVICE)
            trainloader, _ = load_data(
                partition_id=args.partition_id,
                dataset_name=args.dataset,
                dataset_path=None,
                num_clients=num_clients,
            )

        client = DeComFLClient(
            model=net,
            train_loader=trainloader,
            smoothing_param=decomfl_config.smoothing_param,
            device=DEVICE,
        )

        logger.info(f"Connecting to gRPC server at {args.server_address}...")

        try:
            fl.client.start_decomfl_client(
                server_address=args.server_address,
                client=client,
                client_id=client_id,
            )
        except KeyboardInterrupt:
            logger.info(f"[{client_id}] Interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"[{client_id}] Error: {e}")
            raise
        finally:
            logger.info(f"[{client_id}] Client disconnected.")
    else:
        logger.info("Using FedAvg client (standard parameter download)")

        client = ZOSLClient(
            partition_id=args.partition_id,
            dataset_name=args.dataset,
            dataset_path=dataset_path,
            num_clients=num_clients,
        )

        logger.info(f"Connecting to gRPC server at {args.server_address}...")

        try:
            fl.client.start_client(
                server_address=args.server_address,
                client=client,
                client_id=client_id,
            )
        except KeyboardInterrupt:
            logger.info(f"[{client_id}] Interrupted by user. Shutting down...")
        except Exception as e:
            logger.error(f"[{client_id}] Error: {e}")
            raise
        finally:
            logger.info("=== Utilization Summary ===")
            logger.info(
                f"{'Step':25} {'CPU RAM (MB)':15} {'GPU Alloc (MB)':15} "
                f"{'GPU Reserved (MB)':18} {'GPU Util (%)':12}"
            )
            for entry in utilization_log:
                logger.info(
                    f"{entry['step']:25}"
                    f"{entry['cpu_ram_mb']:<15.2f}"
                    f"{(entry['gpu_alloc_mb'] or 0):<15.2f}"
                    f"{(entry['gpu_reserved_mb'] or 0):<18.2f}"
                    f"{(entry['gpu_util_percent'] or 0):<12}"
                )
            logger.info(f"[{client_id}] Client disconnected.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for PyInstaller on macOS (spawn method)
    main()
