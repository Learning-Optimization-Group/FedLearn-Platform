
import os
import sklearn  # Pre-load sklearn to fix ARM64 static TLS block memory allocation issues with libgomp

import argparse
import torch
import time
import threading
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import psutil
import fedlearn as fl
from fedlearn.client import DeComFLClient  # Import DeComFL client from framework

from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from models import CnnNet


try:
    import pynvml
    print('pynvml - ',pynvml.__version__)
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    PYNVML_AVAILABLE = False


utilization_log = []

# --- !! MANUAL FLAG TO SWITCH BETWEEN MODELS !! ---
USE_LLM = True
USE_MLP = False  # NEW: Flag for MLP/ECG
# --------------------------------------------------

# --- Configuration ---
NUM_PARTITIONS = 10
BATCH_SIZE = 32
# TODO: Use torch.backends.mps.is_available() for Mac (MPS) and torch.cuda.is_available() for Nvidia (CUDA) directly in the future instead of hardcoding CPU.
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
MODEL_NAME = "facebook/opt-125m"

# Dataset configurations (matching DeComFL paper)
DATASET_CONFIGS = {
    "cb": {
        "num_classes": 3,
        "learning_rate": 2e-6,
        "local_epochs": 1,  # K=1 as per paper
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
        "local_epochs": 1,  # K=1 as per paper
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
LLM_MAX_GRAD_NORM = 1.0  # Standard value for transformers
LLM_WARMUP_RATIO = 0.1  # 10% of steps for warmup
CNN_LEARNING_RATE = 1e-3
MLP_LEARNING_RATE = 1e-3  # Higher LR for zeroth-order optimization

# Global dataset selection (will be set via argparse)
DATASET_NAME = "sst2"

print(f"Client operating on {DEVICE}")
print(f"--- RUNNING EXPERIMENT: {'LLM (OPT-125M)' if USE_LLM else 'MLP (ECG)' if USE_MLP else 'CNN (CIFAR-10)'} ---")


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

    utilization_log.append(entry)

    # Optional live print
    print(f"[Usage] {step_tag} CPU RAM {cpu_ram:.2f} MB "
          f"GPU alloc {entry['gpu_alloc_mb']} MB "
          f"GPU util {entry['gpu_util_percent']}")


# ==============================================================================
# --- ECG Data Loading ---
# ==============================================================================
def load_ecg_data_from_csv(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG data from CSV file.

    Args:
        dataset_path: Path to ECG CSV file

    Returns:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
    """
    print(f"Loading ECG data from {dataset_path}")

    # Load CSV
    df = pd.read_csv(dataset_path, header=None)

    # Last column is label, rest are features
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    print(f"Loaded ECG data: X shape={X.shape}, y shape={y.shape}")
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
    """Load data with Dirichlet split."""
    if USE_LLM:
        from pathlib import Path
        import pickle
        from torch.utils.data import Subset

        config = DATASET_CONFIGS[dataset_name]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load dataset
        if dataset_name == "cb":
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")
        else:
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")

        print(f"Dataset {dataset_name} loaded: {len(raw_dataset)} samples")

        # Tokenize
        def tokenize_function(examples):
            if config["text2_column"]:
                tokenized = tokenizer(
                    examples[config["text_column"]],
                    examples[config["text2_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"]
                )
            else:
                tokenized = tokenizer(
                    examples[config["text_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"]
                )
            tokenized['labels'] = examples[config["label_column"]]
            return tokenized

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_dataset.column_names
        ).with_format("torch")

        # Get or create Dirichlet split
        labels = np.array(raw_dataset[config["label_column"]])

        cache_dir = Path("./data_splits")
        cache_dir.mkdir(exist_ok=True)

        alpha = 1.0
        split_file = cache_dir / f"{dataset_name}_clients{num_clients}_alpha{alpha}.pkl"

        if split_file.exists():
            print(f"Loading existing split from {split_file}")
            with open(split_file, 'rb') as f:
                client_indices_list = pickle.load(f)
        else:
            print(f"Creating new Dirichlet split (alpha={alpha})")
            client_indices_list = dirichlet_split(labels, num_clients, alpha, seed=42)
            with open(split_file, 'wb') as f:
                pickle.dump(client_indices_list, f)

            # Print distribution
            for i, indices in enumerate(client_indices_list):
                client_labels = labels[indices]
                dist = np.bincount(client_labels, minlength=3)
                print(f"  Client {i}: {len(indices)} samples - Class dist: {dist}")

        # Get this client's data
        client_indices = client_indices_list[partition_id]
        client_dataset = Subset(tokenized_dataset, client_indices)

        # Print client's label distribution
        client_labels = labels[client_indices]
        dist = np.bincount(client_labels, minlength=3)
        print(f"Client {partition_id} label distribution:")
        print(f"  Class 0: {dist[0]} ({dist[0]/len(client_indices)*100:.1f}%)")
        print(f"  Class 1: {dist[1]} ({dist[1]/len(client_indices)*100:.1f}%)")
        print(f"  Class 2: {dist[2]} ({dist[2]/len(client_indices)*100:.1f}%)")

        train_loader = DataLoader(
            client_dataset,
            batch_size=config.get("batch_size_train", 8),
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            client_dataset,
            batch_size=config.get("batch_size_test", 8),
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader
    else:
        # CNN: CIFAR-10 (unchanged)
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_PARTITIONS})
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        return (
            DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE, num_workers=0)
        )


def train(net, trainloader, epochs: int, dataset_name: str, progress_callback=None):
    """
    Train the model with dataset-specific hyperparameters.

    Args:
        net: Neural network model
        trainloader: Training data loader
        epochs: Number of epochs (K=1 for LLM as per paper)
        dataset_name: "cb" or "sst2" for LLM, "ecg" for MLP
        progress_callback: Optional callback function(current_step, total_steps)
    """
    # Get dataset-specific configuration
    if USE_LLM:
        config = DATASET_CONFIGS[dataset_name]
        learning_rate = config["learning_rate"]
    elif USE_MLP:
        from config import get_dataset_config
        config = get_dataset_config("ecg")
        learning_rate = config.learning_rate
    else:
        learning_rate = CNN_LEARNING_RATE

    print(f"\n{'='*60}")
    print(f"TRAINING DEBUG INFO")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Num batches: {len(trainloader)}")
    print(f"  Batch size: {trainloader.batch_size}")
    print(f"  Total samples: {len(trainloader.dataset)}")
    print(f"  Device: {DEVICE}")
    print(f"  Model type: {'LLM' if USE_LLM else 'MLP' if USE_MLP else 'CNN'}")
    print(f"{'='*60}\n")

    # Setup optimizer based on model type
    if USE_LLM:
        # Use regular Adam on CPU for better numerical stability
        if DEVICE == "cpu":
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate
                #    * 10,  # Increase LR for CPU
                # betas=(0.9, 0.999),
                # eps=1e-8
            )
            print(f"  [DEBUG] Actual optimizer LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Using AdamW (CPU mode) with LR={learning_rate:.2e}")
        else:
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=LLM_WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        print(f"  Optimizer: Adam")
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    total_steps = len(trainloader) * epochs
    current_step = 0

    # Setup learning rate scheduler for LLM
    # if USE_LLM:
    #     num_warmup_steps = int(total_steps * LLM_WARMUP_RATIO)
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=total_steps
    #     )
    #     print(f"   [Training] Using warmup for {num_warmup_steps} steps")

    print(f"   [Training] Starting {total_steps} steps for {epochs} epoch(s)...")
    print(f"   [Training] Learning rate: {learning_rate}")
    if USE_LLM or USE_MLP:
        print(f"   [Training] Dataset: {dataset_name}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i, batch in enumerate(trainloader):
            if i == 0:
                print(f"\n   [DEBUG] First batch info:")
                if USE_LLM:
                    print(f"     Batch keys: {list(batch.keys())}")
                    print(f"     Input IDs shape: {batch['input_ids'].shape}")
                    print(f"     Labels shape: {batch['labels'].shape}")
                    print(f"     Sample labels: {batch['labels'][:5]}")
                else:
                    print(f"     Batch type: {type(batch)}")
            optimizer.zero_grad()

            if USE_LLM:
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                if i == 0:
                    print(f"     Keys passed to model: {list(batch.keys())}")
                    print(f"     Input IDs device: {batch['input_ids'].device}")
                    print(f"     Labels device: {batch['labels'].device}")
                # Forward pass with labels
                outputs = net(**batch)
                loss = outputs.loss

                if i == 0:
                    print(f"     Loss: {loss.item():.4f}")
                    print(f"     Logits shape: {outputs.logits.shape}")
                    print(f"     Logits sample: {outputs.logits[0]}")
                    print(f"     Predictions: {torch.argmax(outputs.logits, dim=-1)[:5]}")
                    print(f"     True labels: {batch['labels'][:5]}")

                    print(f"\n     [DETAILED DEBUG]")
                    print(f"     Optimizer type: {type(optimizer).__name__}")
                    print(f"     Optimizer LR: {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"     Model device: {next(net.parameters()).device}")
                    print(f"     Input device: {batch['input_ids'].device}")
                    print(f"     Model training mode: {net.training}")
            elif USE_MLP:
                # MLP: batch is (features, labels) tuple
                features, labels = batch
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = net(features)
                loss = criterion(outputs, labels)
            else:
                # CNN
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)

            # Check for NaN or exploding loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   [Training] WARNING: Invalid loss detected (NaN/Inf), skipping batch")
                continue

            if loss.item() > 100.0:
                print(f"   [Training] WARNING: Extremely high loss ({loss.item():.2f}), skipping batch")
                continue

            loss.backward()

            if current_step % 10 == 0:
                log_processing_usage(f"batch {current_step}")

            if i == 0 and USE_LLM:
                total_norm = 0
                for p in net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"     Gradient norm: {total_norm:.4f}")

                print(f"     Gradient clipping: {'YES (max_norm=1.0)' if False else 'NO'}")  # We removed clipping
                print(f"\n     [WEIGHT UPDATE CHECK]")
                # Get a reference weight before optimizer.step()
                ref_weight_before = net.model.decoder.final_layer_norm.weight.clone()
            # Gradient clipping for LLM to prevent explosion
            # if USE_LLM:
            #
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), LLM_MAX_GRAD_NORM)

            optimizer.step()
            if i == 0 and USE_LLM:
                ref_weight_after = net.model.decoder.final_layer_norm.weight
                weight_change = (ref_weight_after - ref_weight_before).abs().mean().item()
                print(f"     Weight change after step: {weight_change:.6e}")
                print(f"     LR × grad_norm = {optimizer.param_groups[0]['lr'] * total_norm:.6e} (expected change magnitude)")
                # === END ADD ===

            # Step scheduler for LLM
            # if USE_LLM:
            #     scheduler.step()

            current_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            # Update progress for heartbeat
            if progress_callback:
                progress_callback(current_step, total_steps)

            # Log progress every 10 steps
            if (current_step % 10) == 0:
                avg_loss = epoch_loss / epoch_steps
                # current_lr = scheduler.get_last_lr()[0] if USE_LLM else optimizer.param_groups[0]['lr']
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   [Training] Epoch {epoch+1}/{epochs}, "
                      f"Step {current_step}/{total_steps}: "
                      f"Loss = {loss.item():.4f}, "
                      f"Avg Loss = {avg_loss:.4f}, "
                      f"LR = {current_lr:.2e}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(f"   [Training] Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        print(f"   [Training] Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        print(f"   [Training] Total batches processed: {epoch_steps}")

# ==============================================================================
# --- Custom Client Class for FedLearn with Heartbeat Support ---
# ==============================================================================
class ZOSLClient(fl.Client):
    def __init__(self, partition_id: int, dataset_name: str = "sst2", dataset_path: str = None, num_clients: int = 10):
        self.partition_id = partition_id
        self.dataset_name = dataset_name
        self.grpc_client = None  # Will be set by start_client

        if USE_MLP:
            # MLP: ECG model
            from models.ecg_mlp import ECGModel
            from config import get_dataset_config

            config = get_dataset_config("ecg")
            self.net = ECGModel(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes
            ).to(DEVICE)
            print(f"Loaded ECG MLP model ({config.num_classes} classes)")
        elif USE_LLM:
            config = DATASET_CONFIGS[dataset_name]
            self.net = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=config["num_classes"],
                use_safetensors=True
            )
            self.net.to(DEVICE)
            print(f"Loaded {MODEL_NAME} for {dataset_name} ({config['num_classes']} classes)")
        else:
            self.net = CnnNet().to(DEVICE)
            print("Loaded CNN for CIFAR-10")

        log_processing_usage("after model init")

        self.trainloader, self.valloader = load_data(
            partition_id=self.partition_id,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_clients=num_clients
        )
        print(f"Data loaded successfully for client {partition_id}.")

    def set_grpc_client(self, grpc_client):
        """Set the gRPC client for heartbeat updates."""
        self.grpc_client = grpc_client
        print(f"[Client] gRPC client configured for heartbeat updates.")

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.net.state_dict()

    def fit(
            self,
            parameters: OrderedDict[str, torch.Tensor],
            config: dict
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        # Load parameters

        server_round = config.get("server_round", 0)

        if server_round == 1:
            print(f"\n{'='*60}")
            print(f"CLIENT RECEIVED PARAMETERS - ROUND 1")
            print(f"{'='*60}")
            print(f"Total parameters: {len(parameters)}")

            if USE_LLM:
                if 'score.weight' in parameters:
                    print(f"✅ score.weight: {parameters['score.weight'].shape}")
                    if parameters['score.weight'].shape[0] != 3:
                        print(f"❌ WRONG! Expected [3, 768], got {parameters['score.weight'].shape}")
                    else:
                        print(f"✅ Correct shape [3, 768]")
                else:
                    print(f"❌ score.weight MISSING!")

                # Bias might not exist
                if 'score.bias' in parameters:
                    print(f"✅ score.bias: {parameters['score.bias'].shape}")
                else:
                    print(f"ℹ️  score.bias not present (expected for OPT)")

            print(f"{'='*60}\n")

        parameters = OrderedDict({k: v.to(DEVICE) for k, v in parameters.items()})

        initial_params = {k: v.clone() for k, v in parameters.items()}
        print(f"\n[FIT DEBUG] Initial parameter stats:")
        for name, param in list(parameters.items())[:3]:  # First 3 layers
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")


        self.net.load_state_dict(parameters)

        # RIGHT AFTER: self.net.load_state_dict(parameters)

        # Verify parameters were actually loaded
        if USE_LLM:
            loaded_embed = self.net.model.decoder.embed_tokens.weight
            loaded_score = self.net.score.weight

            print(f"\n[VERIFY LOAD] After load_state_dict:")
            print(f"  embed_tokens mean: {loaded_embed.mean().item():.6f}")
            print(f"  embed_tokens std: {loaded_embed.std().item():.6f}")
            print(f"  score.weight mean: {loaded_score.mean().item():.6f}")
            print(f"  score.weight std: {loaded_score.std().item():.6f}")

            # Compare with what was sent
            if 'model.decoder.embed_tokens.weight' in parameters:
                sent_embed = parameters['model.decoder.embed_tokens.weight']
                print(f"\n[VERIFY LOAD] What was sent by server:")
                print(f"  embed_tokens mean: {sent_embed.mean().item():.6f}")
                print(f"  embed_tokens std: {sent_embed.std().item():.6f}")

                if torch.allclose(loaded_embed, sent_embed):
                    print(f"  ✅ MATCH: Parameters loaded correctly")
                else:
                    print(f"  ❌ MISMATCH: Parameters NOT loaded correctly!")

        # Get local epochs from config or use dataset default
        if USE_LLM:
            local_epochs = config.get("local_epochs", DATASET_CONFIGS[self.dataset_name]["local_epochs"])
            print(f'Dataset: {self.dataset_name}')
            print(f'Local epochs: {local_epochs}')
            print(f'Batch size: {self.trainloader.batch_size}')
            print(f'Num batches: {len(self.trainloader)}')
        elif USE_MLP:
            from config import get_dataset_config
            ecg_config = get_dataset_config("ecg")
            local_epochs = config.get("local_epochs", ecg_config.local_epochs)
        else:
            local_epochs = config.get("local_epochs", 1)

        # Define progress callback to update heartbeat status
        def progress_callback(current_step, total_steps):
            if self.grpc_client:
                self.grpc_client.update_status("training", current_step, total_steps)

        import gc
        gc.collect()

        # Train with progress updates
        train(
            self.net,
            self.trainloader,
            epochs=local_epochs,
            dataset_name=self.dataset_name,
            progress_callback=progress_callback
        )

        # DEBUG: Check if parameters changed
        final_params = self.net.state_dict()
        print(f"\n[FIT DEBUG] Final parameter stats:")
        for name, param in list(final_params.items())[:3]:  # First 3 layers
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

        print(f"\n[FIT DEBUG] Parameter changes:")
        total_change = 0
        num_params = 0
        for name in list(initial_params.keys())[:3]:
            change = (final_params[name] - initial_params[name]).abs().mean().item()
            print(f"  {name}: avg change = {change:.6e}")
            total_change += change
            num_params += 1
        print(f"  Average parameter change: {total_change/num_params:.6e}")

        if total_change/num_params < 1e-8:
            print(f"  ⚠️ WARNING: Parameters barely changed! Model may not be training!")

        # Logging after training
        log_processing_usage("after training finished")

        # Right before: return self.net.state_dict(), len(self.trainloader.dataset)

        # === ADD THIS ===
        print(f"\n[JAVA CLIENT DEBUG] Training complete for round {server_round}")
        print(f"  Total batches trained: {len(self.trainloader)}")

        # Check if model actually learned
        if USE_LLM:
            final_score_weight = self.net.score.weight
            print(f"  Final score.weight mean: {final_score_weight.mean().item():.6f}")
            print(f"  Final score.weight std: {final_score_weight.std().item():.6f}")

            # Most importantly - did the model's predictions change?
            self.net.eval()
            with torch.no_grad():
                first_batch = next(iter(self.trainloader))
                batch = {k: v.to(DEVICE) for k, v in first_batch.items()}
                outputs = self.net(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                print(f"  Sample predictions after training: {predictions[:8]}")
                print(f"  Sample labels: {batch['labels'][:8]}")
                accuracy_on_batch = (predictions == batch['labels']).float().mean().item()*100
                print(f"  Accuracy on first training batch: {accuracy_on_batch:.2f}%")
            self.net.train()
        # === END ADD ===

        return self.net.state_dict(), len(self.trainloader.dataset)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_DATASET_PATH = os.path.join(SCRIPT_DIR, "ecg_data", "ecg.csv")  # Hardcoded ECG dataset path
ECG_NUM_CLIENTS = 5  # Hardcoded number of clients for ECG
ECG_STRATEGY = "DeComFL"  # Hardcoded strategy for MLP


def create_decomfl_compatible_loader(original_loader, is_llm=False):
    """
    Wrap a DataLoader to make it compatible with DeComFLClient.

    For LLM: Converts dict batches to (inputs_dict, labels) tuples
    For other models: Returns as-is
    """
    if not is_llm:
        return original_loader

    class LLMBatchWrapper:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                # Extract labels and create inputs dict
                labels = batch.pop('labels')
                inputs = batch  # Contains input_ids, attention_mask, etc.
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
    global USE_LLM, USE_MLP, DATASET_NAME, BATCH_SIZE

    print(f"\n{'='*60}")
    print(f"DEVICE DETECTION")
    print(f"{'='*60}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Selected device: {DEVICE}")
    print(f"{'='*60}\n")
    parser = argparse.ArgumentParser(description="FedLearn gRPC Client with Heartbeat")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--server-address", type=str, required=True, help="gRPC server address")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, NUM_PARTITIONS), help="Client partition ID")
    parser.add_argument("--model-type", type=str, choices=["CNN", "TRANSFORMER", "MLP"], help="Model type")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2", "ecg"], help="Dataset")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="FL strategy (FedAvg or DeComFL)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM (deprecated, use --model-type TRANSFORMER)")

    args = parser.parse_args()

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
        args.strategy = ECG_STRATEGY  # Override strategy for MLP

        print(f"\n{'='*60}")
        print(f"MLP MODEL DETECTED - Using Hardcoded ECG Configuration")
        print(f"{'='*60}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Num clients: {num_clients}")
        print(f"  Strategy: {args.strategy}")
        print(f"{'='*60}\n")
    else:
        dataset_path = None
        num_clients = 2  # Default for CNN/LLM

    # === END HARDCODED OVERRIDE ===

    DATASET_NAME = args.dataset

    # Set batch size based on model type
    if USE_LLM:
        config = DATASET_CONFIGS[args.dataset]
        BATCH_SIZE = config.get("batch_size_train", 8)  # Use config value, default 8
        print(f"  Batch size: {BATCH_SIZE}")
    elif USE_MLP:
        from config import get_dataset_config
        config = get_dataset_config("ecg")
        BATCH_SIZE = config.batch_size_train
    else:
        BATCH_SIZE = 32

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Starting FedLearn Client")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Project ID: {args.project_id}")
    print(f"  Partition ID: {args.partition_id}")
    print(f"  Model Type: {args.model_type or ('LLM' if USE_LLM else 'CNN')}")
    print(f"  Model Name: {args.model_name or MODEL_NAME}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Dataset: {args.dataset.upper()}")

    if USE_LLM:
        print(f"  Num classes: {DATASET_CONFIGS[args.dataset]['num_classes']}")
        print(f"  Learning rate: {DATASET_CONFIGS[args.dataset]['learning_rate']}")
        print(f"  Local epochs: {DATASET_CONFIGS[args.dataset]['local_epochs']}")
    elif USE_MLP:
        from config import get_dataset_config
        config = get_dataset_config("ecg")
        print(f"  Num classes: {config.num_classes}")
        print(f"  Input dim: {config.input_dim}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Local epochs: {config.local_epochs}")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Total clients: {num_clients}")

    print(f"  Device: {DEVICE}")
    print(f"  Server: {args.server_address}")
    print(f"{'='*60}\n")

    # === CREATE CLIENT BASED ON STRATEGY ===
    if args.strategy.lower() == 'decomfl':
        print("Using DeComFL client from framework")

        from config import get_decomfl_config

        if USE_MLP:
            # MLP/ECG with DeComFL
            from models.ecg_mlp import ECGModel
            from config import get_dataset_config

            ecg_config = get_dataset_config("ecg")
            decomfl_config = get_decomfl_config("ecg")

            net = ECGModel(
                input_dim=ecg_config.input_dim,
                hidden_dim=ecg_config.hidden_dim,
                num_classes=ecg_config.num_classes
            ).to(DEVICE)

            # Load data
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
                seed=ecg_config.seed
            )

        elif USE_LLM:
            # LLM with DeComFL
            config = DATASET_CONFIGS[args.dataset]
            decomfl_config = get_decomfl_config("default")

            net = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=config["num_classes"],
                use_safetensors=True
            ).to(DEVICE)

            # Load LLM data (reuse existing load_data function)
            trainloader_original, _ = load_data(
                partition_id=args.partition_id,
                dataset_name=args.dataset,
                dataset_path=None,
                num_clients=num_clients
            )

            # Wrap the loader to make it DeComFL-compatible
            trainloader = create_decomfl_compatible_loader(trainloader_original, is_llm=True)

        else:
            # CNN with DeComFL (if needed)
            decomfl_config = get_decomfl_config("default")
            net = CnnNet().to(DEVICE)

            trainloader, _ = load_data(
                partition_id=args.partition_id,
                dataset_name=args.dataset,
                dataset_path=None,
                num_clients=num_clients
            )

        # Create DeComFL client (works for all model types)
        client = DeComFLClient(
            model=net,
            train_loader=trainloader,
            smoothing_param=decomfl_config.smoothing_param,
            device=DEVICE
        )

        client_id = f"project_{args.project_id}_client_{args.partition_id}"

        print(f"Connecting to gRPC server at {args.server_address}...")

        try:
            # Start DeComFL client
            fl.client.start_decomfl_client(
                server_address=args.server_address,
                client=client,
                client_id=client_id
            )
        except KeyboardInterrupt:
            print(f"\n[{client_id}] Interrupted by user. Shutting down...")
        except Exception as e:
            print(f"[{client_id}] Error: {e}")
            raise
        finally:
            print(f"[{client_id}] Client disconnected.")

    else:
        # Use standard FedAvg client
        print("Using FedAvg client (standard parameter download)")

        client = ZOSLClient(
            partition_id=args.partition_id,
            dataset_name=args.dataset,
            dataset_path=dataset_path,
            num_clients=num_clients
        )
        client_id = f"project_{args.project_id}_client_{args.partition_id}"

        print(f"Connecting to gRPC server at {args.server_address}...")

        try:
            # Start the client
            fl.client.start_client(
                server_address=args.server_address,
                client=client,
                client_id=client_id
            )

        except KeyboardInterrupt:
            print(f"\n[{client_id}] Interrupted by user. Shutting down...")
        except Exception as e:
            print(f"[{client_id}] Error: {e}")
            raise
        finally:
            print("\n=== Utilization Summary ===")
            print(f"{'Step':25} {'CPU RAM (MB)':15} {'GPU Alloc (MB)':15} {'GPU Reserved (MB)':18} {'GPU Util (%)':12}")

            for entry in utilization_log:
                print(f"{entry['step']:25}"
                      f"{entry['cpu_ram_mb']:<15.2f}"
                      f"{(entry['gpu_alloc_mb'] or 0):<15.2f}"
                      f"{(entry['gpu_reserved_mb'] or 0):<18.2f}"
                      f"{(entry['gpu_util_percent'] or 0):<12}")

            print(f"[{client_id}] Client disconnected.")


if __name__ == "__main__":
    main()
