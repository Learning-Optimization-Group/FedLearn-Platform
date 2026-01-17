import argparse
import torch
import time
import threading
import numpy as np
from collections import OrderedDict
from typing import List, Tuple
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import psutil
import fedlearn as fl

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
# --------------------------------------------------

# --- Configuration ---
NUM_PARTITIONS = 10
BATCH_SIZE = 32 if not USE_LLM else 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/opt-125m"

# Dataset configurations (matching DeComFL paper)
DATASET_CONFIGS = {
    "cb": {
        "num_classes": 3,
        "learning_rate": 1e-6,
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
        "learning_rate": 1e-5,
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
LLM_WEIGHT_DECAY = 0.1
LLM_MAX_GRAD_NORM = 1.0  # Standard value for transformers
LLM_WARMUP_RATIO = 0.1  # 10% of steps for warmup
CNN_LEARNING_RATE = 1e-3

# Global dataset selection (will be set via argparse)
DATASET_NAME = "sst2"

print(f"Client operating on {DEVICE}")
print(f"--- RUNNING EXPERIMENT: {'LLM (OPT-125M)' if USE_LLM else 'CNN (CIFAR-10)'} ---")


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
# --- Data and Training Logic ---
# ==============================================================================
def load_data(partition_id: int, dataset_name: str):
    """
    Load data for either CNN (CIFAR-10) or LLM (CB/SST-2).

    Args:
        partition_id: Client partition ID
        dataset_name: "cb" or "sst2" for LLM, ignored for CNN
    """
    if USE_LLM:
        config = DATASET_CONFIGS[dataset_name]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load dataset based on configuration
        if dataset_name == "cb":
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")
        else:  # sst2
            raw_dataset = load_dataset(config["dataset_key"], config["dataset_name"], split="train")

        print(f"Dataset {dataset_name} loaded: {len(raw_dataset)} samples")

        # Partition data across clients
        num_samples = len(raw_dataset)
        samples_per_client = num_samples // NUM_PARTITIONS
        start = partition_id * samples_per_client
        end = (partition_id + 1) * samples_per_client if partition_id < NUM_PARTITIONS - 1 else num_samples
        partition = raw_dataset.select(range(start, end))

        print(f"Client {partition_id} partition: {len(partition)} samples (indices {start}-{end})")

        def tokenize_function(examples):
            """Tokenize based on dataset type (single or pair)."""
            if config["text2_column"]:  # CB: premise + hypothesis
                return tokenizer(
                    examples[config["text_column"]],
                    examples[config["text2_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"]
                )
            else:  # SST-2: single sentence
                return tokenizer(
                    examples[config["text_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"]
                )

        # Tokenize and preserve labels
        tokenized_partition = partition.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=[col for col in partition.column_names if col != config["label_column"]]
        ).rename_column(config["label_column"], "labels").with_format("torch")

        # Split into train/test
        trainloader = DataLoader(tokenized_partition, shuffle=True, batch_size=BATCH_SIZE)
        return trainloader, trainloader
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
        dataset_name: "cb" or "sst2" for LLM
        progress_callback: Optional callback function(current_step, total_steps)
    """

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"   [Training] Trainable params: {trainable:,}")
    print(f"   [Training] Frozen params: {frozen:,}")

    if trainable == 0:
        raise ValueError("ERROR: All model parameters are frozen!")

    # Get dataset-specific configuration
    if USE_LLM:
        config = DATASET_CONFIGS[dataset_name]
        learning_rate = config["learning_rate"]
    else:
        learning_rate = CNN_LEARNING_RATE

    # Setup optimizer based on model type
    if USE_LLM:
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=learning_rate,
            weight_decay=LLM_WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    total_steps = len(trainloader) * epochs
    current_step = 0

    # Setup learning rate scheduler for LLM
    if USE_LLM:
        num_warmup_steps = int(total_steps * LLM_WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        print(f"   [Training] Using warmup for {num_warmup_steps} steps")

    print(f"   [Training] Starting {total_steps} steps for {epochs} epoch(s)...")
    print(f"   [Training] Learning rate: {learning_rate}")
    if USE_LLM:
        print(f"   [Training] Dataset: {dataset_name}")

    # Save initial parameters (OUTSIDE epoch loop)
    first_epoch_params = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Save parameters at START of first epoch
        if epoch == 0:
            first_epoch_params = {name: param.clone().detach() for name, param in net.named_parameters()}
            print(f"   [Training] Saved initial parameters for comparison")

        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()

            if USE_LLM:
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                # Forward pass with labels
                outputs = net(**batch)
                loss = outputs.loss

                if current_step == 0:  # First batch only
                    print(f"   [DEBUG] First batch check:")
                    print(f"   [DEBUG] Loss raw value: {loss}")
                    print(f"   [DEBUG] Loss item: {loss.item()}")
                    print(f"   [DEBUG] Loss requires_grad: {loss.requires_grad}")
                    print(f"   [DEBUG] Batch has labels: {'labels' in batch}")
                    print(
                        f"   [DEBUG] Model trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
            else:
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

            # Gradient clipping for LLM to prevent explosion
            if USE_LLM:
                torch.nn.utils.clip_grad_norm_(net.parameters(), LLM_MAX_GRAD_NORM)

            optimizer.step()

            # Step scheduler for LLM
            if USE_LLM:
                scheduler.step()

            current_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            # Update progress for heartbeat
            if progress_callback:
                progress_callback(current_step, total_steps)

            # Log progress every 10 steps
            if (current_step % 10) == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0] if USE_LLM else optimizer.param_groups[0]['lr']
                print(f"   [Training] Epoch {epoch + 1}/{epochs}, "
                      f"Step {current_step}/{total_steps}: "
                      f"Loss = {loss.item():.6f}, "
                      f"Avg Loss = {avg_loss:.6f}, "
                      f"LR = {current_lr:.2e}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(f"   [Training] Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.6f}")

        # Compare parameters at END of last epoch
        if epoch == epochs - 1 and first_epoch_params is not None:
            param_changes = []
            for name, param in net.named_parameters():
                if name in first_epoch_params:
                    diff = (param - first_epoch_params[name]).abs().mean().item()
                    param_changes.append(diff)

            if param_changes:
                avg_change = sum(param_changes) / len(param_changes)
                max_change = max(param_changes)
                print(f"   [Training] Parameter changes from start:")
                print(f"   [Training]   Average change: {avg_change:.6e}")
                print(f"   [Training]   Maximum change: {max_change:.6e}")

                if avg_change < 1e-10:
                    print("   [Training] ⚠️  WARNING: Parameters barely changed!")
                else:
                    print(f"   [Training] ✓ Parameters updated successfully")

    log_processing_usage("after training finished")


# ==============================================================================
# --- Custom Client Class for FedLearn with Heartbeat Support ---
# ==============================================================================
class ZOSLClient(fl.Client):
    def __init__(self, partition_id: int, dataset_name: str = "sst2"):
        self.partition_id = partition_id
        self.dataset_name = dataset_name
        self.grpc_client = None  # Will be set by start_client

        if USE_LLM:
            config = DATASET_CONFIGS[dataset_name]
            self.net = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=config["num_classes"],
                use_safetensors=True
            )
            self.net.to(DEVICE)
            print(f"Loaded {MODEL_NAME} for {dataset_name} ({config['num_classes']} classes)")
            self.trainloader, _ = load_data(
                partition_id=self.partition_id,
                dataset_name=dataset_name
            )
        else:
            self.net = CnnNet().to(DEVICE)
            print("Loaded CNN for CIFAR-10")

            self.trainloader, self.valloader = load_data(
                partition_id=self.partition_id,
                dataset_name=dataset_name
            )

        log_processing_usage("after model init")


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
        parameters = OrderedDict({k: v.to(DEVICE) for k, v in parameters.items()})
        self.net.load_state_dict(parameters)

        # Get local epochs from config or use dataset default
        if USE_LLM:
            local_epochs = config.get("local_epochs", DATASET_CONFIGS[self.dataset_name]["local_epochs"])
            print('self.dataset_name - ',self.dataset_name)
            print('Client Class epochs - ',local_epochs)
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

        # Logging after training
        log_processing_usage("after training finished")

        return self.net.state_dict(), len(self.trainloader.dataset)


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
def main():
    global USE_LLM, DATASET_NAME, BATCH_SIZE

    parser = argparse.ArgumentParser(description="FedLearn gRPC Client with Heartbeat")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--server-address", type=str, required=True, help="gRPC server address (e.g., localhost:50051)")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, NUM_PARTITIONS), help="Client partition ID")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM model instead of CNN")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2"], help="Dataset for LLM (cb or sst2)")
    args = parser.parse_args()

    # Update global configuration
    USE_LLM = args.use_llm
    DATASET_NAME = args.dataset
    BATCH_SIZE = 1 if USE_LLM else 32

    print(f"\n{'='*60}")
    print(f"Starting FedLearn Client")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Project ID: {args.project_id}")
    print(f"  Partition ID: {args.partition_id}")
    print(f"  Model: {'LLM (OPT-125M)' if USE_LLM else 'CNN (CIFAR-10)'}")
    if USE_LLM:
        print(f"  Dataset: {args.dataset.upper()}")
        print(f"  Num classes: {DATASET_CONFIGS[args.dataset]['num_classes']}")
        print(f"  Learning rate: {DATASET_CONFIGS[args.dataset]['learning_rate']}")
        print(f"  Local epochs: {DATASET_CONFIGS[args.dataset]['local_epochs']}")
    print(f"  Device: {DEVICE}")
    print(f"  Server: {args.server_address}")
    print(f"{'='*60}\n")

    client = ZOSLClient(partition_id=args.partition_id, dataset_name=args.dataset)
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