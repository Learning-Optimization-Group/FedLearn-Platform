import sys
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import logging
import time
import psutil
import gc
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import fedlearn as fl
from fedlearn.server import DeComFL  # Import DeComFL strategy from framework
from models import CnnNet
import sys
sys.path.insert(0, os.path.dirname(__file__))
from init_model import get_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flwr_datasets import FederatedDataset
from datasets import load_dataset
import torchvision.transforms as transforms
import os
import requests
from config import DATASET_CONFIGS, get_dataset_config, get_decomfl_config
from data import load_server_test_data

try:
    base_url = os.environ['AWS_HOST']
    print(f"Host environment variable: {base_url}")
except KeyError:
    base_url = "localhost"
    print("Base url environment variable not found setting to local host.")


# ==============================================================================
# Helper Functions
# ==============================================================================
def load_ecg_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG data from CSV file.

    Args:
        dataset_path: Path to ECG CSV file

    Returns:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
    """
    logging.info(f"Loading ECG data from {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ECG dataset not found at: {dataset_path}")

    # Load CSV
    df = pd.read_csv(dataset_path, header=None)

    # Last column is label, rest are features
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    logging.info(f"Loaded ECG data: X shape={X.shape}, y shape={y.shape}")
    logging.info(f"Label distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

    return X, y


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_DATASET_PATH = os.path.join(SCRIPT_DIR, "ecg_data", "ecg.csv")  # Hardcoded ECG dataset path
ECG_DATASET_NAME = "ecg"
ECG_NUM_CLIENTS = 3  # Hardcoded number of clients for ECG
ECG_STRATEGY = "DeComFL"  # Hardcoded strategy for MLP


# ==============================================================================
# Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FedLearn gRPC Server with Heartbeat for a Project")
    parser.add_argument("--model-path", type=str, required=True, help="Path to initial model weights (.npz)")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--model-type", type=str.upper, required=True, choices=['CNN', 'TRANSFORMER', 'MLP'], help="Model type")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Aggregation strategy")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2", "ecg"], help="Dataset")
    args = parser.parse_args()

    # if args.model_type == 'TRANSFORMER' and args.strategy.lower() == 'decomfl':
    #     args.min_clients = 1
    # print(f"[OVERRIDE] Setting min_clients to 1 for LLM+DeComFL testing")

    is_mlp = args.model_type == 'MLP'

    if is_mlp:
        # Override with hardcoded ECG values
        args.dataset = ECG_DATASET_NAME
        args.strategy = ECG_STRATEGY
        dataset_path = ECG_DATASET_PATH
        num_clients = ECG_NUM_CLIENTS

        print(f"\n{'='*60}")
        print(f"MLP MODEL DETECTED - Using Hardcoded ECG Configuration")
        print(f"{'='*60}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Num clients: {num_clients}")
        print(f"{'='*60}\n")
    else:
        dataset_path = None
        num_clients = None

    logging.info(f"--- Starting gRPC FedLearn Server for Project: {args.project_id} ---")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Server is configured to use device: {DEVICE}")

    # Determine if LLM or MLP
    is_llm = args.model_type == 'TRANSFORMER'
    is_mlp = args.model_type == 'MLP'

    # Get dataset configuration
    if is_llm or is_mlp:
        config = get_dataset_config(args.dataset)
        print(f"\n{'='*60}")
        print(f"Federated Learning Server - {args.dataset.upper()} Dataset")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Num rounds: {args.num_rounds}")

        if is_mlp:
            print(f"  Input dim: {config.input_dim}")
            print(f"  Hidden dim: {config.hidden_dim}")
            print(f"  Num classes: {config.num_classes}")
            print(f"  Batch size: {config.batch_size_train}")
        else:
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Num classes: {config.num_classes}")

        print(f"  Local epochs (K): {config.local_epochs}")
        print(f"  Min clients: {args.min_clients}")
        print(f"  Model: {args.model_name}")
        print(f"{'='*60}\n")

    # Validate ECG dataset path
    # if is_mlp and args.dataset == "ecg":
    #     if not args.dataset_path:
    #         logging.error("--dataset-path is required for ECG dataset")
    #         exit(1)

    # Load model architecture
    net = get_model(args.model_type, args.model_name, DEVICE)

    # Load initial parameters
    initial_parameters = OrderedDict()
    try:
        if not os.path.exists(args.model_path):
            logging.error(f"Model path not found: {args.model_path}")
            exit(1)

        with np.load(args.model_path, allow_pickle=False) as npzfile:
            for key in npzfile.files:
                value = npzfile[key]
                if isinstance(value, np.ndarray):
                    original_key = key.replace('__DOT__', '.')
                    initial_parameters[original_key] = torch.from_numpy(value)
                else:
                    logging.warning(f"Skipping invalid key {key} of type {type(value)}")

        if not initial_parameters:
            logging.error(f"No valid model parameters found in {args.model_path}")
            exit(1)

        logging.info("Model parameters loaded successfully with correct layer names.")

    except Exception as e:
        logging.error(f"Failed to load model parameters from {args.model_path}. Reason: {e}", exc_info=True)
        exit(1)

    # Load test data for server-side evaluation
    if is_mlp and args.dataset == "ecg":
        # Load ECG test data
        from data_loaders.ecg_loader import get_test_loader

        X, y = load_ecg_data(dataset_path)

        # Get number of clients from args or config
        num_clients =  config.num_clients

        test_loader, test_info = get_test_loader(
            X=X,
            y=y,
            num_clients=num_clients,
            batch_size=config.batch_size_test,
            alpha=config.alpha,
            data_fraction=config.data_fraction,
            test_size=config.test_size,
            num_workers=0,  # Set to 0 for server
            seed=config.seed
        )

        logging.info(f"Loaded ECG test data: {test_info['test_samples']} samples")
    else:
        # Load CIFAR-10 or LLM test data
        test_loader = load_server_test_data(is_llm, args.dataset if is_llm else None)

    # Define server-side evaluation function
    def server_side_evaluate(server_round: int, parameters: OrderedDict[str, torch.Tensor]) -> tuple[float, dict]:
        """
        Evaluate the aggregated model on the server's test dataset.
        """
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Server-side Evaluation")
        print(f"{'='*60}")

        # Clear GPU cache before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load parameters into model
        net.load_state_dict(parameters, strict=True)
        net.to(DEVICE)
        net.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_loader:
                if is_llm:
                    # LLM: batch is a dict with input_ids, attention_mask, labels
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    outputs = net(**batch)
                    loss = outputs.loss
                    logits = outputs.logits
                    labels = batch["labels"]
                elif is_mlp:
                    # MLP: batch is (features, labels) tuple
                    features, labels = batch
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = net(features)
                    loss = criterion(outputs, labels)
                    logits = outputs
                else:
                    # CNN: batch is a dict with 'img' and 'label'
                    images = batch["img"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    logits = outputs

                total_loss += loss.item()
                num_batches += 1

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # Average loss per batch (not per sample)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        print(f"Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

        # Compare to target for different datasets
        if is_llm:
            if args.dataset == "cb":
                target = 75.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")
            elif args.dataset == "sst2":
                target = 85.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")
        elif is_mlp and args.dataset == "ecg":
            target = 80.0
            status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
            print(f"  Target (DeComFL): {target:.2f}% {status}")

        print(f"{'='*60}\n")

        # Clear GPU memory after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, {"accuracy": accuracy}

    # Create strategy based on user selection
    if args.strategy.lower() == 'decomfl':
        logging.info("Using DeComFL strategy from framework")

        # Get DeComFL config (use 'ecg' config for ECG dataset, otherwise 'default')
        decomfl_config = get_decomfl_config('ecg' if args.dataset == 'ecg' else 'default')

        strategy = DeComFL(
            initial_parameters=initial_parameters,
            evaluate_fn=server_side_evaluate,
            min_fit_clients=args.min_clients,
            num_local_steps=decomfl_config.num_local_steps,
            num_perturbations=decomfl_config.num_perturbations,
            learning_rate=decomfl_config.learning_rate,
            smoothing_param=decomfl_config.smoothing_param,
            seed=decomfl_config.seed
        )

        logging.info(f"DeComFL initialized with: K={decomfl_config.num_local_steps}, "
                     f"P={decomfl_config.num_perturbations}, "
                     f"η={decomfl_config.learning_rate}, "
                     f"μ={decomfl_config.smoothing_param}")
    else:
        if args.strategy.lower() != 'fedavg':
            logging.warning(f"Strategy '{args.strategy}' not recognized. Defaulting to FedAvg.")

        strategy = fl.FedAvg(
            initial_parameters=initial_parameters,
            evaluate_fn=server_side_evaluate,
            min_fit_clients=args.min_clients
        )

        logging.info("Using FedAvg strategy")

    # Start gRPC server
    server_address = f"0.0.0.0:{args.port}"
    logging.info(f"Starting FedLearn gRPC server on {server_address}...")

    history, final_parameters = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    logging.info("--- Federated Learning session complete. ---")

    # Print training summary
    if history:
        print("\n" + "="*60)
        print(" " * 20 + "Training Summary")
        print("="*60)

        if history and history[0][1]:
            first_round_metrics = history[0][1]
            metric_keys = sorted(first_round_metrics.keys())

            # Print header
            header = f"| {'Round':<5} |"
            for key in metric_keys:
                header += f" {key.capitalize():<12} |"
            print(header)
            print(f"|{'-'*7}|" + f"{'-'*14}|" * len(metric_keys))

            # Print rows
            for r, metrics in history:
                row = f"| {r:<5} |"
                for key in metric_keys:
                    value = metrics.get(key, 'N/A')
                    if isinstance(value, float):
                        row += f" {value:<12.6f} |"
                    else:
                        row += f" {str(value):<12} |"
                print(row)

            print("="*60)

            # Print final results
            if history:
                final_round, final_metrics = history[-1]
                final_accuracy = final_metrics.get('accuracy', 0.0)
                print(f"\nFinal Results (Round {final_round}):")
                print(f"  Accuracy: {final_accuracy:.2f}%")

                if is_llm:
                    if args.dataset == "cb":
                        target = 75.0
                        status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                        print(f"  Target: {target:.2f}% {status}")
                    elif args.dataset == "sst2":
                        target = 85.0
                        status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                        print(f"  Target: {target:.2f}% {status}")
                elif is_mlp and args.dataset == "ecg":
                    target = 80.0
                    status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                    print(f"  Target: {target:.2f}% {status}")

    # Save final model
    if final_parameters:
        logging.info("--- Saving final global model to .npz format... ---")
        save_path = args.model_path

        params_to_save = {
            key.replace('.', '__DOT__'): tensor.cpu().numpy()
            for key, tensor in final_parameters.items()
        }

        try:
            np.savez(save_path, **params_to_save)
            logging.info(f"Final model weights successfully saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save final model to {save_path}. Reason: {e}", exc_info=True)
    else:
        logging.warning("--- No final model parameters to save. ---")

    # Mark Project as completed
    project_complete_url = "http://"+base_url+":8081"+"/api/projects/"+args.project_id+"/stop"

    try:
        response = requests.post(project_complete_url)
        response.raise_for_status()
        print(f"POST request successful. Status Code: {response.status_code}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutdown requested by user.")
    except Exception as e:
        logging.critical("An unhandled exception occurred in the main function.", exc_info=True)
        exit(1)