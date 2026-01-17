import sys
import io

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
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import fedlearn as fl
from models import CnnNet
from init_model import get_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flwr_datasets import FederatedDataset
from datasets import load_dataset
import torchvision.transforms as transforms
import os
import requests
from config import DATASET_CONFIGS
from data import load_server_test_data

try:
    base_url = os.environ['AWS_HOST']
    print(f"Host environment variable: {base_url}")
except KeyError:
    base_url = "localhost"
    print("Base url environment variable not found setting to local host.")



# ==============================================================================
# Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FedLearn gRPC Server with Heartbeat for a Project")
    parser.add_argument("--model-path", type=str, required=True, help="Path to initial model weights (.npz)")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients per round")
    parser.add_argument("--model-type", type=str.upper, required=True, choices=['CNN', 'TRANSFORMER'], help="Model type")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (e.g., facebook/opt-125m)")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Aggregation strategy (currently only FedAvg)")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2"], help="Dataset for LLM evaluation")
    args = parser.parse_args()

    logging.info(f"--- Starting gRPC FedLearn Server for Project: {args.project_id} ---")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Server is configured to use device: {DEVICE}")

    # Display configuration
    is_llm = args.model_type == 'TRANSFORMER'
    if is_llm:
        config = DATASET_CONFIGS[args.dataset]
        print(f"\n{'='*60}")
        print(f"Federated Learning Server - {args.dataset.upper()} Dataset")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Num rounds: {args.num_rounds}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Local epochs (K): {config.local_epochs}")
        print(f"  Num classes: {config.num_classes}")
        print(f"  Min clients: {args.min_clients}")
        print(f"  Model: {args.model_name}")
        print(f"{'='*60}\n")

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

        # Compare to target for LLM datasets
        if is_llm:
            if args.dataset == "cb":
                target = 75.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")
            elif args.dataset == "sst2":
                target = 85.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")

        print(f"{'='*60}\n")

        # Clear GPU memory after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, {"accuracy": accuracy}

    # Create strategy
    if args.strategy.lower() != 'fedavg':
        logging.warning(f"Your framework currently only supports FedAvg. Ignoring '{args.strategy}'.")

    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=server_side_evaluate,
        min_fit_clients=args.min_clients
    )

    # Start gRPC server
    server_address = f"0.0.0.0:{args.port}"
    logging.info(f"Starting FedLearn gRPC server on {server_address}...")

    history, final_parameters = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        # project_id=args.project_id
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