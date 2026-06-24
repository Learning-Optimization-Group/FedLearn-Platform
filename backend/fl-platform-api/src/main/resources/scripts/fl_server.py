import sys
import io
import os

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
from fedlearn.server import DeComFL, FedLoRA  # Import strategies from framework
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

target_ip = os.environ.get('SERVER_HOST') or os.environ.get('AWS_HOST') or 'localhost'
base_url = target_ip  # Preserved to ensure downstream REST logs don't break
bind_address = "[::]"

# Backend URL used for service-to-service callbacks (round results, project-finished
# notifications). Prefer FEDLEARN_BACKEND_URL when set — e.g. the internal ALB or
# service-discovery DNS inside the VPC. Falls back to http://<base_url>:8081 for
# local dev where everything runs on the same host.
BACKEND_URL = os.environ.get('FEDLEARN_BACKEND_URL', f"http://{base_url}:8081").rstrip('/')

# Shared secret that gates /api/internal/** on the backend. Set by the orchestrator
# (FlowerServerManager propagates it into the Fargate task env, or the dev runner
# exports it). We deliberately do NOT default this — missing key means no callback
# will succeed, and the task should surface that loudly.
INTERNAL_API_KEY = os.environ.get('FEDLEARN_INTERNAL_API_KEY', '').strip()


def _internal_headers() -> dict:
    """Headers for /api/internal/** callbacks. Raises if no key is configured."""
    if not INTERNAL_API_KEY:
        raise RuntimeError(
            "FEDLEARN_INTERNAL_API_KEY is not set; refusing to call backend /api/internal/** "
            "without a shared secret."
        )
    return {"X-Internal-Key": INTERNAL_API_KEY, "Content-Type": "application/json"}


if os.environ.get('AWS_HOST'):
    logging.info(f"[NETWORK] Cloud deployment detected. Clients should target AWS Elastic IP: {target_ip}")
elif os.environ.get('SERVER_HOST'):
    logging.info(f"[NETWORK] LAN deployment detected. Clients should target LAN IP: {target_ip}")
else:
    logging.info(f"[NETWORK] Local environment detected. Clients should target: {target_ip}")

logging.info(f"[NETWORK] gRPC Server universally binding to: {bind_address}")
logging.info(f"[NETWORK] Backend callbacks will target: {BACKEND_URL}")


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
    parser.add_argument("--model-type", type=str.upper, required=True, choices=['CNN', 'TRANSFORMER', 'MLP', 'PNEUMONIA_CNN', 'LLM_LORA'], help="Model type")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Aggregation strategy")
    parser.add_argument("--aggregation", type=str, default="FFA_LORA", choices=["FFA_LORA", "FEDIT"], help="LoRA aggregation sub-mode (LLM_LORA only)")
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

    # Load model architecture. LLM_LORA rebuilds its peft model per eval round (eval_net in
    # server_side_evaluate), so skip the eager build here — it would needlessly download + build
    # a full base model that is immediately discarded (and would default to FFA freezing).
    net = None if args.model_type.upper() == "LLM_LORA" else get_model(args.model_type, args.model_name, DEVICE)

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

        logging.info(f"\n{'='*60}")
        logging.info(f"LOADED PARAMETERS FROM .NPZ FILE")
        logging.info(f"{'='*60}")
        logging.info(f"Total parameters loaded: {len(initial_parameters)}")


        # LLM_LORA uses the compact peft key 'base_model.model.score.weight', not
        # the bare 'score.weight', so skip this diagnostic to avoid false error logs.
        if args.model_type.upper() != "LLM_LORA":
            if 'score.weight' in initial_parameters:
                logging.info(f"✅ score.weight found: shape {initial_parameters['score.weight'].shape}")
                logging.info(f"   Expected: torch.Size([3, 768]) for CB")
                if initial_parameters['score.weight'].shape[0] != 3:
                    logging.error(f"   ❌ WRONG NUMBER OF CLASSES: {initial_parameters['score.weight'].shape[0]} instead of 3!")
                else:
                    logging.info(f"   ✅ Correct: 3 classes")
            else:
                logging.error(f"❌ score.weight NOT FOUND!")

            if 'score.bias' in initial_parameters:
                logging.info(f"✅ score.bias found: shape {initial_parameters['score.bias'].shape}")
            else:
                logging.error(f"❌ score.bias NOT FOUND!")

        logging.info(f"\nFirst 10 parameter keys:")
        for i, key in enumerate(list(initial_parameters.keys())[:10]):
            logging.info(f"  {i+1}. {key}: {initial_parameters[key].shape}")

        logging.info(f"\nLast 5 parameter keys:")
        for i, key in enumerate(list(initial_parameters.keys())[-5:]):
            logging.info(f"  {i+1}. {key}: {initial_parameters[key].shape}")

        logging.info(f"{'='*60}\n")

    except Exception as e:
        logging.error(f"Failed to load model parameters from {args.model_path}. Reason: {e}", exc_info=True)
        exit(1)

    # Load test data for server-side evaluation
    is_pneumonia = args.model_type == 'PNEUMONIA_CNN'
    is_llm_lora = args.model_type == 'LLM_LORA'
    if is_llm_lora:
        import recipes
        test_loader = recipes.get_recipe('LLM_LORA').load_server_test_data(model_name=args.model_name)
        logging.info("Loaded LLM_LORA server test data via recipes.LLM_LORA")
    elif is_pneumonia:
        import recipes
        test_loader = recipes.get_recipe('PNEUMONIA_CNN').load_server_test_data(batch_size=32)
        logging.info("Loaded chest X-ray test data via recipes.PNEUMONIA_CNN (NORMAL/PNEUMONIA)")
    elif is_mlp and args.dataset == "ecg":
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
        if is_llm_lora:
            import recipes as _recipes
            from peft import set_peft_model_state_dict
            eval_net = _recipes.get_recipe("LLM_LORA").build_model(
                DEVICE, model_name=args.model_name, aggregation=args.aggregation)
            # peft's set_peft_model_state_dict mutates its input dict in-place; copy so the global
            # adapter params (reused across rounds) are not corrupted during evaluation.
            set_peft_model_state_dict(eval_net, OrderedDict(parameters))
            eval_net.to(DEVICE)
            eval_net.eval()
        else:
            net.load_state_dict(parameters, strict=True)
            net.to(DEVICE)
            net.eval()
            eval_net = net

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):

                if batch_idx == 0:
                    logging.info(f"[EVAL DEBUG] First test batch:")
                    logging.info(f"[Debug] Batch type: {type(batch)}")
                    logging.info(f"  Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'N/A'}")

                if hasattr(batch, 'keys'):
                        logging.info(f"[Debug] Batch keys: {list(batch.keys())}")
                # Handle different batch formats
                if hasattr(batch, 'data'):  # BatchEncoding has a .data attribute
                    batch = dict(batch)
                try:
                    if is_llm or is_llm_lora:
                        # LLM / LLM_LORA: batch should be a dict with input_ids, attention_mask, labels
                        if isinstance(batch, dict):
                            if 'labels' not in batch:
                                raise KeyError(f"LLM batch is dict but missing 'labels' key. Available keys: {list(batch.keys())}")

                            # Move all tensors to device
                            batch = {k: v.to(DEVICE) for k, v in batch.items()}

                            # Forward pass
                            outputs = eval_net(**batch)
                            loss = outputs.loss
                            logits = outputs.logits
                            labels = batch["labels"]

                        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                            # Fallback: batch is (inputs_dict, labels) tuple
                            inputs, labels = batch
                            if isinstance(inputs, dict):
                                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                                labels = labels.to(DEVICE)
                                outputs = eval_net(**inputs, labels=labels)
                                loss = outputs.loss
                                logits = outputs.logits
                            else:
                                raise ValueError(f"Expected LLM inputs to be dict, got {type(inputs)}")
                        else:
                            raise ValueError(f"Unexpected LLM batch format: {type(batch)}")

                    elif is_mlp:
                        # MLP: batch is (features, labels) tuple
                        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                            raise ValueError(f"Expected MLP batch to be (features, labels) tuple, got {type(batch)}")

                        features, labels = batch
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = eval_net(features)
                        loss = criterion(outputs, labels)
                        logits = outputs

                    else:
                        # CNN: batch is a dict with 'img' and 'label'
                        if isinstance(batch, dict):
                            if 'img' not in batch or 'label' not in batch:
                                raise KeyError(f"CNN batch missing keys. Available: {list(batch.keys())}")

                            images = batch["img"].to(DEVICE)
                            labels = batch["label"].to(DEVICE)
                        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                            # Fallback: batch is (images, labels) tuple
                            images, labels = batch
                            images = images.to(DEVICE)
                            labels = labels.to(DEVICE)
                        else:
                            raise ValueError(f"Unexpected CNN batch format: {type(batch)}")

                        outputs = eval_net(images)
                        loss = criterion(outputs, labels)
                        logits = outputs

                    total_loss += loss.item()
                    num_batches += 1

                    # Calculate accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                    if batch_idx == 0 and (is_llm or is_llm_lora):
                        logging.info(f"  Logits shape: {logits.shape}")
                        logging.info(f"  Predictions: {predictions[:5]}")
                        logging.info(f"  True labels: {labels[:5]}")
                        logging.info(f"  Batch correct: {(predictions == labels).sum().item()}/{labels.size(0)}")

                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    logging.error(f"Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        logging.error(f"Batch keys: {list(batch.keys())}")
                    raise

        # Average loss per batch (not per sample)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        print(f"Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

        # Emit JSON structure for frontend LogViewer.tsx telemetry over WebSocket
        import json
        print(json.dumps({
            "level": "INFO",
            "serverRound": server_round,
            "loss": avg_loss,
            "accuracy": accuracy / 100.0,
            "message": f"[Telemetry] Round {server_round} Aggregation Complete: Loss {avg_loss:.4f}, Acc {accuracy/100.0:.4f}"
        }))

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

    # LLM_LORA does not support DeComFL — the zeroth-order path requires a flat
    # float-vector parameter space that is incompatible with adapter-only sync.
    if args.model_type.upper() == "LLM_LORA" and args.strategy.lower() == "decomfl":
        logging.error("LLM_LORA does not support the DeComFL strategy (use FedAvg/FedLoRA).")
        sys.exit(1)

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
    elif args.strategy.lower() == 'fedlora':
        logging.info(f"Using FedLoRA strategy (aggregation={args.aggregation})")
        strategy = FedLoRA(
            initial_parameters=initial_parameters,
            evaluate_fn=server_side_evaluate,
            min_fit_clients=args.min_clients,
            aggregation=args.aggregation,
        )
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
    server_address = f"{bind_address}:{args.port}"
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


    # Report results via the internal callback endpoint (guarded by X-Internal-Key).
    results_url = f"{BACKEND_URL}/api/internal/results/{args.project_id}"
    try:
        headers = _internal_headers()
    except RuntimeError as e:
        logging.error("Cannot report round results: %s", e)
        headers = None

    if history and headers is not None:
        for r, metrics in history:
            acc_metric = float(metrics.get("accuracy", 0.0))
            # The evaluate_fn returns accuracy as a percentage (e.g. 52.74)
            # The database / frontend expects a decimal for precision (e.g. 0.5274)
            decimal_accuracy = acc_metric / 100.0 if acc_metric > 1.0 else acc_metric

            result_payload = {
                "serverRound": r,
                "loss": float(metrics.get("loss", 0.0)),
                "accuracy": decimal_accuracy,
                "gpuUtilization": 0.0,
            }
            try:
                res = requests.post(results_url, json=result_payload, headers=headers, timeout=30)
                res.raise_for_status()
                logging.info(f"Successfully reported results for round {r}")
            except Exception as e:
                logging.error(f"Failed to report results for round {r}: {e}")

    # Mark project as completed. Uses the internal endpoint
    # (POST /api/internal/results/{id}/finished) so the FL-server task does not
    # need a user JWT.
    project_complete_url = f"{BACKEND_URL}/api/internal/results/{args.project_id}/finished"
    if headers is not None:
        try:
            response = requests.post(project_complete_url, headers=headers, timeout=30)
            response.raise_for_status()
            logging.info("Project marked as finished (status=%s)", response.status_code)
        except requests.exceptions.RequestException as e:
            logging.error("Failed to mark project as finished: %s", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutdown requested by user.")
    except Exception as e:
        logging.critical("An unhandled exception occurred in the main function.", exc_info=True)
        exit(1)