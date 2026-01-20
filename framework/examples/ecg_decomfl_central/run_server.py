# =============================================================================
# run_server.py - DeComFL Server Implementation (Algorithm 3)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
import time
import pandas as pd
from pathlib import Path

from config import Config
from model import create_model
from data import get_test_loader, get_ecg_loaders


class DeComFLServer:
    """DeComFL Server implementing Algorithm 3."""

    def __init__(self, config: Config, input_dim: int):
        self.config = config
        self.device = config.DEVICE

        # Create global model
        self.model = create_model(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            device=self.device
        )

        print(f"\n[SERVER] Model created with {self.model.get_num_params():,} parameters")

        # Algorithm 3, Line 2: Initialize history
        self.gradient_history = []
        self.seed_history = []

        # Training history
        self.train_history = {
            'round': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_time': []
        }

        # Random seed for reproducibility
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)

    def generate_seeds(self, round_idx: int) -> List[List[int]]:
        """Generate random seeds for perturbations. Algorithm 3, Line 5"""
        K = self.config.NUM_LOCAL_STEPS
        P = self.config.NUM_PERTURBATIONS

        seeds = []
        for k in range(K):
            k_seeds = []
            for p in range(P):
                seed = np.random.randint(0, 2 ** 31 - 1)
                k_seeds.append(seed)
            seeds.append(k_seeds)

        return seeds

    def aggregate_gradients(
            self,
            client_gradients: Dict[int, List[List[float]]],
            round_idx: int
    ):
        """Aggregate gradients and update global model. Algorithm 3, Lines 10-12"""
        K = self.config.NUM_LOCAL_STEPS
        P = self.config.NUM_PERTURBATIONS
        eta = self.config.LEARNING_RATE
        num_clients = len(client_gradients)

        # Get current model parameters
        x_current = self.model.get_flat_params()

        # For each local step
        for k in range(K):
            delta = torch.zeros_like(x_current)

            # Average gradients across clients
            for client_id, grad_scalars in client_gradients.items():
                for p in range(P):
                    # Regenerate perturbation from seed
                    z = self._generate_perturbation(self.seed_history[round_idx][k][p])

                    # Get gradient scalar for this client
                    g = grad_scalars[k][p]

                    # Accumulate gradient direction
                    delta += g * z

            # Average across clients and perturbations
            delta = delta / (num_clients * P)

            # Update model parameters
            x_current = x_current - eta * delta * P

        # Set updated parameters
        self.model.set_flat_params(x_current)

    def _generate_perturbation(self, seed: int) -> torch.Tensor:
        """Generate perturbation vector from seed."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(
            self.model.get_num_params(),
            generator=generator,
            device=self.device
        )
        return z

    def evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate global model on test set"""
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def save_checkpoint(self, round_idx: int = None):
        """Save model checkpoint"""
        path = self.config.get_model_save_path(round_idx)
        torch.save({
            'round': round_idx,
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.model.fc1.in_features,
                'hidden_dim': self.config.HIDDEN_DIM,
                'num_classes': self.config.NUM_CLASSES
            }
        }, path)
        print(f"[SERVER] Checkpoint saved: {path}")

    def save_results(self):
        """Save training history"""
        path = self.config.get_results_save_path()
        np.savez(
            path,
            rounds=self.train_history['round'],
            test_loss=self.train_history['test_loss'],
            test_accuracy=self.train_history['test_accuracy'],
            train_time=self.train_history['train_time']
        )
        print(f"[SERVER] Results saved: {path}")


def run_centralized_training():
    """Run centralized DeComFL training"""
    print("\n" + "=" * 70)
    print("DeComFL CENTRALIZED TRAINING")
    print("=" * 70)

    Config.print_config()

    # Load ECG data
    print("Loading ECG data...")
    data_path = Config.DATA_PATH

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")

    # Assuming last column is the label
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Get test loader
    test_loader, test_info = get_test_loader(
        X=X, y=y,
        num_clients=Config.NUM_CLIENTS,
        batch_size=Config.BATCH_SIZE_TEST,
        alpha=Config.ALPHA,
        data_fraction=Config.DATA_FRACTION,
        test_size=Config.TEST_SIZE,
        num_workers=0,
        seed=Config.SEED
    )

    input_dim = test_info['input_dim']

    # Initialize server
    server = DeComFLServer(config=Config, input_dim=input_dim)

    # Initial evaluation
    print("\n[SERVER] Initial evaluation...")
    metrics = server.evaluate(test_loader)
    print(f"[SERVER] Initial - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")

    # OPTIMIZATION: Create client ONCE, not every round
    print("\n[SERVER] Initializing client (one-time setup)...")
    from run_client import DeComFLClient

    # Get client's training data (one time)
    train_loader, _, _ = get_ecg_loaders(
        X=X, y=y,
        client_id=0,
        num_clients=Config.NUM_CLIENTS,
        batch_size_train=Config.BATCH_SIZE_TRAIN,
        batch_size_test=Config.BATCH_SIZE_TEST,
        data_fraction=Config.DATA_FRACTION,
        alpha=Config.ALPHA,
        test_size=Config.TEST_SIZE,
        num_workers=0,
        seed=Config.SEED
    )

    # Create client once
    client = DeComFLClient(
        client_id=0,
        config=Config,
        model_params=server.model.get_flat_params(),
        train_loader=train_loader,
        input_dim=input_dim
    )

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for round_idx in range(Config.NUM_ROUNDS):
        round_start_time = time.time()

        # Algorithm 3, Line 5: Generate seeds
        seeds = server.generate_seeds(round_idx)
        server.seed_history.append(seeds)

        # Update client's model parameters from server
        client.x_current = server.model.get_flat_params()
        client.model.set_flat_params(client.x_current)

        # Algorithm 3, Lines 7-9: Client local update
        grad_scalars = client.local_update(seeds, round_idx)

        # Algorithm 3, Line 11: Store gradients
        server.gradient_history.append(grad_scalars)

        # Algorithm 3, Line 12: Update global model
        client_gradients = {0: grad_scalars}
        server.aggregate_gradients(client_gradients, round_idx)

        # Evaluate
        metrics = server.evaluate(test_loader)

        # Record history
        round_time = time.time() - round_start_time
        server.train_history['round'].append(round_idx + 1)
        server.train_history['test_loss'].append(metrics['loss'])
        server.train_history['test_accuracy'].append(metrics['accuracy'])
        server.train_history['train_time'].append(round_time)

        # Print progress
        if (round_idx + 1) % max(1, Config.NUM_ROUNDS // 10) == 0:
            print(f"Round {round_idx + 1}/{Config.NUM_ROUNDS} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Acc: {metrics['accuracy']:.2f}% | "
                  f"Time: {round_time:.2f}s")

        # Save checkpoint periodically
        if (round_idx + 1) % 50 == 0:
            server.save_checkpoint(round_idx + 1)

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    final_metrics = server.evaluate(test_loader)
    print(f"\nFinal Results:")
    print(f"  Test Loss: {final_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"  Best Accuracy: {max(server.train_history['test_accuracy']):.2f}%")

    server.save_checkpoint()
    server.save_results()

    print(f"\nCheckpoints saved to: {Config.CHECKPOINT_DIR}")
    print(f"Results saved to: {Config.RESULTS_DIR}")

    return server


if __name__ == "__main__":
    server = run_centralized_training()