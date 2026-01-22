# examples/decomfl/run_server.py
"""
DeComFL Server Example - ECG Classification
Run federated learning server using DeComFL strategy.
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.fedlearn.server.server import start_server
import fedlearn as fl
from src.fedlearn.server.decomfl_strategy import DeComFL
from config import Config
from model import create_model
from data import get_test_loader


def evaluate_fn(server_round: int, parameters: OrderedDict[str, torch.Tensor]):
    """
    Evaluation function for the global model.

    Args:
        server_round: Current round number
        parameters: Global model parameters

    Returns:
        loss: Test loss
        metrics: Dictionary of metrics
    """
    device = Config.DEVICE

    # Create model and load parameters
    model = create_model(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        device=device
    )
    model.load_state_dict(parameters)
    model.eval()

    # Get test loader (use cached split)
    test_loader, _ = get_test_loader(
        X=evaluate_fn.X_test,
        y=evaluate_fn.y_test,
        num_clients=Config.NUM_CLIENTS,
        batch_size=Config.BATCH_SIZE_TEST,
        alpha=Config.ALPHA,
        data_fraction=Config.DATA_FRACTION,
        test_size=Config.TEST_SIZE,
        num_workers=0,
        seed=Config.SEED
    )

    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"\n{'=' * 70}")
    print(f"ROUND {server_round} EVALUATION")
    print(f"{'=' * 70}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 70}\n")

    return avg_loss, {"accuracy": accuracy}


def main():
    """Main server function."""
    print("\n" + "=" * 70)
    print("DeComFL FEDERATED LEARNING SERVER - ECG Classification")
    print("=" * 70)

    Config.print_config()

    # Load data
    print("\nLoading ECG data...")
    data_path = Config.DATA_PATH

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape}")

    # Extract features and labels
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Store data for evaluation function
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.SEED, stratify=y
    )
    evaluate_fn.X_test = X_test
    evaluate_fn.y_test = y_test

    # Update input dimension in config
    Config.INPUT_DIM = X.shape[1]

    # Create initial model
    print("\nInitializing global model...")
    initial_model = create_model(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE
    )

    initial_parameters = initial_model.state_dict()
    print(f"Model initialized with {sum(p.numel() for p in initial_model.parameters()):,} parameters")

    # Create DeComFL strategy
    print("\nCreating DeComFL strategy...")
    strategy = DeComFL(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        min_fit_clients=Config.MIN_FIT_CLIENTS,
        clients_per_round=Config.CLIENTS_PER_ROUND,
        num_local_steps=Config.NUM_LOCAL_STEPS,
        num_perturbations=Config.NUM_PERTURBATIONS,
        learning_rate=Config.LEARNING_RATE,
        smoothing_param=Config.SMOOTHING_PARAM,
        seed=Config.SEED
    )

    # Start server
    print("\n" + "=" * 70)
    print("STARTING FEDERATED LEARNING SERVER")
    print("=" * 70)
    print(f"Server address: {Config.SERVER_ADDRESS}")
    print(f"Total rounds: {Config.NUM_ROUNDS}")
    print(f"Clients per round: {Config.CLIENTS_PER_ROUND}")
    print(f"Strategy: DeComFL")
    print("=" * 70 + "\n")

    history, _ = fl.server.start_server(
        server_address=Config.SERVER_ADDRESS,
        strategy=strategy,
        config=fl.server.ServerConfig(Config.NUM_ROUNDS)
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model saved to: {Config.CHECKPOINT_DIR}")
    print(f"Results saved to: {Config.RESULTS_DIR}")

    if history:
        final_round, final_metrics = history[-1]
        final_accuracy = final_metrics.get('accuracy', 0.0)
        print(f"\nFinal Results (Round {final_round}):")
        print(f"  Accuracy: {final_accuracy:.2f}%")


        # if args.dataset == "cb":
        #     target = 75.0
        #     status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
        #     print(f"  Target: {target:.2f}% {status}")
        # elif args.dataset == "sst2":
        #     target = 85.0
        #     status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
        #     print(f"  Target: {target:.2f}% {status}")

        # ========== NEW: CREATE TABLE AND PLOT ==========
        import matplotlib.pyplot as plt
        # import pandas as pd
        from datetime import datetime

        # Extract data from history
        rounds = []
        accuracies = []
        losses = []

        for round_num, metrics in history:
            rounds.append(round_num)
            accuracies.append(metrics.get('accuracy', 0.0))
            losses.append(metrics.get('loss', 0.0))

        # Create DataFrame
        df = pd.DataFrame({
            'Round': rounds,
            'Accuracy (%)': accuracies,
            'Loss': losses
        })

        # Print table
        print("\n" + "=" * 60)
        print(" " * 20 + "Training History Table")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)

        # Save table to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"results_ECG_{timestamp}_{Config.CLIENTS_PER_ROUND}_clients.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n✓ Table saved to: {csv_filename}")

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Accuracy over rounds
        ax1.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title(f'ECG - Accuracy over Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add target line
        target_acc = 93.0
        ax1.axhline(y=target_acc, color='r', linestyle='--', linewidth=2, label=f'Target: {target_acc}%')
        ax1.legend()

        # Annotate final accuracy
        ax1.annotate(f'Final: {final_accuracy:.2f}%',
                     xy=(rounds[-1], accuracies[-1]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Plot 2: Loss over rounds
        ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'ECG - Loss over Rounds', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Annotate final loss
        ax2.annotate(f'Final: {losses[-1]:.4f}',
                     xy=(rounds[-1], losses[-1]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()

        # Save plot
        plot_filename = f"plot_ECG_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_filename}")

        # Show plot (comment out if running on server without display)
        # plt.show()

        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print("=" * 60)
        print(f"  Total Rounds: {len(rounds)}")
        print(f"  Initial Accuracy: {accuracies[0]:.2f}%")
        print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
        print(f"  Improvement: {accuracies[-1] - accuracies[0]:.2f}%")
        print(f"  Initial Loss: {losses[0]:.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Best Accuracy: {max(accuracies):.2f}% (Round {rounds[accuracies.index(max(accuracies))]})")
        print(f"  Lowest Loss: {min(losses):.4f} (Round {rounds[losses.index(min(losses))]})")
        print("=" * 60)


if __name__ == "__main__":
    main()