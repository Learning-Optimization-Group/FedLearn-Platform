# =============================================================================
# run_server.py - Federated Learning Server for CB and SST-2
# =============================================================================

import argparse
from collections import OrderedDict
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import fedlearn as fl
from transformers import AutoModelForSequenceClassification
from data import get_llm_loaders
from config import MODEL_NAME, DATASET_CONFIGS
from data import get_test_loader


def create_evaluate_fn(dataset_name: str, data_fraction: float = 1.0):
    """
    Creates a server-side evaluation function for the specified dataset.

    Args:
        dataset_name: "cb" or "sst2"
        data_fraction: Fraction of data used for training

    Returns:
        Evaluation function for the server
    """
    config = DATASET_CONFIGS[dataset_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    testloader, _ = get_test_loader(
        dataset_name=dataset_name,
        data_fraction=data_fraction
    )

    print(f"Server evaluation setup:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Device: {device}")
    print(f"  Test samples: {len(testloader.dataset)}")
    print(f"  Num classes: {config.num_classes}")


    def test_random_head_performance():
        """Test if random classification head gets high accuracy."""
        # Create model with random head
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in testloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct += (predictions == labels).sum().item()
                total += len(labels)

        acc = 100.0 * correct / total
        print(f"[DEBUG] Random head baseline: {acc:.2f}%")
        return acc

    # Call it before training starts
    random_head_acc = test_random_head_performance()
    def server_side_evaluate(server_round: int, parameters: OrderedDict[str, torch.Tensor]):
        """
        Evaluate the global model on the test set.
        Returns accuracy for classification tasks.
        """

        # Verify we're testing on unseen data
        print(f"[DEBUG] Test set size: {len(testloader.dataset)}")
        print(f"[DEBUG] Expected for SST-2 validation: 872 samples")

        # Test random baseline
        import random
        random_correct = sum(
            [random.randint(0, 1) == label.item() for batch in testloader for label in batch['labels']])
        random_acc = 100.0 * random_correct / len(testloader.dataset)
        print(f"[DEBUG] Random guessing would give: {random_acc:.2f}% (should be ~50%)")


        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=config.num_classes
        )
        model.load_state_dict(parameters)
        model.to(device)
        model.eval()

        # Debug prints
        print(f"[DEBUG] Classification head weights mean: {model.score.weight.mean().item():.6f}")
        print(f"[DEBUG] Classification head weights std: {model.score.weight.std().item():.6f}")

        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(testloader):  # Added enumerate

                if i == 0:
                    print(f"[DEBUG] Batch keys: {batch.keys()}")
                    print(f"[DEBUG] Labels shape: {batch['labels'].shape if 'labels' in batch else 'NO LABELS!'}")
                    print(f"[DEBUG] Sample labels: {batch['labels'][:5] if 'labels' in batch else 'NO LABELS'}")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                num_batches += 1

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(f"Round {server_round} Evaluation:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Loss: {avg_loss:.4f}")

        # Compare to target
        if dataset_name == "cb":
            target = 75.0
            status = "✓" if accuracy >= target else "✗"
            print(f"  Target (DeComFL): {target:.2f}% {status}")
        elif dataset_name == "sst2":
            target = 85.0
            status = "✓" if accuracy >= target else "✗"
            print(f"  Target (DeComFL): {target:.2f}% {status}")

        print(f"{'=' * 60}\n")

        return avg_loss, {"accuracy": accuracy}

    return server_side_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLearn LLM Server")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2"], help="Dataset to use")
    parser.add_argument("--num_rounds", type=int, default=None, help="Number of training rounds (default from config)")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--min_fit_clients", type=int, default=8, help="Minimum clients per round")
    parser.add_argument("--clients_per_round", type=int, default=2, help="Trigger federation after these many rounds")
    args = parser.parse_args()

    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    num_rounds = args.num_rounds if args.num_rounds else config.num_rounds

    print(f"\n{'=' * 60}")
    print(f"Federated Learning Server - {args.dataset.upper()} Dataset")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Num rounds: {num_rounds}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Clients per round: {config.clients_per_round}")
    print(f"  Local epochs (K): {config.local_epochs}")
    print(f"  Data fraction: {args.data_fraction * 100}%")
    print(f"  Alpha (Dirichlet): {config.alpha}")
    print(f"{'=' * 60}\n")

    # Initialize model
    net = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=config.num_classes
    )
    initial_parameters = net.state_dict()

    # Create evaluation function
    evaluate_fn = create_evaluate_fn(args.dataset, args.data_fraction)

    # Define strategy (FedAvg)
    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        min_fit_clients=args.min_fit_clients,
        clients_per_round=args.clients_per_round,
    )

    # Start server
    history, _ = fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if history:
        final_round, final_metrics = history[-1]
        final_accuracy = final_metrics.get('accuracy', 0.0)
        print(f"\nFinal Results (Round {final_round}):")
        print(f"  Accuracy: {final_accuracy:.2f}%")


        if args.dataset == "cb":
            target = 75.0
            status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
            print(f"  Target: {target:.2f}% {status}")
        elif args.dataset == "sst2":
            target = 85.0
            status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
            print(f"  Target: {target:.2f}% {status}")

        # ========== NEW: CREATE TABLE AND PLOT ==========
        import matplotlib.pyplot as plt
        import pandas as pd
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
        csv_filename = f"results_{args.dataset}_{timestamp}_{args.clients_per_round}_clients.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n✓ Table saved to: {csv_filename}")

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Accuracy over rounds
        ax1.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title(f'{args.dataset.upper()} - Accuracy over Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add target line
        target_acc = 75.0 if args.dataset == "cb" else 85.0
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
        ax2.set_title(f'{args.dataset.upper()} - Loss over Rounds', fontsize=14, fontweight='bold')
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
        plot_filename = f"plot_{args.dataset}_{timestamp}.png"
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



