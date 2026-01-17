# =============================================================================
# run_server.py - Federated Learning Server for ECG Classification
# =============================================================================

import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import pandas as pd

# Add path to your federated learning library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import fedlearn as fl

# Import local modules
from data import get_test_loader
from config import ECG_CONFIG

# =========================
# Preprocessing functions
# =========================
from scipy.signal import find_peaks

def normalize_signal(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        return np.zeros_like(signal)
    return (signal - mean) / std

def detect_r_peaks(signal, height=0.3, distance_pct=0.05):
    normalized = normalize_signal(signal)
    min_distance = int(len(signal) * distance_pct)
    peaks, _ = find_peaks(normalized, height=height, distance=min_distance)
    return peaks

def align_signal(signal, target_length=140, r_peak_position=0.3):
    normalized = normalize_signal(signal)
    peaks = detect_r_peaks(signal)
    if len(peaks) == 0:
        result = np.zeros(target_length)
        copy_len = min(len(normalized), target_length)
        result[:copy_len] = normalized[:copy_len]
        return result
    main_peak = peaks[0]
    offset = int(target_length * r_peak_position)
    aligned = np.zeros(target_length)
    start_idx = max(0, main_peak - offset)
    end_idx = min(len(signal), main_peak + (target_length - offset))
    for i in range(start_idx, end_idx):
        target_idx = i - main_peak + offset
        if 0 <= target_idx < target_length:
            aligned[target_idx] = normalized[i]
    return aligned

def preprocess_ecg(df, label_col='target', target_length=140):
    signal_cols = [col for col in df.columns if col != label_col]
    processed = []
    for idx in range(len(df)):
        signal = df.iloc[idx][signal_cols].values.astype(float)
        aligned = align_signal(signal, target_length)
        processed.append(aligned)
    new_cols = [f'signal_{i}' for i in range(target_length)]
    df_processed = pd.DataFrame(processed, columns=new_cols)
    df_processed[label_col] = df[label_col].values
    return df_processed

# =========================
# ECG Transformer Model
# =========================
class ECGTransformer(nn.Module):
    def __init__(self, input_dim=140, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# =========================
# Server Evaluation Function
# =========================
def create_evaluate_fn(X, y, num_clients, alpha, data_fraction):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    testloader, test_info = get_test_loader(
        X=X,
        y=y,
        num_clients=num_clients,
        batch_size=ECG_CONFIG['batch_size_test'],
        alpha=alpha,
        data_fraction=data_fraction,
        test_size=ECG_CONFIG['test_size'],
        num_workers=ECG_CONFIG['num_workers']
    )

    criterion = nn.CrossEntropyLoss()

    def evaluate(server_round, parameters: OrderedDict[str, torch.Tensor]):
        model = ECGTransformer(
            input_dim=ECG_CONFIG['input_dim'],
            d_model=ECG_CONFIG['d_model'],
            nhead=ECG_CONFIG['nhead'],
            num_layers=ECG_CONFIG['num_layers'],
            num_classes=ECG_CONFIG['num_classes']
        )
        model.load_state_dict(parameters)
        model.to(device)
        model.eval()

        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds += preds.cpu().numpy().tolist()
                all_labels += y_batch.cpu().numpy().tolist()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(testloader)

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        print(f"Round {server_round} — Acc: {accuracy:.2f}%  Loss: {avg_loss:.4f}")

        return avg_loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return evaluate

# =========================
# MAIN SERVER
# =========================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "ecg_data", "ecg.csv")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min_fit_clients", type=int, default=1)
    parser.add_argument("--clients_per_round", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=default_data_path, 
                        help="Path to raw ECG CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found at: {args.data_path}")
        print(f"Please specify correct path with --data_path argument")
        sys.exit(1)

    print("Loading and preprocessing ECG CSV file...")
    df = pd.read_csv(args.data_path)
    
    # Relabel columns if not properly labeled
    n_cols = len(df.columns)
    df.columns = [f"signal_{i}" for i in range(n_cols - 1)] + ["target"]
    
    df_processed = preprocess_ecg(df, label_col='target', target_length=ECG_CONFIG['input_dim'])

    X = df_processed.drop('target', axis=1).values.astype(float)
    y = df_processed['target'].values.astype(int)

    print(f"Data loaded: X={X.shape}, y={y.shape}")
    print(f"Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

    num_rounds = args.num_rounds or ECG_CONFIG['num_rounds']

    net = ECGTransformer(
        input_dim=X.shape[1],
        d_model=ECG_CONFIG['d_model'],
        nhead=ECG_CONFIG['nhead'],
        num_layers=ECG_CONFIG['num_layers'],
        num_classes=ECG_CONFIG['num_classes']
    )
    initial_parameters = net.state_dict()

    evaluate_fn = create_evaluate_fn(
        X=X,
        y=y,
        num_clients=args.num_clients,
        alpha=args.alpha,
        data_fraction=args.data_fraction
    )

    strategy = fl.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        min_fit_clients=args.min_fit_clients,
        clients_per_round=args.clients_per_round,
    )

    print(f"Starting server on 0.0.0.0:{args.port}")
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
        csv_filename = f"results_ecg_{timestamp}_{args.num_clients}_clients.csv"
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
        target_acc = 97.0
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
        plot_filename = f"plot_ecg_{timestamp}.png"
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