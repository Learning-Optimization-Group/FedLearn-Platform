# =============================================================================
# run_client.py - Federated Learning Client for ECG Classification
# =============================================================================

import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import pandas as pd

# Import your federated learning library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import fedlearn as fl

# Import local modules
from data import get_ecg_loaders
from config import ECG_CONFIG, USE_MIXED_PRECISION, SCHEDULER_CONFIG

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

# =============================================================================
# ECG Transformer Model Definition
# =============================================================================
class ECGTransformer(nn.Module):
    """Optimized Transformer model for ECG classification"""
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
        output = self.classifier(x)
        return output


# =============================================================================
# ECG Federated Learning Client
# =============================================================================
class ECGClient(fl.Client):
    """
    Custom ECG client for binary classification tasks (Normal/Abnormal).
    Implements local training with mixed precision and learning rate scheduling.
    """

    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, 
                 num_clients: int, alpha: float = 0.5, data_fraction: float = 1.0):
        """
        Initialize ECG client.
        
        Args:
            client_id: Unique client identifier
            X: Full ECG feature array
            y: Full label array
            num_clients: Total number of clients in federation
            alpha: Dirichlet parameter for non-IID split
            data_fraction: Fraction of data to use
        """
        self.client_id = client_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.net = ECGTransformer(
            input_dim=ECG_CONFIG['input_dim'],
            d_model=ECG_CONFIG['d_model'],
            nhead=ECG_CONFIG['nhead'],
            num_layers=ECG_CONFIG['num_layers'],
            num_classes=ECG_CONFIG['num_classes']
        ).to(self.device)
        
        self.trainloader, _, self.data_info = get_ecg_loaders(
            X=X,
            y=y,
            client_id=client_id,
            num_clients=num_clients,
            batch_size_train=ECG_CONFIG['batch_size_train'],
            batch_size_test=ECG_CONFIG['batch_size_test'],
            data_fraction=data_fraction,
            alpha=alpha,
            test_size=ECG_CONFIG['test_size'],
            num_workers=ECG_CONFIG['num_workers'],
            seed=ECG_CONFIG['seed']
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if (
            USE_MIXED_PRECISION and torch.cuda.is_available()
        ) else None
        
        print(f"Client {client_id} initialized: {self.data_info['train_samples']} training samples")

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Get current model parameters."""
        return self.net.state_dict()

    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict):
        """
        Train the model locally for K epochs with optimized training loop.
        """
        self.net.load_state_dict(parameters)
        self.net.train()
        
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=ECG_CONFIG['learning_rate'],
            weight_decay=ECG_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=SCHEDULER_CONFIG['max_lr'],
            epochs=ECG_CONFIG['local_epochs'],
            steps_per_epoch=len(self.trainloader),
            pct_start=SCHEDULER_CONFIG['pct_start'],
            anneal_strategy=SCHEDULER_CONFIG['anneal_strategy'],
            div_factor=SCHEDULER_CONFIG['div_factor'],
            final_div_factor=SCHEDULER_CONFIG['final_div_factor']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for epoch in range(ECG_CONFIG['local_epochs']):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.net(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.net(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                epoch_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                epoch_correct += (predictions == y_batch).sum().item()
                epoch_samples += y_batch.size(0)
                num_batches += 1
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            epoch_acc = 100.0 * epoch_correct / epoch_samples
            avg_epoch_loss = epoch_loss / len(self.trainloader)
            print(f"  Client {self.client_id} - Epoch {epoch+1}/{ECG_CONFIG['local_epochs']}: "
                  f"Loss={avg_epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        avg_loss = total_loss / num_batches
        avg_acc = 100.0 * total_correct / total_samples
        
        print(f"  Client {self.client_id} - Overall: "
              f"Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.2f}%")
        
        return self.net.state_dict(), self.data_info['train_samples']


# =============================================================================
# Main Client Execution
# =============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "ecg_data", "ecg.csv")
    
    parser = argparse.ArgumentParser(description="FedLearn ECG Client")
    parser.add_argument("--server_address", type=str, default="localhost:50051", 
                       help="Server address (e.g., localhost:50051)")
    parser.add_argument("--id", type=int, required=True, 
                       help="Client ID (0 to num_clients-1)")
    parser.add_argument("--num_clients", type=int, default=5, 
                       help="Total number of clients")
    parser.add_argument("--data_path", type=str, default=default_data_path, 
                       help="Path to ECG CSV file")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Dirichlet alpha for non-IID split")
    parser.add_argument("--data_fraction", type=float, default=1.0, 
                       help="Fraction of data to use (0.1 = 10%)")
    args = parser.parse_args()
    
    if args.id < 0 or args.id >= args.num_clients:
        raise ValueError(f"Client ID must be between 0 and {args.num_clients-1}")
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data file not found at: {args.data_path}")
        print(f"Please specify correct path with --data_path argument")
        sys.exit(1)
    
    print(f"Loading ECG data from: {args.data_path}")
    
    df = pd.read_csv(args.data_path)
    
    # Relabel columns if not properly labeled
    n_cols = len(df.columns)
    df.columns = [f"signal_{i}" for i in range(n_cols - 1)] + ["target"]
    
    df_processed = preprocess_ecg(df, label_col='target', target_length=ECG_CONFIG['input_dim'])
    
    X = df_processed.drop('target', axis=1).values.astype(float)
    y = df_processed['target'].values.astype(int)
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    print(f"Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    client = ECGClient(
        client_id=args.id,
        X=X,
        y=y,
        num_clients=args.num_clients,
        alpha=args.alpha,
        data_fraction=args.data_fraction
    )
    
    print(f"Connecting to server at {args.server_address}...")
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
        client_id=f"client_{args.id}"
    )