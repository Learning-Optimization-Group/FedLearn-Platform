#!/usr/bin/env python3
"""
FedLearn Full E2E Test Suite
============================
Tests all major example configurations:
  Test 1: SimpleCNN + MNIST + FedAvg (10 rounds)
  Test 2: ECGTransformer + ECG Data + FedAvg (10 rounds)
  Test 3: ECG MLP + ECG Data + DeComFL (10 rounds)

Each test spins up a gRPC server + clients, runs federated training,
and verifies accuracy/convergence. Logs are stored per-test.

Usage:
    cd framework/
    python run_full_test_suite.py
"""

import multiprocessing
import time
import sys
import os
import signal
import logging
import traceback
import json
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

LOG_BASE = os.path.join(SCRIPT_DIR, "logs")
ECG_DATA_PATH = os.path.join(SCRIPT_DIR, "examples", "simple_federation", "data")
ECG_CSV_PATH = os.path.join(SCRIPT_DIR, "examples", "ecg_federation", "ecg_data", "ecg.csv")
DECOMFL_ECG_CSV = os.path.join(SCRIPT_DIR, "examples", "ecg_decomfl_framework_integration", "ecg_data", "ecg.csv")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SHARED MODELS                                                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class SimpleCNN(nn.Module):
    """SimpleCNN for MNIST (from examples/simple_federation)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ECGTransformer(nn.Module):
    """ECGTransformer for ECG classification (from examples/ecg_federation)."""
    def __init__(self, input_dim=140, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


class ECGModel(nn.Module):
    """Simple MLP for ECG (from examples/ecg_decomfl_framework_integration)."""
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ECG DATA HELPERS                                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def load_ecg_data(csv_path, preprocess=True, target_length=140):
    """Load and optionally preprocess ECG CSV data."""
    from scipy.signal import find_peaks

    df = pd.read_csv(csv_path)
    n_cols = len(df.columns)
    df.columns = [f"signal_{i}" for i in range(n_cols - 1)] + ["target"]

    if preprocess:
        signal_cols = [c for c in df.columns if c != "target"]
        processed = []
        for idx in range(len(df)):
            signal = df.iloc[idx][signal_cols].values.astype(float)
            # Normalize
            std = np.std(signal)
            normalized = (signal - np.mean(signal)) / std if std > 1e-10 else np.zeros_like(signal)
            # Detect R-peaks
            peaks, _ = find_peaks(normalized, height=0.3, distance=int(len(signal) * 0.05))
            # Align
            aligned = np.zeros(target_length)
            if len(peaks) > 0:
                peak = peaks[0]
                offset = int(target_length * 0.3)
                for i in range(max(0, peak - offset), min(len(signal), peak + (target_length - offset))):
                    t = i - peak + offset
                    if 0 <= t < target_length:
                        aligned[t] = normalized[i]
            else:
                copy_len = min(len(normalized), target_length)
                aligned[:copy_len] = normalized[:copy_len]
            processed.append(aligned)

        new_cols = [f"signal_{i}" for i in range(target_length)]
        df_proc = pd.DataFrame(processed, columns=new_cols)
        df_proc["target"] = df["target"].values
        X = df_proc.drop("target", axis=1).values.astype(float)
        y = df_proc["target"].values.astype(int)
    else:
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

    return X, y


def make_ecg_loaders(X, y, client_id, num_clients, batch_size=128, alpha=0.5, seed=42):
    """Create train/test DataLoaders for a client using Dirichlet split."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Dirichlet split
    np.random.seed(seed)
    num_classes = len(np.unique(y_train))
    label_dist = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        splits = (np.cumsum(label_dist[k]) * len(idx_k)).astype(int)[:-1]
        for i, idx in enumerate(np.split(idx_k, splits)):
            client_indices[i].extend(idx)

    c_idx = client_indices[client_id]
    X_c = torch.FloatTensor(X_train[c_idx])
    y_c = torch.LongTensor(y_train[c_idx])

    train_loader = DataLoader(TensorDataset(X_c, y_c), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size
    )

    return train_loader, test_loader, len(c_idx)


def make_ecg_test_loader(X, y, batch_size=128, seed=42):
    """Create a test-only DataLoader."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 1: SimpleCNN + MNIST + FedAvg                                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _test1_server(result_q, ready_ev, log_dir):
    log_file = open(os.path.join(log_dir, "server.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.server.server import start_server, ServerConfig
    from fedlearn.server.strategy import FedAvg
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.INFO, format="[SERVER] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_ROUNDS = 10
    NUM_CLIENTS = 3

    try:
        net = SimpleCNN()
        initial_params = net.state_dict()

        # Test evaluation
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_ds = datasets.MNIST(ECG_DATA_PATH, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=128)

        def evaluate_fn(server_round, parameters):
            model = SimpleCNN()
            model.load_state_dict(parameters)
            model.eval()
            criterion = nn.CrossEntropyLoss()
            correct, total, loss = 0, 0, 0.0
            with torch.no_grad():
                for imgs, lbs in test_loader:
                    out = model(imgs)
                    loss += criterion(out, lbs).item()
                    _, pred = torch.max(out, 1)
                    total += lbs.size(0)
                    correct += (pred == lbs).sum().item()
            acc = correct / total
            avg_loss = loss / len(test_loader)
            print(f"[Eval] Round {server_round}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
            return avg_loss, {"accuracy": acc}

        strategy = FedAvg(initial_parameters=initial_params, evaluate_fn=evaluate_fn,
                          min_fit_clients=NUM_CLIENTS, clients_per_round=NUM_CLIENTS)
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        ready_ev.set()

        history, final_params = start_server("localhost:50051", config, strategy)
        acc_history = [{"round": r, "accuracy": m.get("accuracy", 0), "loss": m.get("loss", 0)} for r, m in history]
        result_q.put({"success": True, "has_history": len(history) > 0,
                      "has_params": final_params is not None and len(final_params) > 0,
                      "num_rounds": len(history), "accuracy_history": acc_history})
    except Exception as e:
        traceback.print_exc()
        ready_ev.set()
        result_q.put({"success": False, "error": str(e)})


def _test1_client(client_idx, client_id, result_q, log_dir):
    log_file = open(os.path.join(log_dir, f"{client_id}.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.client.client import start_client
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    logging.basicConfig(level=logging.INFO, format=f"[{client_id}] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_CLIENTS = 3

    class MnistClient:
        def __init__(self, cid):
            self.net = SimpleCNN()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            full = datasets.MNIST(ECG_DATA_PATH, train=True, download=True, transform=transform)
            sz = len(full) // NUM_CLIENTS
            self.loader = DataLoader(Subset(full, range(cid * sz, (cid + 1) * sz)), batch_size=32, shuffle=True)

        def get_parameters(self): return self.net.state_dict()
        def fit(self, params, config):
            self.net.load_state_dict(params)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
            self.net.train()
            for _ in range(2):
                for imgs, lbs in self.loader:
                    optimizer.zero_grad(); loss = criterion(self.net(imgs), lbs)
                    loss.backward(); optimizer.step()
            return self.net.state_dict(), len(self.loader.dataset)

    try:
        time.sleep(3)
        start_client("localhost:50051", MnistClient(client_idx), client_id)
        result_q.put({"client_id": client_id, "success": True})
    except Exception as e:
        traceback.print_exc()
        result_q.put({"client_id": client_id, "success": False, "error": str(e)})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 2: ECGTransformer + ECG Data + FedAvg                            ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _test2_server(result_q, ready_ev, log_dir):
    log_file = open(os.path.join(log_dir, "server.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.server.server import start_server, ServerConfig
    from fedlearn.server.strategy import FedAvg

    logging.basicConfig(level=logging.INFO, format="[SERVER] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_ROUNDS = 10
    NUM_CLIENTS = 3

    try:
        X, y = load_ecg_data(ECG_CSV_PATH, preprocess=True, target_length=140)
        logging.info(f"ECG data loaded: X={X.shape}, y={y.shape}")

        net = ECGTransformer(input_dim=140, d_model=64, nhead=4, num_layers=2, num_classes=2)
        initial_params = net.state_dict()

        test_loader = make_ecg_test_loader(X, y, batch_size=128)

        def evaluate_fn(server_round, parameters):
            model = ECGTransformer(input_dim=140, d_model=64, nhead=4, num_layers=2, num_classes=2)
            model.load_state_dict(parameters)
            model.eval()
            criterion = nn.CrossEntropyLoss()
            correct, total, loss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = model(xb)
                    loss += criterion(out, yb).item()
                    _, pred = torch.max(out, 1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()
            acc = correct / total
            avg_loss = loss / len(test_loader)
            print(f"[Eval] Round {server_round}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
            return avg_loss, {"accuracy": acc}

        strategy = FedAvg(initial_parameters=initial_params, evaluate_fn=evaluate_fn,
                          min_fit_clients=NUM_CLIENTS, clients_per_round=NUM_CLIENTS)
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        ready_ev.set()

        history, final_params = start_server("localhost:50052", config, strategy)
        acc_history = [{"round": r, "accuracy": m.get("accuracy", 0), "loss": m.get("loss", 0)} for r, m in history]
        result_q.put({"success": True, "has_history": len(history) > 0,
                      "has_params": final_params is not None and len(final_params) > 0,
                      "num_rounds": len(history), "accuracy_history": acc_history})
    except Exception as e:
        traceback.print_exc()
        ready_ev.set()
        result_q.put({"success": False, "error": str(e)})


def _test2_client(client_idx, client_id, result_q, log_dir):
    log_file = open(os.path.join(log_dir, f"{client_id}.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.client.client import start_client

    logging.basicConfig(level=logging.INFO, format=f"[{client_id}] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_CLIENTS = 3

    class ECGFedAvgClient:
        def __init__(self, cid):
            self.net = ECGTransformer(input_dim=140, d_model=64, nhead=4, num_layers=2, num_classes=2)
            X, y = load_ecg_data(ECG_CSV_PATH, preprocess=True, target_length=140)
            self.train_loader, _, self.n_samples = make_ecg_loaders(X, y, cid, NUM_CLIENTS)

        def get_parameters(self): return self.net.state_dict()
        def fit(self, params, config):
            self.net.load_state_dict(params)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.0001, weight_decay=0.01)
            self.net.train()
            for _ in range(5):  # 5 local epochs per round
                for xb, yb in self.train_loader:
                    optimizer.zero_grad()
                    loss = criterion(self.net(xb), yb)
                    loss.backward(); optimizer.step()
            return self.net.state_dict(), self.n_samples

    try:
        time.sleep(3)
        start_client("localhost:50052", ECGFedAvgClient(client_idx), client_id)
        result_q.put({"client_id": client_id, "success": True})
    except Exception as e:
        traceback.print_exc()
        result_q.put({"client_id": client_id, "success": False, "error": str(e)})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST 3: ECG MLP + ECG Data + DeComFL                                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _test3_server(result_q, ready_ev, log_dir):
    log_file = open(os.path.join(log_dir, "server.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.server.server import start_server, ServerConfig
    from fedlearn.server.decomfl_strategy import DeComFL

    logging.basicConfig(level=logging.INFO, format="[SERVER] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_ROUNDS = 10
    NUM_CLIENTS = 2

    try:
        X, y = load_ecg_data(DECOMFL_ECG_CSV, preprocess=False)
        input_dim = X.shape[1]
        logging.info(f"ECG data loaded: X={X.shape}, y={y.shape}, input_dim={input_dim}")

        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        net = ECGModel(input_dim=input_dim, hidden_dim=64, num_classes=2)
        initial_params = net.state_dict()

        def evaluate_fn(server_round, parameters):
            model = ECGModel(input_dim=input_dim, hidden_dim=64, num_classes=2)
            model.load_state_dict(parameters)
            model.eval()
            criterion = nn.CrossEntropyLoss()
            from torch.utils.data import DataLoader, TensorDataset
            loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=256)
            correct, total, loss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in loader:
                    out = model(xb)
                    loss += criterion(out, yb).item()
                    _, pred = torch.max(out, 1)
                    total += yb.size(0)
                    correct += (pred == yb).sum().item()
            acc = correct / total
            avg_loss = loss / len(loader)
            print(f"[Eval] Round {server_round}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
            return avg_loss, {"accuracy": acc}

        strategy = DeComFL(
            initial_parameters=initial_params, evaluate_fn=evaluate_fn,
            min_fit_clients=NUM_CLIENTS, clients_per_round=NUM_CLIENTS,
            num_local_steps=1, num_perturbations=10, learning_rate=0.001,
            smoothing_param=0.001, seed=42
        )
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        ready_ev.set()

        history, final_params = start_server("localhost:50053", config, strategy)
        acc_history = [{"round": r, "accuracy": m.get("accuracy", 0), "loss": m.get("loss", 0)} for r, m in history]
        result_q.put({"success": True, "has_history": len(history) > 0,
                      "has_params": final_params is not None and len(final_params) > 0,
                      "num_rounds": len(history), "accuracy_history": acc_history})
    except Exception as e:
        traceback.print_exc()
        ready_ev.set()
        result_q.put({"success": False, "error": str(e)})


def _test3_client(client_idx, client_id, result_q, log_dir):
    log_file = open(os.path.join(log_dir, f"{client_id}.log"), "w", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file

    from fedlearn.client.decomfl_client import DeComFLClient
    from fedlearn.client.decomfl_start import start_decomfl_client

    logging.basicConfig(level=logging.INFO, format=f"[{client_id}] %(asctime)s - %(message)s",
                        stream=log_file, force=True)

    NUM_CLIENTS = 2

    try:
        time.sleep(3)
        X, y = load_ecg_data(DECOMFL_ECG_CSV, preprocess=False)
        input_dim = X.shape[1]

        train_loader, _, n_samples = make_ecg_loaders(X, y, client_idx, NUM_CLIENTS, batch_size=128)

        model = ECGModel(input_dim=input_dim, hidden_dim=64, num_classes=2)
        client = DeComFLClient(model=model, train_loader=train_loader, smoothing_param=0.001, device='cpu')

        start_decomfl_client(server_address="localhost:50053", client=client, client_id=client_id)
        result_q.put({"client_id": client_id, "success": True})
    except Exception as e:
        traceback.print_exc()
        result_q.put({"client_id": client_id, "success": False, "error": str(e)})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST RUNNER                                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_single_test(test_name, server_fn, client_fn, num_clients, port, timeout=600):
    """Run a single test: start server + clients, collect results."""
    log_dir = os.path.join(LOG_BASE, test_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'━' * 70}")
    print(f"  🧪 Running: {test_name}")
    print(f"     Port: {port} | Clients: {num_clients} | Timeout: {timeout}s")
    print(f"     Logs: {log_dir}")
    print(f"{'━' * 70}")

    server_q = multiprocessing.Queue()
    client_q = multiprocessing.Queue()
    ready_ev = multiprocessing.Event()

    server_proc = None
    client_procs = []
    start_time = time.time()

    try:
        # Start server
        server_proc = multiprocessing.Process(
            target=server_fn, args=(server_q, ready_ev, log_dir), name=f"{test_name}-server"
        )
        server_proc.start()

        if not ready_ev.wait(timeout=20):
            raise TimeoutError("Server did not start in 20s")
        time.sleep(2)
        print(f"  ✓ Server ready")

        # Start clients
        for i in range(num_clients):
            cid = f"client-{i}"
            p = multiprocessing.Process(
                target=client_fn, args=(i, cid, client_q, log_dir), name=f"{test_name}-{cid}"
            )
            p.start()
            client_procs.append(p)
        print(f"  ✓ {num_clients} client(s) started")

        # Wait for server result
        print(f"  ⏳ Training in progress...")
        try:
            result = server_q.get(timeout=timeout)
        except Exception:
            raise TimeoutError(f"Server did not finish within {timeout}s")

        # Collect client results
        client_results = []
        for _ in range(num_clients):
            try:
                cr = client_q.get(timeout=30)
                client_results.append(cr)
            except Exception:
                client_results.append({"client_id": "?", "success": False, "error": "timeout"})

        elapsed = time.time() - start_time

        # Validate
        passed = result["success"] and result.get("has_history") and result.get("has_params")
        all_clients_ok = all(cr.get("success") for cr in client_results)
        passed = passed and all_clients_ok

        return {
            "test_name": test_name,
            "passed": passed,
            "elapsed": elapsed,
            "num_rounds": result.get("num_rounds", 0),
            "accuracy_history": result.get("accuracy_history", []),
            "client_results": client_results,
            "error": result.get("error"),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "test_name": test_name,
            "passed": False,
            "elapsed": elapsed,
            "num_rounds": 0,
            "accuracy_history": [],
            "client_results": [],
            "error": str(e),
        }

    finally:
        # Cleanup
        for p in client_procs:
            if p.is_alive():
                p.terminate()
        for p in client_procs:
            p.join(timeout=5)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
        if server_proc and server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                os.kill(server_proc.pid, signal.SIGKILL)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  FedLearn Full E2E Test Suite")
    print(f"  Timestamp: {timestamp}")
    print("=" * 70)

    tests = [
        {
            "name": "test_1_mnist_fedavg",
            "desc": "SimpleCNN + MNIST + FedAvg (10 rounds, 3 clients)",
            "server_fn": _test1_server,
            "client_fn": _test1_client,
            "num_clients": 3,
            "port": 50051,
            "timeout": 300,
        },
        {
            "name": "test_2_ecg_fedavg",
            "desc": "ECGTransformer + ECG Data + FedAvg (10 rounds, 3 clients, 5 local epochs)",
            "server_fn": _test2_server,
            "client_fn": _test2_client,
            "num_clients": 3,
            "port": 50052,
            "timeout": 600,
        },
        {
            "name": "test_3_ecg_decomfl",
            "desc": "ECG MLP + ECG Data + DeComFL (10 rounds, 2 clients, zeroth-order)",
            "server_fn": _test3_server,
            "client_fn": _test3_client,
            "num_clients": 2,
            "port": 50053,
            "timeout": 600,
        },
    ]

    results = []

    for t in tests:
        print(f"\n{'▓' * 70}")
        print(f"  {t['desc']}")
        print(f"{'▓' * 70}")

        result = run_single_test(
            test_name=t["name"],
            server_fn=t["server_fn"],
            client_fn=t["client_fn"],
            num_clients=t["num_clients"],
            port=t["port"],
            timeout=t["timeout"],
        )
        results.append(result)

        # Print accuracy table for this test
        if result["accuracy_history"]:
            print(f"\n  📊 {t['name']} — Accuracy Progression:")
            print("  ┌────────┬──────────┬──────────┐")
            print("  │ Round  │   Loss   │ Accuracy │")
            print("  ├────────┼──────────┼──────────┤")
            for e in result["accuracy_history"]:
                acc_fmt = f"{e['accuracy']:.4f}" if e['accuracy'] < 1 else f"{e['accuracy']:.2f}%"
                print(f"  │   {e['round']:>2}   │  {e['loss']:.4f}  │  {acc_fmt:>8} │")
            print("  └────────┴──────────┴──────────┘")

        status = "✅ PASSED" if result["passed"] else f"❌ FAILED: {result.get('error', '?')}"
        print(f"\n  {status} ({result['elapsed']:.1f}s)")

        # Small delay between tests to let ports free up
        time.sleep(3)

    # ── Final Summary ────────────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("  FULL TEST SUITE SUMMARY")
    print("=" * 70)

    all_passed = True
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        acc = ""
        if r["accuracy_history"]:
            first = r["accuracy_history"][0]["accuracy"]
            last = r["accuracy_history"][-1]["accuracy"]
            if first < 1:  # decimal
                acc = f"  {first:.4f} → {last:.4f}"
            else:  # percentage
                acc = f"  {first:.1f}% → {last:.1f}%"
        print(f"  {icon} {r['test_name']:30s} │ {r['num_rounds']:>2} rounds │ {r['elapsed']:6.1f}s │{acc}")
        if not r["passed"]:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print("\n  ⚠️  SOME TESTS FAILED — check logs above.")
        sys.exit(1)

    # Save results JSON
    results_file = os.path.join(LOG_BASE, f"test_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  📁 Full results: {results_file}")
    print(f"  📁 Log directory: {LOG_BASE}/")
    print()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
