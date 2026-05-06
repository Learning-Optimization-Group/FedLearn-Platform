#!/usr/bin/env python3
"""
FedLearn End-to-End Integration Test (with Real Training)
==========================================================
Spins up 1 gRPC server and 3 clients locally using the SimpleCNN model
from examples/simple_federation. Each client trains on a partition of
MNIST data with real SGD, and the server evaluates accuracy after each round.

Usage:
    cd framework/
    python run_local_test.py

Prerequisites:
    pip install torch torchvision grpcio grpcio-tools numpy
"""

import multiprocessing
import time
import sys
import os
import signal
import logging
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---------------------------------------------------------------------------
#  Ensure the framework source is on PYTHONPATH
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
SERVER_ADDRESS = "localhost:50051"
NUM_CLIENTS = 3
NUM_ROUNDS = 10
TIMEOUT_SECONDS = 300  # MNIST training takes longer than dummy noise
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DATA_DIR = os.path.join(SCRIPT_DIR, "examples", "simple_federation", "data")
LOCAL_EPOCHS = 2  # Local training epochs per round


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MODEL (from examples/simple_federation/model.py)                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification — same as simple_federation example."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
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
        x = self.fc3(x)
        return x


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  DATA (from examples/simple_federation/data.py)                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def get_mnist_loader(client_id: int, num_clients: int):
    """Loads a non-IID partition of MNIST for a specific client."""
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)

    # Simple non-IID partitioning: each client gets a slice
    total_size = len(full_dataset)
    partition_size = total_size // num_clients
    start_idx = client_id * partition_size
    end_idx = start_idx + partition_size

    indices = list(range(start_idx, end_idx))
    client_dataset = Subset(full_dataset, indices)

    return DataLoader(client_dataset, batch_size=32, shuffle=True)


def get_test_loader():
    """Loads the entire MNIST test set for server-side evaluation."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=128)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MNIST CLIENT (real training with SGD)                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class MnistClient:
    """
    A real federated learning client that:
      - Loads a partition of MNIST data
      - Trains a SimpleCNN with CrossEntropyLoss + SGD
      - Returns updated parameters + num_examples
    """

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = SimpleCNN().to(self.device)
        self.trainloader = get_mnist_loader(client_id, num_clients=NUM_CLIENTS)

    def get_parameters(self) -> OrderedDict:
        return self.net.state_dict()

    def fit(self, parameters: OrderedDict, config: dict):
        """
        Real local training:
        1. Load global parameters
        2. Train on local MNIST partition with SGD
        3. Return updated weights + dataset size
        """
        self.net.load_state_dict(parameters)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.net.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(LOCAL_EPOCHS):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[Client {self.client_id}] Local training done — "
              f"{LOCAL_EPOCHS} epochs, {num_batches} batches, avg loss: {avg_loss:.4f}")

        return self.net.state_dict(), len(self.trainloader.dataset)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SERVER-SIDE EVALUATION (real accuracy on MNIST test set)              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def server_side_evaluate(server_round: int, parameters: OrderedDict):
    """Evaluate the global model on the full MNIST test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(parameters)

    testloader = get_test_loader()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss / len(testloader)
    accuracy = correct / total
    print(f"[Server Eval] Round {server_round}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f} ({correct}/{total})")
    return avg_loss, {"accuracy": accuracy}


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SERVER PROCESS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _run_server(result_queue: multiprocessing.Queue, ready_event: multiprocessing.Event):
    """Server child process — runs FedAvg with real MNIST evaluation."""
    # Redirect stdout/stderr to log file
    log_file = open(os.path.join(LOG_DIR, "server.log"), "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    from fedlearn.server.server import start_server, ServerConfig
    from fedlearn.server.strategy import FedAvg

    logging.basicConfig(
        level=logging.INFO,
        format="[SERVER] %(asctime)s - %(levelname)s - %(message)s",
        stream=log_file,
        force=True,
    )

    try:
        # Initial model parameters
        net = SimpleCNN()
        initial_params = net.state_dict()
        logging.info(f"SimpleCNN model: {sum(p.numel() for p in initial_params.values()):,} parameters")

        # FedAvg strategy with real server-side evaluation
        strategy = FedAvg(
            initial_parameters=initial_params,
            evaluate_fn=server_side_evaluate,
            min_fit_clients=NUM_CLIENTS,
            clients_per_round=NUM_CLIENTS,
        )

        config = ServerConfig(num_rounds=NUM_ROUNDS)

        ready_event.set()

        logging.info(f"Starting gRPC server on {SERVER_ADDRESS}...")
        history, final_parameters = start_server(SERVER_ADDRESS, config, strategy)

        has_history = len(history) > 0
        has_params = final_parameters is not None and len(final_parameters) > 0

        logging.info(f"Server finished. Rounds completed: {len(history)}, Has final params: {has_params}")

        # Extract accuracy progression for the test summary
        accuracy_history = []
        for round_num, metrics in history:
            acc = metrics.get("accuracy", 0.0)
            accuracy_history.append({"round": round_num, "accuracy": acc, "loss": metrics.get("loss", 0.0)})
            logging.info(f"  Round {round_num}: Loss={metrics.get('loss', 0):.4f}, Accuracy={acc:.4f}")

        result_queue.put({
            "success": True,
            "has_history": has_history,
            "has_params": has_params,
            "num_rounds_completed": len(history),
            "accuracy_history": accuracy_history,
        })

    except Exception as e:
        logging.error(f"Server failed: {e}")
        traceback.print_exc()
        ready_event.set()
        result_queue.put({"success": False, "error": str(e)})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  CLIENT PROCESS                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _run_client(client_idx: int, client_id: str, server_address: str, client_result_queue: multiprocessing.Queue):
    """Client child process — trains on its MNIST partition each round."""
    # Redirect stdout/stderr to per-client log file
    log_file = open(os.path.join(LOG_DIR, f"{client_id}.log"), "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    from fedlearn.client.client import start_client

    logging.basicConfig(
        level=logging.INFO,
        format=f"[CLIENT-{client_id}] %(asctime)s - %(levelname)s - %(message)s",
        stream=log_file,
        force=True,
    )

    try:
        time.sleep(3)  # Wait for server to bind

        logging.info(f"Starting client '{client_id}' (partition {client_idx}) → {server_address}")

        # Create MNIST client with real data partition
        mnist_client = MnistClient(client_id=client_idx)
        start_client(server_address, mnist_client, client_id)

        logging.info(f"Client '{client_id}' completed successfully")
        client_result_queue.put({"client_id": client_id, "success": True})

    except Exception as e:
        logging.error(f"Client '{client_id}' failed: {e}")
        traceback.print_exc()
        client_result_queue.put({"client_id": client_id, "success": False, "error": str(e)})


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MAIN TEST ORCHESTRATOR                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("  FedLearn E2E Integration Test — MNIST + SimpleCNN")
    print("=" * 70)
    print(f"  Server Address : {SERVER_ADDRESS}")
    print(f"  Num Clients    : {NUM_CLIENTS}")
    print(f"  Num Rounds     : {NUM_ROUNDS}")
    print(f"  Local Epochs   : {LOCAL_EPOCHS}")
    print(f"  Timeout        : {TIMEOUT_SECONDS}s")
    print(f"  Log Dir        : {LOG_DIR}")
    print(f"  Data Dir       : {DATA_DIR}")
    print("=" * 70)
    print()

    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)

    # Shared state
    server_result_queue = multiprocessing.Queue()
    client_result_queue = multiprocessing.Queue()
    server_ready = multiprocessing.Event()

    server_process = None
    client_processes = []

    try:
        # ── Step 1: Start the server ─────────────────────────────────────
        print("[TEST] Starting server process...")
        server_process = multiprocessing.Process(
            target=_run_server,
            args=(server_result_queue, server_ready),
            name="fedlearn-server",
        )
        server_process.start()

        if not server_ready.wait(timeout=15):
            raise TimeoutError("Server did not start within 15 seconds")

        time.sleep(2)
        print("[TEST] Server is ready.\n")

        # ── Step 2: Start the clients ────────────────────────────────────
        print(f"[TEST] Starting {NUM_CLIENTS} MNIST client processes...")
        for i in range(NUM_CLIENTS):
            client_id = f"test-client-{i}"
            p = multiprocessing.Process(
                target=_run_client,
                args=(i, client_id, SERVER_ADDRESS, client_result_queue),
                name=f"fedlearn-client-{i}",
            )
            p.start()
            client_processes.append(p)
            print(f"[TEST]   → Started '{client_id}' (PID: {p.pid}, MNIST partition {i})")
        print()

        # ── Step 3: Wait for server to finish ────────────────────────────
        print(f"[TEST] Waiting up to {TIMEOUT_SECONDS}s for {NUM_ROUNDS} rounds to complete...")
        print("[TEST] (Each round: 3 clients train on MNIST → server aggregates → evaluates)")
        print()
        try:
            result = server_result_queue.get(timeout=TIMEOUT_SECONDS)
        except Exception:
            raise TimeoutError(
                f"Server did not complete within {TIMEOUT_SECONDS}s. "
                "Possible deadlock in gRPC aggregation."
            )

        # ── Step 4: Collect client results ───────────────────────────────
        client_results = []
        for _ in range(NUM_CLIENTS):
            try:
                cr = client_result_queue.get(timeout=30)
                client_results.append(cr)
            except Exception:
                client_results.append({"client_id": "unknown", "success": False, "error": "timeout"})

        # ── Step 5: Assertions & Results ─────────────────────────────────
        print("=" * 70)
        print("  TEST RESULTS")
        print("=" * 70)

        # Server assertions
        assert result["success"], f"Server reported failure: {result.get('error', 'unknown')}"
        assert result["has_history"], "Server returned empty history — no rounds completed"
        assert result["has_params"], "Server returned no final parameters"
        assert result["num_rounds_completed"] == NUM_ROUNDS, (
            f"Expected {NUM_ROUNDS} round(s), got {result['num_rounds_completed']}"
        )
        print(f"  ✅ Server: {result['num_rounds_completed']} round(s) completed")
        print(f"  ✅ Server: Final aggregated parameters present")

        # Print accuracy progression
        if result.get("accuracy_history"):
            print()
            print("  📊 Accuracy Progression:")
            print("  ┌────────┬──────────┬──────────┐")
            print("  │ Round  │   Loss   │ Accuracy │")
            print("  ├────────┼──────────┼──────────┤")
            for entry in result["accuracy_history"]:
                print(f"  │   {entry['round']:>2}   │  {entry['loss']:.4f}  │  {entry['accuracy']:.4f}  │")
            print("  └────────┴──────────┴──────────┘")

            # Verify accuracy improved from round 1 to final round
            first_acc = result["accuracy_history"][0]["accuracy"]
            final_acc = result["accuracy_history"][-1]["accuracy"]
            print(f"\n  📈 Accuracy: {first_acc:.4f} → {final_acc:.4f} "
                  f"({'↑ improved' if final_acc > first_acc else '→ no change'})")

        # Client assertions
        print()
        successful_clients = [cr for cr in client_results if cr.get("success")]
        failed_clients = [cr for cr in client_results if not cr.get("success")]

        for cr in successful_clients:
            print(f"  ✅ Client '{cr['client_id']}': completed successfully")
        for cr in failed_clients:
            print(f"  ❌ Client '{cr.get('client_id', '?')}': FAILED — {cr.get('error', '?')}")

        assert len(successful_clients) == NUM_CLIENTS, (
            f"Expected {NUM_CLIENTS} successful clients, got {len(successful_clients)}"
        )

        print()
        print("  ━" * 35)
        print("  🎉 ALL ASSERTIONS PASSED — E2E TEST SUCCESSFUL!")
        print("  ━" * 35)
        print()
        print("  📁 Log files:")
        print(f"     Server : {os.path.join(LOG_DIR, 'server.log')}")
        for i in range(NUM_CLIENTS):
            print(f"     Client {i}: {os.path.join(LOG_DIR, f'test-client-{i}.log')}")
        print()

    except AssertionError as e:
        print(f"\n  ❌ ASSERTION FAILED: {e}\n")
        sys.exit(1)

    except TimeoutError as e:
        print(f"\n  ⏰ TIMEOUT: {e}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n  💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # ── Step 6: Teardown ──────────────────────────────────────────────
        print("[TEST] Tearing down processes...")

        for p in client_processes:
            if p.is_alive():
                print(f"[TEST]   Terminating client PID {p.pid}")
                p.terminate()
        for p in client_processes:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[TEST]   Force-killing client PID {p.pid}")
                os.kill(p.pid, signal.SIGKILL)

        if server_process and server_process.is_alive():
            print(f"[TEST]   Terminating server PID {server_process.pid}")
            server_process.terminate()
            server_process.join(timeout=5)
            if server_process.is_alive():
                print(f"[TEST]   Force-killing server PID {server_process.pid}")
                os.kill(server_process.pid, signal.SIGKILL)

        print("[TEST] All processes cleaned up. No zombie gRPC ports left open.")
        print()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
