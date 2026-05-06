# 04 — Client Internals

## Table of Contents
- [Overview](#overview)
- [The Client ABC](#the-client-abc)
- [The start_client() Loop](#the-start_client-loop)
  - [Registration Phase](#registration-phase)
  - [Heartbeat Thread](#heartbeat-thread)
  - [Training Loop](#training-loop)
  - [Polling Strategy and Known Limitations](#polling-strategy-and-known-limitations)
  - [Error Handling and Reconnection](#error-handling-and-reconnection)
- [GrpcClient — The Transport Layer](#grpcclient--the-transport-layer)
  - [Channel Lifecycle](#channel-lifecycle)
  - [Status Tracking](#status-tracking)
- [Implementing a Custom Client](#implementing-a-custom-client)
  - [Standard FedAvg Client Example (CNN)](#standard-fedavg-client-example-cnn)
  - [Standard FedAvg Client Example (LLM)](#standard-fedavg-client-example-llm)
- [DeComFL Client](#decomfl-client)
  - [start_decomfl_client()](#start_decomfl_client)
  - [DeComFLClient.fit() Mechanics](#decomflclientfit-mechanics)
  - [Model Rebuild Protocol](#model-rebuild-protocol)
- [Progress Reporting to the Backend](#progress-reporting-to-the-backend)
- [Data Loading Patterns](#data-loading-patterns)

---

## Overview

The client side of the framework has two flavours:

| Flavour | Entry Point | Client Class | Use Case |
|---------|------------|-------------|---------|
| Standard FL | `start_client()` | `Client` ABC | FedAvg, any gradient-based optimizer |
| DeComFL | `start_decomfl_client()` | `DeComFLClient` | Zeroth-order, communication-efficient |

Both flavours share the same `GrpcClient` transport layer and the same outer loop structure. The difference is *what* gets exchanged: standard clients exchange full parameter tensors; DeComFL clients exchange tiny gradient scalar arrays.

---

## The Client ABC

```python
# client/client.py
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
from typing import Tuple

class Client(ABC):
    """Abstract base class for federated learning clients."""

    @abstractmethod
    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return the current local model parameters."""
        pass

    @abstractmethod
    def fit(
        self,
        parameters: OrderedDict[str, torch.Tensor],
        config: dict
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        """
        Train the local model using the provided global parameters.

        Args:
            parameters: Global model weights from the server
            config:     Round-specific configuration dict

        Returns:
            Tuple of (updated_parameters, num_training_examples)
        """
        pass
```

**Design rationale:** By requiring only these two methods, the framework remains compatible with any training approach — backpropagation, meta-learning, quantisation-aware training, etc. The user controls the optimizer, loss function, number of local epochs, and data loading.

---

## The start_client() Loop

```
start_client(server_address, client, client_id)
    │
    ├── 1. Register with server
    │       GrpcClient.register()
    │       → RegisterClient RPC
    │
    ├── 2. Start heartbeat daemon thread
    │       GrpcClient.start_heartbeat()
    │
    ├── 3. Pass GrpcClient to client (for progress updates)
    │       if hasattr(client, 'set_grpc_client'): ...
    │
    └── 4. Training loop (forever until completion signal)
            │
            ├── a. Fetch global model
            │       GrpcClient.get_global_model()
            │       → GetGlobalModelStream RPC
            │
            ├── b. Check completion signal (server_round == -1)
            │
            ├── c. Check for new round (server_round > last_completed_round)
            │
            ├── d. Local training
            │       client.fit(parameters, config)
            │
            ├── e. Submit update
            │       GrpcClient.submit_update(new_params, num_examples, round)
            │       → SubmitModelUpdate or SubmitModelUpdateStream RPC
            │
            └── f. Wait 2 seconds, then poll again
```

### Registration Phase

```python
comm_client = GrpcClient(client_id=client_id, server_address=server_address)

if not comm_client.register():
    log.error("[%s] Could not register with server; exiting", client_id)
    return   # exit cleanly

log.info("[%s] Registered with server; starting heartbeat", client_id)
comm_client.start_heartbeat()
```

If registration fails (e.g., server is not yet running), the client exits rather than entering a broken loop. The orchestrator (Electron app or Spring Boot backend) is responsible for ensuring the server is ready before clients attempt to connect.

### Heartbeat Thread

The heartbeat daemon thread is started immediately after registration:

```python
# GrpcClient.start_heartbeat()
self.heartbeat_active = True
self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
self.heartbeat_thread.start()
```

`daemon=True` means the thread exits automatically when the main process exits, with no explicit cleanup needed. The interval is 5 seconds by default (`heartbeat_interval = 5`).

The heartbeat payload includes real-time progress:

```python
def send_heartbeat(self) -> bool:
    req = HeartbeatRequest(
        client_id=self.client_id,
        status=self.current_status,   # "idle", "training", "submitting_update", etc.
        current_step=self.current_step,
        total_steps=self.total_steps,
        current_round=self.current_round,
    )
    res = self.heartbeat_stub.Heartbeat(req, timeout=30.0)
    if res.should_stop:
        return False   # server wants us to stop
    return res.acknowledged
```

### Training Loop

The main loop is a `while True` that polls continuously:

```python
last_completed_round = -1

while True:
    # Phase 1: Fetch model
    comm_client.update_status("fetching_model", 0, 0)
    parameters, server_round, config = comm_client.get_global_model()

    # Phase 2: Check for completion
    if server_round == -1:
        break

    # Phase 3: Check if this is a new round
    if server_round > last_completed_round:
        comm_client.current_round = server_round
        comm_client.update_status("training", 0, 1)

        # Phase 4: Local training
        new_parameters, num_examples = client.fit(parameters, config)

        # Phase 5: Submit update
        comm_client.update_status("submitting_update", 0, 0)
        success = comm_client.submit_update(new_parameters, num_examples, server_round)

        if success:
            last_completed_round = server_round
            comm_client.update_status("idle", 0, 0)
        else:
            comm_client.update_status("error", 0, 0)
    else:
        # Server hasn't advanced yet — other clients are still training
        comm_client.update_status("waiting", 0, 0)

    time.sleep(2)  # poll interval
```

Status values flow through as heartbeat `status` strings:

| Status String | Meaning |
|--------------|---------|
| `"fetching_model"` | Downloading global model from server |
| `"training"` | Running `client.fit()` |
| `"submitting_update"` | Uploading updated parameters |
| `"idle"` | Waiting for next round to start |
| `"waiting"` | Submitted update, waiting for other clients |
| `"error"` | Last operation failed |

### Polling Strategy and Known Limitations

Currently, clients determine if a new round is available by **polling** `GetGlobalModelStream` every 2 seconds. This has two drawbacks:

1. **Wasteful:** Each poll downloads the full model again even if the round hasn't changed.
2. **Latency:** There's a 0–2 second delay between the server advancing a round and the client discovering it.

A better design (noted as a TODO in the code) would use a **server-streaming RPC** called `WaitForNextRound` that blocks until `coordinator._round_complete_event` fires and then pushes the new round metadata. This would give near-instant round notification with zero polling overhead.

### Error Handling and Reconnection

The loop catches errors at two levels:

```python
try:
    # inner loop
    ...
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        log.warning("[%s] Server unavailable; shutting down", client_id)
        break  # server has shut down after training — exit cleanly
    log.warning("[%s] gRPC error (code=%s): %s. Retrying in 10s", ...)
    comm_client.update_status("error", 0, 0)
    time.sleep(10)   # backoff before retrying
except Exception:
    log.exception("[%s] Unexpected client error; shutting down", client_id)
    comm_client.update_status("error", 0, 0)
    break  # unknown error = exit so orchestrator can restart cleanly
```

`UNAVAILABLE` from `GetGlobalModel` typically means the server's `grpc_server.stop()` was called (normal post-training shutdown). This is treated as a clean exit.

All other `grpc.RpcError` codes get a 10-second retry. Non-gRPC exceptions (Python bugs, out-of-memory) exit immediately — letting the orchestrator restart with a clean slate is safer than continuing in an unknown state.

---

## GrpcClient — The Transport Layer

`GrpcClient` wraps all network I/O and presents a clean Python API to the `start_client()` loop:

```python
class GrpcClient:
    def register(self) -> bool
    def get_global_model(self) -> (params, round, config)
    def submit_update(self, params, num_examples, round_number) -> bool
    def send_heartbeat(self) -> bool
    def start_heartbeat(self)
    def stop_heartbeat(self)
    def update_status(self, status, current_step, total_steps)
    def close(self)

    # DeComFL-specific
    def get_decomfl_config(self) -> (round, seeds, rebuild_history, config)
    def submit_gradient_scalars(self, gradient_scalars, num_examples, round_num) -> bool
```

### Channel Lifecycle

```python
def __init__(self, client_id, server_address):
    grpc_options = [...]

    # Primary channel for all heavy operations
    self.channel = _build_channel(server_address, grpc_options)
    self.stub = FederatedLearningServiceStub(self.channel)

    # Dedicated heartbeat channel
    self.heartbeat_channel = _build_channel(server_address, grpc_options)
    self.heartbeat_stub = FederatedLearningServiceStub(self.heartbeat_channel)

def close(self):
    self.stop_heartbeat()
    self.channel.close()
    if self.heartbeat_channel:
        self.heartbeat_channel.close()
```

Channels are created once and reused for the lifetime of the client. gRPC handles connection pooling and reconnection internally.

### Status Tracking

The client maintains a small piece of mutable state for heartbeat payloads:

```python
def update_status(self, status: str, current_step: int, total_steps: int):
    self.current_status = status
    self.current_step = current_step
    self.total_steps = total_steps
```

This is called from two places:
1. **`start_client()` loop** — for coarse-grained status transitions (`"training"`, `"idle"`, etc.)
2. **Inside `client.fit()`** — for fine-grained step-level progress, if the client has a reference to `grpc_client`

---

## Implementing a Custom Client

### Standard FedAvg Client Example (CNN)

```python
import fedlearn as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1   = nn.Linear(9216, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MNISTClient(fl.Client):
    def __init__(self, model, train_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.grpc_client = None  # set by framework via set_grpc_client()

    def set_grpc_client(self, grpc_client):
        """Optional: receive reference to GrpcClient for progress updates."""
        self.grpc_client = grpc_client

    def get_parameters(self):
        """Return current model weights."""
        return self.model.state_dict()

    def fit(self, parameters, config):
        """
        Load global parameters, run local training, return updated params.

        Args:
            parameters: Global model state_dict from server
            config:     Dict with round-specific settings (currently empty for FedAvg)

        Returns:
            (updated_state_dict, num_training_samples)
        """
        # 1. Load global parameters into local model
        self.model.load_state_dict(parameters)
        self.model.train()

        local_epochs = 1
        total_steps = local_epochs * len(self.train_loader)
        step = 0

        # 2. Local training
        for epoch in range(local_epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                step += 1
                # 3. Report progress back to server via heartbeat
                if self.grpc_client:
                    self.grpc_client.update_status("training", step, total_steps)

        # 4. Return updated weights and dataset size
        return self.model.state_dict(), len(self.train_loader.dataset)


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="client_0")
    parser.add_argument("--server_address", default="localhost:50051")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNNModel()
    client = MNISTClient(model, train_loader, device="cpu")

    fl.client.start_client(
        server_address=args.server_address,
        client=client,
        client_id=args.id,
    )
```

### Standard FedAvg Client Example (LLM)

LLM clients are nearly identical, but `fit()` works with a HuggingFace `Trainer` or custom loop:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import OrderedDict

class LLMClient(fl.Client):
    def __init__(self, model_name, train_dataset, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.train_dataset = train_dataset
        self.grpc_client = None

    def set_grpc_client(self, gc):
        self.grpc_client = gc

    def get_parameters(self):
        return self.model.state_dict()

    def fit(self, parameters, config):
        # Load global parameters
        self.model.load_state_dict(parameters)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)  # HuggingFace returns a dict with 'loss'
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.model.state_dict(), len(self.train_dataset)
```

> **Important:** When using HuggingFace models, the `GrpcClient.submit_update()` will automatically detect transformer parameter names (containing `attention`, `encoder`, etc.) and use streaming upload. No special handling needed.

---

## DeComFL Client

### start_decomfl_client()

```python
# client/decomfl_start.py
def start_decomfl_client(server_address, client, client_id):
    comm_client = GrpcClient(client_id=client_id, server_address=server_address)
    last_completed_round = -1

    if not comm_client.register():
        return

    comm_client.start_heartbeat()
    client.set_grpc_client(comm_client)

    while True:
        try:
            # Fetch seeds and rebuild history instead of model weights
            current_round, seeds, rebuild_history, config = comm_client.get_decomfl_config()

            if current_round == -1:
                break   # training complete

            if current_round > last_completed_round:
                # Replay missed rounds (rebuilds local model from seeds+gradients)
                if rebuild_history:
                    client.rebuild_model(rebuild_history, float(config.get('learning_rate', 0.001)))

                # Inject seeds into config for fit()
                config['seeds'] = seeds

                # Local training (returns gradient scalars, not updated weights)
                gradient_scalars, num_examples = client.fit(parameters=None, config=config)

                # Submit O(K×P) scalars instead of full model
                success = comm_client.submit_gradient_scalars(
                    gradient_scalars, num_examples, current_round
                )

                if success:
                    last_completed_round = current_round
            else:
                time.sleep(2)   # wait for other clients

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                break
            time.sleep(10)

    comm_client.stop_heartbeat()
    comm_client.close()
```

### DeComFLClient.fit() Mechanics

The `fit()` method on `DeComFLClient` implements **Algorithm 4** from the DeComFL paper. Instead of computing gradients via backpropagation, it uses a **forward-difference zeroth-order gradient estimator**:

```
g = (f(x + μz; ξ) - f(x; ξ)) / μ
```

where:
- `x` = current flat model parameters
- `z` = random perturbation vector sampled from N(0, I)
- `μ` (mu) = smoothing parameter (typically 0.001)
- `ξ` = mini-batch of training data
- `f(·)` = loss function

Full implementation:

```python
def fit(self, parameters, config):
    seeds = config['seeds']    # List[List[int]] — [local_step][perturbation]
    K = len(seeds)             # Number of local steps
    P = len(seeds[0])          # Number of perturbations per step
    eta = float(config.get('learning_rate', 0.001))

    total_perturbation = torch.zeros_like(self.x_current)
    gradient_scalars = []
    data_iter = iter(self.train_loader)

    for k in range(K):
        delta = torch.zeros_like(self.x_current)
        k_gradient_scalars = []

        # Get data batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_loader)
            batch = next(data_iter)

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Progress reporting
        if self.grpc_client:
            self.grpc_client.update_status("training", k + 1, K)

        for p in range(P):
            # Generate z from shared seed (reproducible on server + client)
            z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))

            # Compute ZO gradient scalar
            g = self.zo_estimator.compute_gradient_scalar(
                self.model, self.x_current, z, inputs, targets
            )
            k_gradient_scalars.append(g)

            # Accumulate update direction
            delta += g * z

        # Apply local step update
        step_update = (eta / P) * delta
        self.x_current -= step_update
        total_perturbation -= step_update

        gradient_scalars.append(k_gradient_scalars)

    # CRITICAL: Revert x_current to pre-round state
    # The server will apply the update using its own reconstruction.
    # The client must NOT keep the locally perturbed model.
    self.x_current -= total_perturbation
    self.zo_estimator._set_flat_params(self.model, self.x_current)

    return gradient_scalars, len(self.train_loader.dataset)
```

> **Why revert `x_current`?** After local training, the client reverts its model to the state before this round's perturbations. This is necessary because the client's model will be advanced by the server's averaged update (communicated as seeds + average gradients) through the `rebuild_model()` call at the start of the *next* round. If the client kept its locally perturbed model, it would diverge from the global model that the server constructs.

### ZerothOrderEstimator.compute_gradient_scalar()

```python
def compute_gradient_scalar(self, model, flat_params, perturbation, inputs, targets):
    model.eval()
    with torch.no_grad():
        # Baseline loss: f(x; ξ)
        self._set_flat_params(model, flat_params)
        if isinstance(inputs, dict):   # LLM path
            loss_x = model(**inputs, labels=targets).loss
        else:                          # CNN/MLP path
            loss_x = nn.CrossEntropyLoss()(model(inputs), targets)

        # Perturbed loss: f(x + μz; ξ)
        perturbed_params = flat_params + self.mu * perturbation
        self._set_flat_params(model, perturbed_params)
        if isinstance(inputs, dict):
            loss_x_perturbed = model(**inputs, labels=targets).loss
        else:
            loss_x_perturbed = nn.CrossEntropyLoss()(model(inputs), targets)

        # ZO gradient scalar
        g = (loss_x_perturbed - loss_x) / self.mu

    return g.item()   # Python float
```

This requires **two forward passes** per perturbation and **no backward pass**. For a model with `d` parameters and `P` perturbations, communication is `O(K × P)` scalars vs. `O(d)` for standard FL — a massive reduction for LLMs.

### Model Rebuild Protocol

When a client joins late or misses rounds (e.g., temporary network disconnection), it must replay the missed rounds to synchronise its local model:

```python
def rebuild_model(self, rebuild_history, learning_rate):
    """
    Replays missed rounds using server-provided seeds and averaged gradients.
    Each missed round is re-simulated deterministically.
    """
    for round_data in rebuild_history:
        seeds = round_data['seeds']       # List[List[int]]
        avg_gradients = round_data['gradients']  # List[List[float]]

        K = len(seeds)
        P = len(seeds[0])

        for k in range(K):
            delta = torch.zeros_like(self.x_current)
            for p in range(P):
                # Regenerate exact same perturbation vector
                z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))
                g = avg_gradients[k][p]   # averaged gradient from server
                delta += g * z

            # Apply the averaged global update for this step
            self.x_current -= (learning_rate / P) * delta

    # Sync model weights
    self.zo_estimator._set_flat_params(self.model, self.x_current)
```

The rebuild is deterministic because both the client and server use the same random seeds to regenerate perturbation vectors. The only information transmitted is the scalar gradient values — the full perturbation vectors are regenerated locally.

---

## Progress Reporting to the Backend

The Spring Boot backend displays real-time training progress to the user. This works through a chain:

```
client.fit() step N
    │
    ▼
grpc_client.update_status("training", N, total_steps)
    │
    ▼ (every 5s, heartbeat thread)
HeartbeatRequest { status="training", current_step=N, total_steps=total_steps }
    │
    ▼ (gRPC to Python server)
FLCoordinator.update_client_heartbeat()
    │ updates client_heartbeats dict
    │
    ▼ (Spring backend polls /api/projects/{id}/clients or reads server stdout logs)
React frontend receives progress update
```

For the `GrpcClient` reference to be available inside `fit()`, the client must implement `set_grpc_client()`:

```python
class MyClient(fl.Client):
    def __init__(self, ...):
        self.grpc_client = None   # will be set by framework

    def set_grpc_client(self, gc):
        self.grpc_client = gc     # called by start_client() before training loop

    def fit(self, parameters, config):
        for step, batch in enumerate(self.train_loader):
            # ... training ...
            if self.grpc_client:
                self.grpc_client.update_status("training", step, len(self.train_loader))
```

---

## Data Loading Patterns

### Non-IID Dirichlet Partitioning

For realistic federated scenarios, data should be non-uniformly distributed across clients:

```python
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets

def partition_dataset_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition dataset using Dirichlet distribution for non-IID splits.

    Args:
        dataset:     Full dataset
        num_clients: Number of clients
        alpha:       Dirichlet concentration parameter
                     - alpha → 0: one class per client (extreme non-IID)
                     - alpha → ∞: uniform IID split
        seed:        Random seed for reproducibility

    Returns:
        List of indices for each client
    """
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(set(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        # Dirichlet distribution determines how this class is split
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        for i, split_indices in enumerate(np.split(class_indices, proportions)):
            client_indices[i].extend(split_indices.tolist())

    return [Subset(dataset, indices) for indices in client_indices]


# Usage
full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
client_datasets = partition_dataset_dirichlet(full_dataset, num_clients=5, alpha=0.5)

# Client 0 gets a DataLoader over its personal partition
client_0_loader = DataLoader(client_datasets[0], batch_size=32, shuffle=True)
```
