# 04 — Client Internals

## Table of Contents
- [Overview](#overview)
- [The Client ABC](#the-client-abc)
- [LocalTrainer — The Shipped First-Order Client](#localtrainer--the-shipped-first-order-client)
- [The start_client() Loop](#the-start_client-loop)
  - [Registration Phase](#registration-phase)
  - [Heartbeat Thread](#heartbeat-thread)
  - [Training Loop](#training-loop)
  - [Server-Driven Stop (FR-10)](#server-driven-stop-fr-10)
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
  - [Syncing the Initial Model and the Applied-Round Watermark](#syncing-the-initial-model-and-the-applied-round-watermark)
  - [DeComFLClient.fit() Mechanics](#decomflclientfit-mechanics)
  - [Model Rebuild Protocol](#model-rebuild-protocol)
- [Progress Reporting to the Backend](#progress-reporting-to-the-backend)
- [Data Loading Patterns](#data-loading-patterns)

---

## Overview

The client side of the framework has two flavours:

| Flavour | Entry Point | Client Class | Use Case |
|---------|------------|-------------|---------|
| Standard FL | `start_client()` | `Client` ABC — or the shipped `LocalTrainer` | FedAvg / FedProx / FedOpt / FedLoRA / robust |
| DeComFL | `start_decomfl_client()` | `DeComFLClient` | Zeroth-order, communication-efficient |

Both flavours share the same `GrpcClient` transport layer and the same outer loop structure. The
difference is *what* gets exchanged: standard clients exchange full parameter tensors; DeComFL
clients exchange tiny gradient scalar arrays. There is a third way to run either one with no
transport at all — the in-process simulator calls `client.fit(...)` directly (see
[01 — The In-Process Simulator](01_architecture_overview.md#the-in-process-simulator)); a `Client`
that works under `start_client()` works there unchanged.

> **`start_decomfl_client()` returns a terminal outcome; `start_client()` returns `None`.** The
> DeComFL entry point returns one of `OUTCOME_COMPLETED` / `OUTCOME_DISCONNECTED` /
> `OUTCOME_ERROR`, so a caller (or a test) can tell a normal end-of-run from a real disconnect and
> pick an exit code. `start_client()` has no such distinction yet.
>
> One gap to code defensively against: the two **FR-1 initial-download failure paths**
> (`decomfl_start.py:68` and `:77` — the `get_global_model` RPC raised, or the server returned no
> global model) `return` bare, so the caller sees `None` rather than `OUTCOME_ERROR`. Treat any
> non-`OUTCOME_COMPLETED` return, `None` included, as a failure.

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

Two **optional** hooks the framework calls if present (neither is on the ABC):

| Hook | Called by | Purpose |
|---|---|---|
| `set_grpc_client(gc)` | `start_client()` / `start_decomfl_client()` before the loop | gives `fit()` a handle for progress updates and for polling `should_stop_training()` |
| `get_client_config()` — *server-side*, on the **Strategy** | `FLCoordinator._strategy_client_config()` | how `proximal_mu` / `learning_rate` / `local_epochs` reach `fit()`'s `config` |

---

## LocalTrainer — The Shipped First-Order Client

You do not have to hand-write a `Client` for the first-order family.
`fedlearn.client.local_trainer.LocalTrainer` (exported as `fl.LocalTrainer`) is a concrete `Client`
that runs ordinary minibatch SGD and is the counterpart to `FedProx` / `FedOpt` on the server:

```python
import fedlearn as fl

client = fl.LocalTrainer(model=MyModel(), train_loader=loader, device="cpu")
fl.client.start_client("localhost:50051", client, "client_0")
```

It reads three keys out of the round `config`, all of which may arrive as `str` (they travel through
a protobuf `map<string,string>`) and are coerced:

| Key | Default | Effect |
|---|---|---|
| `learning_rate` | `0.01` | SGD learning rate |
| `local_epochs` | `1` | passes over `train_loader` |
| `proximal_mu` | `0.0` | FedProx penalty strength; `0.0` is exactly the FedAvg client |

**How the FedProx term is applied — it is a gradient, not a loss addition.** `LocalTrainer` snapshots
the round's starting global model as an anchor (only when `mu > 0`), then after `loss.backward()`
adds the penalty's exact gradient contribution in place before stepping:

```python
global_anchor = [p.detach().clone() for p in self.model.parameters()] if mu > 0.0 else None
...
loss.backward()
if mu > 0.0:
    for p, w0 in zip(self.model.parameters(), global_anchor):
        if p.grad is not None:
            p.grad.add_(p.detach() - w0, alpha=mu)     # mu * (w - w_global)
optimizer.step()
```

Two consequences worth knowing:

- **`mu = 0` skips the anchor entirely**, so a FedProx run with `mu = 0` is bitwise the FedAvg
  client — there is no residual cost.
- The penalty's own iteration is `w ← w − lr·mu·(w − w_global)`, a linear map with multiplier
  `(1 − lr·mu)`. It contracts toward the anchor only while `0 < lr·mu < 2`. `FedProx` refuses to
  construct outside that envelope — see
  [05 — FedProx](05_strategies.md#fedprox--proximal-regularisation-shipped).

It returns `(state_dict as detached CPU clones, len(train_loader.dataset))`, and polls
`should_stop_training()` between minibatches (FR-10).

---

## The start_client() Loop

```
start_client(server_address, client, client_id)
    │
    ├── 1. Register with server
    │       GrpcClient.register()
    │       → RegisterClient RPC       (exits immediately if this fails)
    │
    ├── 2. Start heartbeat daemon thread
    │       GrpcClient.start_heartbeat()
    │
    ├── 3. Pass GrpcClient to client (for progress updates + the stop latch)
    │       if hasattr(client, 'set_grpc_client'): ...
    │
    ├── 4. Training loop (until a completion signal, a stop, or a fatal error)
    │       │
    │       ├── 0. FR-10: if should_stop_training() → break BEFORE the round
    │       │
    │       ├── a. Fetch global model
    │       │       GrpcClient.get_global_model()
    │       │       → GetGlobalModelStream RPC
    │       │
    │       ├── b. Check completion signal (server_round == -1) → break
    │       │
    │       ├── c. New round? (server_round > last_completed_round)
    │       │       NO  → status "waiting", sleep 2s, poll again
    │       │       YES → continue
    │       │
    │       ├── d. Local training
    │       │       client.fit(parameters, config)
    │       │
    │       ├── e. FR-10: if should_stop_training() → DISCARD the partial
    │       │       update and break (never submit a half-round)
    │       │
    │       └── f. Submit update
    │               GrpcClient.submit_update(new_params, num_examples, round)
    │               → SubmitModelUpdate or SubmitModelUpdateStream RPC
    │
    └── 5. finally: stop_heartbeat() + close()   ← always runs, on every exit path
```

> **The 2-second sleep is only in the "no new round" branch.** A client that just completed a round
> loops straight back to `get_global_model()` with no delay.

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

`daemon=True` means the thread exits automatically when the main process exits, with no explicit cleanup needed. The interval is 5 seconds by default (`heartbeat_interval = 5`), and the inter-beat wait is an **interruptible `Event.wait`**, not `time.sleep` — `stop_heartbeat()` ends it at once instead of the thread having to sleep out the full interval (FR-10).

The heartbeat payload includes real-time progress, read through a lock so it can never observe a
torn write:

```python
def send_heartbeat(self) -> bool:
    status, current_step, total_steps, current_round = self._status_snapshot()  # under _status_lock
    req = HeartbeatRequest(
        client_id=self.client_id,
        status=status,                # "idle", "training", "submitting_update", …
        current_step=current_step,
        total_steps=total_steps,
        current_round=current_round,
    )
    res = self.heartbeat_stub.Heartbeat(req, timeout=30.0)
    if res.should_stop:
        self._stop_training.set()     # FR-10: LATCH it for the training thread
        return False
    return res.acknowledged
```

> **Why the status triple needs a lock.** `update_status()` is called from the *training* thread and
> `_status_snapshot()` from the *heartbeat* thread. Three bare attribute stores are not atomic as a
> unit, so without `_status_lock` a heartbeat could report the status of one phase with the step
> count of another — a confusing dashboard, and a misleading progress signal to the backend.

### Training Loop

The main loop is a `while True` that polls continuously:

```python
last_completed_round = -1

while True:
    # Phase 0: FR-10 — server-driven stop, checked before starting another round
    if comm_client.should_stop_training():
        break

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

        # Phase 4b: FR-10 — fit() aborted early, so these params are a PARTIAL round.
        # Discard them; never submit a half-round.
        if comm_client.should_stop_training():
            break

        # Phase 5: Submit update
        comm_client.update_status("submitting_update", 0, 0)
        if comm_client.submit_update(new_parameters, num_examples, server_round):
            last_completed_round = server_round
            comm_client.update_status("idle", 0, 0)
        else:
            comm_client.update_status("error", 0, 0)
    else:
        # Server hasn't advanced yet — other clients are still training.
        # The sleep lives HERE, in this branch only: a client that just finished a round
        # loops straight back to get_global_model() with no delay.
        comm_client.update_status("waiting", 0, 0)
        time.sleep(2)
```

Status values flow through as heartbeat `status` strings. Note the first two rows: `start_client()`
sets `"fetching_model"`, then `GrpcClient.get_global_model()` **immediately overwrites it** with
`"downloading_model"`, so `"fetching_model"` is essentially never observed on the wire — a heartbeat
during a download reports `"downloading_model"`.

| Status String | Set by | Meaning |
|--------------|--------|---------|
| `"fetching_model"` | `start_client()` | About to download — overwritten within microseconds |
| `"downloading_model"` | `GrpcClient.get_global_model()` | Downloading the global model |
| `"training"` | `start_client()` then `client.fit()` | Running local training; `fit()` refines `current_step`/`total_steps` |
| `"submitting_update"` | `start_client()` **and** `GrpcClient.submit_update()` | Uploading updated parameters |
| `"idle"` | `start_client()` | Update accepted; waiting for the next round |
| `"waiting"` | `start_client()` | Server has not advanced yet — other clients still training |
| `"error"` | `start_client()` | Last operation failed |
| `"fetching_config"` / `"rebuilding"` | `start_decomfl_client()` | DeComFL-only phases |

### Server-Driven Stop (FR-10)

The server can end a client's round mid-flight, and the mechanism is worth understanding because it
crosses threads *and* channels.

```
heartbeat thread                          training thread
────────────────                          ───────────────
Heartbeat RPC (heartbeat stub) ──►
   HeartbeatResponse.should_stop=True
        │
        └─► self._stop_training.set()  ────────► should_stop_training()  polled between
            (a threading.Event, LATCHING)         local steps / minibatches
                                                       │
                                                       └─► abort the round
```

- The **heartbeat stub is the only channel that can reach a busy client**: `fit()` blocks the
  training stub for the whole round. That is the deeper reason for the dual-channel design, beyond
  keeping liveness flowing.
- The Event is **latching** — once set it stays set; a server-driven stop ends the run.
- It is polled in three places: `LocalTrainer.fit()` between minibatches,
  `DeComFLClient.fit()` between local steps, and `start_client()` / `start_decomfl_client()` at the
  top of each loop iteration.
- **A partial round is never submitted.** After `fit()` returns, both entry points re-check
  `should_stop_training()` and discard the parameters (or the partial, non-K×P scalar grid) rather
  than uploading them.
- Abort latency is bounded by ~one heartbeat interval + one local step, rather than the full round.

Historically `_heartbeat_loop` discarded the response, which made the server's stop request a silent
no-op. Preserve both stubs *and* this latch when touching client lifecycle code.

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

Whatever the exit path, a `finally:` block always runs `stop_heartbeat()` then `close()`, so the
daemon thread and both channels are released.

> **`start_decomfl_client()` handles this materially better, and the difference is deliberate.**
> Rather than treating `UNAVAILABLE`/`CANCELLED` as a clean exit, it first calls
> `comm_client.server_reports_complete()` — a `GetServerStatus` probe on the *heartbeat* channel
> (the training stub may be in a bad state after a cancelled call). If the server reports
> `TRAINING_COMPLETE` it returns `OUTCOME_COMPLETED`; otherwise the server went away without
> finishing and it returns `OUTCOME_DISCONNECTED`. Other transient errors get a **bounded** rejoin
> budget (`_MAX_CONSECUTIVE_FAILURES = 3`, `_RETRY_DELAY_SECONDS = 10`) rather than an unbounded
> retry loop, and a successful poll resets the counter. This is what makes "the run finished" and
> "the server died" distinguishable — `start_client()` cannot currently tell them apart.

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
    def should_stop_training(self) -> bool          # FR-10 latch, polled by fit()
    def update_status(self, status, current_step, total_steps)
    def close(self)

    # Terminal-state probes (used by start_decomfl_client)
    def get_server_status(self) -> Optional[GetServerStatusResponse]   # None if unreachable
    def server_reports_complete(self) -> bool

    # DeComFL-specific
    def get_decomfl_config(self) -> (round, seeds, rebuild_history, config)
    def submit_gradient_scalars(self, gradient_scalars, num_examples, round_num) -> bool
```

### Channel Lifecycle

```python
def __init__(self, client_id, server_address):
    grpc_options = [...]

    # Primary channel for all heavy operations.
    # maybe_wrap_channel attaches FEDLEARN_CONNECTION_TOKEN if set (SE-1); no-op when unset.
    self.channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
    self.stub = FederatedLearningServiceStub(self.channel)

    # Dedicated heartbeat channel — never blocked by ongoing transfers.
    self.heartbeat_channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
    self.heartbeat_stub = FederatedLearningServiceStub(self.heartbeat_channel)

def close(self):
    self.stop_heartbeat()
    self.channel.close()
    if hasattr(self, 'heartbeat_channel') and self.heartbeat_channel:
        self.heartbeat_channel.close()
```

Channels are created once and reused for the lifetime of the client. gRPC handles connection pooling
and reconnection internally. `_build_channel()` returns an `insecure_channel` unless
`FEDLEARN_GRPC_USE_TLS=1`, in which case it builds `ssl_channel_credentials` (with optional mTLS key
and cert) and a `secure_channel` — see
[02 — TLS Configuration](02_grpc_communication.md#client-side-tls-setup). Transport encryption and
client authentication are **orthogonal**: `maybe_wrap_channel` attaches the connection token
regardless of whether TLS is on.

> `register()` currently sends only `client_id` — `run_id`, `protocol_version` and
> `enrollment_token` are left at their proto defaults, so a Python client always takes the server's
> permissive version branch. See
> [02 — Protocol Version Negotiation](02_grpc_communication.md#protocol-version-negotiation).

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
def start_decomfl_client(server_address, client, client_id) -> str:
    comm_client = GrpcClient(client_id=client_id, server_address=server_address)
    last_completed_round, consecutive_failures = -1, 0
    outcome = OUTCOME_ERROR

    if not comm_client.register():
        return OUTCOME_ERROR

    comm_client.start_heartbeat()
    client.set_grpc_client(comm_client)

    # FR-1: adopt the server's global model FIRST. DeComFL's core invariant is that every party
    # shares the same x_0 — without this the client's gradient scalars are directional derivatives
    # of a DIFFERENT function than the server's global model, so the aggregate is meaningless.
    # This is the one-shot O(d) download the paper assumes; per-round comms stay O(K*P).
    global_params, server_round_at_download, _ = comm_client.get_global_model()
    if not global_params:
        ...   # exit rather than train from an unsynced x_0
    client.load_global_model(global_params,
                             synced_through_round=server_round_at_download - 1)   # FR-16

    try:
        while True:
            if comm_client.should_stop_training():          # FR-10
                break
            try:
                current_round, seeds, rebuild_history, config = comm_client.get_decomfl_config()
                consecutive_failures = 0

                # MO-19/FR-14: FATAL, not transient — a dimension mismatch can never clear.
                if config.get('model_dim'):
                    client.assert_dim_matches(int(config['model_dim']))

                if current_round == -1:
                    outcome = OUTCOME_COMPLETED
                    break

                if current_round > last_completed_round:
                    if rebuild_history:
                        client.rebuild_model(rebuild_history,
                                             float(config.get('learning_rate', 0.001)))

                    training_config = dict(config)
                    training_config['seeds'] = seeds
                    gradient_scalars, num_examples = client.fit(None, training_config)

                    if comm_client.should_stop_training():   # FR-10: partial grid — do NOT submit
                        break

                    if comm_client.submit_gradient_scalars(gradient_scalars, num_examples,
                                                           current_round):
                        last_completed_round = current_round
                else:
                    time.sleep(5)   # wait for other clients

            except grpc.RpcError as e:
                # A finished run tears the RPCs down -> CANCELLED/UNAVAILABLE. ASK before assuming.
                if comm_client.server_reports_complete():
                    outcome = OUTCOME_COMPLETED
                    break
                if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED):
                    outcome = OUTCOME_DISCONNECTED
                    break
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:   # 3
                    outcome = OUTCOME_DISCONNECTED
                    break
                time.sleep(_RETRY_DELAY_SECONDS)                        # 10
    finally:
        comm_client.stop_heartbeat()
        comm_client.close()
    return outcome
```

Three things distinguish this from `start_client()`:

- **The one-shot O(d) sync is mandatory** (FR-1) and happens *before* the loop. Failing to download
  the global model is a hard exit, not a retry: training from an unsynced `x_0` produces gradient
  scalars for the wrong function, and nothing downstream would notice.
- **A dimension mismatch is fatal, not transient** (FR-14). Retrying a condition that can never
  clear just burns a federation training garbage.
- **The DeComFL poll interval is 5 s**, not the 2 s of `start_client()`.

### Syncing the Initial Model and the Applied-Round Watermark

`DeComFLClient.load_global_model(parameters, synced_through_round=None)` does more than a
`load_state_dict`:

```python
incoming, known = set(parameters), set(self.model.state_dict())
if incoming - known:                       # an unexpected key is always an error
    raise ValueError(...)
trainable_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
if trainable_keys - incoming:              # a MISSING TRAINABLE key silently misaligns z
    raise ValueError(...)
self.model.load_state_dict(parameters, strict=False)
self.x_current = self.zo_estimator._get_flat_params(self.model).to(self.device)
if synced_through_round is not None:
    self._synced_through = synced_through_round
```

**Why non-strict, and why then re-tightened by hand.** DeComFL synchronises only the
`requires_grad`-filtered trainable layout — the exact `d`-vector the shared-seed perturbation `z`
indexes. Frozen parameters (a frozen backbone, a partial fine-tune) and buffers are *not* sent; they
keep the net's deterministic build-time init, which every peer reproduces identically. A strict
`load_state_dict` therefore crashes on those legitimately-absent keys. So the load is non-strict but
the invariant that actually matters is enforced explicitly and model-agnostically: **every trainable
param must be present, and no unexpected key may appear.** For a fully-trainable model this is
exactly as strict as before, with no per-model special case.

`_synced_through` (FR-15 / FR-16) is the highest round whose *averaged* update is already folded into
`x_current`. It exists because the server advances its per-client baseline only on aggregation while
the client mutates `x_current` at config-fetch time — so a dropped submission makes the server
re-hand a round the client already applied. Setting it at download time also covers a **restart**: a
client that reuses its deterministic `client_id` and re-downloads `x_{r-1}` would otherwise reset the
watermark to `-1` while the server still remembers its pre-crash baseline, and double-apply
everything it was re-handed.

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
    seeds = config.get('seeds', [])   # List[List[int]] — [local_step][perturbation]
    K = len(seeds)                    # Number of local steps
    P = len(seeds[0]) if K else 0     # Number of perturbations per step
    eta = float(config.get('learning_rate', 0.001))

    # FR-10: mu is SERVER-AUTHORITATIVE. A mismatched mu makes the gradient scalars
    # derivatives of a different smoothed function than the one the server reconstructs.
    if config.get('smoothing_param') is not None:
        self.zo_estimator.mu = float(config['smoothing_param'])

    total_perturbation = torch.zeros_like(self.x_current)
    gradient_scalars = []
    data_iter = iter(self.train_loader)

    for k in range(K):
        # FR-10: honour a server-driven stop BETWEEN local steps. total_perturbation only
        # accumulates APPLIED steps, so the revert below stays exact for a partial run.
        if self.grpc_client is not None and self.grpc_client.should_stop_training():
            break

        delta = torch.zeros_like(self.x_current)
        k_gradient_scalars = []

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_loader)   # cycle the loader
            batch = next(data_iter)

        inputs, targets = batch
        inputs = ({k: v.to(self.device) for k, v in inputs.items()}
                  if isinstance(inputs, dict) else inputs.to(self.device))
        targets = targets.to(self.device)

        if self.grpc_client:
            self.grpc_client.update_status("training", k + 1, K)

        # f(x; xi) is the SAME for all P perturbations of this step — x and the batch are both
        # fixed here, only z varies. Hoist it: P+1 forward passes per step instead of 2P.
        base_loss = self.zo_estimator.compute_base_loss(self.model, self.x_current, inputs, targets)

        for p in range(P):
            z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))
            g = self.zo_estimator.compute_gradient_scalar(
                self.model, self.x_current, z, inputs, targets, base_loss=base_loss
            )
            k_gradient_scalars.append(g)
            delta += g * z

        step_update = (eta / P) * delta
        self.x_current -= step_update
        total_perturbation -= step_update

        gradient_scalars.append(k_gradient_scalars)

    # CRITICAL: Revert x_current to pre-round state
    self.x_current -= total_perturbation
    self.zo_estimator._set_flat_params(self.model, self.x_current)

    return gradient_scalars, len(self.train_loader.dataset)
```

> **Why revert `x_current`?** After local training, the client reverts its model to the state before this round's perturbations. This is necessary because the client's model will be advanced by the server's averaged update (communicated as seeds + average gradients) through the `rebuild_model()` call at the start of the *next* round. If the client kept its locally perturbed model, it would diverge from the global model that the server constructs.

### ZerothOrderEstimator.compute_gradient_scalar()

```python
def compute_base_loss(self, model, flat_params, inputs, targets) -> float:
    """f(x; xi) once, for reuse across a local step's P perturbations."""
    model.eval()
    with torch.no_grad():
        self._set_flat_params(model, flat_params)
        return self._evaluate_loss(model, inputs, targets).item()

def compute_gradient_scalar(self, model, flat_params, perturbation, inputs, targets,
                            base_loss=None) -> float:
    model.eval()
    with torch.no_grad():
        if base_loss is None:                       # back-compatible: recompute f(x; xi)
            self._set_flat_params(model, flat_params)
            loss_x = self._evaluate_loss(model, inputs, targets).item()
        else:
            loss_x = base_loss

        self._set_flat_params(model, flat_params + self.mu * perturbation)
        loss_x_perturbed = self._evaluate_loss(model, inputs, targets).item()

        g = (loss_x_perturbed - loss_x) / self.mu
    return g            # already a Python float

def _evaluate_loss(self, model, inputs, targets):
    """One forward pass at the model's CURRENT params. Caller owns eval()/no_grad()."""
    if isinstance(inputs, dict):                    # LLM: unpack as kwargs
        return model(**inputs, labels=targets).loss
    return self.criterion(model(inputs), targets)   # CNN/MLP: CrossEntropyLoss
```

**Cost is P+1 forward passes per local step, not 2P** — and no backward pass at all. Within one
local step both the base point `x` and the batch `ξ` are fixed, so `f(x; ξ)` is literally the same
number for every perturbation; hoisting it matches the authors' reference implementation. The scalar
is bit-identical either way, because the base loss is deterministic under `eval()` + `no_grad()` —
which is exactly the condition. **Pass `base_loss=None` for any model whose forward is stochastic at
inference**, or the cached value will be wrong.

Communication is `O(K × P)` scalars vs. `O(d)` for standard FL — a massive reduction for LLMs. Note
the two `loss.item()` calls: the scalar leaves the autograd/tensor world immediately, so no `d`-sized
gradient tensor is ever materialised.

### Model Rebuild Protocol

When a client joins late or misses rounds (e.g., temporary network disconnection), it must replay the missed rounds to synchronise its local model:

```python
def rebuild_model(self, rebuild_history, learning_rate):
    """
    Replays missed rounds using server-provided seeds and averaged gradients.
    Each missed round is re-simulated deterministically. IDEMPOTENT by watermark.
    """
    for round_data in rebuild_history:
        round_num = round_data['round_number']

        # FR-15: skip any round already folded into x_current. The server advances a client's
        # baseline only when its submission is AGGREGATED, so a dropped/straggler client is
        # re-handed a round it already applied; replaying it would double-apply and silently
        # diverge the local model from the global trajectory.
        if round_num <= self._synced_through:
            continue

        seeds = round_data['seeds']              # List[List[int]]
        avg_gradients = round_data['gradients']  # List[List[float]]
        K = len(seeds)
        P = len(seeds[0]) if K else 0

        for k in range(K):
            delta = torch.zeros_like(self.x_current)
            for p in range(P):
                # Regenerate the exact same perturbation vector from the shared seed
                z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))
                delta += avg_gradients[k][p] * z

            self.x_current = self.x_current - (learning_rate / P) * delta

        self._synced_through = round_num         # FR-15: record it as folded in

    self.zo_estimator._set_flat_params(self.model, self.x_current)
```

The rebuild is deterministic because both the client and server regenerate `z` from the same seed
through the same canonical CPU-only RNG path (see
[06 — Seed Generation and Sharing](06_decomfl.md#seed-generation-and-sharing)). The only information
transmitted is the scalar gradient values; the `d`-dimensional perturbation vectors are never stored
or sent.

> **The `_synced_through` watermark is what makes the replay idempotent**, and it is not a
> micro-optimisation — without it, an overlapping rebuild history double-applies a round's averaged
> update and the client's model drifts off the global trajectory with no error raised anywhere. The
> server has a complementary fail-loud guard for the opposite failure: if it cannot supply the seeds
> *and* averaged gradients for every round in the catch-up range it raises `DeComFLRebuildGap`
> rather than handing back a torn chain (FR-4).

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
    │ updates client_heartbeats dict (and returns should_stop — see FR-10 above)
    │
    ▼ (the FL server's JSON stdout is captured by the Spring backend)
React frontend receives progress update over STOMP
```

There is a **second, richer telemetry channel** that does not go through the heartbeat:
`ReportClientMetrics`. It carries per-round `loss`, `accuracy`, `current_step`, `total_steps`,
`client_type` (`"desktop"` | `"docker"` | `"mobile"`) and `compute_ms`, and the coordinator appends
it to `client_metrics_log`. It is best-effort by design — a telemetry failure returns
`acknowledged=False` and never fails the round. The Python `GrpcClient` does **not** currently call
it; it exists to close the mobile observability gap, and the native clients use it.

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

### Use the Shipped Partitioners

Do **not** hand-roll a Dirichlet split. `fedlearn.simulation.partition` ships four seeded
partitioners with a tested completeness/disjointness/determinism contract; they return
`List[np.ndarray]` of *index arrays*, so they are cheap to hold for thousands of clients and can be
recorded verbatim in a result's `meta` block.

```python
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from fedlearn.simulation.partition import dirichlet_partition, partition_report

full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

parts = dirichlet_partition(
    labels=full_dataset.targets.numpy(),
    num_clients=5,
    alpha=0.5,                # lower = more label skew
    seed=42,
    min_partition_size=16,    # an EMPTY client is rejected at coordinator ingress
)

print(partition_report(parts, full_dataset.targets.numpy()))
client_0_loader = DataLoader(Subset(full_dataset, parts[0]), batch_size=32, shuffle=True)
```

`min_partition_size` matters more than it looks: low `α` naturally produces empty clients, and an
empty client is not a benign edge case — `num_examples == 0` is rejected at coordinator ingress, so
it silently shrinks the effective cohort. The repair is deterministic (move samples from the largest
donor) rather than redraw-until-it-fits, because a variable number of RNG draws would make the
*rest* of the run's randomness depend on how many redraws happened.

See [07 — Data Partitioning](07_data_partitioning.md) for all four partitioners, the α guide, and
`partition_report`'s fields.

### Per-Client Randomness

If you are driving clients yourself (or through the simulator), derive each client's randomness from
`fedlearn.simulation.rng.RunRng` rather than a global seed — `ClientRng.torch_generator(round)` hands
a DataLoader a private generator, so one client's shuffle order cannot perturb another's. See
[01 — The In-Process Simulator](01_architecture_overview.md#the-in-process-simulator).
