# 03 — Server Internals

## Table of Contents
- [Overview](#overview)
- [Entry Point: start_server()](#entry-point-start_server)
- [Server Configuration](#server-configuration)
- [JSON Structured Logging](#json-structured-logging)
- [The FLCoordinator](#the-flcoordinator)
  - [State Machine](#state-machine)
  - [Thread Safety Model](#thread-safety-model)
  - [Round Lifecycle](#round-lifecycle)
  - [Update Submission and Aggregation Trigger](#update-submission-and-aggregation-trigger)
  - [Stale Update Rejection](#stale-update-rejection)
  - [Client Registration](#client-registration)
- [Heartbeat Management](#heartbeat-management)
- [DeComFL Path in the Coordinator](#decomfl-path-in-the-coordinator)
- [The gRPC Servicer](#the-grpc-servicer)
- [Server Thread Pool Sizing](#server-thread-pool-sizing)
- [Graceful Shutdown](#graceful-shutdown)
- [Environment Variables Reference](#environment-variables-reference)

---

## Overview

The server-side of the framework comprises three cooperating objects:

```
start_server()
    │
    ├── creates FLCoordinator       (business logic / state)
    ├── creates grpc.Server          (network I/O)
    └── creates FederatedLearningServiceServicer (RPC dispatcher)
                │
                └── delegates all calls to FLCoordinator
```

The key architectural principle is **strict separation of concerns**:
- **`FLCoordinator`** never touches networking.
- **`FederatedLearningServiceServicer`** never touches model parameters directly — it only calls coordinator methods.
- **`start_server()`** only sets up infrastructure and runs the outer training loop.

---

## Entry Point: start_server()

```python
# server.py
def start_server(
    server_address: str,
    config: ServerConfig,
    strategy: Strategy
) -> tuple[list, dict]:
    """
    Args:
        server_address: e.g. "0.0.0.0:50051"
        config: ServerConfig(num_rounds=10)
        strategy: An instance of FedAvg, DeComFL, or a custom Strategy subclass

    Returns:
        history: List of (round_num, metrics_dict) for each completed round
        final_parameters: The global model state_dict after all rounds
    """
```

Call sequence:

1. **Create coordinator** — initialized with the strategy and client count settings
2. **Set initial parameters** — pushes `strategy.initialize_parameters()` into coordinator
3. **Create gRPC server** — with ThreadPoolExecutor sized for expected client count
4. **Register servicer** — `add_FederatedLearningServiceServicer_to_server(servicer, grpc_server)`
5. **Bind address** — insecure or TLS depending on env vars
6. **Start server** — `grpc_server.start()` begins accepting connections
7. **Run training loop** — iterates rounds, blocking on `coordinator.wait_for_round_to_complete()`
8. **Return** history and final parameters
9. **Cleanup** — `grpc_server.stop(grace=5)` in the finally block

---

## Server Configuration

```python
@dataclass
class ServerConfig:
    num_rounds: int = 3
```

Currently a minimal dataclass. Extended configuration (e.g., client selection fraction, convergence criteria) is expected to be passed through the `Strategy` object's constructor arguments rather than `ServerConfig`.

Typical usage:

```python
import fedlearn as fl

model = MyCNN()
strategy = fl.FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_global_model,
    min_fit_clients=2,
    clients_per_round=3,
)

history, final_params = fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

---

## JSON Structured Logging

The server configures Python's root logger to emit JSON on startup:

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_obj["stackTrace"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]
```

**Why JSON logging?** The Spring Boot backend captures the Python process's stdout and parses it line-by-line to extract structured log objects, which are then broadcast to the React frontend via STOMP WebSocket. Plain text logs would require fragile regex parsing.

Example output:
```json
{"timestamp": "2024-01-15T10:23:45.123Z", "level": "INFO", "message": "Starting round 1/10"}
{"timestamp": "2024-01-15T10:23:46.456Z", "level": "INFO", "message": "All 3 clients reported for round 1; aggregating"}
{"timestamp": "2024-01-15T10:23:46.789Z", "level": "INFO", "message": "[Server] Round 1 complete. Metrics: {'loss': 0.43, 'accuracy': 0.87}"}
```

---

## The FLCoordinator

`FLCoordinator` is the stateful core of the server. It owns:
- The current global model parameters
- The current round number
- The list of received updates for the current round
- Client registration and heartbeat records

### State Machine

```
              start_round()
                   │
    ┌──────────────▼──────────────┐
    │         COLLECTING          │ ← waiting for client updates
    │                             │
    │  submit_client_update() ────┤
    │  (called per client)        │
    └──────────────┬──────────────┘
                   │ len(updates) == clients_per_round
                   ▼
    ┌──────────────────────────────┐
    │   AGGREGATING (brief)        │
    │                             │
    │  strategy.aggregate_fit()   │
    │  strategy.evaluate()        │
    │  current_round += 1         │
    │  _round_complete_event.set()│
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  wait_for_round_to_complete  │
    │  unblocks in server.py loop  │
    └─────────────────────────────┘
```

### Thread Safety Model

The coordinator is accessed concurrently by:
- **The main server thread** — calling `start_round()`, `wait_for_round_to_complete()`, `get_latest_metrics()`
- **One gRPC thread per active client** — calling `submit_client_update()`, `get_global_model_for_client()`, `register_client()`, `update_client_heartbeat()`

Two separate locks protect different parts of the state:

```python
class FLCoordinator:
    def __init__(self, ...):
        self._lock = threading.Lock()        # protects round state + model params
        self._round_complete_event = threading.Event()  # signals round completion

        self.heartbeat_lock = Lock()         # protects heartbeat dict only
        self.client_heartbeats: Dict[str, dict] = {}
```

> **Critical threading contract:** `_trigger_aggregation_and_evaluation()` is always called **while `_lock` is held** (inside `submit_client_update`). The round counter increment and `_round_complete_event.set()` are therefore atomic — no waiting thread can observe an inconsistent state.

### Round Lifecycle

```python
def start_round(self):
    """Called by main loop to begin a new round."""
    with self._lock:
        self._client_updates_received.clear()  # flush any stale updates
    self._round_complete_event.clear()          # reset completion signal

def wait_for_round_to_complete(self):
    """Blocking wait on main loop thread. Checks stop flag every 1 second."""
    while not self._round_complete_event.wait(timeout=1.0):
        if self.stop_requested:
            break
```

The 1-second timeout in `wait()` allows the server to respond to `signal_stop()` without hanging indefinitely.

### Update Submission and Aggregation Trigger

```python
def submit_client_update(self, client_id, params, num_examples, trained_on_round):
    with self._lock:
        # Stale update rejection (see below)
        if trained_on_round < self.current_round:
            return
        if trained_on_round > self.current_round:
            return

        # Sanitise num_examples
        if num_examples <= 0:
            log.warning("Invalid num_examples from %s; skipping", client_id)
            return
        num_examples = min(num_examples, self.MAX_NUM_EXAMPLES)  # cap at 100,000

        self._client_updates_received.append((params, num_examples))

        # Trigger aggregation when we have all expected updates
        if len(self._client_updates_received) == self.clients_per_round:
            self._trigger_aggregation_and_evaluation()
            # Note: this is still inside _lock — intentional

def _trigger_aggregation_and_evaluation(self):
    """MUST be called while self._lock is held."""
    results = list(self._client_updates_received)
    self._client_updates_received.clear()  # free memory immediately

    aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

    if aggregated_parameters is not None:
        self._global_model_params = aggregated_parameters
        loss, metrics = self.strategy.evaluate(self.current_round, self._global_model_params)
        self.latest_metrics = {"loss": loss, **metrics}
    else:
        log.warning("Aggregation for round %d failed", self.current_round)
        self.latest_metrics = None

    # Advance round counter LAST, then signal
    self.current_round += 1
    self._round_complete_event.set()
```

> **Memory note:** `self._client_updates_received.clear()` is called immediately after copying to `results`. This frees all client parameter tensors as soon as aggregation completes, which is critical when training large models across many clients.

### Stale Update Rejection

The coordinator rejects updates that don't match the current round:

```python
if trained_on_round < self.current_round:
    return  # Stale: client trained on an old model, ignore
if trained_on_round > self.current_round:
    return  # Illegal: client is claiming to be ahead of the server
```

**Why stale updates happen:** If a client is slow and submits its update after the round has already advanced (because other clients submitted faster), the update is stale. Accepting it would corrupt the next round's aggregation with parameters from a different global model version.

**Why "ahead" updates are rejected:** This shouldn't happen in correct protocol execution. If it does, it indicates a bug or a malicious client. Rejecting silently is safe.

### Client Registration

```python
def register_client(self, client_id: str) -> bool:
    with self._lock:
        self._registered_clients.add(client_id)
        return True  # currently always accepts
```

The registration is lightweight — it just records the client ID. Future versions could implement capacity limits (reject if `len(registered_clients) >= max_clients`).

---

## Heartbeat Management

The coordinator maintains a `client_heartbeats` dictionary keyed by `client_id`:

```python
{
    "client_0": {
        "status": "training",
        "current_step": 42,
        "total_steps": 100,
        "current_round": 3,
        "last_seen": 1705312345.678  # unix timestamp
    }
}
```

Updated by `update_client_heartbeat()` on every heartbeat RPC (every 5 seconds per client):

```python
def update_client_heartbeat(self, client_id, status, current_step, total_steps, current_round):
    with self.heartbeat_lock:
        self.client_heartbeats[client_id] = {
            'status': status,
            'current_step': current_step,
            'total_steps': total_steps,
            'current_round': current_round,
            'last_seen': time.time()
        }
    # ...
    should_stop = False
    return True, should_stop, f"Heartbeat received for {client_id}"
```

> **`should_stop` field:** Included for future use — the server can signal a client to stop mid-training by returning `should_stop=True` from the heartbeat response. Currently always `False`.

Active client detection uses a 300-second timeout:

```python
def get_active_clients(self) -> list[str]:
    current_time = time.time()
    with self.heartbeat_lock:
        return [
            client_id
            for client_id, data in self.client_heartbeats.items()
            if current_time - data['last_seen'] < self.heartbeat_timeout  # 300 seconds
        ]
```

---

## DeComFL Path in the Coordinator

When `DeComFL` is the active strategy, the coordinator has a separate update path:

```python
def submit_decomfl_update(self, client_id, gradient_scalars, num_examples, trained_on_round):
    with self._lock:
        # same stale rejection logic as FedAvg path
        ...
        # Store as 3-tuple: (client_id, gradient_scalars, num_examples)
        self._client_updates_received.append((client_id, gradient_scalars, num_examples))

        if len(self._client_updates_received) >= self.clients_per_round:
            self._trigger_decomfl_aggregation_and_evaluation()

def _trigger_decomfl_aggregation_and_evaluation(self):
    results = list(self._client_updates_received)
    self._client_updates_received.clear()

    aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

    if aggregated_parameters is not None:
        self._global_model_params = aggregated_parameters

        # Store gradient history for client model rebuilding
        avg_gradients = self._calculate_average_gradients(results)
        if hasattr(self.strategy, 'gradient_history'):
            self.strategy.gradient_history.append(avg_gradients)

        loss, metrics = self.strategy.evaluate(...)
        self.latest_metrics = {"loss": loss, **metrics}

    self.current_round += 1
    self._round_complete_event.set()
```

The `_calculate_average_gradients()` computes element-wise average gradient scalars across all clients:

```python
def _calculate_average_gradients(self, results):
    # results = [(client_id, grad_scalars[K][P], num_examples), ...]
    _, first_grads, _ = results[0]
    K = len(first_grads)   # local steps
    P = len(first_grads[0]) # perturbations

    avg = [[0.0] * P for _ in range(K)]
    for _, grads, _ in results:
        for k in range(K):
            for p in range(P):
                avg[k][p] += grads[k][p]

    num_clients = len(results)
    for k in range(K):
        for p in range(P):
            avg[k][p] /= num_clients

    return avg
```

These averaged gradients are stored in `strategy.gradient_history` so that late-joining clients can replay missed rounds and rebuild their local model to match the current global state.

---

## The gRPC Servicer

`FederatedLearningServiceServicer` is a thin I/O adapter. Every method:
1. Deserialises the protobuf request
2. Calls the appropriate coordinator method
3. Serialises the protobuf response

There is intentionally no business logic in the servicer. Example:

```python
class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):

    def __init__(self, coordinator: FLCoordinator):
        self.coordinator = coordinator

    def RegisterClient(self, request, context):
        success = self.coordinator.register_client(request.client_id)
        status = (RegisterClientResponse.Status.ACCEPTED if success
                  else RegisterClientResponse.Status.REJECTED)
        return RegisterClientResponse(status=status, message="...")

    def SubmitModelUpdate(self, request, context):
        params, num_examples = proto_to_parameters(request.parameters)
        self.coordinator.submit_client_update(
            request.client_id, params, num_examples, request.trained_on_round
        )
        return SubmitModelUpdateResponse(received=True)
```

The `GetGlobalModelStream` and `SubmitModelUpdateStream` RPCs use Python generators — `yield` for server streaming, iterator consumption for client streaming.

---

## Server Thread Pool Sizing

The thread pool must be large enough to handle all concurrent RPCs:

```python
max_expected_clients = int(os.environ.get('MAX_CLIENTS', 50))
optimal_workers = (max_expected_clients * 2) + 10
# 50 clients → 110 workers
# 10 clients → 30 workers
```

The `* 2` factor accounts for the fact that each client typically has **two concurrent RPCs** at any point:
1. The primary training RPC (e.g., a long-running `GetGlobalModelStream`)
2. The heartbeat RPC (fires every 5 seconds)

The `+ 10` provides overhead for registration RPCs, status polls, etc.

---

## Graceful Shutdown

The server shuts down in a controlled sequence:

```python
# server.py
try:
    history = []
    for round_num in range(1, config.num_rounds + 1):
        coordinator.start_round()
        coordinator.wait_for_round_to_complete()

        if coordinator.stop_requested:
            break

        metrics = coordinator.get_latest_metrics()
        history.append((round_num, metrics))

    final_parameters = coordinator.get_global_model_params()
    return history, final_parameters

except KeyboardInterrupt:
    return [], {}
except Exception as e:
    coordinator.signal_stop()   # releases any waiting threads
    return [], {}
finally:
    grpc_server.stop(grace=5)   # waits up to 5s for active RPCs to complete
```

`signal_stop()` sets `stop_requested = True` and also calls `_round_complete_event.set()` to unblock `wait_for_round_to_complete()`:

```python
def signal_stop(self):
    self.stop_requested = True
    self._round_complete_event.set()  # unblock the waiting main thread
```

After training completes, the server broadcasts a completion signal to clients by returning `current_round = -1` from `get_global_model_for_client()`:

```python
def get_global_model_for_client(self):
    with self._lock:
        if self.stop_requested:
            return None, -1, {}   # sentinel: -1 means "training complete"
        return self._global_model_params, self.current_round, {}
```

Clients detect this and exit cleanly:

```python
# client.py
parameters, server_round, config = comm_client.get_global_model()
if server_round == -1:
    log.info("Server signalled training complete; shutting down")
    break
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CLIENTS` | `50` | Used to size the gRPC thread pool |
| `FEDLEARN_GRPC_USE_TLS` | `"0"` | Set to `"1"` to enable TLS |
| `FEDLEARN_GRPC_SERVER_KEY` | — | Path to server private key (PEM) |
| `FEDLEARN_GRPC_SERVER_CERT` | — | Path to server certificate (PEM) |
| `FEDLEARN_GRPC_ROOT_CERT` | — | CA cert for verifying clients (mTLS) |
| `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` | `"0"` | `"1"` to require client certs |
| `FEDLEARN_USE_COMPRESSION` | `"0"` | `"1"` to enable LZ4 compression |
| `FEDLEARN_CHUNK_SIZE_MB` | `"4"` | Chunk size for streaming serialisation |
