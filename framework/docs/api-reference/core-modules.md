# Core Modules Reference

This document provides detailed explanations of all core modules in the `src/fedlearn` package. These modules form the foundation of the FedLearn framework.

## Package Structure

```
src/fedlearn/
├── client/                   # Client-side implementations
│   ├── __init__.py
│   ├── client.py             # Base client class and main loop
│   ├── decomfl_client.py     # DeComFL zeroth-order client
│   ├── local_trainer.py      # Local training loop
│   └── grpc_client.py        # gRPC communication wrapper
├── server/                   # Server-side coordination
│   ├── __init__.py
│   ├── server.py             # Server entry point
│   ├── coordinator.py        # Round management & synchronization
│   ├── strategy.py           # Aggregation strategies (FedAvg, FedLoRA, FedProx)
│   ├── strategy_factory.py   # Strategy construction by name
│   ├── decomfl_strategy.py   # DeComFL server strategy
│   ├── robust_aggregation.py # Byzantine-robust aggregators
│   ├── subset_federation.py  # Client subset selection
│   └── grpc_servicer.py      # gRPC service handlers
├── communication/            # gRPC and serialization
│   ├── generated/            # Auto-generated Protocol Buffer code
│   │   ├── fedlearn_pb2.py
│   │   ├── fedlearn_pb2_grpc.py
│   │   ├── fot_pb2.py
│   │   └── fot_pb2_grpc.py
│   ├── protos/
│   │   └── fedlearn.proto      # Protocol Buffer definitions (package fedlearn.v2)
│   ├── safetensors_codec.py  # safetensors encode/decode (the wire format)
│   └── serializer.py         # Tensor serialization/deserialization
├── backbone/                 # Frozen-backbone distribution
├── bundle/                   # Adapter-bundle manifest + schema
├── data/                     # Data handling utilities
├── estimators/               # Custom gradient estimators
├── fot/                      # Federation over Text (research mode)
├── privacy/                  # DP mechanism + RDP accountant
└── security/                 # TLS policy, interceptors, token verification
```

---

## Client Module (`fedlearn/client/`)

### `client.py`

**Purpose**: Base client class and client lifecycle management for federated learning.

**Key Classes**:

#### `Client` (Abstract Base Class)
Base class that all federated learning clients must inherit from.

```python
class Client(ABC):
    @abstractmethod
    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return current model parameters"""
        pass
    
    @abstractmethod
    def fit(
        self, 
        parameters: OrderedDict[str, torch.Tensor], 
        config: dict
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        """Train model locally and return updated parameters"""
        pass
```

**Methods**:
- `get_parameters()`: Returns current model weights as OrderedDict
- `fit(parameters, config)`: Trains model locally for K epochs
  - **Args**:
    - `parameters`: Global model parameters from server
    - `config`: Training configuration (epochs, learning rate, etc.)
  - **Returns**: `(updated_parameters, num_examples)`

**Usage Example**:
```python
class CNNClient(Client):
    def __init__(self, model, train_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self):
        return self.model.state_dict()
    
    def fit(self, parameters, config):
        # Load global model
        self.model.load_state_dict(parameters)
        self.model.train()
        
        # Train for specified epochs
        epochs = config.get('local_epochs', 1)
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        # Return updated model and number of training samples
        return self.model.state_dict(), len(self.train_loader.dataset)
```

**Key Function**:

#### `start_client()`
```python
def start_client(
    server_address: str, 
    client: Client, 
    client_id: str
):
    """
    Starts a client that connects to a server with heartbeat support.
    
    Args:
        server_address: gRPC server address (e.g., "localhost:50051")
        client: The Client instance that implements fit() and get_parameters()
        client_id: Unique identifier for this client
    """
```

**Client Lifecycle**:
1. **Registration**: Register with server
2. **Heartbeat**: Start background heartbeat thread
3. **Main Loop**:
   - Fetch global model from server
   - Train locally using `fit()`
   - Submit update to server
   - Wait for new round or termination
4. **Cleanup**: Stop heartbeat and close connection

**Status Updates**:
The client tracks and reports its status:
- `"fetching_model"`: Downloading global model
- `"training"`: Local training in progress
- `"submitting_update"`: Uploading trained model
- `"waiting"`: Waiting for new round to start
- `"idle"`: Nothing to do
- `"error"`: Error occurred

**When to Extend**:
- Different training loops (e.g., reinforcement learning, generative models)
- Custom optimization strategies
- Specialized privacy mechanisms (differential privacy)
- Multi-task learning scenarios
- Custom data augmentation during training

---

### `grpc_client.py`

**Purpose**: gRPC communication layer for client-server interaction with support for large model streaming.

**Key Classes**:

#### `GrpcClient`
Handles all gRPC communication with the server, including model download/upload and heartbeats.

**Initialization**:
```python
class GrpcClient:
    def __init__(self, client_id: str, server_address: str):
        # Main channel for model transfer. _build_channel() picks TLS vs plaintext;
        # maybe_wrap_channel() attaches the SE-1 connection token when one is present.
        self.channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.stub = FederatedLearningServiceStub(self.channel)

        # Separate channel for heartbeats (non-blocking)
        self.heartbeat_channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.heartbeat_stub = FederatedLearningServiceStub(self.heartbeat_channel)
```

**Why Two Channels?**
- Main channel: For model downloads/uploads (can take minutes for large models)
- Heartbeat channel: For keep-alive signals (needs to be non-blocking)

**Transport security (`_build_channel`)**:
```python
def _build_channel(server_address: str, grpc_options: list) -> grpc.Channel:
    """Builds a gRPC channel. Uses TLS when FEDLEARN_GRPC_USE_TLS=1."""
    use_tls = os.environ.get("FEDLEARN_GRPC_USE_TLS", "0") == "1"
    if not use_tls:
        return grpc.insecure_channel(server_address, options=grpc_options)
    credentials = grpc.ssl_channel_credentials(...)   # FEDLEARN_GRPC_ROOT_CERT / _CLIENT_CERT / _CLIENT_KEY
    return grpc.secure_channel(server_address, credentials, options=grpc_options)
```

> ⚠️ **The transport is plaintext by default.** TLS is implemented but **opt-in** via
> `FEDLEARN_GRPC_USE_TLS=1` (add `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` for mTLS). The server
> side fails closed: with `FEDLEARN_REQUIRE_TLS=1` — which the backend sets on deployed
> profiles — it refuses to serve without TLS. Client authentication is separate:
> `FEDLEARN_CONNECTION_TOKEN` (SE-1/SE-14), fetched from
> `GET /api/client/projects/{id}/connection`.

**Key Methods**:

##### `register() -> bool`
Register client with server.
```python
def register(self) -> bool:
    req = fedlearn_pb2.RegisterClientRequest(client_id=self.client_id)
    res = self.stub.RegisterClient(req)
    return res.status == RegisterClientResponse.Status.ACCEPTED
```

##### `get_global_model() -> Tuple[OrderedDict, int, dict]`
Download global model using streaming (supports large models like LLMs).
```python
def get_global_model(self):
    # Uses GetGlobalModelStream RPC
    # Returns: (parameters, current_round, config)
```

**Streaming Process**:
1. Request model from server
2. Receive model in 50MB chunks
3. Reconstruct full model from chunks
4. Return parameters, round number, and config

##### `submit_update() -> bool`
Upload trained model. Automatically chooses between unary and streaming based on model size.
```python
def submit_update(
    self, 
    params: OrderedDict[str, torch.Tensor], 
    num_examples: int, 
    round_number: int
) -> bool:
    # Decision logic:
    # - If model > 100MB OR is transformer: use streaming
    # - Otherwise: use unary (single message)
```

**Adaptive Streaming**:
```python
STREAMING_THRESHOLD_MB = 100
ALWAYS_STREAM_TRANSFORMERS = True

# Detects transformer models by layer names
is_transformer = any(
    keyword in name.lower()
    for name in params.keys()
    for keyword in ['transformer', 'bert', 'gpt', 'opt', 'attention', 'encoder', 'decoder']
)
```

##### Heartbeat Methods
```python
def start_heartbeat():
    """Start background thread that sends periodic heartbeats"""
    
def send_heartbeat() -> bool:
    """Send single heartbeat with current status"""
    
def stop_heartbeat():
    """Stop heartbeat thread"""
    
def update_status(status: str, current_step: int, total_steps: int):
    """Update status that will be sent in next heartbeat"""
```

**Heartbeat Interval**: 5 seconds (configurable)

**gRPC Configuration**:
```python
grpc_options = [
    # Message size limits (1GB)
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024),

    # Keepalive tuned to survive AWS NLB / ALB idle-connection culling.
    ('grpc.keepalive_time_ms', 60000),     # Ping every minute
    ('grpc.keepalive_timeout_ms', 20000),  # Wait 20s for pong
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 0),

    # Connection timeouts
    ('grpc.max_connection_idle_ms', 7200000),      # 2 hours
    ('grpc.max_connection_age_ms', 14400000),      # 4 hours
    ('grpc.max_connection_age_grace_ms', 600000),  # 10 minutes
]
```

**When to Modify**:
- Custom compression algorithms
- Additional metadata transmission
- Connection pooling
- Retry logic
- Authentication/authorization

---

## Server Module (`fedlearn/server/`)

### `server.py`

**Purpose**: Main server orchestration and federated round management.

**Key Components**:

#### `ServerConfig`
```python
@dataclass
class ServerConfig:
    num_rounds: int = 3  # Number of federated learning rounds
```

#### `start_server()`
```python
def start_server(
    server_address: str,
    config: ServerConfig,
    strategy: Strategy
) -> tuple[list, dict]:
    """
    Start a gRPC Federated Learning server.
    
    Args:
        server_address: Address to bind (e.g., "0.0.0.0:50051")
        config: Server configuration
        strategy: Aggregation strategy (FedAvg, custom, etc.)
    
    Returns:
        (history, final_parameters): Training history and final global model
    """
```

**Server Lifecycle**:

1. **Initialization**:
   ```python
   coordinator = FLCoordinator(
       strategy=strategy,
       min_clients_for_aggregation=strategy.min_fit_clients,
       clients_per_round=strategy.clients_per_round,
   )
   coordinator.set_initial_parameters(strategy.initial_parameters)
   ```

2. **gRPC Server Setup**:
   ```python
   grpc_server = grpc.server(
       futures.ThreadPoolExecutor(max_workers=10),
       options=[...]
   )
   grpc_server.add_insecure_port(server_address)
   grpc_server.start()
   ```

3. **Training Loop**:
   ```python
   for round_num in range(1, config.num_rounds + 1):
       coordinator.start_round()
       coordinator.wait_for_round_to_complete()  # BLOCKS until aggregation
       
       metrics = coordinator.get_latest_metrics()
       history.append((round_num, metrics))
       
       coordinator.current_round += 1
   ```

4. **Cleanup**:
   ```python
   final_parameters = coordinator.get_global_model_params()
   grpc_server.stop(grace=5)
   return history, final_parameters
   ```

**gRPC Configuration**:
- **Thread pool**: 10 workers (handles concurrent client requests)
- **Message size**: 1GB limit (for large models)
- **Keepalive**: Prevents idle connection timeouts
- **Connection limits**: Long-lived connections for training

**When to Modify**:
- Implement client selection strategies
- Add adaptive timeout mechanisms
- Implement checkpoint saving
- Add TensorBoard logging
- Implement warm restarts

---

### `coordinator.py`

**Purpose**: Coordinates federated learning rounds and manages client updates.

**Key Classes**:

#### `FLCoordinator`
Manages the state and synchronization of federated learning.

**Initialization**:
```python
class FLCoordinator:
    def __init__(
        self,
        strategy: Strategy,
        min_clients_for_aggregation: int,
        clients_per_round: int,
        round_timeout_s: Optional[float] = None,
        # SE-3 poisoning defense: clamp each DeComFL gradient scalar at ingress. None disables.
        grad_clip_threshold: Optional[float] = 1000.0,
        # SE-3 (FedAvg path): optional server-side L2 clip of each client's update delta. Opt-in.
        client_update_l2_clip: Optional[float] = None,
    ):
        self.strategy = strategy
        self.min_clients = min_clients_for_aggregation
        self.clients_per_round = clients_per_round
        
        # Synchronization
        self._lock = threading.Lock()
        self._round_complete_event = threading.Event()
        
        # State
        self._global_model_params = None
        self._client_updates_received = []
        self._registered_clients = set()
        self.current_round = 1
        
        # Heartbeat tracking
        self.client_heartbeats = {}
        self.heartbeat_timeout = 300  # 5 minutes
```

**Key Methods**:

##### Round Management
```python
def start_round(self):
    """Called by server to begin a new round"""
    self._round_complete_event.clear()

def wait_for_round_to_complete(self):
    """Blocks until current round finishes"""
    while not self._round_complete_event.wait(timeout=1.0):
        if self.stop_requested:
            break
```

##### Client Update Handling
```python
def submit_client_update(
    self, 
    client_id: str, 
    params: OrderedDict[str, torch.Tensor], 
    num_examples: int,
    trained_on_round: int
):
    """
    Accept update from client and trigger aggregation if enough updates received.
    """
    with self._lock:
        # Ignore stale updates
        if trained_on_round < self.current_round:
            return
        
        # Ignore future updates (shouldn't happen)
        if trained_on_round > self.current_round:
            return
        
        # Add update to list
        self._client_updates_received.append((params, num_examples))
        
        # Check if we have enough clients
        if len(self._client_updates_received) == self.clients_per_round:
            self._trigger_aggregation_and_evaluation()
```

##### Aggregation & Evaluation
```python
def _trigger_aggregation_and_evaluation(self):
    """Core logic for advancing a round"""
    # Get all updates
    results = list(self._client_updates_received)
    self._client_updates_received.clear()
    
    # Aggregate using strategy
    aggregated_parameters = self.strategy.aggregate_fit(
        self.current_round, 
        results
    )
    
    # Update global model
    if aggregated_parameters is not None:
        self._global_model_params = aggregated_parameters
        
        # Evaluate
        loss, metrics = self.strategy.evaluate(
            self.current_round,
            self._global_model_params
        )
        self.latest_metrics = {"loss": loss, **metrics}
    
    # Signal round complete
    self._round_complete_event.set()
```

##### Heartbeat Management
```python
def update_client_heartbeat(
    self, 
    client_id: str, 
    status: str, 
    current_step: int, 
    total_steps: int, 
    current_round: int
) -> tuple[bool, bool, str]:
    """Update the last heartbeat time for a client"""
    with self.heartbeat_lock:
        self.client_heartbeats[client_id] = {
            'status': status,
            'current_step': current_step,
            'total_steps': total_steps,
            'current_round': current_round,
            'last_seen': time.time()
        }
    return True, False, f"Heartbeat received for {client_id}"

def is_client_alive(self, client_id: str) -> bool:
    """Check if client is still alive based on heartbeat"""
    with self.heartbeat_lock:
        if client_id not in self.client_heartbeats:
            return False
        last_seen = self.client_heartbeats[client_id]['last_seen']
        return (time.time() - last_seen) < self.heartbeat_timeout
```

**Thread Safety**:
- Uses `threading.Lock()` for state mutations
- Uses `threading.Event()` for round completion signaling
- Separate `heartbeat_lock` for heartbeat operations

**When to Modify**:
- Implement asynchronous aggregation
- Add client selection logic
- Implement stragglers mitigation
- Add round timeout handling
- Implement client weighting schemes

---

### `strategy.py`

**Purpose**: Defines aggregation strategies for combining client updates.

**Key Classes**:

#### `Strategy` (Abstract Base Class)
```python
class Strategy(ABC):
    @abstractmethod
    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Return initial global model parameters"""
        pass
    
    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Aggregate training results from clients"""
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        server_round: int, 
        parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        """Evaluate the global model"""
        pass
```

#### `FedAvg` (Federated Averaging Strategy)
```python
class FedAvg(Strategy):
    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        evaluate_fn: Optional[Callable] = None,
        min_fit_clients: int = 1,
        clients_per_round: int = None
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        # Defaults to min_fit_clients when not given
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregator = FedAvgAggregator()
```

`strategy.py` also ships `FedLoRA` (LoRA aggregation + central DP) and `FedProx`; `strategy_factory.py`
constructs a strategy by name.

**Methods**:

##### `initialize_parameters()`
```python
def initialize_parameters(self):
    return self.initial_parameters
```

##### `aggregate_fit()`
```python
def aggregate_fit(self, server_round, results):
    if not results:
        return None
    return self.aggregator.aggregate(results)
```

##### `evaluate()`
```python
def evaluate(self, server_round, parameters):
    if self.evaluate_fn is None:
        return None
    
    loss, metrics = self.evaluate_fn(server_round, parameters)
    print(f"Strategy Evaluation (Round {server_round}): Loss={loss:.4f}, Metrics={metrics}")
    return loss, metrics
```

**Custom Evaluation Function Example**:
```python
def evaluate_fn(server_round, parameters):
    # Load parameters into model
    model.load_state_dict(parameters)
    model.eval()
    
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, {'accuracy': accuracy}

# Use in strategy
strategy = FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    clients_per_round=5
)
```

#### `FedAvgAggregator`
Implements the weighted averaging algorithm.

```python
class FedAvgAggregator:
    MAX_SAMPLES = 100_000  # Cap to prevent model poisoning via inflated num_examples

    def aggregate(self, updates):
        """
        Weighted average of client models.

        Args:
            updates: accepted wire shapes (2-/3-tuples, JSON-encoded params), coerced by
                     normalize_updates() into (client_id, state_dict, num_examples)

        Returns:
            Aggregated parameters
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        updates = normalize_updates(updates)

        _, template_params, _ = updates[0]
        template_params = {k: v.to(device) for k, v in template_params.items()}
        aggregated_params = OrderedDict(
            [(key, torch.zeros_like(tensor, dtype=torch.float32)) for key, tensor in template_params.items()])

        # Sanitize num_examples: cap and reject invalid values
        sanitized_updates = [(cid, p, min(n, self.MAX_SAMPLES)) for cid, p, n in updates if n > 0]
        if not sanitized_updates:
            raise ValueError("No valid updates after sanitization.")

        total_examples = sum(num_examples for _, _, num_examples in sanitized_updates)

        for client_id, params, num_examples in sanitized_updates:
            weight = num_examples / total_examples
            for key in aggregated_params:
                if key in params:
                    torch.add(aggregated_params[key], params[key].to(device).float(),
                              alpha=weight, out=aggregated_params[key])
            params.clear()  # Aggressively free client memory buffer

        return aggregated_params
```

> `num_examples` is capped at `MAX_SAMPLES` (100,000) so a single client cannot seize the whole
> average by inflating its reported sample count.

**Mathematical Formula**:
```
w_global = Σ(w_i * n_i) / Σ(n_i)

where:
  w_i = client i's model parameters
  n_i = number of training samples at client i
```

**Example**:
```
Client 1: 1000 samples, weight = 0.333
Client 2: 500 samples,  weight = 0.167
Client 3: 1500 samples, weight = 0.500

Aggregated = 0.333*params_1 + 0.167*params_2 + 0.500*params_3
```

**When to Extend**:
- Implement FedProx (proximal term)
- Implement FedOpt (server-side optimization)
- Implement adaptive learning rates
- Add client selection strategies
- Add differential privacy mechanisms

---

### `grpc_servicer.py`

**Purpose**: Implements gRPC service handlers (server-side RPC methods).

**Key Classes**:

#### `FederatedLearningServiceServicer`
```python
class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):
    def __init__(self, coordinator: FLCoordinator):
        self.coordinator = coordinator
```

**RPC Handlers**:

##### `RegisterClient()`
```python
def RegisterClient(self, request, context):
    client_id = request.client_id
    success = self.coordinator.register_client(client_id)
    
    if success:
        return fedlearn_pb2.RegisterClientResponse(
            status=fedlearn_pb2.RegisterClientResponse.Status.ACCEPTED,
            message=f"Client '{client_id}' registered successfully."
        )
```

##### `GetGlobalModelStream()` (Server Streaming)
```python
def GetGlobalModelStream(self, request, context):
    """Stream global model to client in chunks"""
    # Get model from coordinator
    params, current_round, config = self.coordinator.get_global_model_for_client()

    # FR-8: serialize as a deterministic SAFETENSORS blob — the same libtorch-free wire the
    # upload path and the mobile C++ core use. Never a torch.save pickle blob.
    data_to_send = state_dict_to_safetensors(params, num_examples=0)
    download_codec = "safetensors"

    # Declare the sha256 of the FULL payload so receivers verify the reassembled blob
    # (format-agnostically, before any deserialization). Set on EVERY chunk.
    payload_sha256 = hashlib.sha256(data_to_send).hexdigest()

    # Split into 50MB chunks
    chunk_size = 50 * 1024 * 1024
    total_size = len(data_to_send)
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    # Stream chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_size)

        yield fedlearn_pb2.ModelChunk(
            chunk_index=i,
            total_chunks=num_chunks,
            chunk_data=data_to_send[start:end],
            is_final_chunk=(i == num_chunks - 1),
            current_round=current_round,
            config=config if i == 0 else {},
            codec=download_codec,
            total_bytes=total_size,
            sha256=payload_sha256,
        )
```

> **Download chunk size is 50 MB and fixed here** — distinct from the *upload* path, whose chunk
> size comes from `serializer.CHUNK_SIZE` (`FEDLEARN_CHUNK_SIZE_MB`, default **4 MB**).

##### `SubmitModelUpdateStream()` (Client Streaming)
```python
def SubmitModelUpdateStream(self, request_iterator, context):
    """Receive streamed model update from client"""
    chunks = []
    client_id = None
    
    # Receive all chunks
    for chunk in request_iterator:
        if client_id is None:
            client_id = chunk.client_id
            round_num = chunk.trained_on_round
        chunks.append(chunk.chunk_data)
    
    # Reconstruct model
    full_data = b''.join(chunks)
    parameters, num_examples = chunks_to_parameters(full_data)
    
    # Submit to coordinator
    self.coordinator.submit_client_update(
        client_id, 
        parameters, 
        num_examples, 
        round_num
    )
    
    return fedlearn_pb2.SubmitModelUpdateResponse(received=True)
```

##### `Heartbeat()`
```python
def Heartbeat(self, request, context):
    """Handle heartbeat from client (fast, non-blocking)"""
    acknowledged, should_stop, message = self.coordinator.update_client_heartbeat(
        request.client_id,
        request.status,
        request.current_step,
        request.total_steps,
        request.current_round
    )
    
    return fedlearn_pb2.HeartbeatResponse(
        acknowledged=acknowledged,
        should_stop=should_stop,
        message=message
    )
```

**Error Handling**:
- All handlers include try-except blocks
- Detailed error logging with stack traces
- Graceful error reporting to clients via `context.abort()`

**When to Modify**:
- Add authentication/authorization
- Implement rate limiting
- Add request validation
- Implement custom error handling
- Add metrics collection

---

## Communication Module (`fedlearn/communication/`)

### `fedlearn.proto` (Protocol Buffers)

**Purpose**: Defines the communication contract between clients and server.

The contract is `package fedlearn.v2`. The canonical source is the top-level `proto/fedlearn/v2/fedlearn.proto`;
this file is a byte-identical mirror of it, enforced in CI by `scripts/check_proto_mirror.sh`.

**Service Definition**:
```protobuf
service FederatedLearningService {
  // --- lifecycle / control ---
  rpc RegisterClient        (RegisterClientRequest)        returns (RegisterClientResponse);
  rpc GetServerStatus       (GetServerStatusRequest)       returns (GetServerStatusResponse);
  rpc Heartbeat             (HeartbeatRequest)             returns (HeartbeatResponse);

  // --- model transfer (FedAvg path) ---
  rpc GetGlobalModel        (GetGlobalModelRequest)        returns (GetGlobalModelResponse);
  rpc GetGlobalModelStream  (GetGlobalModelRequest)        returns (stream ModelChunk);
  rpc SubmitModelUpdate     (SubmitModelUpdateRequest)     returns (SubmitModelUpdateResponse);
  rpc SubmitModelUpdateStream(stream ModelUpdateChunk)     returns (SubmitModelUpdateResponse);

  // --- DeComFL path (scalars + seeds only; no weights on the wire) ---
  rpc GetDeComFLConfig      (GetDeComFLConfigRequest)      returns (GetDeComFLConfigResponse);
  rpc SubmitGradientScalars (SubmitGradientScalarsRequest) returns (SubmitGradientScalarsResponse);

  // --- telemetry ---
  rpc ReportClientMetrics   (ReportClientMetricsRequest)   returns (ReportClientMetricsResponse);
}
```

**Core Messages**:
```protobuf
// A single typed tensor. The ONLY weight-bearing wire type; no torch.save blobs.
message Tensor {
  bytes          data  = 1;   // raw bytes, dtype+dims interpret them
  repeated int64 dims  = 2;
  string         dtype = 3;   // whitelist: "float32","float64","int32","int64","uint8","bool"
}

message ModelParameters {
  map<string, Tensor> tensors              = 1;  // layer_name -> Tensor
  int64               num_examples_trained = 2;  // Number of training samples
}

message ModelChunk {
  int32  chunk_index    = 1;
  int32  total_chunks   = 2;
  bytes  chunk_data     = 3;
  bool   is_final_chunk = 4;
  int32  current_round  = 5;
  map<string,string> config = 6;
  // --- v2 framing fields ---
  string codec          = 7;   // "safetensors" (typed; NOT torch.save) — required, validated
  bool   compressed     = 8;   // on the wire, not inferred from env; codec="lz4+safetensors" if true
  int64  total_bytes    = 9;   // full reassembled size; receiver bounds-checks cumulative
  string sha256         = 10;  // hash of the full reassembled blob; receiver verifies
}

message HeartbeatRequest {
  string client_id     = 1;
  string run_id        = 2;
  string status        = 3;   // free-text client phase, e.g. "TRAINING","IDLE"
  int32  current_step  = 4;
  int32  total_steps   = 5;
  int32  current_round = 6;
}
```

The DeComFL path carries **seeds and gradient scalars only** — no weights on the wire
(`PerturbationSeeds`, `GradientScalars`, `RebuildHistory`). See the proto for the full set.

**Regenerating Code**:

Stubs are generated from the **canonical** `proto/fedlearn/v2/fedlearn.proto` via `buf`
(`proto/buf.yaml`, `proto/buf.gen.yaml`), which emits Python, Java, TypeScript, and C++ targets
from one config. The local generation is equivalent to:

```bash
python -m grpc_tools.protoc \
    -I src/fedlearn/communication/protos \
    --python_out=src/fedlearn/communication/generated \
    --grpc_python_out=src/fedlearn/communication/generated \
    src/fedlearn/communication/protos/fedlearn.proto
```

**When to Modify**:

Edit the **canonical** `proto/fedlearn/v2/fedlearn.proto` — never this mirror. Then re-sync the
mirrors and verify with `scripts/check_proto_mirror.sh`; CI (`proto.yml`) runs buf lint, a
breaking-change check against `main`, a regenerate-no-op check, and the mirror check.

- Add new RPC methods
- Add new message fields
- Change serialization format
- Add versioning

---

### `serializer.py`

**Purpose**: Convert PyTorch tensors to/from bytes for network transmission.

**Key Functions**:

#### For Small Models (Unary Transfer)
```python
def parameters_to_proto(
    parameters: OrderedDict[str, torch.Tensor], 
    num_examples: int
) -> ModelParameters:
    """Convert PyTorch state_dict to protobuf"""
    tensors = {}
    for name, tensor in parameters.items():
        np_array = tensor.cpu().detach().numpy()
        tensors[name] = Tensor(
            data=np_array.tobytes(),
            dims=list(np_array.shape),
            dtype=str(np_array.dtype),
        )
    return ModelParameters(tensors=tensors, num_examples_trained=num_examples)

def proto_to_parameters(proto: ModelParameters) -> tuple[OrderedDict, int]:
    """Convert protobuf to PyTorch state_dict. Validates before trusting the wire."""
    parameters = OrderedDict()
    for name, tensor_proto in proto.tensors.items():
        # 1. dtype whitelist — prevents arbitrary dtype injection
        if tensor_proto.dtype not in _SAFE_DTYPES:
            raise ValueError(f"Unsafe dtype '{tensor_proto.dtype}' for tensor '{name}'")

        np_array = np.frombuffer(tensor_proto.data, dtype=np.dtype(tensor_proto.dtype))

        # 2. dims must be positive and consistent with the payload length
        expected_size = 1
        for d in tensor_proto.dims:
            if d <= 0:
                raise ValueError(f"Invalid dimension {d} for tensor '{name}'")
            expected_size *= d
        if expected_size != len(np_array):
            raise ValueError(f"Shape mismatch for tensor '{name}'")

        np_array = np_array.reshape(tensor_proto.dims).copy()

        # 3. reject NaN/Inf (SE-3 poisoning defense)
        _reject_non_finite(name, np_array)
        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained
```

#### For Large Models (Streaming Transfer)

The wire format is **safetensors, not pickle** (`communication/safetensors_codec.py`). It is
**float32-only** — any other dtype raises rather than being silently cast.

```python
CHUNK_SIZE = int(os.environ.get("FEDLEARN_CHUNK_SIZE_MB", "4")) * 1024 * 1024

def parameters_to_chunks(
    params: OrderedDict[str, torch.Tensor],
    num_examples: int,
    chunk_size: int = CHUNK_SIZE,          # default 4 MB; override with FEDLEARN_CHUNK_SIZE_MB
    compress: Optional[bool] = None,       # None -> USE_COMPRESSION
) -> Generator[Dict, None, None]:
    """Memory-efficient serialization using the safetensors wire format."""
    if compress is None:
        compress = USE_COMPRESSION

    # 1. Serialize entire model (safetensors; num_examples rides in the metadata)
    serialized = state_dict_to_safetensors(params, num_examples)

    # 2. Optional compression
    if compress and LZ4_AVAILABLE:
        data_to_send = lz4.frame.compress(serialized, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
    else:
        data_to_send = serialized

    # 3. Split into chunks and yield
    total_size = len(data_to_send)
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_size)

        yield {
            'chunk_index': i,
            'total_chunks': num_chunks,
            'chunk_data': data_to_send[start:end],
            'is_final_chunk': (i == num_chunks - 1),
            'num_examples': num_examples,
        }

def chunks_to_parameters(
    chunks_data: bytes,
    compressed: Optional[bool] = None,
) -> Tuple[OrderedDict, int]:
    """Reconstruct a state_dict from a safetensors blob (optionally lz4-compressed)."""
    if compressed is None:
        compressed = USE_COMPRESSION

    # 1. Decompress if needed
    data = lz4.frame.decompress(chunks_data) if (compressed and LZ4_AVAILABLE) else chunks_data

    # 2. Reject legacy pickle/zip blobs loudly rather than silently mis-reading.
    #    torch.save produces a zip starting with PK\x03\x04; raw pickle starts with 0x80.
    if len(data) >= 2 and (data[:2] == b"PK" or data[0] == 0x80):
        raise ValueError("Received a legacy pickle/zip blob (torch.save format). "
                         "Only safetensors wire format is accepted.")

    # 3. Decode; every tensor is screened for NaN/Inf (SE-3 poisoning defense)
    named_arrays, meta = load_safetensors(data)
    params = OrderedDict()
    for name, arr in named_arrays:
        _reject_non_finite(name, arr)
        params[name] = torch.tensor(arr)

    return params, int(meta["num_examples"])
```

**Legacy-format policy (asymmetric, by design)**:
- **Ingest** (`chunks_to_parameters`, used by the upload/receive path) **rejects** torch.save/pickle blobs outright.
- The **client download** path (`grpc_client.get_global_model`) still *accepts* them via
  `torch.load(..., weights_only=True)` so a new client keeps working against an old server
  during a staged rollout. The `codec` field is the primary signal; a magic-byte sniff is the backstop.

**Compression**:
- Optional LZ4 compression (2-3x size reduction)
- **Opt-in via env var**: `USE_COMPRESSION = LZ4_AVAILABLE and os.environ.get("FEDLEARN_USE_COMPRESSION", "0") == "1"`
  — off by default even when lz4 is installed, for parity with existing deployments.

**When to Modify**:
- Implement quantization (int8, int4)
- Add different compression algorithms (zstd, brotli)
- Implement sparse tensor serialization
- Add encryption

---

## Usage Examples

### Complete Server Example
```python
import torch
from fedlearn.server import start_server, ServerConfig
from fedlearn.server.strategy import FedAvg

# Define model
model = SimpleCNN()

# Define evaluation function
def evaluate_fn(round_num, parameters):
    model.load_state_dict(parameters)
    # Evaluate on test set
    return test_loss, {'accuracy': accuracy}

# Create strategy
strategy = FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    clients_per_round=5
)

# Create config
config = ServerConfig(num_rounds=10)

# Start server
history, final_params = start_server(
    server_address="0.0.0.0:50051",
    config=config,
    strategy=strategy
)

# Save final model
torch.save(final_params, "final_model.pt")
```

### Complete Client Example
```python
import torch
from fedlearn.client import Client, start_client

class MNISTClient(Client):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    def get_parameters(self):
        return self.model.state_dict()
    
    def fit(self, parameters, config):
        self.model.load_state_dict(parameters)
        self.model.train()
        
        for epoch in range(5):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict(), len(self.train_loader.dataset)

# Create client
model = SimpleCNN()
client = MNISTClient(model, train_loader)

# Connect to server
start_client(
    server_address="localhost:50051",
    client=client,
    client_id="client_1"
)
```

---

## Best Practices

### Code Organization
- Keep client and server logic separate
- Use type hints for all public APIs
- Document all public classes and methods
- Write unit tests for new functionality

### Performance
- Profile before optimizing
- Use vectorized operations (NumPy, PyTorch)
- Enable mixed precision training when possible
- Batch operations when appropriate

### Debugging
- Add logging at key decision points
- Use meaningful error messages
- Validate inputs early
- Test with 1-2 clients before scaling

### Security
- Validate client inputs
- Implement rate limiting
- Add authentication/authorization
- Use TLS for production deployments

---

## Extension Points

### Custom Client
```python
class MyCustomClient(Client):
    def fit(self, parameters, config):
        # Your custom training logic
        pass
```

### Custom Strategy
```python
class MyCustomStrategy(Strategy):
    def aggregate_fit(self, server_round, results):
        # Your custom aggregation logic
        pass
```

### Custom Serialization
```python
def my_custom_serializer(parameters):
    # Your custom serialization logic
    pass
```

---

## Troubleshooting

### Common Issues

**Client can't connect to server**:
- Check server address and port
- Ensure server is running
- Check firewall settings

**Model transfer fails**:
- Check message size limits in gRPC options
- Enable streaming for large models
- Check network bandwidth

**Round doesn't complete**:
- Check `clients_per_round` setting
- Verify clients are submitting updates
- Check heartbeat status

**Memory issues**:
- Enable streaming for large models
- Reduce batch size
- Use gradient checkpointing

---

## Next Steps

For more detailed information, see:
- **Architecture**: [architecture.md](architecture.md)
- **API Reference**: [server.md](server.md) · [client.md](client.md) · [strategies.md](strategies.md)
- **Examples**: [examples/](../examples/)
- **Advanced Topics**: [advanced/](../advanced/)