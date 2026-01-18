# Core Modules Reference

This document provides detailed explanations of all core modules in the `src/fedlearn` package. These modules form the foundation of the FedLearn framework.

## Package Structure

```
src/fedlearn/
├── client/              # Client-side implementations
│   ├── __init__.py
│   ├── client.py        # Base client class and main loop
│   └── grpc_client.py   # gRPC communication wrapper
├── server/              # Server-side coordination
│   ├── __init__.py
│   ├── server.py        # Server entry point
│   ├── coordinator.py   # Round management & synchronization
│   ├── strategy.py      # Aggregation strategies (FedAvg)
│   └── grpc_servicer.py # gRPC service handlers
├── communication/       # gRPC and serialization
│   ├── generated/       # Auto-generated Protocol Buffer code
│   │   ├── fedlearn_pb2.py
│   │   └── fedlearn_pb2_grpc.py
│   ├── protos/
│   │   └── fedlearn.proto  # Protocol Buffer definitions
│   └── serializer.py    # Tensor serialization/deserialization
├── core/               # Core utilities
├── data/               # Data handling utilities
└── estimators/         # Custom gradient estimators
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
        # Main channel for model transfer
        self.channel = grpc.insecure_channel(server_address, options=grpc_options)
        self.stub = FederatedLearningServiceStub(self.channel)
        
        # Separate channel for heartbeats (non-blocking)
        self.heartbeat_channel = grpc.insecure_channel(server_address, options=grpc_options)
        self.heartbeat_stub = FederatedLearningServiceStub(self.heartbeat_channel)
```

**Why Two Channels?**
- Main channel: For model downloads/uploads (can take minutes for large models)
- Heartbeat channel: For keep-alive signals (needs to be non-blocking)

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
    for keyword in ['transformer', 'bert', 'gpt', 'attention']
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
    
    # Keepalive settings
    ('grpc.keepalive_time_ms', 120000),  # Ping every 2 minutes
    ('grpc.keepalive_timeout_ms', 60000),  # Wait 1 minute for pong
    ('grpc.keepalive_permit_without_calls', True),
    
    # Connection timeouts
    ('grpc.max_connection_idle_ms', 7200000),  # 2 hours
    ('grpc.max_connection_age_ms', 14400000),  # 4 hours
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
        clients_per_round: int
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
        clients_per_round: int = 2
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round
        self.aggregator = FedAvgAggregator()
```

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
    def aggregate(self, updates):
        """
        Weighted average of client models.
        
        Args:
            updates: List of (parameters, num_examples) tuples
        
        Returns:
            Aggregated parameters
        """
        # Calculate total examples
        total_examples = sum(num_examples for _, num_examples in updates)
        
        # Initialize aggregated parameters
        aggregated_params = OrderedDict()
        
        # Weighted sum
        for params, num_examples in updates:
            weight = num_examples / total_examples
            for key in params:
                if key not in aggregated_params:
                    aggregated_params[key] = torch.zeros_like(params[key])
                aggregated_params[key] += params[key] * weight
        
        return aggregated_params
```

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
    
    # Serialize
    buffer = io.BytesIO()
    model_data = {'parameters': params, 'num_examples': 0}
    torch.save(model_data, buffer)
    data_to_send = buffer.getvalue()
    
    # Split into 50MB chunks
    chunk_size = 50 * 1024 * 1024
    num_chunks = (len(data_to_send) + chunk_size - 1) // chunk_size
    
    # Stream chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(data_to_send))
        
        yield fedlearn_pb2.ModelChunk(
            chunk_index=i,
            total_chunks=num_chunks,
            chunk_data=data_to_send[start:end],
            is_final_chunk=(i == num_chunks - 1),
            current_round=current_round,
            config=config if i == 0 else {}
        )
```

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

**Service Definition**:
```protobuf
service FederatedLearningService {
  rpc RegisterClient(RegisterClientRequest) returns (RegisterClientResponse);
  rpc GetGlobalModel(GetGlobalModelRequest) returns (GetGlobalModelResponse);
  rpc GetGlobalModelStream(GetGlobalModelRequest) returns (stream ModelChunk);
  rpc SubmitModelUpdate(SubmitModelUpdateRequest) returns (SubmitModelUpdateResponse);
  rpc SubmitModelUpdateStream(stream ModelUpdateChunk) returns (SubmitModelUpdateResponse);
  rpc GetServerStatus(GetServerStatusRequest) returns (GetServerStatusResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
}
```

**Core Messages**:
```protobuf
message Tensor {
  bytes data = 1;           // Raw tensor bytes
  repeated int64 dims = 2;  // Shape [batch, channels, height, width]
  string dtype = 3;         // "float32", "int64", etc.
}

message ModelParameters {
  map<string, Tensor> tensors = 1;  // layer_name -> Tensor
  int64 num_examples_trained = 2;   // Number of training samples
}

message ModelChunk {
  int32 chunk_index = 1;
  int32 total_chunks = 2;
  bytes chunk_data = 3;      // 50MB of serialized model
  bool is_final_chunk = 4;
  int32 current_round = 5;
  map<string, string> config = 6;
}

message HeartbeatRequest {
  string client_id = 1;
  string status = 2;         // "training", "idle", etc.
  int32 current_step = 3;
  int32 total_steps = 4;
  int32 current_round = 5;
}
```

**Regenerating Code**:
```bash
python -m grpc_tools.protoc \
    -I communication/protos \
    --python_out=communication/generated \
    --grpc_python_out=communication/generated \
    communication/protos/fedlearn.proto
```

**When to Modify**:
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
    """Convert protobuf to PyTorch state_dict"""
    parameters = OrderedDict()
    for name, tensor_proto in proto.tensors.items():
        np_array = np.frombuffer(tensor_proto.data, dtype=tensor_proto.dtype)
        np_array = np_array.reshape(tensor_proto.dims).copy()
        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained
```

#### For Large Models (Streaming Transfer)
```python
def parameters_to_chunks(
    params: OrderedDict[str, torch.Tensor],
    num_examples: int,
    chunk_size: int = 50 * 1024 * 1024,  # 50 MB
    compress: bool = False
) -> Generator[Dict, None, None]:
    """
    Memory-efficient serialization using torch.save.
    Yields chunks of serialized model.
    """
    # 1. Serialize entire model
    buffer = io.BytesIO()
    model_data = {'parameters': params, 'num_examples': num_examples}
    torch.save(model_data, buffer)
    serialized = buffer.getvalue()
    buffer.close()
    
    # 2. Optional compression
    if compress and LZ4_AVAILABLE:
        serialized = lz4.frame.compress(serialized)
    
    # 3. Split into chunks and yield
    total_size = len(serialized)
    num_chunks = (total_size + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_size)
        
        yield {
            'chunk_index': i,
            'total_chunks': num_chunks,
            'chunk_data': serialized[start:end],
            'is_final_chunk': (i == num_chunks - 1),
            'num_examples': num_examples
        }

def chunks_to_parameters(
    chunks_data: bytes, 
    compressed: bool = False
) -> Tuple[OrderedDict, int]:
    """Reconstruct model from concatenated chunks"""
    # 1. Decompress if needed
    if compressed and LZ4_AVAILABLE:
        chunks_data = lz4.frame.decompress(chunks_data)
    
    # 2. Load using torch
    buffer = io.BytesIO(chunks_data)
    model_data = torch.load(buffer, map_location='cpu')
    buffer.close()
    
    return model_data['parameters'], model_data['num_examples']
```

**Compression**:
- Optional LZ4 compression (2-3x size reduction)
- `USE_COMPRESSION = False` by default (set to True if lz4 installed)

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
- **API Reference**: [api_reference.md](api_reference.md)
- **Examples**: [examples/](../examples/)
- **Advanced Topics**: [advanced/](../advanced/)