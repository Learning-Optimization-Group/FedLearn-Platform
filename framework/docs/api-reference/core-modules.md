# Core Modules Reference

This document provides detailed explanations of all core modules in the `src/fedlearn` package. These modules form the foundation of the FedLearn framework.

## Package Structure

```
src/fedlearn/
├── client/              # Client-side implementations
├── server/              # Server-side coordination
├── communication/       # gRPC and serialization
├── core/               # Core utilities
├── data/               # Data handling
└── estimators/         # DeComFL estimators
```

---

## Client Module (`fedlearn/client/`)

### `client.py`

**Purpose**: Base client class and client lifecycle management.

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
    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict):
        """Train model locally and return updated parameters"""
        pass
```

**Methods**:
- `get_parameters()`: Returns current model weights as OrderedDict
- `fit(parameters, config)`: Trains model locally for K epochs
  - **Args**:
    - `parameters`: Global model parameters from server
    - `config`: Training configuration (epochs, learning rate, etc.)
  - **Returns**: `(updated_parameters, num_samples)`

**Usage Example**:
```python
class MyClient(fl.Client):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
    
    def get_parameters(self):
        return self.model.state_dict()
    
    def fit(self, parameters, config):
        self.model.load_state_dict(parameters)
        # Training logic here
        return self.model.state_dict(), len(self.data_loader.dataset)
```

**When to Modify**: Create custom client classes for:
- Different training loops (RL, unsupervised learning)
- Custom optimization strategies
- Specialized privacy mechanisms
- Multi-task learning

---

### `decomfl_client.py`

**Purpose**: Client implementation for DeComFL (Decomposed Federated Learning) with Byzantine-robust aggregation.

**Key Classes**:

#### `DeComFLClient`
Specialized client that computes gradient estimators for DeComFL strategy.

```python
class DeComFLClient(Client):
    def __init__(self, model, data_loader, estimator_type='zo'):
        self.estimator_type = estimator_type  # 'zo' or 'fo'
        # ... initialization
```

**Key Methods**:
- `compute_zo_gradients()`: Zero-order gradient estimation
- `compute_fo_gradients()`: First-order gradient estimation
- `fit()`: Trains with decomposed gradient computation

**Estimator Types**:
- **ZO (Zero-Order)**: Gradient-free optimization using function evaluations
- **FO (First-Order)**: Traditional gradient-based optimization

**Usage**:
```python
client = fl.DeComFLClient(
    model=model,
    data_loader=loader,
    estimator_type='zo'
)
```

**When to Use**: 
- Byzantine fault tolerance needed
- Non-differentiable models
- Black-box optimization scenarios

---

### `grpc_client.py`

**Purpose**: gRPC communication layer for client-server interaction.

**Key Classes**:

#### `GRPCClient`
Handles low-level gRPC communication with server.

**Methods**:
- `connect(server_address)`: Establish connection to server
- `send_parameters(parameters)`: Send model updates
- `receive_parameters()`: Receive global model
- `disconnect()`: Close connection gracefully

**Configuration**:
```python
# gRPC options
options = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ('grpc.keepalive_time_ms', 10000),
]
```

**When to Modify**:
- Custom serialization protocols
- Additional metadata transmission
- Connection pooling
- Compression algorithms

---

## Server Module (`fedlearn/server/`)

### `server.py`

**Purpose**: Main server orchestration and federated round management.

**Key Functions**:

#### `start_server()`
```python
def start_server(
    server_address: str,
    config: ServerConfig,
    strategy: Strategy,
) -> Tuple[History, float]:
    """
    Start federated learning server.
    
    Args:
        server_address: Address to bind (e.g., "0.0.0.0:50051")
        config: Server configuration
        strategy: Aggregation strategy (FedAvg, DeComFL, etc.)
    
    Returns:
        (history, elapsed_time): Training history and total time
    """
```

**ServerConfig**:
```python
@dataclass
class ServerConfig:
    num_rounds: int = 10
    round_timeout: float = 600.0  # seconds
    min_fit_clients: int = 2
    min_available_clients: int = 2
```

**Server Lifecycle**:
1. Initialize global model
2. Wait for clients to connect
3. For each round:
   - Select clients
   - Send global model
   - Receive and aggregate updates
   - Evaluate global model
4. Return training history

**When to Modify**:
- Custom round selection logic
- Adaptive timeout mechanisms
- Multi-stage training pipelines
- Hierarchical aggregation

---

### `strategy.py`

**Purpose**: Base strategy class for model aggregation.

**Key Classes**:

#### `Strategy` (Abstract Base Class)
```python
class Strategy(ABC):
    @abstractmethod
    def aggregate_fit(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Optional[Parameters]:
        """Aggregate training results from clients"""
        pass
    
    @abstractmethod
    def configure_fit(
        self, 
        round_num: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training round"""
        pass
```

**Built-in Strategies**:

1. **FedAvg** (Federated Averaging)
   - Weighted average of client models
   - Weight = number of training samples
   
2. **DeComFL** (Decomposed Federated Learning)
   - Byzantine-robust aggregation
   - Uses gradient estimators

**When to Modify**:
- Implement custom aggregation (FedProx, FedOpt)
- Add client selection strategies
- Implement adaptive learning rates
- Add differential privacy

---

### `decomfl_strategy.py`

**Purpose**: DeComFL aggregation strategy with Byzantine robustness.

**Key Classes**:

#### `DeComFL`
```python
class DeComFL(Strategy):
    def __init__(
        self,
        initial_parameters,
        byzantine_threshold: float = 0.2,
        estimator_type: str = 'zo'
    ):
        self.byzantine_threshold = byzantine_threshold
        self.estimator_type = estimator_type
```

**Aggregation Process**:
1. Collect gradient estimates from clients
2. Detect Byzantine clients using statistical methods
3. Aggregate only trusted gradients
4. Update global model

**Byzantine Detection**:
- Krum algorithm
- Median-based filtering
- Trimmed mean

**Usage**:
```python
strategy = fl.DeComFL(
    initial_parameters=model.state_dict(),
    byzantine_threshold=0.2,  # Tolerate 20% Byzantine clients
    estimator_type='zo'
)
```

---

### `coordinator.py` & `async_coordinator.py`

**Purpose**: Manage client connections and round coordination.

**Key Classes**:

#### `Coordinator`
Synchronous coordination - waits for all clients.

```python
class Coordinator:
    def wait_for_clients(self, min_clients: int):
        """Block until minimum clients connect"""
        pass
    
    def broadcast_parameters(self, parameters):
        """Send parameters to all clients"""
        pass
```

#### `AsyncCoordinator`
Asynchronous coordination - doesn't wait for stragglers.

```python
class AsyncCoordinator:
    def configure_round(self, timeout: float):
        """Configure round with timeout"""
        pass
```

**When to Use**:
- `Coordinator`: Small number of reliable clients
- `AsyncCoordinator`: Large-scale, heterogeneous environments

---

### `grpc_servicer.py`

**Purpose**: gRPC server implementation (RPC handlers).

**Key Methods**:
- `GetParameters()`: Handle parameter requests
- `SendUpdate()`: Handle client updates
- `RegisterClient()`: Handle client registration

**When to Modify**:
- Add custom RPC methods
- Implement authentication
- Add rate limiting
- Custom error handling

---

## Communication Module (`fedlearn/communication/`)

### `generated/` (Protocol Buffers)

**Purpose**: Auto-generated gRPC stubs from `.proto` files.

**Files**:
- `fedlearn_pb2.py`: Message definitions
- `fedlearn_pb2_grpc.py`: Service definitions

**Message Types**:
```protobuf
message Parameters {
    map<string, bytes> tensors = 1;
}

message FitRequest {
    Parameters parameters = 1;
    map<string, string> config = 2;
}

message FitResponse {
    Parameters parameters = 1;
    int64 num_examples = 2;
}
```

**When to Modify**:
- Change `.proto` file, then regenerate:
```bash
python -m grpc_tools.protoc \
    -I protos \
    --python_out=. \
    --grpc_python_out=. \
    protos/fedlearn.proto
```

---

### `serializer.py`

**Purpose**: Serialize/deserialize PyTorch tensors for transmission.

**Key Functions**:

```python
def parameters_to_proto(parameters: OrderedDict) -> ParametersProto:
    """Convert PyTorch state_dict to protobuf"""
    pass

def proto_to_parameters(proto: ParametersProto) -> OrderedDict:
    """Convert protobuf to PyTorch state_dict"""
    pass
```

**Serialization Format**:
1. Tensor → NumPy array → bytes (via pickle)
2. Wrap in protobuf message
3. Transmit via gRPC

**When to Modify**:
- Use different serialization (MessagePack, FlatBuffers)
- Add compression (gzip, zstd)
- Implement quantization
- Add encryption

---

## Core Module (`fedlearn/core/`)

**Purpose**: Shared utilities and helper functions.

**Common Contents**:
- Logging utilities
- Configuration management
- Type definitions
- Common exceptions

**When to Modify**:
- Add framework-wide utilities
- Implement custom loggers
- Add metrics collection

---

## Data Module (`fedlearn/data/`)

**Purpose**: Data loading, partitioning, and preprocessing utilities.

**Key Functions**:

```python
def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data using Dirichlet distribution for non-IID split.
    
    Args:
        X: Feature array
        y: Label array
        num_clients: Number of clients
        alpha: Dirichlet parameter (lower = more heterogeneous)
        seed: Random seed
    
    Returns:
        List of (X_client, y_client) tuples
    """
```

**Data Distribution Types**:
1. **IID** (α → ∞): Each client gets random uniform sample
2. **Non-IID** (α < 1): Each client gets skewed label distribution
3. **Extreme Non-IID** (α < 0.1): Very heterogeneous

**Usage**:
```python
from fedlearn.data import partition_data

client_data = partition_data(
    X=X_train, 
    y=y_train,
    num_clients=5,
    alpha=0.5
)

for i, (X_client, y_client) in enumerate(client_data):
    print(f"Client {i}: {len(X_client)} samples")
```

**When to Modify**:
- Implement different partitioning strategies
- Add data augmentation
- Implement federated datasets
- Add privacy-preserving techniques

---

## Estimators Module (`fedlearn/estimators/`)

**Purpose**: Gradient estimators for DeComFL optimization.

**Key Components**:

### Zero-Order (ZO) Estimators
```python
class ZOEstimator:
    def estimate_gradient(self, loss_fn, parameters, delta=0.01):
        """
        Estimate gradient using finite differences.
        
        ∇f(θ) ≈ [f(θ + δe) - f(θ - δe)] / (2δ)
        """
```

### First-Order (FO) Estimators
```python
class FOEstimator:
    def compute_gradient(self, loss, parameters):
        """
        Compute exact gradient using autograd.
        """
```

**When to Use**:
- **ZO**: Non-differentiable objectives, black-box optimization
- **FO**: Standard differentiable deep learning

**When to Modify**:
- Implement SPSA (Simultaneous Perturbation Stochastic Approximation)
- Add variance reduction techniques
- Implement adaptive step sizes

---

## File Modification Guidelines

### When to Modify Files

**Client-Side** (`client/`):
- Custom training loops
- Different model architectures
- Privacy mechanisms (DP, secure aggregation)

**Server-Side** (`server/`):
- Aggregation strategies
- Client selection policies
- Evaluation metrics

**Communication** (`communication/`):
- Protocol changes
- Compression algorithms
- Security features

**Data** (`data/`):
- Data loading pipelines
- Preprocessing steps
- Augmentation strategies

### How to Extend

1. **Inherit from base classes**
```python
class MyStrategy(fl.Strategy):
    def aggregate_fit(self, results):
        # Custom logic
        pass
```

2. **Override specific methods**
```python
class MyClient(fl.Client):
    def fit(self, parameters, config):
        # Custom training
        return parameters, num_samples
```

3. **Add to existing modules**
```python
# In fedlearn/data/utils.py
def my_custom_partitioning(data, num_clients):
    # Implementation
    pass
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
- Enable mixed precision training
- Batch operations when possible

### Debugging
- Add logging at key points
- Use meaningful error messages
- Validate inputs early
- Test with 1-2 clients first

---

## Next Steps

- **Server API**: [server.md](server.md)
- **Client API**: [client.md](client.md)
- **Strategies**: [strategies.md](strategies.md)
- **Custom Strategies**: [../advanced/custom-strategies.md](../advanced/custom-strategies.md)