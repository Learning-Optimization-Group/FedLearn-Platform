# 09 — Developer Guide: Extending the Framework

## Table of Contents
- [Installation for Development](#installation-for-development)
- [Adding a Custom Strategy](#adding-a-custom-strategy)
- [Adding a Custom Client](#adding-a-custom-client)
- [Modifying the Proto Contract](#modifying-the-proto-contract)
- [Testing Guidelines](#testing-guidelines)
- [Logging Best Practices](#logging-best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Contributing Checklist](#contributing-checklist)

---

## Installation for Development

```bash
cd FedLearn-Platform/framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install in editable mode so changes take effect immediately
pip install -e .

# Install dev extras
pip install grpcio-tools pytest pytest-timeout
```

Verify installation:
```python
import fedlearn as fl
print(fl.__file__)   # should point to src/fedlearn/__init__.py
```

---

## Adding a Custom Strategy

**1. Create the strategy file:**

```python
# src/fedlearn/server/my_strategy.py
from .strategy import Strategy
from collections import OrderedDict
from typing import Optional, Tuple, List
import torch

class MyStrategy(Strategy):

    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        evaluate_fn=None,
        min_fit_clients: int = 1,
        clients_per_round: int = 2,
        # add your custom hyperparameters here
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[OrderedDict, int]],
    ) -> Optional[OrderedDict]:
        if not results:
            return None
        # --- your aggregation logic here ---
        # results = [(params_dict, num_examples), ...]
        # must return an OrderedDict[str, torch.Tensor]
        raise NotImplementedError

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
```

**2. Export it from the package:**

```python
# src/fedlearn/__init__.py  — add this line
from .server.my_strategy import MyStrategy
```

**3. Use it:**

```python
import fedlearn as fl

strategy = fl.MyStrategy(
    initial_parameters=model.state_dict(),
    clients_per_round=3,
)
fl.server.start_server("0.0.0.0:50051", fl.server.ServerConfig(num_rounds=10), strategy)
```

---

## Adding a Custom Client

**1. Subclass `Client`:**

```python
# my_client.py
import fedlearn as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple

class MyClient(fl.Client):

    def __init__(self, model: nn.Module, train_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.grpc_client = None   # set by framework if you implement set_grpc_client

    def set_grpc_client(self, gc):
        """Optional but recommended — enables heartbeat progress updates."""
        self.grpc_client = gc

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.model.state_dict()

    def fit(
        self,
        parameters: OrderedDict[str, torch.Tensor],
        config: dict,
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        # 1. Load global model
        self.model.load_state_dict(parameters)
        self.model.train()

        total_steps = len(self.train_loader)

        # 2. Local training
        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()

            # 3. Report progress
            if self.grpc_client:
                self.grpc_client.update_status("training", step + 1, total_steps)

        # 4. Return updated params + dataset size
        return self.model.state_dict(), len(self.train_loader.dataset)


# Entry point
if __name__ == "__main__":
    client = MyClient(MyModel(), my_train_loader, device="cpu")
    fl.client.start_client("localhost:50051", client, "client_0")
```

---

## Modifying the Proto Contract

If you add a new RPC or message type:

```bash
# 1. Edit the source proto
vim src/fedlearn/communication/protos/fedlearn.proto

# 2. Regenerate Python stubs
python -m grpc_tools.protoc \
    -I src/fedlearn/communication/protos \
    --python_out=src/fedlearn/communication/generated \
    --grpc_python_out=src/fedlearn/communication/generated \
    src/fedlearn/communication/protos/fedlearn.proto

# 3. Fix relative import in generated grpc file
sed -i '' 's/^import fedlearn_pb2/from . import fedlearn_pb2/' \
    src/fedlearn/communication/generated/fedlearn_pb2_grpc.py

# 4. Implement the handler in grpc_servicer.py
# 5. Add coordinator method if needed
# 6. Add GrpcClient method if clients need to call the new RPC
```

---

## Testing Guidelines

### Unit test pattern

```python
# tests/test_aggregation.py
import pytest
import torch
from collections import OrderedDict
from fedlearn.server.strategy import FedAvgAggregator

def make_params(values: dict) -> OrderedDict:
    return OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()})

def test_fedavg_weighted_average():
    agg = FedAvgAggregator()
    results = [
        (make_params({"w": [1.0, 2.0]}), 100),  # weight = 100/150
        (make_params({"w": [4.0, 5.0]}), 50),   # weight =  50/150
    ]
    output = agg.aggregate(results)
    expected = (100/150) * torch.tensor([1.0, 2.0]) + (50/150) * torch.tensor([4.0, 5.0])
    assert torch.allclose(output["w"], expected)

def test_fedavg_rejects_zero_examples():
    agg = FedAvgAggregator()
    results = [(make_params({"w": [1.0]}), 0)]
    with pytest.raises(ValueError, match="No valid updates"):
        agg.aggregate(results)

def test_fedavg_caps_num_examples():
    agg = FedAvgAggregator()
    # 200,000 > MAX_SAMPLES=100,000; should be capped silently
    results = [
        (make_params({"w": [1.0]}), 200_000),
        (make_params({"w": [3.0]}), 100_000),
    ]
    output = agg.aggregate(results)
    # Both capped to 100,000 → equal weights → average = 2.0
    assert torch.allclose(output["w"], torch.tensor([2.0]))
```

Run tests:
```bash
pytest tests/ -v --timeout=60
```

### Integration test pattern

```python
# tests/test_integration.py
import threading, time, torch
from collections import OrderedDict
import fedlearn as fl

def make_simple_client(train_steps=5):
    class ToyClient(fl.Client):
        def get_parameters(self):
            return OrderedDict({"w": torch.zeros(10)})
        def fit(self, parameters, config):
            # Simulate training: shift weights slightly
            new_params = OrderedDict({k: v + 0.1 for k, v in parameters.items()})
            return new_params, 100
    return ToyClient()

def test_two_round_federation():
    model = torch.nn.Linear(10, 1)
    strategy = fl.FedAvg(
        initial_parameters=model.state_dict(),
        min_fit_clients=2,
        clients_per_round=2,
    )

    history_holder = []

    def run_server():
        h, _ = fl.server.start_server(
            "localhost:59876",
            fl.server.ServerConfig(num_rounds=2),
            strategy,
        )
        history_holder.extend(h)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.0)   # allow server to bind

    c0 = threading.Thread(
        target=fl.client.start_client,
        args=("localhost:59876", make_simple_client(), "c0"),
        daemon=True,
    )
    c1 = threading.Thread(
        target=fl.client.start_client,
        args=("localhost:59876", make_simple_client(), "c1"),
        daemon=True,
    )

    c0.start(); c1.start()
    server_thread.join(timeout=30)

    assert len(history_holder) == 2, "Expected 2 completed rounds"
```

---

## Logging Best Practices

The framework uses Python's standard `logging` module throughout. Follow these rules:

```python
import logging
log = logging.getLogger(__name__)   # always use module-level logger

# Choose the right level:
log.debug(...)    # step-level detail (chatty), off in production
log.info(...)     # key lifecycle events (round start, aggregation, shutdown)
log.warning(...)  # recoverable anomalies (stale update, zero examples)
log.error(...)    # operation failed (submit failed, RPC error)
log.exception(...)# unexpected exception — includes full traceback

# Include structured context in messages:
log.info("Aggregating %d updates for round %d", len(results), server_round)
# NOT: log.info(f"Aggregating {len(results)} for round {server_round}")
# (%-style is preferred — avoids string formatting when log level is disabled)
```

The server's JSON formatter will capture all of these and relay them to the Spring Boot backend.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgetting `params.clear()` after aggregation | OOM on large models | Always clear client param dicts after use |
| Using `torch.load` without `weights_only=True` | Security warning / pickle execution | Always pass `weights_only=True` |
| Calling `_trigger_aggregation_and_evaluation()` outside the lock | Race condition, double-aggregation | Only call while `self._lock` is held |
| Shared heartbeat and data channel | Heartbeats queue behind model uploads | Use `GrpcClient`'s dual-channel design |
| Not implementing `set_grpc_client` | No progress in UI | Implement the hook; framework calls it automatically |
| Training on GPU but submitting CPU tensors | Serialiser calls `.cpu()` anyway, but wasteful | Call `.cpu().detach()` before `state_dict()` if using GPU |
| α too low in Dirichlet partition | Some clients get 0 samples | Set `min_samples_per_client=10` guard |

---

## Contributing Checklist

Before submitting a pull request:

- [ ] All three abstract methods implemented (`initialize_parameters`, `aggregate_fit`, `evaluate`)
- [ ] `params.clear()` called after using client updates in aggregation
- [ ] New RPC handlers added to both `grpc_servicer.py` and `GrpcClient`
- [ ] Proto stubs regenerated and relative import fix applied
- [ ] Module-level logger used (`log = logging.getLogger(__name__)`)
- [ ] Unit tests added under `tests/`
- [ ] Example added or updated in `examples/` if adding a major feature
- [ ] Public symbols exported from `src/fedlearn/__init__.py`
- [ ] `CONTRIBUTING.md` reviewed for code style requirements
