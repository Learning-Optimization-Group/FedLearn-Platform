# 05 — Aggregation Strategies

## Table of Contents
- [Overview](#overview)
- [The Strategy Abstract Base Class](#the-strategy-abstract-base-class)
- [FedAvg — Federated Averaging](#fedavg--federated-averaging)
  - [Mathematical Foundation](#mathematical-foundation)
  - [FedAvg Constructor Parameters](#fedavg-constructor-parameters)
  - [FedAvgAggregator — Weighted Averaging](#fedavgaggregator--weighted-averaging)
  - [Security: num_examples Sanitisation](#security-num_examples-sanitisation)
  - [Memory Management in Aggregation](#memory-management-in-aggregation)
  - [Evaluation Hook](#evaluation-hook)
- [FedAvg vs. Local Training](#fedavg-vs-local-training)
- [Implementing a Custom Strategy](#implementing-a-custom-strategy)
  - [FedProx Example](#fedprox-example)
  - [Trimmed Mean (Byzantine-Robust) Example](#trimmed-mean-byzantine-robust-example)
- [Strategy Lifecycle in the Server](#strategy-lifecycle-in-the-server)
- [Strategy Selection Decision Tree](#strategy-selection-decision-tree)

---

## Overview

The strategy system is the primary extension point for the FedLearn framework. A `Strategy` object controls:
1. **Initialization:** How the global model is initially distributed
2. **Aggregation:** How client updates are combined into a new global model
3. **Evaluation:** How the global model is evaluated after each round

The framework ships with two built-in strategies:
- **`FedAvg`** — Weighted federated averaging (McMahan et al., 2017)
- **`DeComFL`** — Dimension-free gradient scalar aggregation (documented separately in [06 — DeComFL](06_decomfl.md))

---

## The Strategy Abstract Base Class

```python
# server/strategy.py
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
from collections import OrderedDict
import torch

class Strategy(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """
        Return the initial global model parameters.
        Called once before the first round.
        """
        pass

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        """
        Aggregate training results from multiple clients.

        Args:
            server_round: Current round number (1-indexed)
            results:      List of (parameters, num_examples) tuples

        Returns:
            Aggregated global model parameters, or None if aggregation failed
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        server_round: int,
        parameters: OrderedDict[str, torch.Tensor],
    ) -> Optional[Tuple[float, dict]]:
        """
        Evaluate the global model after aggregation.

        Args:
            server_round: Current round number
            parameters:   Global model parameters to evaluate

        Returns:
            (loss, metrics_dict) or None if no evaluation function is set
        """
        pass
```

### Method Call Order (Per Round)

```
coordinator._trigger_aggregation_and_evaluation()
    │
    ├── strategy.aggregate_fit(round, results)
    │       → new global parameters
    │
    └── strategy.evaluate(round, new_parameters)
            → (loss, metrics)
```

### Return Type Contract

| Method | Return Value | Effect if None Returned |
|--------|-------------|------------------------|
| `initialize_parameters()` | Initial state_dict | Server aborts with `UNAVAILABLE` |
| `aggregate_fit()` | Updated state_dict | Round marked as failed; metrics=None |
| `evaluate()` | (loss, metrics dict) | Logged as warning; training continues |

---

## FedAvg — Federated Averaging

### Mathematical Foundation

**FedAvg** (Federated Averaging) computes a weighted average of client model parameters:

```
w_global = Σ(n_i / N) × w_i

where:
  w_i = parameters submitted by client i
  n_i = number of training examples used by client i
  N   = total training examples = Σ n_i
```

This is mathematically equivalent to gradient descent on the global objective function when:
- All clients train for exactly one local step (E=1)
- Data distribution is IID across clients

In practice, multiple local epochs (E>1) and non-IID data make FedAvg approximate, but it remains effective in most federated settings.

### FedAvg Constructor Parameters

```python
class FedAvg(Strategy):
    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        evaluate_fn: Optional[Callable] = None,
        min_fit_clients: int = 1,
        clients_per_round: int = 2,
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round
        self.aggregator = FedAvgAggregator()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_parameters` | `OrderedDict[str, Tensor]` | Starting model state_dict |
| `evaluate_fn` | `Callable(round, params) → (loss, dict)` | Optional evaluation function |
| `min_fit_clients` | `int` | Minimum clients for aggregation (guards against premature rounds) |
| `clients_per_round` | `int` | Exact number of clients expected per round |

> **Note:** `min_fit_clients` is currently read by the `FLCoordinator` for its `min_clients` attribute, but the coordinator uses `clients_per_round` for the actual aggregation trigger. Semantically, `clients_per_round` should always be ≥ `min_fit_clients`.

### FedAvgAggregator — Weighted Averaging

The `FedAvgAggregator` is separated from the `FedAvg` strategy to allow reuse:

```python
class FedAvgAggregator:
    MAX_SAMPLES = 100_000  # Security cap — prevents poisoning via inflated counts

    def aggregate(self, updates):
        """
        Args:
            updates: List of (params, num_examples) OR (client_id, params, num_examples)
        Returns:
            Weighted average of params
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Normalise to 3-tuple format
        deserialized_updates = []
        for entry in updates:
            if len(entry) == 3:
                client_id, params, num_examples = entry
            else:
                params, num_examples = entry
                client_id = None

            # Handle JSON-serialised params (from proto round-trip)
            if isinstance(params, str):
                params = json.loads(params)
                params = OrderedDict({k: torch.tensor(v) for k, v in params.items()})

            deserialized_updates.append((client_id, params, num_examples))

        # Security: discard updates with invalid num_examples
        sanitized = [
            (cid, p, min(n, self.MAX_SAMPLES))
            for cid, p, n in deserialized_updates if n > 0
        ]
        if not sanitized:
            raise ValueError("No valid updates after sanitisation.")

        # Compute weighted sum
        total_examples = sum(n for _, _, n in sanitized)

        # Initialise accumulator to zeros matching the template tensor shapes
        _, template_params, _ = sanitized[0]
        template_params = {k: v.to(device) for k, v in template_params.items()}
        aggregated = OrderedDict(
            (key, torch.zeros_like(tensor, dtype=torch.float32))
            for key, tensor in template_params.items()
        )

        for client_id, params, num_examples in sanitized:
            weight = num_examples / total_examples  # normalised weight

            for key in aggregated:
                if key in params:
                    torch.add(
                        aggregated[key],
                        params[key].to(device).float(),
                        alpha=weight,           # in-place weighted accumulation
                        out=aggregated[key],
                    )

            # Free client buffer immediately after use
            params.clear()

        return aggregated
```

**Step-by-step walkthrough for 3 clients:**

```
Client A: 100 examples, weight = 100/250 = 0.40
Client B:  75 examples, weight =  75/250 = 0.30
Client C:  75 examples, weight =  75/250 = 0.30

For each parameter key "layer.weight":
  aggregated["layer.weight"] = 0
  aggregated["layer.weight"] += 0.40 × A["layer.weight"]
  aggregated["layer.weight"] += 0.30 × B["layer.weight"]
  aggregated["layer.weight"] += 0.30 × C["layer.weight"]
```

### Security: num_examples Sanitisation

The aggregation contains two layers of protection against **model poisoning via inflated dataset size**:

1. **In `FLCoordinator.submit_client_update()`:**
   ```python
   MAX_NUM_EXAMPLES = 100_000
   num_examples = min(num_examples, self.MAX_NUM_EXAMPLES)
   ```

2. **In `FedAvgAggregator.aggregate()`:**
   ```python
   MAX_SAMPLES = 100_000
   sanitized = [(cid, p, min(n, self.MAX_SAMPLES)) for cid, p, n in updates if n > 0]
   ```

Without these caps, a malicious client could claim to have trained on 10 billion examples, making its parameters dominate the weighted average (weight ≈ 1.0, others ≈ 0.0), effectively replacing the global model.

### Memory Management in Aggregation

After each client's parameters are incorporated into the weighted sum, they are explicitly freed:

```python
for client_id, params, num_examples in sanitized:
    # ... accumulate weighted sum ...
    params.clear()   # free the OrderedDict immediately
```

For large models (e.g., OPT-125M at ~500 MB), keeping all client copies in memory simultaneously would require `num_clients × model_size` RAM. This `params.clear()` pattern ensures each client's memory is released as soon as its contribution is folded in.

### Evaluation Hook

The `evaluate_fn` callback is optional. If provided, it is called after each aggregation:

```python
def evaluate(self, server_round, parameters):
    if self.evaluate_fn is None:
        return None

    loss, metrics = self.evaluate_fn(server_round, parameters)
    log.info("FedAvg eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics)
    return loss, metrics
```

The evaluation function signature:

```python
def my_evaluation_function(
    server_round: int,
    parameters: OrderedDict[str, torch.Tensor]
) -> Tuple[float, dict]:
    """
    Evaluate the global model on a held-out test set.

    Returns:
        (loss, metrics) where metrics is any JSON-serialisable dict
    """
    model.load_state_dict(parameters)
    model.eval()

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            total_loss += F.cross_entropy(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum().item()

    loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    return loss, {"accuracy": accuracy}
```

---

## FedAvg vs. Local Training

It's useful to understand the difference between what each client does and what the strategy computes:

```
Client A trains locally for E=3 epochs:
  w_A_start (global) → [3 local gradient steps] → w_A_end

Client B trains locally for E=3 epochs:
  w_B_start (global) → [3 local gradient steps] → w_B_end

FedAvg aggregation:
  w_global_new = 0.5 × w_A_end + 0.5 × w_B_end   (assuming equal samples)

Next round, both clients start from w_global_new, NOT from their own w_A_end or w_B_end.
```

This "reset to global" behaviour at the start of each round is what prevents client models from diverging permanently. It also means the local optimizer state (momentum, Adam's m/v buffers) is discarded between rounds — or must be explicitly preserved by the client implementation.

---

## Implementing a Custom Strategy

Any class inheriting from `Strategy` and implementing the three abstract methods is a valid strategy.

### FedProx Example

FedProx adds a proximal term `μ/2 ||w - w_global||²` to the local loss to limit client drift:

```python
from fedlearn.server.strategy import Strategy
import torch
from collections import OrderedDict
from typing import Optional, Tuple, List

class FedProx(Strategy):
    """
    FedProx strategy: FedAvg with proximal regularisation.
    Reduces client drift in non-IID settings.
    """

    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        evaluate_fn=None,
        min_fit_clients: int = 1,
        clients_per_round: int = 2,
        proximal_mu: float = 0.01,   # regularisation strength
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round
        self.proximal_mu = proximal_mu   # passed to clients via config

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[OrderedDict, int]],
    ) -> Optional[OrderedDict]:
        """FedProx uses same weighted averaging as FedAvg."""
        if not results:
            return None

        total_examples = sum(n for _, n in results)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialise accumulator
        template = {k: v.to(device) for k, v in results[0][0].items()}
        aggregated = OrderedDict(
            (k, torch.zeros_like(t, dtype=torch.float32))
            for k, t in template.items()
        )

        for params, num_examples in results:
            weight = num_examples / total_examples
            for key in aggregated:
                aggregated[key] += weight * params[key].to(device).float()
            params.clear()

        return aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
```

For FedProx to work, the client's `fit()` method must add the proximal term to its loss:

```python
class FedProxClient(fl.Client):
    def fit(self, parameters, config):
        global_weights = parameters  # save for proximal term

        self.model.load_state_dict(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        for inputs, targets in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)

            # Proximal term: μ/2 × ||w - w_global||²
            proximal_term = 0.0
            for name, param in self.model.named_parameters():
                proximal_term += torch.norm(param - global_weights[name])**2
            loss += (config.get('proximal_mu', 0.01) / 2) * proximal_term

            loss.backward()
            optimizer.step()

        return self.model.state_dict(), len(self.train_loader.dataset)
```

### Trimmed Mean (Byzantine-Robust) Example

For Byzantine-robust aggregation, replace FedAvg's weighted mean with a coordinate-wise trimmed mean:

```python
class TrimmedMeanStrategy(Strategy):
    """
    Coordinate-wise trimmed mean for Byzantine robustness.
    Removes the top and bottom `trim_fraction` of updates per coordinate.
    """

    def __init__(self, initial_parameters, evaluate_fn=None,
                 min_fit_clients=1, clients_per_round=5, trim_fraction=0.1):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round
        self.trim_fraction = trim_fraction   # remove 10% from each end by default

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(self, server_round, results):
        if not results:
            return None

        n = len(results)
        k = int(n * self.trim_fraction)  # number to trim from each end

        aggregated = OrderedDict()
        param_names = list(results[0][0].keys())

        for name in param_names:
            stacked = torch.stack([params[name].float() for params, _ in results])
            # stacked shape: [n_clients, *param_shape]

            # Sort along client dimension, trim k from each end
            sorted_stacked, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_stacked[k:n-k]   # shape: [n-2k, *param_shape]

            aggregated[name] = trimmed.mean(dim=0)

        for params, _ in results:
            params.clear()

        return aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
```

---

## Strategy Lifecycle in the Server

```python
# server.py — full strategy usage trace

# 1. Strategy is instantiated by user code
strategy = FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    clients_per_round=3,
)

# 2. Coordinator is created, strategy is injected
coordinator = FLCoordinator(
    strategy=strategy,
    min_clients_for_aggregation=strategy.min_fit_clients,
    clients_per_round=strategy.clients_per_round,
)

# 3. Initial parameters from strategy are set on coordinator
coordinator.set_initial_parameters(strategy.initial_parameters)
# equivalent to: strategy.initialize_parameters()

# 4. Training loop
for round_num in range(1, config.num_rounds + 1):
    coordinator.start_round()
    coordinator.wait_for_round_to_complete()
    # ... inside coordinator, during this wait:
    #   strategy.aggregate_fit(round_num, results) is called
    #   strategy.evaluate(round_num, new_params) is called

    metrics = coordinator.get_latest_metrics()
    history.append((round_num, metrics))
```

---

## Strategy Selection Decision Tree

```
Are you training LLMs or very large models?
│
├── YES → DeComFL (06_decomfl.md)
│           - Communication: O(K×P) scalars per round
│           - No backpropagation required
│           - Works on memory-constrained devices
│
└── NO  → FedAvg (this document)
            │
            ├── Is your data IID and clients trusted?
            │   └── YES → Standard FedAvg
            │
            ├── Do you have many local epochs (E>1) and non-IID data?
            │   └── YES → FedProx (add proximal term)
            │
            └── Do you have potentially malicious/Byzantine clients?
                └── YES → TrimmedMean or Krum (custom strategy)
```
