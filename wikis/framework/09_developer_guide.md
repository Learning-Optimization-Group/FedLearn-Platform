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

> **Before extending anything, read [05 — Strategies](05_strategies.md) and
> [01 — Module Inventory](01_architecture_overview.md#module-inventory--all-57-modules).** A large
> share of what people set out to add here — FedProx, FedOpt, trimmed-mean aggregation, a Dirichlet
> partitioner, a differential-privacy accountant, a multi-thousand-client simulator — is already
> implemented and registered.

---

## Installation for Development

```bash
cd FedLearn-Platform/framework

# Create a virtual environment. Python 3.12 is what CI tests; setup.py declares a 3.10 floor.
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install in editable mode so changes take effect immediately
pip install -e .

# Dev extras are declared in setup.py's extras_require
pip install -e ".[dev]"    # pytest, pytest-asyncio, pytest-cov, ruff
pip install grpcio-tools   # only needed if you regenerate proto stubs
```

> **`setup.py` filters `torch*` and `torchvision*` out of `install_requires`** — PyTorch is installed
> separately so a consumer can pick a CUDA/CPU build. `requirements.txt` still pins
> `torch==2.12.0`, and that pin is load-bearing: the DeComFL golden fixtures and the ExecuTorch
> native extension were built against it, and `test_torch_version_matches_manifest` gates on it.
> `torchvision`/`torchaudio` are deliberately **absent** from `requirements.txt` — the framework does
> not import them, and pulling `torchvision` from PyPI against a pytorch-index `torch` build causes
> an ABI mismatch (`operator torchvision::nms does not exist`) that breaks the transformers-importing
> tests. Install them from the matched pytorch index alongside `torch` only where a consumer needs
> them.

Verify installation:
```python
import fedlearn as fl
print(fl.__file__)   # should point to src/fedlearn/__init__.py
```

Note the two paths that both work and mean different things: `pip install -e .` puts `fedlearn` on
the path permanently; `PYTHONPATH=src` (what CI and `fl-runtime` use) resolves it per-invocation
without installing.

---

## Adding a Custom Strategy

> **First check the [registry](05_strategies.md#the-strategy-registry).** Six strategies already
> ship — `fedavg`, `fedprox`, `fedopt`, `fedlora`, `decomfl`, `robust`. FedProx, FedOpt and
> Byzantine-robust aggregation are **not** things you need to write; earlier revisions of the
> strategies page presented them as examples, and that was wrong.

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
        clients_per_round: int = None,   # match the shipped convention, NOT a literal default
        # add your custom hyperparameters here
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = (
            clients_per_round if clients_per_round is not None else min_fit_clients
        )

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[OrderedDict, int]],
    ) -> Optional[OrderedDict]:
        if not results:
            return None
        # Accept every wire shape the other strategies do (2-/3-tuples, JSON params).
        from fedlearn.server._update_normalize import normalize_updates
        updates = normalize_updates(results)   # [(client_id, state_dict, num_examples), ...]
        # --- your aggregation logic here; must return an OrderedDict[str, torch.Tensor] ---
        raise NotImplementedError

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None                       # MUST be None, not a (0.0, {}) placeholder
        return self.evaluate_fn(server_round, parameters)

    # OPTIONAL: push client-side hyperparameters. Values must be strings — the proto
    # config is map<string,string> and the client trainer coerces them back.
    def get_client_config(self) -> dict:
        return {"learning_rate": str(self.learning_rate)}
```

**2. Register it so it is selectable by name:**

```python
# src/fedlearn/server/strategy_factory.py
from .my_strategy import MyStrategy

STRATEGY_REGISTRY = {
    ...,
    "mystrategy": MyStrategy,      # matched case-insensitively, - and _ ignored
}
```

**3. Export it (optional but conventional):**

```python
# src/fedlearn/server/__init__.py
from .my_strategy import MyStrategy
```

Top-level `fedlearn/__init__.py` is deliberately minimal — `FedLoRA` and `RobustAggregator` are only
on `fedlearn.server`, so a new strategy belongs there too unless it is core to the public API.

**4. Use it:**

```python
import fedlearn as fl
from fedlearn.server import create_strategy

strategy = create_strategy("my-strategy", initial_parameters=model.state_dict(), clients_per_round=3)
fl.server.start_server("0.0.0.0:50051", fl.server.ServerConfig(num_rounds=10), strategy)
```

### Four rules the coordinator relies on

1. **`aggregate_fit` runs while the coordinator's lock is held**, on the gRPC thread that submitted
   the last update. No blocking I/O, no network calls. The same applies to `evaluate_fn`.
2. **Return `None` from `aggregate_fit` to fail a round non-fatally** — the coordinator keeps the
   prior global model, logs a warning, and continues. That is the contract
   `RobustAggregator`'s Byzantine guard uses.
3. **Free client buffers** (`params.clear()`) after folding each one in; for large models the peak
   is otherwise `num_clients × model_size`.
4. **Expose `min_fit_clients` and `clients_per_round` as attributes** — `server.start_server` reads
   them off the strategy object to construct the coordinator, and reads `initial_parameters` too.

---

## Adding a Custom Client

> **For the first-order family you probably do not need one.** `fl.LocalTrainer` is a shipped,
> concrete `Client` that runs minibatch SGD, applies the FedProx proximal gradient when the server
> sends `proximal_mu > 0`, reads `learning_rate` / `local_epochs` out of the round config, and polls
> the FR-10 stop latch. Write a custom client when your training loop genuinely differs — a
> HuggingFace `Trainer`, a custom loss, quantisation-aware training, meta-learning. See
> [04 — LocalTrainer](04_client_internals.md#localtrainer--the-shipped-first-order-client).

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

> **STOP — `src/fedlearn/communication/protos/fedlearn.proto` is a MIRROR, not a source.** Editing it
> directly fails CI. `scripts/check_proto_mirror.sh` byte-compares three in-tree mirrors against the
> canonical contract and prints the exact `cp` that fixes a drift.

The canonical contracts live in `proto/` and are governed by `buf`:

| Canonical | Mirrors gated by `check_proto_mirror.sh` |
|---|---|
| `proto/fedlearn/v2/fedlearn.proto` | `framework/src/fedlearn/communication/protos/fedlearn.proto`<br>`mobile_client/proto/fedlearn/v2/fedlearn.proto` |
| `proto/fedlearn/fot/v1/fot.proto` | `framework/src/fedlearn/communication/protos/fot.proto` |

The flow is **edit canonical → buf → sync mirrors → regenerate stubs**:

```bash
# 1. Edit proto/fedlearn/v2/fedlearn.proto (the single source of truth), then:
cd proto
buf lint
buf breaking --against '.git#branch=main'   # a wire-breaking change fails the PR
buf generate

# 2. Sync the in-tree mirrors back out from canonical
cd ..
cp proto/fedlearn/v2/fedlearn.proto framework/src/fedlearn/communication/protos/fedlearn.proto
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
cp proto/fedlearn/fot/v1/fot.proto  framework/src/fedlearn/communication/protos/fot.proto

# 3. Regenerate the framework's committed Python stubs
cd framework
python -m grpc_tools.protoc \
    -I src/fedlearn/communication/protos \
    --python_out=src/fedlearn/communication/generated \
    --pyi_out=src/fedlearn/communication/generated \
    --grpc_python_out=src/fedlearn/communication/generated \
    src/fedlearn/communication/protos/fedlearn.proto

# 4. Fix the absolute import protoc emits in the _grpc file
sed -i '' 's/^import fedlearn_pb2/from . import fedlearn_pb2/' \
    src/fedlearn/communication/generated/fedlearn_pb2_grpc.py

# 5. Verify no mirror has drifted (this is the CI gate)
cd .. && ./scripts/check_proto_mirror.sh
```

Then implement it: handler in `grpc_servicer.py` → coordinator method if it touches round state →
`GrpcClient` method if clients call it. If the RPC should require authentication, **add its method
name to `security/interceptor.PROTECTED_METHODS`** — the interceptor matches by name and silently
exempts anything not listed.

CI enforces all of this as `proto.yml`: `buf lint`, `buf breaking` against `main`, a `buf generate`
**freshness** check (regeneration must be a no-op, so committed stubs cannot silently rot), and the
mirror check.

> **The protobuf floor is pinned to the newest gencode, not to a preference.** `fot_pb2.py` is
> generated at 5.29.0 and `fedlearn_pb2.py` at 4.25.1; protobuf requires runtime ≥ gencode, so
> `requirements.txt` pins `protobuf>=5.29.0,<6.0.0`. `tests/test_protobuf_gencode_pin.py` guards it.
> Regenerating with a newer protoc raises that floor — check that test before committing new stubs.

---

## Testing Guidelines

### How the suite is actually run

```bash
cd framework
PYTHONPATH=src python -m pytest -q          # exactly how CI runs it
```

`pytest.ini` is authoritative and does two things beyond a bare pytest:

```ini
addopts = -m "not slow" --cov=fedlearn --cov-report=term-missing --cov-fail-under=73
markers =
    slow: tests that download models or run full training (deselect with -m "not slow")
```

- **`-m "not slow"`** — the default run deselects tests that download models or run full training.
  Mark anything in that category `@pytest.mark.slow`.
- **Coverage is enforced at 73%.** Measured line coverage is ~77%, so the floor guards against
  regression rather than being an aspirational target. **Running a hand-picked subset will report
  low coverage and trip the floor by design — pass `--no-cov` for a subset run.**

Tests live in `framework/tests/` (~100 modules), with frozen fixtures under `tests/fixtures/`
— including `decomfl_golden/`, the Python↔C++ RNG parity contract. Do not regenerate those
casually; see [06 — Cross-Architecture Determinism](06_decomfl.md#cross-architecture-determinism-golden-vector-parity).

Useful existing tests to copy the shape of:

| If you are touching… | Read first |
|---|---|
| aggregation | `test_fedavg_aggregator.py`, `test_strategy_adversarial_audit.py` |
| the wire / codec | `test_safetensors_codec.py`, `test_wire_adversarial_hardening.py`, `test_serializer_safetensors_sniff.py` |
| the coordinator | `test_coordinator.py`, `test_round_timeout.py`, `test_coordinator_failed_round.py` |
| DeComFL | `test_decomfl_correctness.py`, `test_decomfl_lr_guard.py`, `test_decomfl_seed_lifecycle.py` |
| security | `test_token_interceptor_e2e.py`, `test_tls_policy.py`, `test_client_identity_binding.py` |
| the simulator | `test_simulation_federation.py`, `test_simulation_rng.py`, `test_simulation_partition.py` |

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

def test_fedavg_renormalises_a_subset_held_key():
    """FR-18: a key only some clients carry is weighted over THOSE clients, not all."""
    agg = FedAvgAggregator()
    output = agg.aggregate([
        (make_params({"shared": [1.0]}), 100),                   # no "extra"
        (make_params({"shared": [3.0], "extra": [4.0]}), 100),
    ])
    assert torch.allclose(output["shared"], torch.tensor([2.0]))  # 0.5/0.5 over both
    assert torch.allclose(output["extra"],  torch.tensor([4.0]))  # 1.0 over the ONE holder
```

Run a focused subset (remember to disable the coverage floor):
```bash
PYTHONPATH=src pytest tests/test_fedavg_aggregator.py -v --no-cov
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
| **Editing `communication/protos/*.proto` directly** | CI fails on `check_proto_mirror.sh` | Edit `proto/…` and `cp` out — the script prints the exact command |
| Forgetting `params.clear()` after aggregation | OOM on large models | Always clear client param dicts after use |
| Using `torch.load` without `weights_only=True` | Security warning / pickle execution | Always pass `weights_only=True`. Better: don't reintroduce a pickle path at all |
| Calling `_trigger_aggregation_and_evaluation()` outside the lock | Race condition, double-aggregation | Only call while `self._lock` is held |
| Calling `_trigger_aggregation_and_evaluation()` on a **DeComFL** run | Clients silently desync — `gradient_history[round]` is never written | Use `_trigger_round_completion()`, which dispatches on strategy type (FR-4) |
| Shared heartbeat and data channel | Heartbeats queue behind model uploads; the server's `should_stop` can never reach a busy client | Keep `GrpcClient`'s dual-channel design **and** the `_stop_training` latch |
| Not implementing `set_grpc_client` | No progress in UI, and `fit()` cannot see a server stop | Implement the hook; the framework calls it automatically |
| Training on GPU but submitting CPU tensors | Serialiser calls `.cpu()` anyway, but wasteful | Call `.cpu().detach()` before `state_dict()` if using GPU |
| Choosing the aggregation device by `torch.cuda.is_available()` | Never selects MPS; force-migrates a CPU run on a CUDA box and crashes FedOpt | Derive it from the incoming tensors (`_first_tensor_device`) |
| Templating an aggregate on `results[0]`'s keys | Keys only later clients carry are dropped; subset-held keys decay toward zero | Template on the **union** with per-key totals (FR-18) |
| α too low in a Dirichlet partition | Some clients get 0 samples; `num_examples == 0` is rejected at ingress, silently shrinking the cohort | Pass `min_partition_size` to `dirichlet_partition` |
| Passing `model.state_dict()` as a DeComFL server's `initial_parameters` | `d_server > d_client`; the shared-seed `z` misaligns and the model diverges | Use `estimators.params.trainable_state(model)` (FR-14) |
| Reusing a DeComFL `eta` at a larger model dimension | The run learns, then explodes to ~1e19 loss | The constructor now refuses the measured-divergent regime; use `suggested_eta(d)` |
| Adding a new RPC without listing it in `PROTECTED_METHODS` | The RPC is silently exempt from connection-token auth | Add the method **name** to `security/interceptor.PROTECTED_METHODS` |
| Configuring root logging at import time | Hijacks a host application's logger | Do it in the entry point (`configure_logging()`), as `start_server` does (FR-9) |

---

## Contributing Checklist

Before submitting a pull request:

- [ ] All three abstract methods implemented (`initialize_parameters`, `aggregate_fit`, `evaluate`),
      with `evaluate` returning `None` when `evaluate_fn` is `None`
- [ ] `params.clear()` called after using client updates in aggregation
- [ ] Incoming updates routed through `_update_normalize.normalize_updates`
- [ ] New strategies added to `STRATEGY_REGISTRY` (not just exported)
- [ ] New RPC handlers added to `grpc_servicer.py` **and** `GrpcClient` **and**
      `security/interceptor.PROTECTED_METHODS`
- [ ] Proto edited at `proto/` (never a mirror), `buf lint` + `buf breaking` clean, mirrors `cp`-ed,
      stubs regenerated, `./scripts/check_proto_mirror.sh` green
- [ ] Module-level logger used (`log = logging.getLogger(__name__)`), `%`-style formatting
- [ ] Unit tests added under `framework/tests/`; anything that downloads a model or trains for real
      is marked `@pytest.mark.slow`
- [ ] `PYTHONPATH=src python -m pytest -q` passes, including the coverage floor
- [ ] Anything measurable has a **seeded harness** under `benchmarks/` that writes a JSON record —
      a number without a re-runnable harness is not a result
- [ ] Example added or updated in `examples/` if adding a major feature
- [ ] Public symbols exported from `src/fedlearn/server/__init__.py` (or the top-level
      `__init__.py` only if genuinely core)
- [ ] `CONTRIBUTING.md` reviewed for code style requirements
