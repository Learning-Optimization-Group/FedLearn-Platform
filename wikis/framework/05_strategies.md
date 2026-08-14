# 05 — Aggregation Strategies

## Table of Contents
- [Overview](#overview)
- [The Strategy Registry](#the-strategy-registry)
- [The Strategy Abstract Base Class](#the-strategy-abstract-base-class)
- [FedAvg — Federated Averaging](#fedavg--federated-averaging)
  - [Mathematical Foundation](#mathematical-foundation)
  - [FedAvg Constructor Parameters](#fedavg-constructor-parameters)
  - [FedAvgAggregator — Weighted Averaging](#fedavgaggregator--weighted-averaging)
  - [Security: num_examples Sanitisation](#security-num_examples-sanitisation)
  - [Memory Management in Aggregation](#memory-management-in-aggregation)
  - [Evaluation Hook](#evaluation-hook)
- [FedAvg vs. Local Training](#fedavg-vs-local-training)
- [FedProx — Proximal Regularisation (shipped)](#fedprox--proximal-regularisation-shipped)
- [FedOpt — Server-Side Adaptive Optimisation (shipped)](#fedopt--server-side-adaptive-optimisation-shipped)
- [FedLoRA — Adapter-Only Federation (shipped)](#fedlora--adapter-only-federation-shipped)
- [Byzantine-Robust Aggregation (FR-12)](#byzantine-robust-aggregation-fr-12)
- [Central Differential Privacy (FR-13)](#central-differential-privacy-fr-13)
- [Trainable-Subset Federation (DA-11)](#trainable-subset-federation-da-11)
- [Adapter Bundles (DA-9)](#adapter-bundles-da-9)
- [Implementing a Custom Strategy](#implementing-a-custom-strategy)
- [Strategy Lifecycle in the Server](#strategy-lifecycle-in-the-server)
- [Strategy Selection Decision Tree](#strategy-selection-decision-tree)

---

## Overview

The strategy system is the primary extension point for the FedLearn framework. A `Strategy` object controls:
1. **Initialization:** How the global model is initially distributed
2. **Aggregation:** How client updates are combined into a new global model
3. **Evaluation:** How the global model is evaluated after each round

The framework ships with **six registered strategies**, not two. All six are real, tested
implementations selectable per project — none of them is an example you are expected to write
yourself.

---

## The Strategy Registry

`server/strategy_factory.py` is the single dispatch point. Adding a strategy is a one-line registry
entry, not another `if`/`elif` branch scattered across the launch code.

```python
STRATEGY_REGISTRY: Dict[str, Callable[..., Strategy]] = {
    "fedavg":  FedAvg,
    "fedprox": FedProx,
    "fedopt":  FedOpt,
    "fedlora": FedLoRA,
    "decomfl": DeComFL,
    "robust":  RobustAggregator,
}
```

| Name | Class | Module | What it does |
|---|---|---|---|
| `fedavg` | `FedAvg` | `strategy.py` | num-examples-weighted mean of client models |
| `fedprox` | `FedProx` | `strategy.py` | **identical** server aggregation; adds a client-side proximal penalty |
| `fedopt` | `FedOpt` | `strategy.py` | server-side Adam/Yogi step on a pseudo-gradient, moments persisted across rounds |
| `fedlora` | `FedLoRA` | `strategy.py` | aggregates **only** LoRA adapter keys; optional central DP |
| `decomfl` | `DeComFL` | `decomfl_strategy.py` | zeroth-order gradient scalars — see [06](06_decomfl.md) |
| `robust` | `RobustAggregator` | `robust_aggregation.py` | coordinate-wise median / β-trimmed mean + norm clipping |

```python
from fedlearn.server import create_strategy
strategy = create_strategy("fed_prox", initial_parameters=params, proximal_mu=0.1)
```

**Name matching is case-insensitive and ignores hyphens and underscores** (`_normalize` lowercases
and strips both), so `"FedAvg"`, `"fed-avg"` and `"fed_avg"` all resolve. An unregistered name raises
`ValueError` listing the available names.

`strategy_factory.py` lives in its own module specifically to break an import cycle: `strategy.py`
must not import `decomfl_strategy` (which imports `strategy`), so the factory — which needs both —
sits above them.

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
| `initialize_parameters()` | Initial state_dict | The servicer aborts `UNAVAILABLE` ("Server not initialized") on the first model request |
| `aggregate_fit()` | Updated state_dict | Round marked as failed; global model **untouched**; `latest_metrics = None`; the loop continues |
| `evaluate()` | `(loss, metrics)` | `latest_metrics = None`; the round still completes (FR-22 guards the unpack) |

> **`evaluate()` returning `None` is the normal case, not an error path.** `evaluate_fn` defaults to
> `None` on FedAvg, FedProx, FedOpt, FedLoRA and RobustAggregator. Unpacking that `None` as a
> 2-tuple used to raise a `TypeError` *inside the coordinator's lock*, after the round's updates
> were cleared — wedging the round. Both triggers now guard it (FR-22).

### The Optional `get_client_config()` Hook

Not on the ABC, but read by the coordinator every time it serves the global model. A strategy that
needs to push client-side hyperparameters implements it:

```python
def get_client_config(self) -> dict:
    # The proto config is map<string,string>, so values are STRINGIFIED here
    # and coerced back by the client trainer.
    return {"proximal_mu": str(self.mu),
            "learning_rate": str(self.learning_rate),
            "local_epochs": str(self.local_epochs)}
```

`FedProx` and `FedOpt` implement it. `FedAvg`, `FedLoRA`, `DeComFL` and `RobustAggregator` do not, so
`FLCoordinator._strategy_client_config()` returns `{}` for them and behaviour is unchanged. This is
the params-path analogue of DeComFL's `GetDeComFLConfig`.

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
        clients_per_round: int = None,      # None -> falls back to min_fit_clients
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregator = FedAvgAggregator()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_parameters` | `OrderedDict[str, Tensor]` | Starting model state_dict |
| `evaluate_fn` | `Callable(round, params) → (loss, dict)` | Optional evaluation function; `None` means no server-side eval |
| `min_fit_clients` | `int` | Minimum clients for aggregation (the floor a force-resolved round must clear) |
| `clients_per_round` | `int` | Exact number of clients expected per round. **Defaults to `min_fit_clients` when omitted** — every shipped strategy uses that same `clients_per_round if … is not None else min_fit_clients` pattern |

> **Note:** `min_fit_clients` is read by the `FLCoordinator` as its `min_clients`, but the *inline*
> aggregation trigger fires on `clients_per_round`. `min_clients` is what matters on the
> **timeout** path: `resolve_round_incomplete` force-aggregates only if at least
> `max(1, min_clients)` reported. Semantically, `clients_per_round` should always be ≥
> `min_fit_clients`.

### FedAvgAggregator — Weighted Averaging

The `FedAvgAggregator` is separated from the `FedAvg` strategy so `FedProx`, `FedOpt` and `FedLoRA`
can reuse it verbatim rather than re-deriving the mean:

```python
class FedAvgAggregator:
    MAX_SAMPLES = 100_000  # Security cap — prevents poisoning via inflated counts

    def aggregate(self, updates):
        """
        Args:
            updates: List of (params, num_examples) OR (client_id, params, num_examples);
                     params may also be a JSON string (decoded by the shared normalizer).
        Returns:
            Weighted average of params
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        # ONE shared normalizer for all four aggregation sites (FedAvg, FedLoRA's key probe,
        # RobustAggregator, and the DP path) — see server/_update_normalize.py.
        updates = normalize_updates(updates)

        # Aggregate on the device the DATA is already on.
        device = _first_tensor_device(updates)

        sanitized = [(cid, p, min(n, self.MAX_SAMPLES)) for cid, p, n in updates if n > 0]
        if not sanitized:
            raise ValueError("No valid updates after sanitization.")

        # FR-18: template on the UNION of client keys, and total examples PER KEY.
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        key_totals: dict[str, int] = {}
        for _cid, params, num_examples in sanitized:
            for key, tensor in params.items():
                if key not in aggregated:
                    aggregated[key] = torch.zeros_like(tensor.to(device), dtype=torch.float32)
                key_totals[key] = key_totals.get(key, 0) + num_examples

        for client_id, params, num_examples in sanitized:
            for key in params:
                if key in aggregated:
                    weight = num_examples / key_totals[key]      # PER-KEY renormalisation
                    torch.add(aggregated[key], params[key].to(device).float(),
                              alpha=weight, out=aggregated[key])
            params.clear()          # free the client buffer immediately

        return aggregated
```

**Two corrections in this method are load-bearing. Do not "simplify" either one back.**

**1. FR-18 — union of keys, and totals *per key*.** Templating the accumulator on `updates[0]`
alone silently dropped any key that only a later client carried. Worse, weighting every key by
`num_examples / total_over_ALL_clients` while summing only the clients that *have* that key scaled a
subset-held key by a weight share `< 1` — decaying it toward zero every round — and let a client
with an inflated `num_examples` bypass the per-client L2 clip on the keys it omitted. Renormalising
each key over the clients that actually provided it fixes both, and when every client holds every
key it reduces **exactly** to the previous weighted mean.

**2. Device is taken from the data, not from global availability.** The old rule,
`"cuda" if torch.cuda.is_available() else "cpu"`, was wrong in two distinct ways: it never selected
MPS at all (so on Apple Silicon every GPU-trained update was silently copied to CPU every round),
and on a CUDA host it force-migrated aggregates to CUDA regardless of where the run actually lived —
which is what crashed `FedOpt`, whose `_global` stays on the device `initial_parameters` arrived on,
so `g = old - x_bar` mixed CPU and CUDA tensors. `_first_tensor_device(updates)` is both correct and
intent-preserving: a run the caller placed on CPU stays on CPU even on a CUDA box.

**Step-by-step walkthrough for 3 clients (all holding every key):**

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

And with a key only B and C carry, the denominator for **that key** is `75 + 75 = 150`, so B and C
weight 0.5 each and A contributes nothing to it — rather than B and C summing to 0.6 of a value that
should be 1.0.

### Accepted Update Shapes

Four aggregation sites accept the same wire shapes and coerce them identically, through the one
implementation in `server/_update_normalize.py`:

| Input | Normalised to |
|---|---|
| `(client_id, params, num_examples)` | unchanged |
| `(params, num_examples)` | `client_id = None` |
| `params` as a JSON string `{name: list}` | decoded to `OrderedDict[str, Tensor]`; a decode failure raises `ValueError` **naming the offending client** |

Callers that do not need `num_examples` (the DP uniform average, the FedLoRA key probe) simply
ignore the third element.

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

Note the two are **not** redundant. The coordinator's cap protects the live gRPC path; the
aggregator's protects every *other* caller — the in-process simulator, benchmark harnesses, and unit
tests all reach `aggregate()` without passing through `submit_client_update`. A cap in one place only
would be bypassable by construction.

The other two strategies that do not use `FedAvgAggregator` handle this differently rather than
inheriting the cap: `RobustAggregator` validates `num_examples > 0` and then **ignores it entirely**
(unweighted by design, see [above](#aggregation-is-unweighted-and-that-is-deliberate)), and the DP
path also ignores it (uniform average — weighting would void the ε claim).

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

## FedProx — Proximal Regularisation (shipped)

`FedProx` (Li et al. 2020, [arXiv:1812.06127](https://arxiv.org/abs/1812.06127)) is a **real class in
`server/strategy.py`**, registered as `"fedprox"`. Do not reimplement it.

**Its server aggregation is bitwise identical to FedAvg** — it reuses `FedAvgAggregator` with no
reimplementation. FedProx's entire difference is *client-side*: each client minimises

```
min_w  F_i(w) + (mu/2) * || w - w_global ||^2
```

Because that term lives in the client's loss, `mu` never touches server aggregation, so `mu = 0`
makes aggregation bitwise-identical to FedAvg. `mu` reaches the client through
`get_client_config() → config["proximal_mu"]`, read by `LocalTrainer.fit()` exactly the way
`DeComFLClient.fit()` reads `config["learning_rate"]`.

```python
FedProx(initial_parameters, evaluate_fn=None, min_fit_clients=1, clients_per_round=None,
        proximal_mu=0.0, learning_rate=0.01, local_epochs=1, allow_unstable_mu=False)
```

### The (mu, lr) stability envelope — measured, then enforced

`LocalTrainer` applies the penalty as an explicit gradient term `mu * (w - w_global)` added to
`.grad` before the SGD step. The penalty's own iteration is therefore

```
w <- w - lr*mu*(w - w_global)
```

a linear map with multiplier `(1 - lr*mu)`, which contracts toward the anchor **only while
`0 < lr*mu < 2`**. At `lr*mu >= 2` the penalty oscillates *outward*: the term whose entire purpose is
to bound client drift instead amplifies it, and nothing raises. Measured on a 3-class linear task,
one local epoch, drift = `||w_local - w_global||`:

| | `lr*mu = 1.9` | `lr*mu = 10` |
|---|---|---|
| `lr=0.5` | `mu=3.8` → drift **0.517** | `mu=20` → drift **2.4e4** |
| `lr=0.1` | `mu=19` → drift **0.109** | `mu=100` → drift **4.8e3** |
| `lr=0.01` | `mu=190` → drift **0.011** | `mu=1000` → drift **4.8e2** |

The minimum sits at `lr*mu ≈ 2` in all three rows and `mu = 10/lr` is catastrophic in all three, so
the boundary is a property of the **discretisation**, not of the task.

`_check_prox_stability()` therefore runs at **construction**, not mid-run, so a doomed configuration
fails before it burns a federation:

| `s = lr*mu` | Behaviour |
|---|---|
| `mu <= 0` | return immediately — `mu = 0` is exactly FedAvg |
| `s < PROX_STABILITY_WARN` (1.0) | silent |
| `1.0 <= s < PROX_STABILITY_LIMIT` (2.0) | **warn** — stable but near the boundary; drift-vs-`mu` is no longer monotone here |
| `s >= 2.0` | **raise `ValueError`**, unless `allow_unstable_mu=True` (then warn loudly) |

The reason this is an error rather than a note: the failure is silent *and inverted*. A run
configured to reduce drift produces far more of it and still completes — nothing in an accuracy
curve says "unstable". It looks like an ordinary bad-hyperparameter run.

---

## FedOpt — Server-Side Adaptive Optimisation (shipped)

`FedOpt` implements Reddi et al. 2021's Algorithm 2
([arXiv:2003.00295](https://arxiv.org/abs/2003.00295)) — **FedAdam** and **FedYogi**. Clients do
ordinary local SGD (`proximal_mu = 0`); the adaptivity is entirely server-side.

```python
FedOpt(initial_parameters, evaluate_fn=None, min_fit_clients=1, clients_per_round=None,
       server_learning_rate=1.0, beta1=0.9, beta2=0.99, tau=1e-3, variant="adam",
       learning_rate=0.01, local_epochs=1, bias_correction=False)
```

Per round: aggregate the client models into `x_bar` with the usual weighted mean, form a
pseudo-gradient, then take an Adam-style step, **persisting `(m, v)` across rounds**:

```
g_t = w_global(old) - x_bar                          # == -Delta_t in the paper
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
FedAdam:  v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
FedYogi:  v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g_t^2) * g_t^2
w_global(new) = w_global(old) - eta * m_t / (sqrt(v_t) + tau)
```

The paper defines `Delta_t = x_bar - w_global` and *ascends* with `+Delta`; using `g_t` and
*descending* is algebraically identical. Moments initialise to zero and accumulate, so a round's step
depends on the whole history — **the same aggregated input produces a different update at round 2
than at round 20.**

### `bias_correction` — off by default, and that is a real comparability trap

Kingma & Ba's `alpha_t = sqrt(1 - b2^(t+1)) / (1 - b1^(t+1))` scales the server learning rate.
Flower's `FedAdam` applies it; **Reddi et al.'s Algorithm 2 — which this class implements literally —
does not.** So it defaults to `False`, and (mirroring Flower's own asymmetry) it applies to the
`adam` variant only.

That factor is **not** a small correction. At `b1=0.9, b2=0.99` it is `0.74` at round 1, bottoms near
`0.47` around round 12, and is still only `0.93` at round 200 — roughly a **2× difference in
effective server step for the whole of any realistic run**. Turn it on to compare like-for-like
against a Flower `FedAdam` baseline; leaving it off while quoting a Flower `FedAdam` number as the
referent compares two optimisers running at ~2× different effective learning rates. This asymmetry
is also what makes this FedYogi cross-validate against Flower's at float32 epsilon while FedAdam
does not.

### Device migration

`_global` and the moments are migrated **once**, on the first round, to whatever device the
aggregated tensors landed on — rather than converting `x_bar` every round. It is idempotent (after
round 1 the devices agree and `.to()` is a no-op) and costs no per-round transfer.
`aggregate_fit` returns a **clone**, so a downstream mutation of the served model cannot corrupt
server state.

---

## FedLoRA — Adapter-Only Federation (shipped)

`FedLoRA` federates **only LoRA adapter parameters** — never the base model. Two aggregation modes:

| `aggregation` | Communicated | Note |
|---|---|---|
| `"FFA_LORA"` (default) | `B` + head | `A` is frozen and shared, so it is **not** aggregated. The strategy captures `A` from `initial_parameters` and re-attaches it to every aggregated global, unchanged — which is what makes `avg(B) @ A == avg(B @ A)` exact |
| `"FedIT"` | `A` + `B` + head | plain weighted mean over all adapter keys |

`FFA_LORA` **raises at construction** if `initial_parameters` carries no `lora_A` keys — the global
adapter must be the *full* adapter (A+B+head), or the frozen-A invariant has nothing to preserve.

### Two server-side guards

- **`_assert_homogeneous`** — every client must agree on the adapter key set *and* per-key shape
  (homogeneous rank/config), else `ValueError`.
- **`_assert_client_keys_allowed` (FR-23)** — a server-side **allowlist**: every client key must
  already exist in `initial_parameters`. Homogeneity alone only checks clients against *each other*,
  so a `min_clients=1` client (or a colluding full cohort) could append keys outside the adapter —
  poisoned base-model weights under their peft state-dict names — which would then be averaged into
  the global, broadcast to every peer, and packaged into the registry bundle. Clients may send a
  **subset** (FFA re-attaches the frozen A) but never a **superset**.

---

## Byzantine-Robust Aggregation (FR-12)

`RobustAggregator` (`server/robust_aggregation.py`, registered as `"robust"`) replaces FedAvg's
weighted mean with a robust estimator and hardens the ingress.

```python
RobustAggregator(initial_parameters, evaluate_fn=None, min_fit_clients=1, clients_per_round=None,
                 method="median",        # or "trimmed_mean"
                 trim_ratio=0.1,         # beta, in [0, 0.5)
                 clip_norm=None,         # L2 bound S on each client's DELTA; None disables
                 byzantine_fraction=0.0) # the operator's ESTIMATE
```

### The two estimators

Both from Yin et al. 2018, ["Byzantine-Robust Distributed
Learning"](https://arxiv.org/abs/1803.01498):

```python
def coordinate_wise_median(stacked):     # stacked: [num_clients, *param_shape]
    return torch.quantile(stacked.float(), 0.5, dim=0)

def trimmed_mean(stacked, trim_ratio):
    n = stacked.shape[0]
    k = int(trim_ratio * n)                       # floor
    if 2 * k >= n:
        raise ValueError(...)                     # trimming would remove every value
    ordered, _ = torch.sort(stacked.float(), dim=0)
    return ordered[k: n - k].mean(dim=0)
```

> **The median is `torch.quantile(..., 0.5)`, deliberately NOT `torch.median`.** For an even client
> count `torch.median` returns the *lower* of the two central order statistics, which is a biased
> estimator; the true median — the one Yin et al. analyse — is the mean of the two central order
> statistics, which is what `quantile` with linear interpolation gives. It still reduces to the
> middle element for an odd count.

### The five layers

1. **Coordinate-wise median** — tolerates up to (not including) half the clients being Byzantine.
2. **β-trimmed mean** — drops `k = floor(β·n)` from **each** end; tolerates up to `β`. `β = 0`
   recovers the plain mean.
3. **Non-finite rejection** — a client carrying NaN/Inf is dropped *before* aggregation, reusing the
   canonical `serializer._reject_non_finite` so this second layer shares one definition of "poisoned"
   with the wire path rather than re-implementing an `isfinite` test that could drift.
4. **Delta-space L2 clipping** — each client's *delta from the current global* is clipped to `S`
   using the same global-norm convention as `torch.nn.utils.clip_grad_norm_` (all tensors
   concatenated, so a sprawl-across-many-layers attack is bounded jointly), then the model is
   reconstructed as `global + clipped_delta`. Clipping the **delta**, not the raw model, is what
   bounds each client's per-round *pull*. Both estimators are translation-equivariant, so with
   clipping off this reduces exactly to the robust estimator over the raw client models.
5. **Byzantine-fraction guard** — if `byzantine_fraction > tolerance` (0.5 for median, `β` for
   trimmed-mean) the round **refuses to aggregate**, leaves the global untouched, sets
   `last_round_failed` / `last_round_message`, and returns `None`. The coordinator already treats
   `None` as a non-fatal round failure.

**FR-19** adds a sixth: `_conforms_to_global` drops any update whose key set or per-key shape
differs from the global model's. Without it, `torch.stack` raises on a shape mismatch (crashing the
aggregation thread *after* the client was accepted), and an empty or mis-keyed `clients[0]` templates
the reduction to a smaller key set — silently dropping those parameters and, once persisted, **wiping
them from the global model**. The reduction is templated on `self._global.keys()` for the same
reason.

### Aggregation is UNWEIGHTED, and that is deliberate

`num_examples` is validated (`> 0`, else the client is dropped) but never used as a weight. This
matches the robust-statistics literature: **an attacker controls its own reported `num_examples`**,
so a weighted median or trimmed-mean would hand the adversary back exactly the leverage these
estimators exist to remove.

### Honest scope

Median tolerates `< 1/2` Byzantine clients; trimmed-mean tolerates `<= β`. Both are **large-cohort**
defenses and degrade at the 1–3 client cohorts this platform often runs — which is precisely why the
estimator is opt-in per project rather than the default. The measured breakdown point is what
`benchmarks/robust_breakdown_point.py` exists to establish; see [08 — Benchmarks](08_examples.md).

---

## Central Differential Privacy (FR-13)

Client-level (user-level) `(ε, δ)`-DP against an honest-but-curious server, added at aggregation
time. **One client = one privacy unit.** Two modules, both dependency-free:

- `privacy/dp_mechanism.py` — `dp_aggregate()`, the mechanism
- `privacy/dp_accountant.py` — a from-scratch RDP accountant, **no opacus / tensorflow-privacy at
  runtime**

Currently reachable only through `FedLoRA(dp_enabled=True, …)`.

### The mechanism

Over the *aggregatable* keys only — adapter `B` + head, i.e. every client key that is **not** a
frozen `lora_A` key:

1. `delta_i = client[k] - global[k]`
2. Clip `delta_i` **jointly** to L2 norm `S` (the same `clip_l2_norm` the FR-12 path uses, so DP
   clipping and Byzantine clipping share one definition of "an update's norm"). One client's
   sensitivity to the summed delta is then exactly `S`.
3. **Uniform** average (weight `1/N`) — **not** num-examples-weighted.
4. Add `N(0, (z·S/N)²)` per coordinate. `z = 0` is the noiseless clip+average sanity path.
5. Return `{k: global[k] + mean_delta[k]}`; the caller re-attaches the frozen `A` **bit-identically**
   (zero noise on `A` keeps the FFA invariant exact).

> **Step 3 is a security property, not a simplification.** Weighting by an attacker-reported example
> count would inflate that client's sensitivity above `S` and void the ε claim. The DP path
> deliberately drops the weighting.

A non-finite coordinate is rejected loudly: it makes the L2 norm non-finite, so the clip becomes a
no-op (`min(1.0, S/NaN) == 1.0`) or produces NaN (`Inf*0`) — silently defeating the sensitivity bound
the whole guarantee rests on.

### The noise generator is isolated on purpose

```python
self._dp_generator = torch.Generator()
if self.dp_seed is not None:
    self._dp_generator.manual_seed(self.dp_seed)     # reproducible: tests / audits
else:
    self._dp_generator.seed()                        # FRESH OS entropy, independent of global RNG
```

This isolation is load-bearing for the guarantee. `fl_server.resolve_run_seed` calls
`torch.manual_seed(S)` for data/model-init reproducibility and **discloses `S`** on the eval card and
in the logs. If the DP noise were drawn from the global default generator, it would become a
deterministic function of that disclosed seed — an adversary holding the card could replay the run,
**strip the noise**, recover the un-noised client-level aggregate, and void DP entirely (a DA-3 ×
FR-13 interaction). The generator is also persisted across rounds, so advancing it never reuses
identical noise.

### The RDP accountant

Implements Mironov, Talwar & Zhang 2019, ["Rényi Differential Privacy of the Sampled Gaussian
Mechanism"](https://arxiv.org/abs/1908.10530) — the analysis Opacus and TF-Privacy implement — in
pure `math` + numpy (numpy only for array plumbing).

```python
DEFAULT_ORDERS                                    # Opacus' grid: 1.1..10.9, 12..63, 128, 256, 512
compute_rdp(q, noise_multiplier, steps, orders)   # -> list[float], already × steps
get_epsilon(rdp, delta, orders)                   # -> (epsilon, best_order)
required_noise_multiplier(target_eps, q, steps, delta)   # -> smallest z meeting the budget
RDPAccountant                                     # step() per round, get_privacy_spent(delta)
```

Implementation notes that matter if you compare against another library:

- **Integer vs fractional α take different paths.** Integer α uses the exact finite binomial sum in
  log space; fractional α uses the signed infinite series, with the fractional binomial carried as a
  running float (`C(α, i+1) = C(α, i)·(α−i)/(i+1)`) so gamma functions of non-positive arguments are
  never formed. Truncation is by log-magnitude (`-30.0`, matching Opacus) plus a hard 10,000-term
  cap Opacus does not have.
- **`log(erfc(x))` is stdlib-only and tail-accurate.** Below `x = 25` it defers to `math.erfc`;
  beyond that (where `erfc` underflows to 0) it uses the classical asymptotic expansion. This is what
  `scipy.special.log_ndtr` would provide, without the dependency.
- **The ε conversion is intentionally conservative.** `get_epsilon` uses the *classic* Mironov (2017)
  bound `ε = min_α [ rdp(α) + ln(1/δ)/(α − 1) ]`. Opacus uses the tighter Balle et al. (2020) bound
  and therefore reports a **smaller** ε for the same RDP curve. The per-order RDP here matches Opacus
  to ~1e-9; the reported ε never *under*-reports privacy loss.
- `q = 1` (no subsampling) short-circuits to the closed form `α / (2σ²)`, and `q = 1` is the
  conservative default when the enrolled population is unknown.
- `required_noise_multiplier` uses **geometric** bisection because `z` ranges over several orders of
  magnitude, and raises if the target is infeasible even at `z_max = 1e6`.

### Calibration in FedLoRA

Supply **exactly one** of `dp_noise_multiplier` (z directly) or `dp_target_epsilon` (solved via
`required_noise_multiplier`); supplying both, or neither, raises. `dp_target_epsilon` additionally
requires `dp_delta` and `dp_rounds`. The subsampling rate is
`q = clients_per_round / dp_num_clients` (or `1.0` when the population is unknown), and
`dp_accounted_epsilon` records the ε the accountant certifies for the chosen `z` — compare it against
the requested target.

### The honest negative result

The measured privacy–utility finding at laptop scale is a **collapse**, and it is documented rather
than tuned away. The utility SNR is `N / (z·√d)`; on the FedLoRA adapter (`d = 26112` aggregatable
coordinates, `√d ≈ 162`) with a small cohort, the per-coordinate signal `~S/√d` sits far below the
noise floor `z·S/N`. **The SNR is independent of the clip `S` — it cancels — so no clip tuning
helps.** A usable gradient needs SNR near 1: many more clients (`N ≈ √d`), subsampling
amplification, or a lower-dimensional adapter. `benchmarks/dp_epsilon_accuracy.py`,
`dp_snr_crossing.py`, `dp_on_head*.py` and `dp_subsampling_amplification.py` are the committed
harnesses that establish this; the accountant certifies the accounted ε back to the requested budget
exactly in all of them. See [08 — Benchmarks](08_examples.md#the-committed-benchmark-harnesses).

---

## Trainable-Subset Federation (DA-11)

Not a strategy — a set of contracts that let plain FedAvg federate only a model's **trainable**
subset over a shared **frozen backbone**. Three modules cooperate.

### `estimators/params.py` — the canonical layout (FR-14)

The single source of truth for how a model's parameters are enumerated and ordered:
`named_parameters()` order, filtered by `requires_grad`.

| Function | Returns |
|---|---|
| `param_layout(model)` | ordered `(name, shape, numel)` for the trainable params — the flat vector's manifest |
| `flat_params` / `set_flat_params` | flatten / unflatten in that order |
| `num_trainable(model)` | the flat vector's length `d` |
| `trainable_state(model)` | trainable params as an `OrderedDict` — **the correct `initial_parameters` for a DeComFL server** |
| `frozen_state(model)` | the F32-only complement: frozen params, then all *float* buffers — the bytes a `BASE_REF` backbone blob carries |
| `federable_state(state)` | the float32 subset of any state — what can cross the wire |
| `non_federable_names(state)` | what `federable_state` withheld, so a run can **log** the exclusion instead of dropping silently |

> Passing a full `model.state_dict()` where `trainable_state()` belongs includes buffers and frozen
> params, so `d_server > d_client` and the shared-seed perturbation `z` silently misaligns. That is
> the failure `DeComFL.validate_participant_dim` / `DeComFLClient.assert_dim_matches` exist to catch.

### `server/subset_federation.py` — the fail-loud guard

`validate_subset_update(update, model)` raises `SubsetDimMismatch` unless the update matches the
model's trainable layout on **both** axes — key set *and* per-key shape.

**Key comparison is order-INSENSITIVE (set-based), deliberately.** The FedAvg subset path is entirely
by-name (the aggregator averages per key; `apply_trainable_subset` writes back via
`load_state_dict(strict=False)`), so order carries no safety signal — and it cannot be relied on
anyway: a small non-transformer head takes the **unary** upload path, and a protobuf
`map<string, Tensor>` iterates in an unspecified order. An order-sensitive check false-rejected every
legitimate head update on that path.

`guard_client_updates(payloads, model)` must run **per-client, before aggregation**. Validating the
*aggregated* output cannot catch a non-first client's bad payload.

> **A caveat in the source, flagged honestly:** `subset_federation.py`'s module docstring still
> explains that guard in terms of `FedAvgAggregator` deriving its key set from the *first* client and
> silently skipping keys later clients omit. That description predates **FR-18**, which changed the
> aggregator to template on the *union* of client keys with per-key totals (see
> [FedAvgAggregator](#fedavgaggregator--weighted-averaging) above). The guard is still correct and
> still worth running per-client — a wrong-shape update must not reach `torch.add` — but its stated
> rationale is out of date relative to the aggregator it describes.

### `backbone/distribution.py` — shipping the frozen backbone

```python
blob   = serialize_backbone(model)          # deterministic safetensors of frozen_state(model)
sha    = backbone_sha256(blob)              # the content address
path   = BackboneCache(dir).get_or_fetch(sha, fetch)   # fetch is an injected Callable[[], bytes]
model  = reconstruct_frozen_backbone(model, path.read_bytes())
```

Fail-loud throughout, which is the whole point:

- `get_or_fetch` **verifies the sha256 of the fetched bytes against the requested key** and raises
  `BackboneIntegrityError` rather than caching a mismatch.
- A cache file whose bytes no longer hash to its own name is treated as a **miss and re-fetched** —
  self-healing.
- Writes are **atomic** (temp file + `os.replace`), so a crash mid-write never leaves a half-written
  blob under its final content-addressed name.
- `reconstruct_frozen_backbone` requires the blob's key **set** to equal `frozen_state(model)`'s.
  An unexpected key (a blob carrying something the model does not declare frozen) or a missing key
  (a truncated blob) raises `BackboneKeyMismatch` rather than silently loading a partial backbone.
  It then re-freezes the loaded parameters, so afterwards the model's only trainable — hence only
  federated — subset is its head.

The fetch source is a `Callable[[], bytes]`, so this framework contract is independent of *how* the
bytes arrive.

---

## Adapter Bundles (DA-9)

`bundle/manifest.py` builds the content-addressed manifest for a specialised model — the unit of
delivery / serving / on-device training.

```python
adapter_to_safetensors(state_dict, metadata) -> bytes    # deterministic order, stored float32
safetensors_to_state_dict(blob) -> Dict[str, np.ndarray]
sha256_hex(data) -> str                                  # the content address
build_manifest(artifact_sha256=…, kind=…, recipe_key=…, base_model_ref=…, license_tag=…,
               lora=…, eval_card_ref=…, files=[…], provenance=…) -> dict
```

- `SCHEMA_VERSION = "1.0"`; the manifest validates against `adapter_bundle.schema.json`, committed
  **next to the module** rather than under `docs/` (which is gitignored).
- `kind` is `"LORA_ADAPTER"` or `"FULL_CHECKPOINT"`; anything else raises.
- A `LORA_ADAPTER` **must** name its `base_model_ref` and carry a `lora` config — the same invariant
  the registry enforces for the `ADAPTER_OF` edge.
- Adapter weights ship as **safetensors**, never `torch.save`/pickle. Full checkpoints stay in the
  imaging air-gap export format.
- `artifact_sha256` is the same content hash the registry row stores, which is what aligns a bundle
  with its provenance record across languages.

See `framework/src/fedlearn/bundle/BUNDLE_FORMAT.md` for the human-readable spec, including the
fixture-MVP boundary where real export is not yet wired into the mobile bundle path.

---

## Implementing a Custom Strategy

Check the [registry](#the-strategy-registry) first — FedProx, FedOpt, FedLoRA and Byzantine-robust
aggregation are already shipped, and the older versions of this page presenting them as "examples to
write yourself" were wrong. Write a new strategy only for an algorithm none of the six covers.

Any class inheriting `Strategy` and implementing the three abstract methods is valid. The skeleton
and the registration step live in
[09 — Adding a Custom Strategy](09_developer_guide.md#adding-a-custom-strategy). Four conventions to
follow so it composes with the rest of the framework:

1. **Accept the same constructor front-matter** — `initial_parameters`, `evaluate_fn`,
   `min_fit_clients`, `clients_per_round=None` (falling back to `min_fit_clients`). The coordinator
   reads `min_fit_clients` and `clients_per_round` off the strategy object, and `server.py` reads
   `initial_parameters`.
2. **Route incoming updates through `normalize_updates`** rather than unpacking tuples yourself, so
   your strategy accepts every wire shape the others do.
3. **Reuse `FedAvgAggregator`** if your algorithm's aggregation *is* the weighted mean (FedProx,
   FedOpt and FedLoRA all do). Reuse `clip_l2_norm` if you need a norm bound, so there is one
   definition of "an update's norm" across FR-12, FR-13 and SE-3.
4. **Return `None` from `evaluate()` when `evaluate_fn` is `None`**, and free client buffers
   (`params.clear()`) after folding them in.

If your strategy needs client-side hyperparameters, add `get_client_config()` returning a
`str -> str` dict — no coordinator or proto change is required.

---

## Strategy Lifecycle in the Server

```python
# server.py — full strategy usage trace

# 1. Strategy is instantiated by user code, or by name through the factory
strategy = FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_fn,
    min_fit_clients=2,
    clients_per_round=3,
)
# strategy = create_strategy("fedavg", initial_parameters=..., clients_per_round=3)

# 2. Coordinator is created, strategy is injected
coordinator = FLCoordinator(
    strategy=strategy,
    min_clients_for_aggregation=strategy.min_fit_clients,
    clients_per_round=strategy.clients_per_round,
)

# 3. Initial parameters are read off the strategy ATTRIBUTE
coordinator.set_initial_parameters(strategy.initial_parameters)
# initialize_parameters() returns the same object; the attribute is what start_server uses

# 4. Training loop
for round_num in range(1, config.num_rounds + 1):
    coordinator.start_round()
    coordinator.wait_for_round_to_complete()
    # ... inside coordinator, during this wait, on a gRPC thread (or on the main thread
    #     if the dropout deadline fired first):
    #   strategy.get_client_config()  — each time the global model is served
    #   strategy.aggregate_fit(round_num, results)
    #   strategy.evaluate(round_num, new_params)

    metrics = coordinator.get_latest_metrics()
    if metrics:                       # a round with no evaluate_fn appends NOTHING
        history.append((round_num, metrics))
```

> **`aggregate_fit` runs while the coordinator lock is held**, on whichever gRPC thread submitted the
> Nth update. Keep it non-blocking: no network calls, no user prompts. Heavy per-round evaluation
> belongs in `evaluate_fn`, which runs in the same critical section — so it is subject to the same
> rule.

---

## Strategy Selection Decision Tree

All six leaves below are shipped and selectable by name; none requires you to write a strategy.

```
Are you training LLMs / very large models, or on a device where
backprop memory is the binding constraint?
│
├── YES → "decomfl"  (06_decomfl.md)
│           - Communication: O(K×P) scalars per round, independent of d
│           - No backpropagation; P+1 forward passes per local step
│           - Mind the eta*sqrt(d) stability envelope — it is NOT dimension-transferable
│
└── NO  → is only an ADAPTER (LoRA) being trained?
            │
            ├── YES → "fedlora"        (+ dp_enabled=True for central DP, FR-13)
            │
            └── NO  → are some clients potentially malicious?
                        │
                        ├── YES → "robust"   (median / trimmed-mean + delta clipping)
                        │           Large-cohort defense; near a no-op at 1-3 clients
                        │
                        └── NO  → is the data non-IID with multiple local epochs?
                                    │
                                    ├── Client drift is the problem   → "fedprox"
                                    │     (watch lr*mu < 2; the strategy enforces it)
                                    ├── Server-side convergence speed → "fedopt"
                                    │     (adam | yogi; mind bias_correction when
                                    │      comparing against a Flower baseline)
                                    └── Neither                       → "fedavg"
```

Orthogonal to the choice above: **federate only a trainable subset over a frozen backbone**
([DA-11](#trainable-subset-federation-da-11)) — that composes with `fedavg`, `fedprox`, `fedopt` and
`robust` rather than replacing any of them.
