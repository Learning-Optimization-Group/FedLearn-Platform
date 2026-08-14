# 06 — DeComFL: Dimension-Free Federated Learning

## Table of Contents
- [Motivation and Background](#motivation-and-background)
- [The Core Idea: Zeroth-Order Gradient Estimation](#the-core-idea-zeroth-order-gradient-estimation)
- [Communication Comparison](#communication-comparison)
- [Algorithm Overview](#algorithm-overview)
  - [Algorithm 3: Server Protocol](#algorithm-3-server-protocol)
  - [Algorithm 4: Client Protocol](#algorithm-4-client-protocol)
- [Implementation Deep Dive](#implementation-deep-dive)
  - [Seed Generation and Sharing](#seed-generation-and-sharing)
  - [ZerothOrderEstimator](#zerothorderestimator)
  - [DeComFLClient.fit()](#decomflclientfit)
  - [DeComFL.aggregate_fit()](#decomflaggregate_fit)
  - [Model Rebuild for Missed Rounds](#model-rebuild-for-missed-rounds)
  - [History Pruning](#history-pruning)
- [The Learning-Rate Stability Envelope](#the-learning-rate-stability-envelope)
- [Dimension Agreement (FR-14 / MO-19)](#dimension-agreement-fr-14--mo-19)
- [gRPC Protocol for DeComFL](#grpc-protocol-for-decomfl)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Trade-offs and Limitations](#trade-offs-and-limitations)
- [Running a DeComFL Experiment](#running-a-decomfl-experiment)
- [Connection to the DeComFL Paper](#connection-to-the-decomfl-paper)

---

## Motivation and Background

Standard federated learning (FedAvg) requires each client to transmit a full copy of the model parameters after every round. For large language models:

| Model | Parameters | Size (float32) | Bandwidth per round per client |
|-------|-----------|----------------|-------------------------------|
| CNN (MNIST) | 1.2M | 4.8 MB | Negligible |
| GPT-2 Small | 117M | 468 MB | 468 MB upload + 468 MB download |
| OPT-125M | 125M | 500 MB | 500 MB upload + 500 MB download |
| LLaMA-7B | 7B | 28 GB | 28 GB upload + 28 GB download |

For edge devices (mobile phones, laptops, IoT), uploading 500 MB every training round is often impractical.

**DeComFL** (Dimension-Free Communication Federated Learning) solves this by transmitting only a small number of scalar values per round instead of the full parameter vector.

---

## The Core Idea: Zeroth-Order Gradient Estimation

Instead of computing gradients via backpropagation and transmitting the gradient vector (which has the same dimension `d` as the model), DeComFL uses **zeroth-order (ZO) gradient estimation**:

```
g = (f(x + μz; ξ) - f(x; ξ)) / μ
```

Where:
- `f(x; ξ)` = loss on mini-batch `ξ` with parameters `x`
- `z ~ N(0, I_d)` = random perturbation vector
- `μ` = small smoothing parameter (e.g., 0.001)
- `g` = **a single scalar** — the estimated directional gradient along `z`

This requires only **forward passes** — no backward pass — and produces a scalar `g` that fits in
8 bytes. Two forward passes per perturbation in the naive form; in practice `f(x; ξ)` is hoisted
above the perturbation loop, giving `P + 1` forward passes per local step rather than `2P`.

The update direction is then reconstructed as `g × z`. The key insight is that if both server and client know the seed used to generate `z`, they can both regenerate `z` independently — so only the scalar `g` needs to be transmitted.

---

## Communication Comparison

| Method | Upload per round | Download per round | Notes |
|--------|-----------------|-------------------|-------|
| FedAvg | d × 4 bytes | d × 4 bytes | d = number of parameters |
| DeComFL | K × P × 8 bytes | K × P × (4+8) bytes | K=local steps, P=perturbations |

For OPT-125M (d=125M) with K=5 local steps and P=10 perturbations:

| | FedAvg | DeComFL |
|--|--------|---------|
| Upload | 500 MB | 400 bytes (5×10×8B) |
| Download | 500 MB | ~1.6 KB (seeds + scalars) |
| Reduction | — | **~1.25 million× smaller** |

The communication is truly **O(K×P)** — independent of model dimension `d`.

---

## Algorithm Overview

### Algorithm 3: Server Protocol

```
Inputs: T (rounds), K (local steps), P (perturbations), η (lr), μ (smoothing)
Initialize: x_0 (model), seed_history = [], gradient_history = []

For round t = 1, ..., T:
  1. Generate seeds S^t[k][p] for k=1..K, p=1..P
  2. Store seeds: seed_history.append(S^t)
  3. Send S^t and rebuild_history to each client
  4. Receive gradient scalars G^t_i[k][p] from each client i
  5. Compute average: G^t[k][p] = mean_i(G^t_i[k][p])
  6. Store averages: gradient_history.append(G^t)
  7. For k = 1..K:
       Δ = (1/(N×P)) × Σ_i Σ_p G^t_i[k][p] × z(S^t[k][p])
       x_t = x_{t-1} - η × Δ   (Δ already averaged by 1/(N×P); matches the client's (η/P)×δ)
  8. Broadcast updated seed_history + gradient_history for rebuild
```

### Algorithm 4: Client Protocol

```
Inputs: local model x (maintained across rounds), seeds S^t from server

Procedure per round:
  0. (Once, at startup) Download the server's global model — every party shares x_0 (FR-1)
  1. (If missed rounds) Rebuild model from seed+gradient history, SKIPPING any round
     already <= _synced_through (FR-15 idempotency)
  2. For k = 1..K:
     a. Sample mini-batch ξ
     b. base = f(x; ξ)                ← evaluated ONCE per step, reused for all P
     c. For p = 1..P:
          z = randn(seed=S^t[k][p])   ← regenerated on CPU, NOT transmitted
          g = (f(x + μz; ξ) - base) / μ
          δ += g × z
     d. x_local = x_local - (η/P) × δ
  3. Revert x_local to pre-round state (server will advance it)
  4. Submit gradient_scalars G[k][p] = {g for each k, p}
```

> Step 0 is why DeComFL is dimension-free **per round** rather than end-to-end: the initial sync is
> O(d) by construction.

---

## Implementation Deep Dive

### Seed Generation and Sharing

Seeds are generated by the server **once per round** and shared by every client in that round.

```python
# decomfl_strategy.py
def generate_seeds(self, round_idx: int) -> List[List[int]]:
    """seeds[k][p] = seed for local step k, perturbation p"""
    return [[int(self._seed_rng.integers(0, 2 ** 31 - 1)) for _ in range(self.P)]
            for _ in range(self.K)]

def get_or_create_seeds(self, round_idx: int) -> List[List[int]]:
    """Idempotent + thread-safe. THE entry point the servicer must call."""
    with self._seed_lock:
        seeds = self.seed_history.get(round_idx)
        if seeds is None:
            seeds = self.generate_seeds(round_idx)
            self.seed_history[round_idx] = seeds
        return seeds
```

```python
# grpc_servicer.py — GetDeComFLConfig handler
seeds = strategy.get_or_create_seeds(current_round)   # NOT generate_seeds + append
```

Three details here are corrections to earlier, broken behaviour — all of them worth preserving:

- **`get_or_create_seeds`, never `generate_seeds`, from the servicer.** Seeds were previously
  regenerated and list-appended on *every client's* RPC, which handed each client a **different
  perturbation direction** — breaking DeComFL's shared-seed invariant outright — and corrupted
  `seed_history` indexing. It is idempotent and guarded by `self._seed_lock` because concurrent
  client RPCs race here.
- **`seed_history` and `gradient_history` are `Dict[int, …]` keyed by round number** (1-based,
  matching `coordinator.current_round`), not lists. The old list-append produced N entries per round
  and off-by-one indexing.
- **The seed RNG is a private `np.random.default_rng(seed)`**, not `np.random.seed` /
  `torch.manual_seed`. The old global seeding corrupted reproducibility for anything else sharing
  the interpreter. No global torch seeding is needed either, since `canonical_perturbation` uses its
  own local CPU generator.

Clients receive seeds via `GetDeComFLConfig` RPC:

```python
# GetDeComFLConfigResponse
{
    current_round: 3,
    current_seeds: PerturbationSeeds {
        local_steps: [
            LocalStepSeeds { seeds: [1234567, 8901234, ...] },   # k=0  (int64 on the wire)
            LocalStepSeeds { seeds: [5678901, 2345678, ...] },   # k=1
            ...  # K steps total
        ]
    },
    rebuild_history: RebuildHistory { rounds: [...] },
    config: {                       # ALL FIVE keys are sent
        learning_rate:      "0.001",   # strategy.eta
        smoothing_param:    "0.001",   # strategy.mu  — SERVER-AUTHORITATIVE (FR-10)
        num_local_steps:    "5",       # strategy.K
        num_perturbations:  "10",      # strategy.P
        model_dim:          "1026",    # strategy.model_dim — the fail-loud handshake (FR-14)
    },
    torch_version:        "2.12.0",  # advisory; the mobile RandnEngine is version-independent
    grad_estimate_method: "forward", # "forward" | "central"
    golden_vector_sha256: "",        # empty => the client skips the RNG-parity check
}
```

> `smoothing_param` is not decorative: `DeComFLClient.fit()` **overwrites its own `mu`** with the
> server's value when present. A mismatched `mu` would make the client's gradient scalars
> derivatives of a *different* smoothed function than the one the server reconstructs.

### ZerothOrderEstimator

The estimator provides two core primitives:

**Perturbation generation** — deterministic from seed:
```python
def generate_perturbation(self, seed: int, num_params: int) -> torch.Tensor:
    generator = torch.Generator(device=self.device)
    generator.manual_seed(seed)
    # N(0, I_d) sample — same device, same generator, same output
    return torch.randn(num_params, generator=generator, device=self.device)
```

The canonical, device-independent source of truth for this `z` is `canonical_perturbation(seed, num_params)` in `framework/src/fedlearn/estimators/perturbation.py`:

```python
def canonical_perturbation(seed, num_params, dtype=torch.float32) -> torch.Tensor:
    if num_params <= 0:
        raise ValueError(...)
    generator = torch.Generator(device="cpu")     # LOCAL generator, never the process-global RNG
    generator.manual_seed(int(seed))
    return torch.randn(num_params, generator=generator, dtype=dtype, device="cpu")
```

Always CPU, always a local generator, `dtype` **fixed at float32 for parity** — do not pass a model's
dtype, that would break the golden-vector contract. The result is bit-stable across CPU/CUDA/MPS for
a given seed; callers move it to their compute device at the use site (`.to(device)`). Both the
server (`decomfl_strategy._generate_perturbation`) and the client
(`estimators/zeroth_order.generate_perturbation`) are thin delegators to it.

The native mobile core reproduces this contract in **`mobile_client/shared/include/fedlearn/RandnEngine.h`**
(header-only), with **`mobile_client/shared/tests/randn_parity_test.cpp`** as the release gate
(`ASSERT_NEAR(..., 1e-6f)`).

> **Two stale paths to be aware of.** `perturbation.py`'s own module docstring still cites
> `mobile_client/shared/src/Perturbation.cpp` and `rng_parity_test.cpp`; neither file exists. The
> real files are the two named above. The *contract* the docstring describes is correct — only the
> filenames drifted.

#### Cross-Architecture Determinism (golden-vector parity)

The Python↔C++ RNG contract is pinned by a frozen golden-vector fixture under `framework/tests/fixtures/decomfl_golden/` (raw little-endian float32 vectors + a `manifest.json` of seeds, lengths, and SHA-256s), regenerated by `generate.py` from `canonical_perturbation`. The parity test `framework/tests/test_perturbation.py` is **architecture-aware**:

- `torch.randn`'s CPU kernel is **bit-reproducible on the same CPU architecture**, but only **~1-ULP reproducible across architectures** — its vectorized Box–Muller transcendentals differ by a last bit between x86-64 and Apple-Silicon arm64.
- The fixtures are frozen on **one arch**, recorded as `manifest.platform_machine` (currently `x86_64`, the CI runner). So the test asserts:
  - **Bit-exact** (`assert_array_equal`) when `platform.machine() == manifest.platform_machine` — the strongest guarantee and the **CI gate**.
  - **ULP-tolerance** (`assert_allclose`, `atol=2e-6`, `rtol=0`) on any other arch (e.g. Apple Silicon) — the empirical x86↔arm64 spread peaks at ~1.4e-6, so 2e-6 gives margin. Real RNG drift is still caught bit-exact on the freeze arch.

This mirrors the C++ mobile release gate `mobile_client/shared/tests/randn_parity_test.cpp`
(`ASSERT_NEAR(..., 1e-6f)`). A companion check, `test_torch_version_matches_manifest`, asserts
`torch.__version__.split("+")[0] == manifest["torch_version"]` so an unintentional `torch` bump
cannot silently change the contract — re-freezing the fixture must be deliberate, with the new
`torch_version` recorded in the manifest so the C++ parity gate re-validates. **This is why
`framework/requirements.txt` pins `torch==2.12.0` rather than a range.**

The manifest currently records `torch_version: 2.12.0`, `numpy_version: 2.1.2`,
`platform_machine: x86_64`, and four frozen cases — `(seed, num_params)` of `(0, 16)`, `(1, 100)`,
`(1234567, 1000)` and `(2147483646, 4096)` — each with a raw little-endian float32 file, a sha256,
and its first eight values.

**Gradient scalar computation** — `P + 1` forward passes per local step, not `2P`:

```python
def compute_base_loss(self, model, flat_params, inputs, targets) -> float:
    """f(x; xi) ONCE per local step — x and the batch are fixed here, only z varies."""
    model.eval()
    with torch.no_grad():
        self._set_flat_params(model, flat_params)
        return self._evaluate_loss(model, inputs, targets).item()

def compute_gradient_scalar(self, model, flat_params, perturbation, inputs, targets,
                            base_loss=None) -> float:
    model.eval()
    with torch.no_grad():
        loss_x = base_loss if base_loss is not None else (
            self._set_flat_params(model, flat_params)
            or self._evaluate_loss(model, inputs, targets).item())

        self._set_flat_params(model, flat_params + self.mu * perturbation)
        loss_x_perturbed = self._evaluate_loss(model, inputs, targets).item()

        g = (loss_x_perturbed - loss_x) / self.mu
    return g                                   # already a Python float

def _evaluate_loss(self, model, inputs, targets):
    """One forward pass at the model's CURRENT params. Caller owns eval()/no_grad()."""
    if isinstance(inputs, dict):                    # LLM: unpack as kwargs
        return model(**inputs, labels=targets).loss
    return self.criterion(model(inputs), targets)   # CNN/MLP: CrossEntropyLoss
```

Hoisting `f(x; ξ)` above the perturbation loop matches the authors' reference implementation, which
computes `pert_minus_loss` once for the forward-difference method. The scalar is **bit-identical**
either way, because the base loss is deterministic under `eval()` + `no_grad()` — which is exactly
the precondition. Pass `base_loss=None` for any model whose forward is stochastic at inference, or
the cached value is wrong.

**Flat parameter helpers** — thin, back-compatible delegators to the FR-14 canonical manifest
(`estimators/params.py`), so the client, the estimator and the mobile ExecuTorch export all share
**one** `requires_grad`-filtered `named_parameters()` order:

```python
@staticmethod
def _get_flat_params(model): return params.flat_params(model)
@staticmethod
def _set_flat_params(model, flat): params.set_flat_params(model, flat)
@staticmethod
def get_num_params(model): return params.num_trainable(model)
```

Note the filter: `named_parameters()` **with `requires_grad`**, in that order. A mismatch between
server and client here silently misaligns `z` and the model diverges with no error — which is what
the dimension handshake below exists to catch.

### DeComFLClient.fit()

The full training loop with annotations matching the algorithm:

```python
def fit(self, parameters, config):
    seeds = config.get('seeds', [])  # List[List[int]] — KxP
    K = len(seeds)
    P = len(seeds[0]) if K else 0
    eta = float(config.get('learning_rate', 0.001))

    # FR-10: mu is SERVER-AUTHORITATIVE — the scalars must estimate the SAME smoothed
    # function the server reconstructs. Keep our default only if the server omits it.
    if config.get('smoothing_param') is not None:
        self.zo_estimator.mu = float(config['smoothing_param'])

    # Track cumulative update for clean revert at end
    total_perturbation = torch.zeros_like(self.x_current)
    gradient_scalars = []   # Will be KxP list of floats
    data_iter = iter(self.train_loader)

    # Algorithm 4, Line 14: For each local step k = 1..K
    for k in range(K):
        # FR-10: honour a server-driven stop BETWEEN local steps. total_perturbation only
        # accumulates APPLIED steps, so the revert below stays exact for a partial run —
        # and the caller must NOT submit the partial (non-KxP) grid.
        if self.grpc_client is not None and self.grpc_client.should_stop_training():
            break

        delta = torch.zeros_like(self.x_current)  # Σ g×z for this step
        k_gradient_scalars = []

        # Get data batch (cycling through loader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_loader)
            batch = next(data_iter)

        inputs, targets = batch
        # Handle both tensor inputs (CNN) and dict inputs (LLM)
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Progress update for heartbeat
        if self.grpc_client:
            self.grpc_client.update_status("training", k + 1, K)

        # f(x; xi) is the SAME for all P perturbations of this step — x and the batch are
        # both fixed here, only z varies. Evaluate ONCE: P+1 forward passes, not 2P.
        base_loss = self.zo_estimator.compute_base_loss(
            self.model, self.x_current, inputs, targets
        )

        # Algorithm 4, Line 16: For each perturbation p = 1..P
        for p in range(P):
            # Line 17: Generate z from shared seed (server has same z)
            z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))

            # Line 18: Compute ZO gradient scalar
            g = self.zo_estimator.compute_gradient_scalar(
                self.model, self.x_current, z, inputs, targets, base_loss=base_loss
            )
            k_gradient_scalars.append(g)  # this scalar is what we'll transmit

            # Line 19: Accumulate update direction
            delta += g * z

        # Line 21: Apply local step
        step_update = (eta / P) * delta
        self.x_current -= step_update
        total_perturbation -= step_update  # track cumulative for revert

        gradient_scalars.append(k_gradient_scalars)

    # CRITICAL: Revert to pre-round state
    # The server will advance the global model independently.
    # The client syncs via rebuild_model() at the start of the next round.
    self.x_current -= total_perturbation
    self.zo_estimator._set_flat_params(self.model, self.x_current)

    # Algorithm 4, Line 24: Return scalar gradients (KxP floats, not d-dimensional)
    return gradient_scalars, len(self.train_loader.dataset)
```

> **Memory note:** The critical insight is that `z` (shape `[d]`) is generated and immediately used to compute `g` (a scalar) and update `delta`. The `z` tensor is then released by Python's garbage collector. No KxP tensors of size d are ever held in memory simultaneously.

### DeComFL.aggregate_fit()

The server aggregates gradient scalars and reconstructs the global model update:

```python
# decomfl_strategy.py
def aggregate_fit(self, server_round, results):
    """
    Args:
        results: List of (client_id, gradient_scalars[K][P], num_examples)
    """
    if not results:
        return None

    # 1. Organize scalars by client
    client_gradients = {}
    for client_id, grad_scalars, num_examples in results:
        client_gradients[client_id] = grad_scalars
        # FR-2: synced THROUGH server_round - 1, not server_round. See below.
        self.client_last_round[client_id] = server_round - 1

    x_current = self.global_params_flat.clone()
    num_clients = len(client_gradients)

    # 2. Reconstruct the update direction
    for k in range(self.K):
        delta = torch.zeros_like(x_current)
        for p in range(self.P):
            # z depends only on (k, p), NOT on the client — regenerate it ONCE and sum the
            # gradient scalars across clients. O(K*P) perturbations, not O(K*P*N).
            z = self._generate_perturbation(self.seed_history[server_round][k][p])
            g_sum = sum(grad_scalars[k][p] for grad_scalars in client_gradients.values())
            delta += g_sum * z

        # Average across clients AND perturbations, then apply. There is NO extra "* self.P"
        # here: the 1/P averaging IS the paper's update. The v1 code cancelled it and stepped
        # the global model P times too far, off the rebuild trajectory (B1-C1).
        delta = delta / (num_clients * self.P)
        x_current = x_current - self.eta * delta

    self.global_params_flat = x_current
    self._prune_history(server_round)
    return self._unflatten_params(x_current, self.initial_parameters)
```

**The `z`-once restructuring is not just a speedup.** `z` is a `d`-dimensional tensor and depends
only on `(k, p)`. Generating it inside the per-client loop cost `K·P·N` perturbations per round
instead of `K·P`; summing the scalars first is algebraically identical (`Σ_i g_i·z = (Σ_i g_i)·z`) and
is what makes a large cohort tractable.

**Why `seed_history` is indexed by `server_round`:** `get_or_create_seeds(t)` records round `t`'s
K×P seeds under key `t` the first time any client asks for them. When `aggregate_fit(t, …)` runs it
looks up `seed_history[t]` to regenerate exactly those perturbations. Because it is a **dict keyed by
round** (not a list appended per client), the index is unambiguous and aligns with
`gradient_history` and `get_rebuild_history`.

**Why `client_last_round[client_id] = server_round - 1` (FR-2)** — this is subtle and was a real bug.
`fit()` reverts all K local steps, so after participating in round `r` the client's `x_current` still
reflects `x_{r-1}`, the model it *started* the round from. It applies round `r`'s **averaged** update
only at the start of round `r+1`, via `get_rebuild_history`, which returns
`range(last_round + 1, r + 1)`. Recording `server_round` made that range empty (the guard is
`last_round >= current_round - 1`) and pinned a fully-participating client at `x_0` **forever**.
Recording `server_round - 1` makes it replay round `r` exactly once.

### Model Rebuild for Missed Rounds

If a client disconnects and reconnects, it must reconstruct the global model updates for all missed rounds:

```python
# decomfl_client.py
def rebuild_model(self, rebuild_history, learning_rate):
    """
    Replays missed rounds deterministically. IDEMPOTENT via the _synced_through watermark.

    rebuild_history = [
        { 'round_number': 5, 'seeds': [[...], ...], 'gradients': [[...], ...] },
        { 'round_number': 6, 'seeds': [[...], ...], 'gradients': [[...], ...] },
    ]
    """
    for round_data in rebuild_history:
        round_num = round_data['round_number']

        # FR-15: skip any round already folded into x_current.
        if round_num <= self._synced_through:
            continue

        seeds = round_data['seeds']              # K×P ints from the server
        avg_gradients = round_data['gradients']  # K×P floats (server-averaged, 1/N)
        K = len(seeds)
        P = len(seeds[0]) if K else 0

        for k in range(K):
            delta = torch.zeros_like(self.x_current)
            for p in range(P):
                z = self.zo_estimator.generate_perturbation(seeds[k][p], len(self.x_current))
                delta += avg_gradients[k][p] * z
            self.x_current = self.x_current - (learning_rate / P) * delta

        self._synced_through = round_num         # FR-15: record it

    self.zo_estimator._set_flat_params(self.model, self.x_current)
```

The rebuild requires only the seeds (`K × P` int64s on the wire) and the averaged gradient scalars
(`K × P` float64s). Both are stored on the server and sent via `RebuildHistory`. The `d`-dimensional
perturbation vectors are **never stored or sent** — they are regenerated from seeds on demand.

### Two guards on opposite sides of the same failure

| Guard | Side | Prevents |
|---|---|---|
| `_synced_through` watermark (FR-15/FR-16) | client | **double-applying** a round the client already folded in. The server advances a client's baseline only on aggregation, while the client mutates `x_current` at config-fetch time — so a dropped submission gets a round re-handed. A restart is the same hazard: a client reusing its deterministic `client_id` re-downloads `x_{r-1}`, and without the watermark set at download time it would reset to `-1` while the server still remembers its pre-crash baseline. |
| `DeComFLRebuildGap` (FR-4) | server | **under-applying** — handing back a torn chain. If the server lacks the seeds *and/or* averaged gradients for any round in the catch-up range `last_round+1 .. current_round-1`, it raises rather than silently dropping that round, which would let the client rebuild on an incomplete update chain and diverge with no error surfaced anywhere. |

`get_rebuild_history` also handles first contact correctly: a client with no recorded baseline has
*just downloaded the current global model*, so it is already synced through `current_round - 1`, and
that is what gets recorded. A **late joiner** therefore does not replay pre-join rounds on top of
the model it just downloaded (FR-1). For a client present from round 1 this is `0`, correctly
yielding an empty history.

### History Pruning

`seed_history` and `gradient_history` would otherwise grow as O(rounds) — 20,000-round runs are
routine here — which is a poor look for an algorithm whose entire claim is O(1) communication.

```python
def _prune_history(self, server_round):
    if not self.client_last_round:
        return                                  # nobody has synced yet; nothing is provably dead
    floor = min(self.client_last_round.values())
    # every round at or below the floor is unreachable by get_rebuild_history
    ...
```

The floor is derived from **what clients actually need**, not a fixed window, so a round a lagging
client still requires is never discarded — which is what keeps pruning from turning a legitimate
rejoin into a `DeComFLRebuildGap`.

**The cost, stated out loud rather than hidden:** one stalled client pins the floor and blocks all
pruning. That is correct behaviour, but it is worth knowing, so it *warns*: every
`HISTORY_PIN_WARN_LAG = 16` rounds of lag, the strategy logs which client(s) are pinning the floor,
how far behind they are, and how many rounds of history are being retained for them.

---

## The Learning-Rate Stability Envelope

**DeComFL's learning rate is not dimension-transferable, and the failure is silent.** This is the
single most important operational fact on this page.

The zeroth-order estimate has `||ĝ|| ~ sqrt(d/P) · ||∇f||`, so it is systematically *longer* than the
true gradient as `d` grows. An `eta` tuned at one dimension overshoots at a larger one and the run
diverges. The invariant held constant along the scaling law is

```
S = eta * sqrt(d)
```

`decomfl_strategy.py` encodes the measured envelope as module constants:

| Constant | Value | Meaning |
|---|---|---|
| `LR_REFERENCE_D` | 1026 | the shipped frozen head, `Linear(512 -> 2)` |
| `LR_REFERENCE_ETA` | 0.01 | measured stable there |
| `LR_MEASURED_STABLE_D` | 20602 | largest `d` that converged at the reference eta |
| `LR_MEASURED_DIVERGENT_D` | 30902 | smallest `d` that diverged at the reference eta |
| `LR_STABLE_MAX_S` | ≈ 1.435 | `0.01 * sqrt(20602)` |
| `LR_DIVERGENT_MIN_S` | ≈ 1.758 | `0.01 * sqrt(30902)` |

```python
def lr_stability_statistic(eta, d): return eta * math.sqrt(d)
def suggested_eta(d):               return LR_REFERENCE_ETA * math.sqrt(LR_REFERENCE_D / d)
```

`_validate_learning_rate()` runs at **construction** and is deliberately **three-tiered** rather than
a single threshold — below the largest measured-stable `S` it says nothing; above the smallest
measured-*divergent* `S` it refuses; and the band between them is **unmeasured**, so it warns instead
of pretending to know:

| `S = eta·√d` | Behaviour |
|---|---|
| `S <= 1.435` | silent |
| `1.435 < S < 1.758` | **warn** — unmeasured band; may diverge. Logs `suggested_eta(d)` |
| `S >= 1.758` | **raise `ValueError`**, unless `allow_unstable_lr=True` (then warn: "expect the run to learn and then explode") |

**Why an error and not a warning.** The measured record (`research/results/decomfl/`) shows the same
divergence hit twice, and shows *why* an accuracy curve cannot catch it:

- `mu_eta_dimension_scaling.json` — at `d = 103,002` the reference eta diverges to loss ~1e19.
  Scaling eta by `sqrt(d0/d)` **alone** restores it (0.9815 AUC). Scaling `mu` alone does not.
- `stability_ladder.json` — at the reference eta: stable through `d = 20,602`, diverged from
  `d = 30,902`. The `d = 30,902` cell **reaches 0.9805 AUC and then explodes to loss 9.2e18**, so the
  accuracy column cannot see it coming; only the shared-seed replay check caught it.

`suggested_eta(d)` reproduces the value that rescued the diverged `d = 103,002` cell exactly
(`0.0009980466738393954`).

---

## Dimension Agreement (FR-14 / MO-19)

The server and every client must agree on `d` — the length of the flat vector the shared-seed `z`
spans. If they do not, `z` misaligns and the model diverges with **no error**. Both sides check.

```python
# server: decomfl_strategy.py
@property
def model_dim(self) -> int:            # len(self.global_params_flat)
def validate_participant_dim(self, client_flat_dim, client_id=""): ...   # raises ValueError

# client: decomfl_client.py
def assert_dim_matches(self, server_model_dim): ...                       # raises ValueError
```

The server advertises `model_dim` in the `GetDeComFLConfig` config map; `start_decomfl_client` calls
`assert_dim_matches` on every poll and treats a mismatch as **fatal, not transient** — it exits with
`OUTCOME_ERROR` rather than retrying a condition that can never clear.

**The cause is almost always the same mistake**, and both error messages say so: a full
`model.state_dict()` was passed as the server's `initial_parameters` where
`estimators.params.trainable_state(model)` belongs. A `state_dict` includes buffers
(`running_mean` / `running_var` / …) and frozen params (a LoRA base, a partial fine-tune) that the
client's `requires_grad` flatten omits — so `d_server > d_client`.

`DeComFL._flatten_params` documents that contract explicitly; it is not defensive paranoia but the
one invariant that makes the whole protocol work.

---

## gRPC Protocol for DeComFL

DeComFL uses two dedicated RPCs that bypass the standard model parameter path:

### Round Protocol Flow

```
Client                              Server
  │                                   │
  │  GetDeComFLConfig(client_id)       │
  │──────────────────────────────────►│
  │                                   │ generate_seeds(round)
  │                                   │ seed_history.append(seeds)
  │                                   │ get_rebuild_history(client_id, round)
  │  GetDeComFLConfigResponse          │
  │◄──────────────────────────────────│
  │  { round, seeds, rebuild_history, config }
  │                                   │
  │ (if rebuild_history not empty)    │
  │   rebuild_model(history, lr)      │
  │                                   │
  │ fit(config) → grad_scalars[K][P]  │
  │                                   │
  │  SubmitGradientScalars(scalars)   │
  │──────────────────────────────────►│
  │                                   │ submit_decomfl_update(client_id, scalars, n, round)
  │                                   │ (when all clients submitted):
  │                                   │   aggregate_fit()
  │                                   │   round_complete_event.set()
  │  SubmitGradientScalarsResponse    │
  │◄──────────────────────────────────│
  │                                   │
  │  (sleep 5s, poll again)           │
  │──────────────────────────────────►│
```

### Wire Format for Gradient Scalars

```proto
// Upload: K×P float64 scalars (~400 bytes for K=5, P=10)
message SubmitGradientScalarsRequest {
  string          client_id        = 1;
  string          run_id           = 2;
  int32           trained_on_round = 3;
  GradientScalars gradients        = 4;   // K×P doubles
  int64           num_examples     = 5;   // collected; aggregation is UNWEIGHTED
  PerturbationSeeds perturbation_seeds = 6; // the client's ECHO of the server-issued seeds
}

message GradientScalars   { repeated LocalStepGradients local_steps = 1; }
message LocalStepGradients { repeated double scalars = 1; }  // P scalars for this local step
```

> **The seed echo is advisory here.** The server reconstructs `z` from its own `seed_history`
> (`get_or_create_seeds`) and never re-derives from the echo — the servicer only logs the echoed step
> count, as observability and a hook for a future integrity cross-check. The field exists for a
> FedAvg ZO-SGD variant in which the *client* generates the seeds, so the server would genuinely
> need them.

### Wire Format for Seed Download

```proto
// Download: K×P int64 seeds + rebuild history + the determinism contract
message GetDeComFLConfigResponse {
  int32              current_round   = 1;
  PerturbationSeeds  current_seeds   = 2;
  RebuildHistory     rebuild_history = 3;
  map<string,string> config          = 4;  // learning_rate, smoothing_param,
                                           // num_local_steps, num_perturbations, model_dim
  string torch_version        = 5;
  string grad_estimate_method = 6;   // "forward" | "central"
  string golden_vector_sha256 = 7;   // empty => the client skips the RNG-parity check
}

message PerturbationSeeds { repeated LocalStepSeeds local_steps = 1; }
message LocalStepSeeds    { repeated int64 seeds = 1; }   // int64, matching the C++ int64_t
```

### Server-Side Ingress Validation

Before a submission is stored, the coordinator applies three DeComFL-specific checks (see
[03 — DeComFL Path](03_server_internals.md#decomfl-path-in-the-coordinator)):

| Check | On failure |
|---|---|
| The grid is exactly `strategy.K × strategy.P` | raises `MalformedDeComFLSubmission` → `INVALID_ARGUMENT`. Without it a malformed grid reaches `aggregate_fit`'s `grad_scalars[k][p]` and crashes the aggregation thread *after* the client was ACKed |
| FR-5 dedup — this `client_id` already submitted this round | ignored (first accepted wins); a duplicate would double-count in the average |
| Non-finite scalars (SE-3) | the whole update is dropped |
| Finite-but-huge scalars | **clamped** to `±grad_clip_threshold` (default `1e3`) at ingress, before storage, so `aggregate_fit` and `_calculate_average_gradients` read identical values |

---

## Hyperparameter Guide

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|--------------|--------|
| `num_local_steps` | K | 1–10 | More steps = more computation per round, less communication overhead |
| `num_perturbations` | P | 5–50 | More perturbations = better gradient estimation, more computation |
| `learning_rate` | η | **dimension-dependent** | NOT transferable across `d`. Constrained by `eta*sqrt(d)`; see [the stability envelope](#the-learning-rate-stability-envelope) and use `suggested_eta(d)` as the starting point |
| `smoothing_param` | μ | 1e-4 to 1e-2 | Smaller = more accurate ZO estimate but noisier; 0.001 is a good default. **Server-authoritative** — the client adopts the server's value |
| `seed` | — | Any int | Seeds the strategy's *private* `np.random.default_rng` for seed generation. It does **not** touch the process-global NumPy/torch RNG |
| `allow_unstable_lr` | — | `False` | Downgrades the measured-divergent LR check from an error to a warning. Set deliberately only |

### Communication vs. Convergence Trade-off

```
Increasing P (perturbations):
  ✓ Better gradient estimate (lower variance)
  ✗ More computation (2P forward passes per local step)
  = Same communication (K×P scalars)

Increasing K (local steps):
  ✓ More model update per round (faster convergence)
  ✗ More computation (K data batches)
  = Same communication (K×P scalars)

Decreasing μ (smoothing):
  ✓ More accurate gradient direction
  ✗ Noisier estimates (finite-difference approximation error)
  = No effect on communication
```

---

## Trade-offs and Limitations

### Advantages
- **Massive communication reduction:** O(K×P) vs. O(d) — ~1M× for LLMs
- **Backprop-free:** Can work on devices/APIs where gradients are unavailable (black-box models)
- **Memory-efficient:** No gradient tensors stored during training
- **Natural compatibility with model rebuilding:** Missed rounds can be replayed without full model transmission

### Limitations
1. **Convergence rate:** ZO methods converge ~1/d× slower than gradient-based methods in the worst case. In practice, the gap is much smaller.
2. **Forward-pass cost:** `K × (P + 1)` forward passes per round — the base loss is hoisted per local
   step — versus 1 forward + 1 backward per batch for FedAvg. Still no backward pass at all.
3. **High variance with small P:** If P is too small (P < 5), gradient estimates are noisy and training can be unstable.
4. **Smoothing bias:** The ZO estimator introduces a bias term proportional to μ². Smaller μ reduces bias but increases variance.
5. **The learning rate does not transfer across model dimension**, and the failure is silent — a run
   learns first and then explodes. This is the most likely way to lose a DeComFL experiment; the
   constructor now refuses the measured-divergent regime. See
   [the stability envelope](#the-learning-rate-stability-envelope).
6. **History growth is bounded but pinnable.** `_prune_history` drops every round at or below
   `min(client_last_round)`, so history is not O(rounds) in general — but a single stalled client
   pins the floor and blocks all pruning. The strategy warns every 16 rounds of lag, naming the
   pinning clients.
7. **The initial download is O(d).** Per-round communication is O(K×P), but every client must first
   adopt the server's `x_0` (FR-1) — a one-shot full-model download the paper assumes. DeComFL is
   dimension-free *per round*, not end-to-end.
8. **Aggregation is unweighted.** `num_examples` is collected but not used, so a client with 10× the
   data has the same influence as one with 10 samples.

---

## Running a DeComFL Experiment

### ECG Classification Example

There are two ECG DeComFL examples and they are **not** interchangeable — check which one you are in
before copying a command:

- **`examples/ecg_decomfl_multiclient/`** is a *single-process simulation*: `run_server.py` drives its
  own in-file `DeComFLServer`/`DeComFLClient` classes with no gRPC and **no argparse at all** —
  everything (K, P, η, μ, rounds, client count) is set in `config.py`. Its `run_client.py` is a class
  module with no `__main__`, so it is imported, never executed. `ecg_decomfl_central` has the same
  shape.
- **`examples/ecg_decomfl_framework_integration/`** is the one that actually runs the framework over
  gRPC (`fl.server.start_server` + `start_decomfl_client`), and is the one to copy.

```bash
# Terminal 1: server (no CLI flags — edit examples/ecg_decomfl_framework_integration/config.py
# for NUM_ROUNDS / NUM_LOCAL_STEPS / NUM_PERTURBATIONS / LEARNING_RATE / SMOOTHING_PARAM
# and SERVER_ADDRESS, which defaults to localhost:50051)
cd examples/ecg_decomfl_framework_integration
python run_server.py

# Terminal 2-4: clients — only two flags exist
python run_client.py --client-id client_0
python run_client.py --client-id client_1 --server localhost:50051
python run_client.py --client-id client_2 --server localhost:50051
```

```bash
# The single-process simulation, for contrast — no flags, config.py is the only knob
cd examples/ecg_decomfl_multiclient && python run_server.py
```

### Minimal Custom DeComFL Setup

```python
# server side
import fedlearn as fl
from fedlearn.estimators.params import trainable_state, num_trainable
from fedlearn.server.decomfl_strategy import suggested_eta

model = MyModel()

strategy = fl.DeComFL(
    # CRITICAL (FR-14): trainable_state(), NOT model.state_dict(). A full state_dict includes
    # buffers and frozen params, inflating the server's flat vector past the client's and
    # silently misaligning the shared-seed perturbation z.
    initial_parameters=trainable_state(model),
    evaluate_fn=my_eval_fn,
    min_fit_clients=2,
    clients_per_round=3,
    num_local_steps=5,                            # K
    num_perturbations=10,                         # P
    learning_rate=suggested_eta(num_trainable(model)),   # η — dimension-scaled, NOT a constant
    smoothing_param=0.001,                        # μ
)

fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)

# client side
from fedlearn import DeComFLClient
from fedlearn.client.decomfl_start import start_decomfl_client

class MyDeComFLClient(DeComFLClient):
    pass  # DeComFLClient.fit() is already fully implemented

client = MyDeComFLClient(
    model=MyModel(),
    train_loader=my_loader,
    smoothing_param=0.001,   # overwritten by the server's smoothing_param when it sends one
    device="mps",            # Apple Silicon
)

outcome = start_decomfl_client(              # returns a terminal outcome string
    server_address="localhost:50051",
    client=client,
    client_id="client_0",
)
# OUTCOME_COMPLETED | OUTCOME_DISCONNECTED | OUTCOME_ERROR — map it to an exit code
```

`start_decomfl_client` performs the FR-1 initial global-model download itself, so the client's
constructor-time random init is discarded before any training happens. It also calls
`assert_dim_matches` against the server's advertised `model_dim` on every poll and treats a mismatch
as fatal.

---

## Connection to the DeComFL Paper

The implementation maps directly to the algorithms in the DeComFL paper:

| Paper Reference | Code Location |
|----------------|---------------|
| Algorithm 3, Server protocol | `decomfl_strategy.py` `aggregate_fit()` + `get_or_create_seeds()` |
| Algorithm 4, Client protocol | `decomfl_client.py` `fit()` |
| Algorithm 2, Model rebuilding | `decomfl_client.py` `rebuild_model()` + `decomfl_strategy.get_rebuild_history()` |
| ZO estimator (Eq. 1) | `estimators/zeroth_order.py` `compute_gradient_scalar()` |
| The canonical `z` RNG (cross-language contract) | `estimators/perturbation.py` `canonical_perturbation()` |
| Seed history `S^t` | `decomfl_strategy.py` `self.seed_history` — `Dict[round, seeds]` |
| Gradient history `G^t` | `decomfl_strategy.py` `self.gradient_history` — written by `coordinator._trigger_decomfl_aggregation_and_evaluation` |

### Where the implementation goes beyond the paper

These are this platform's additions, not the paper's, and they are worth separating out:

| Addition | Why |
|---|---|
| `_validate_learning_rate` + `suggested_eta` | the paper does not give an operational stability envelope; this one is **measured** on this codebase |
| `_prune_history` | the paper's O(1) claim is per-round communication, not server memory |
| `DeComFLRebuildGap` (FR-4) and the `_synced_through` watermark (FR-15/16) | making dropout and restart safe rather than silently divergent |
| `model_dim` handshake (FR-14) | catching the flat-layout mismatch that silently misaligns `z` |
| The scalar clamp + non-finite reject (SE-3) | the paper assumes honest clients |
| The frozen golden-vector RNG fixture | cross-language (Python ↔ C++) parity is a deployment requirement here, not a paper concern |

### Citation

```bibtex
@inproceedings{li2025decomfl,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Dong, Chaosheng and Yang, Haibo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

[arXiv:2405.15861](https://arxiv.org/abs/2405.15861). Developed at Rochester Institute of Technology
under Professor Haibo Yang. Reference implementation:
[ZidongLiu/DeComFL](https://github.com/ZidongLiu/DeComFL) (Apache-2.0) — that attribution and
license are retained in `decomfl_strategy.py` and `zeroth_order.py` per Apache-2.0 section 4.
