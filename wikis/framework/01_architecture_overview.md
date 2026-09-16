# 01 — Architecture & Package Overview

## Table of Contents
- [What Is the Framework?](#what-is-the-framework)
- [Technology Stack](#technology-stack)
- [Package Layout](#package-layout)
- [Module Inventory — All 57 Modules](#module-inventory--all-57-modules)
- [Module Dependency Graph](#module-dependency-graph)
- [How the Components Fit Together](#how-the-components-fit-together)
- [Federated Learning Lifecycle at a Glance](#federated-learning-lifecycle-at-a-glance)
- [The In-Process Simulator](#the-in-process-simulator)
- [Integration with the Rest of the Platform](#integration-with-the-rest-of-the-platform)
- [Public API Surface](#public-api-surface)

---

## What Is the Framework?

The `framework/` directory contains `fedlearn` — a self-contained Python library that implements the core federated learning logic. It is the component that actually performs model training, parameter aggregation, and client-server communication. Everything else in the platform (the Spring Boot backend, the React frontend, the Electron desktop app) ultimately delegates to this library.

The library is intentionally decoupled from the platform orchestration layer. You can run it standalone on a laptop with three terminals, inside Docker containers, as processes spawned by the Java backend on a deployed VM, or **entirely inside one Python process** through `fedlearn.simulation` — the library itself doesn't know or care.

---

## Technology Stack

| Layer | Technology | Version / Notes |
|-------|------------|-----------------|
| Language | Python | `python_requires='>=3.10'` in `setup.py` (uses `list[X]` built-in generics); CI tests 3.12 |
| Deep Learning | PyTorch | Pinned to `torch==2.12.0` — it is the version the DeComFL golden fixtures and the ExecuTorch toolchain were built against, and `tests/test_perturbation.py::test_torch_version_matches_manifest` gates on it |
| Communication | gRPC + Protobuf | `grpcio>=1.75.1`, `protobuf>=5.29.0,<6.0.0` |
| Wire format | safetensors (own codec) | `communication/safetensors_codec.py` — deterministic, float32-only, no pickle |
| Data Science | NumPy | `numpy==2.1.2`. Array manipulation, Dirichlet sampling, `SeedSequence` streams |
| LLM Support | HuggingFace Transformers | `transformers==4.55.2`, `peft>=0.11` (LoRA adapters) |
| Security | PyJWT | `PyJWT>=2.8,<3` — connection-token verification (SE-1); `cryptography>=46.0.6,<47.0` |
| Compression | lz4 (optional) | Not in `requirements.txt`; imported in a `try/except`. Activated by `FEDLEARN_USE_COMPRESSION=1` |
| Packaging | setuptools | `setup.py` + `pyproject.toml`. `setup.py` filters `torch*`/`torchvision*` out of `install_requires` |

The library has **zero dependency on the Spring Boot backend**. Its only external communication is gRPC between the Python server and Python clients.

> **No Flower — not the semantics, and no longer the package.** The FL framework is entirely
> custom: its own protobuf contract (`fedlearn.v2`) and its own FedAvg / FedProx / FedOpt /
> FedLoRA / DeComFL / robust strategies. The Java-side supervisor was renamed from the historical
> `flower` names to `orchestration` / `FlServerManager` (DA-12).
>
> **`flwr` and `flwr-datasets` are now gone platform-wide** (commit `65048b6`), not merely absent
> from `framework/requirements.txt`. They survived only to build one CIFAR-10 IID shard; that shard
> is now produced natively in `fl-runtime/recipes.py` (`CNN_SHUFFLE_SEED = 42`, which is the
> `flwr` `FederatedDataset` default it replaced) straight from `datasets`, and the replacement was
> verified **byte-identical per partition** by `research/benchmarks/verify_flwr_shard_equivalence.py`.
>
> Two dependency caps fell with it, and both are now **cleared**, not residual:
> - **`cryptography`** — `flwr` 1.20.0 capped it at `<45.0.0`, which made this framework's own
>   `>=46.0.6` security floor unreachable elsewhere in the platform (the SE-22 residual).
>   `backend/fl-platform-api/requirements.txt:16` now pins `cryptography==46.0.7`, at the floor.
> - **`protobuf`** — the same package capped it at `<5.0.0`, which silently made the FoT path
>   *uninstallable* in the backend and Docker lockfiles (`fot_pb2.py` is gencode 5.29.0, and
>   protobuf requires runtime ≥ gencode). The backend now pins `protobuf==5.29.5`, matching the
>   framework's `>=5.29.0,<6.0.0`.
>
> `fl-runtime/tests/test_requirements_security_floors.py::test_flwr_stays_out_of_the_lockfiles` is
> the regression guard: re-adding `flwr` re-imposes both caps at once.
>
> `transformers` *is* pinned in the framework — it backs the HuggingFace model loaders and the LoRA
> recipes. (FoT itself is torch-free by design.)

---

## Package Layout

```
framework/
├── src/
│   └── fedlearn/                   ← Installable Python package (57 modules)
│       ├── __init__.py             ← Public API (re-exports key symbols)
│       │
│       ├── server/                 ← Server-side logic
│       │   ├── __init__.py         ← Re-exports incl. FedLoRA, RobustAggregator, STRATEGY_REGISTRY
│       │   ├── server.py           ← start_server() entry point + JSON logging + TLS/auth wiring
│       │   ├── coordinator.py      ← FLCoordinator (round state, dropout deadline, ingress defenses)
│       │   ├── strategy.py         ← Strategy ABC + FedAvg / FedLoRA / FedProx / FedOpt + FedAvgAggregator
│       │   ├── strategy_factory.py ← create_strategy() + STRATEGY_REGISTRY (six names)
│       │   ├── decomfl_strategy.py ← DeComFL strategy (Algorithm 3) + LR stability envelope
│       │   ├── robust_aggregation.py ← Byzantine-robust aggregators + clip_l2_norm (FR-12)
│       │   ├── subset_federation.py← Trainable-subset guards for frozen-backbone FedAvg (DA-11)
│       │   ├── _update_normalize.py← The one client-update wire-shape normaliser, shared by 4 paths
│       │   └── grpc_servicer.py    ← FederatedLearningServiceServicer (RPC handlers, SE-18 caps)
│       │
│       ├── client/                 ← Client-side logic
│       │   ├── __init__.py
│       │   ├── client.py           ← Client ABC + start_client() entry point
│       │   ├── grpc_client.py      ← GrpcClient (transport layer, dual channel, retry)
│       │   ├── local_trainer.py    ← LocalTrainer (the shipped first-order FedAvg/FedProx/FedOpt client)
│       │   ├── decomfl_client.py   ← DeComFLClient (Algorithm 4 + Algorithm 2 rebuild)
│       │   └── decomfl_start.py    ← start_decomfl_client() entry point + terminal outcomes
│       │
│       ├── communication/          ← Protocol layer
│       │   ├── __init__.py
│       │   ├── serializer.py       ← Proto ↔ PyTorch tensor conversion + safetensors chunking
│       │   ├── safetensors_codec.py← Deterministic float32 safetensors wire codec (+ decode hardening)
│       │   ├── protos/             ← BYTE-IDENTICAL MIRRORS of proto/ — never hand-edit
│       │   │   ├── fedlearn.proto  ← Mirror of proto/fedlearn/v2/fedlearn.proto (fedlearn.v2)
│       │   │   └── fot.proto       ← Mirror of proto/fedlearn/fot/v1/fot.proto (fedlearn.fot.v1)
│       │   └── generated/          ← Code generated by protoc (committed)
│       │       ├── fedlearn_pb2.py / fedlearn_pb2.pyi / fedlearn_pb2_grpc.py
│       │       └── fot_pb2.py / fot_pb2_grpc.py
│       │
│       ├── estimators/             ← Gradient estimation + the trainable-parameter manifest
│       │   ├── __init__.py
│       │   ├── params.py           ← FR-14 canonical layout: param_layout / flat_params /
│       │   │                          trainable_state / frozen_state / federable_state
│       │   ├── perturbation.py     ← canonical_perturbation() — the frozen Python↔C++ RNG contract
│       │   └── zeroth_order.py     ← ZerothOrderEstimator (DeComFL forward-difference)
│       │
│       ├── simulation/             ← In-process federation (P0-1) — no gRPC, no ports, no subprocesses
│       │   ├── __init__.py
│       │   ├── federation.py       ← SimulatedFederation / SimulationResult / RoundRecord
│       │   ├── partition.py        ← iid / dirichlet / shard / pathological + partition_report
│       │   └── rng.py              ← ClientRng / RunRng / torch_rng_scope (per-client isolation)
│       │
│       ├── privacy/                ← Central DP (FR-13)
│       │   ├── __init__.py
│       │   ├── dp_mechanism.py     ← dp_aggregate(): clip → uniform average → Gaussian
│       │   └── dp_accountant.py    ← From-scratch RDP accountant for the Sampled Gaussian Mechanism
│       │
│       ├── security/               ← FL-boundary security
│       │   ├── __init__.py
│       │   ├── tls.py              ← SE-2 fail-closed TLS policy
│       │   ├── token_verify.py     ← SE-1 connection-token verification (PyJWT, HMAC family)
│       │   ├── identity.py         ← SE-15 partition extraction from RPC metadata
│       │   ├── interceptor.py      ← Server-side ConnectionTokenInterceptor (+ FR-7 run binding)
│       │   └── client_interceptor.py ← maybe_wrap_channel() — attaches the token outbound
│       │
│       ├── backbone/               ← DA-11 frozen-backbone distribution
│       │   ├── __init__.py
│       │   └── distribution.py     ← serialize_backbone / BackboneCache / reconstruct_frozen_backbone
│       │
│       ├── bundle/                 ← Adapter-bundle manifest (DA-9)
│       │   ├── __init__.py
│       │   ├── manifest.py         ← build_manifest / adapter_to_safetensors / sha256_hex
│       │   ├── adapter_bundle.schema.json
│       │   └── BUNDLE_FORMAT.md
│       │
│       ├── fot/                    ← Federation over Text (separate, torch-free research mode)
│       │   └── agent.py backend.py distiller.py fot_server.py fot_servicer.py model.py
│       │       provenance.py redaction.py round.py trace_guard.py
│       │
│       └── data/                   ← Raw MNIST IDX files downloaded by the examples
│           └── MNIST/raw/          ← NO Python code lives here (see simulation/partition.py)
│
├── tests/                          ← ~100 pytest modules + tests/fixtures/ (golden vectors)
│
├── examples/                       ← End-to-end runnable examples
│   ├── simple_federation/          ← MNIST + CNN (FedAvg)
│   ├── llm_federation/             ← OPT-125M (FedAvg)
│   ├── ecg_federation/             ← ECG Transformer (FedAvg)
│   ├── ecg_decomfl_central/        ← ECG + DeComFL (centralised baseline)
│   ├── ecg_decomfl_multiclient/    ← ECG + DeComFL (multi-client)
│   ├── ecg_decomfl_framework_integration/
│   └── fot_text_federation/        ← FoT offline demo (run_fot.py)
│
├── benchmarks/                     ← 17 committed, seeded experiment harnesses + the shared
│                                      wire_bytes.py accounting module (see 08).
│                                      results/ is generated, NOT tracked
│
├── requirements.txt                ← Python dependencies
├── setup.py                        ← Package metadata + install configuration
├── pyproject.toml                  ← Minimal build-system config
├── pytest.ini                      ← -m "not slow" + --cov=fedlearn --cov-fail-under=73
├── run_full_test_suite.py          ← Comprehensive e2e test runner
├── run_local_test.py               ← Quick local sanity test
└── run_platform_e2e_test.py        ← Integration test (talks to Spring backend)
```

> **The `protos/` directory is a mirror, not a source.** The canonical contract lives at
> `proto/fedlearn/v2/fedlearn.proto` and `proto/fedlearn/fot/v1/fot.proto`, governed by `buf`.
> `scripts/check_proto_mirror.sh` diff-gates three in-tree mirrors (the two above plus
> `mobile_client/proto/fedlearn/v2/fedlearn.proto`) and fails CI on drift. See
> [02 — gRPC Communication](02_grpc_communication.md#regenerating-generated-code).

---

## Module Inventory — All 57 Modules

Every `.py` file under `src/fedlearn/`, and where it is documented. `__init__.py` files are
counted but only listed where they carry re-exports worth knowing.

| Package | Modules | Documented in |
|---|---|---|
| *(root)* | `__init__.py` | [Public API Surface](#public-api-surface) |
| `server/` (10) | `__init__.py`, `server.py`, `coordinator.py`, `grpc_servicer.py` | [03 — Server Internals](03_server_internals.md) |
| | `strategy.py`, `strategy_factory.py`, `robust_aggregation.py`, `_update_normalize.py` | [05 — Strategies](05_strategies.md) |
| | `decomfl_strategy.py` | [06 — DeComFL](06_decomfl.md) |
| | `subset_federation.py` | [05 — Strategies](05_strategies.md#trainable-subset-federation-da-11) |
| `client/` (6) | `__init__.py`, `client.py`, `grpc_client.py`, `local_trainer.py` | [04 — Client Internals](04_client_internals.md) |
| | `decomfl_client.py`, `decomfl_start.py` | [04](04_client_internals.md#decomfl-client) + [06](06_decomfl.md) |
| `communication/` (8) | `serializer.py`, `safetensors_codec.py`, `__init__.py` | [02 — gRPC Communication](02_grpc_communication.md) |
| | `generated/` (5: `__init__`, `fedlearn_pb2`, `fedlearn_pb2_grpc`, `fot_pb2`, `fot_pb2_grpc`) | [02](02_grpc_communication.md#regenerating-generated-code) |
| `estimators/` (4) | `__init__.py`, `zeroth_order.py`, `perturbation.py` | [06 — DeComFL](06_decomfl.md) |
| | `params.py` | [05 — Strategies](05_strategies.md#trainable-subset-federation-da-11) |
| `simulation/` (4) | `__init__.py`, `federation.py`, `rng.py` | [The In-Process Simulator](#the-in-process-simulator) |
| | `partition.py` | [07 — Data Partitioning](07_data_partitioning.md) |
| `privacy/` (3) | `__init__.py`, `dp_mechanism.py`, `dp_accountant.py` | [05 — Strategies](05_strategies.md#central-differential-privacy-fr-13) |
| `security/` (6) | `__init__.py`, `tls.py`, `token_verify.py`, `identity.py`, `interceptor.py`, `client_interceptor.py` | [02](02_grpc_communication.md#tls-configuration) + [03](03_server_internals.md#security-wiring-in-start_server) |
| `backbone/` (2) | `__init__.py`, `distribution.py` | [05 — Strategies](05_strategies.md#trainable-subset-federation-da-11) |
| `bundle/` (2) | `__init__.py`, `manifest.py` | [05 — Strategies](05_strategies.md#adapter-bundles-da-9) |
| `fot/` (11) | `__init__.py`, `agent.py`, `backend.py`, `distiller.py`, `fot_server.py`, `fot_servicer.py`, `model.py`, `provenance.py`, `redaction.py`, `round.py`, `trace_guard.py` | Not covered by this wiki — see the FoT caveats below |

**Coverage: 46 of 57 modules are documented in these nine pages.** The eleven `fot/` modules are
deliberately out of scope here: Federation over Text is an *additive, orthogonal* research path that
shares no code with the gradient path (it is torch-free by design so it cannot perturb gradient
correctness), and — critically — **no LLM has ever run through it**. `fot/backend.get_backend()`
wires only a `DeterministicStubBackend`; the `local-http` / `vllm` / `ollama` options raise
`BackendError`. Its tests assert plumbing, not semantics, and there is no reproduced FoT result
anywhere in this repo. Treat it as scaffolding, not a validated capability.

---

## Module Dependency Graph

```
                         ┌──────────────────────────────┐
                         │   user script / example /     │
                         │   run_server.py               │
                         └───────────────┬──────────────┘
                                         │ calls
                                         ▼
                         ┌──────────────────────────────┐
                         │    server.start_server()      │
                         │    server.py                  │
                         └──────┬──────────────┬─────────┘
                                │              │ creates
                         creates│              ▼
                                │   ┌──────────────────────┐
                                │   │   FLCoordinator       │
                                │   │   coordinator.py      │◄──────────────┐
                                │   └──────────────────────┘               │
                         ┌──────▼──────────────────────┐                   │
                         │ FederatedLearningServicer    │                   │
                         │ grpc_servicer.py             │──── delegates ────┘
                         └──────────────────────────────┘
                                        ▲
                            gRPC/HTTP2  │
                         ┌──────────────┴──────────────┐
                         │      GrpcClient              │
                         │      client/grpc_client.py   │
                         └──────────────┬───────────────┘
                                        │ used by
                                        ▼
                         ┌──────────────────────────────┐
                         │  client.start_client()       │
                         │  client.py                   │
                         └──────────────────────────────┘
                                        ▲
                                        │ abstract
                         ┌──────────────┴──────────────┐
                         │     User Client Subclass     │
                         │     (e.g. MNISTClient)       │
                         └──────────────────────────────┘

Side chains:
  strategy.py ──► FedAvgAggregator ──► _update_normalize.normalize_updates
  strategy.py (FedLoRA, DP on) ──► privacy.dp_mechanism ──► robust_aggregation.clip_l2_norm
  decomfl_strategy.py ──► estimators.perturbation.canonical_perturbation
  zeroth_order.py ──► estimators.perturbation.canonical_perturbation + estimators.params
  serializer.py ──► communication/safetensors_codec.py
  serializer.py ──► communication/generated/fedlearn_pb2.py

Second driver (no gRPC at all):
  simulation/federation.SimulatedFederation ──► FLCoordinator (direct method calls)
                                            └─► Strategy (the same production objects)
```

Note the shape of that last chain: the simulator does **not** re-implement anything. `FLCoordinator`
turned out to be transport-free — `register_client`, `get_global_model_for_client`,
`submit_client_update` and `start_round` are ordinary methods and gRPC lives entirely in the
servicer that wraps them — so the simulator is a driver loop over the production objects.

---

## How the Components Fit Together

### The Server Side

1. **`server.py`** — the single public entry point. It configures JSON logging (at the entry point, *not* at import time — FR-9), creates a `FLCoordinator`, builds the gRPC server with the optional SE-1 auth interceptor, applies the SE-2 TLS policy, and runs the outer training loop.
2. **`FLCoordinator`** — the brain of the server. It tracks round state, client registrations, heartbeats and update submissions; enforces the ingress defenses (dedup, non-finite rejection, shape checking, optional delta clipping); triggers aggregation when all expected clients have submitted; and **resolves the round anyway** if the per-round dropout deadline elapses first.
3. **`FederatedLearningServiceServicer`** — the gRPC I/O layer. Each incoming RPC call is a thin dispatcher that validates the request and calls a method on `FLCoordinator`. It also owns the SE-18 streamed-upload resource caps and the SE-15 identity binding.
4. **`Strategy`** — pluggable aggregation and evaluation logic. The coordinator delegates `aggregate_fit()` and `evaluate()` to the active strategy, and — if the strategy exposes `get_client_config()` — ships that strategy's client-side hyperparameters down with the global model.

### The Client Side

1. **`client.py`** — the single public entry point for standard FL clients. `start_client()` wraps the entire training loop: register → fetch model → `client.fit()` → submit update → poll for next round, with a server-driven stop checked at every stage (FR-10).
2. **`GrpcClient`** — handles all network I/O. It manages **two** gRPC channels (training + heartbeat), the heartbeat thread, retry logic with exponential backoff, and automatically selects between unary and streaming RPC for model uploads.
3. **`Client` ABC** — the interface that application code must implement. Only two methods: `get_parameters()` and `fit()`.
4. **`LocalTrainer`** — the shipped concrete `Client` for the first-order family. Plain minibatch SGD, plus the FedProx proximal gradient when the server sends `proximal_mu > 0`.
5. **`DeComFLClient`** — a specialised subclass for the DeComFL protocol. Instead of returning updated model weights, it returns gradient scalars, making per-round communication O(K×P) rather than O(d).

---

## Federated Learning Lifecycle at a Glance

```
Server starts
    │
    ▼
coordinator.set_initial_parameters(strategy.initial_parameters)
    │   (start_server reads the attribute directly; initialize_parameters() returns the same object)
    │
    └─► For each round r = 1 … N:
            │
            ├── coordinator.start_round()   ← clears stale state + resets the dropout deadline
            │
            │   ┌──────────────────────────────────────────────┐
            │   │ Meanwhile, clients are polling               │
            │   │                                              │
            │   │  GrpcClient.get_global_model()               │
            │   │     → streams global model weights           │
            │   │                                              │
            │   │  client.fit(parameters, config)              │
            │   │     → local training (backprop or ZO)        │
            │   │                                              │
            │   │  GrpcClient.submit_update(params, n)         │
            │   │     → unary or streaming upload              │
            │   └──────────────────────────────────────────────┘
            │
            ├── coordinator.wait_for_round_to_complete()
            │       (blocks until clients_per_round updates arrive, OR until the
            │        round_timeout_s deadline elapses → resolve_round_incomplete())
            │
            ├── strategy.aggregate_fit(round, results)
            │       → new global model params  (or None → round marked failed)
            │
            ├── strategy.evaluate(round, params)
            │       → (loss, metrics) or None when no evaluate_fn is configured
            │
            └── coordinator.current_round += 1
                coordinator._round_complete_event.set()

coordinator.mark_training_complete()  ← TRAINING_COMPLETE + the -1 sentinel
sleep(FEDLEARN_COMPLETION_DRAIN_SECONDS, default 3)  ← let clients observe it
Server stops gRPC server (grace=5), returns (history, final_parameters)
```

---

## The In-Process Simulator

`fedlearn.simulation` (`simulation/federation.py`, `partition.py`, `rng.py`) runs a federation of
arbitrarily many clients **inside one Python process**: no gRPC channel, no TCP port, no subprocess.

### Why it exists

The deployed path reserves one TCP port per FL server from `50000-50010` (set in the backend's
`application.properties`), which caps the platform at **11 concurrent federations** and makes a
1,000-client experiment inexpressible. That is a deployment constraint leaking into the science —
essentially every FL result worth comparing against is quoted at client counts far above 11.

### It drives the production objects

`SimulatedFederation` constructs a real `FLCoordinator` and calls a real `Strategy`. A simulated run
therefore exercises the same aggregation, the same poisoning defenses, and the same round
bookkeeping a deployed run does; only the transport is elided — and even that can be put back.

```python
from collections import OrderedDict
import torch
from fedlearn.server.strategy import FedAvg
from fedlearn.simulation.federation import SimulatedFederation
from fedlearn.simulation.partition import dirichlet_partition, partition_report

parts = dirichlet_partition(labels, num_clients=1000, alpha=0.5, seed=7, min_partition_size=16)

def make_client(client_id, client_rng):
    # Called ONCE PER PARTICIPATION — close over already-partitioned indices (cheap),
    # do not re-read a dataset from disk here.
    return MyClient(model=build_model(), train_loader=loader_for(parts[client_id]))

sim = SimulatedFederation(
    strategy=FedAvg(initial_parameters=model.state_dict(), evaluate_fn=my_eval, clients_per_round=100),
    client_factory=make_client,
    num_clients=1000,
    clients_per_round=100,
    seed=7,
    client_config={"learning_rate": 0.1, "local_epochs": 1},
    wire_in_the_loop=0.0,     # fraction of updates routed through the real safetensors codec
    dropout_rate=0.0,         # modelled, not waited for
)
result = sim.run(num_rounds=50)
json.dump(result.to_json(), open("run.json", "w"))   # {"meta": {...}, "per_round": [...]}
```

### The three properties it is responsible for

1. **Determinism from `(seed, client_id, round)` alone.** `simulation/rng.py` folds the identity
   into a `numpy.random.SeedSequence` **entropy tuple** — `SeedSequence(entropy=(run_seed,
   client_id))` and `(run_seed, client_id, round)`. That is deliberately *not*
   `SeedSequence.spawn(n)`: spawning `n` children folds `n` into each child's entropy, so the same
   client would draw differently in a bigger cohort, and `clients_per_round=10` would not be
   comparable to `clients_per_round=100`. Round-scoped (rather than sequential) streams mean round 5
   is reproducible **without replaying rounds 1–4**, which is what makes a single anomalous round
   re-examinable. Torch's global RNG drives dropout, weight init and DataLoader shuffling, so
   `torch_rng_scope(seed)` saves and restores it (CPU **and** CUDA) around every seeded block —
   including the whole run, because a strategy's `evaluate_fn` is user code that also touches it.

2. **No wall-clock dependence.** A full cohort aggregates inline on the submit that completes it. A
   round with modelled dropout is resolved immediately via `FLCoordinator.resolve_round_incomplete`
   rather than by sleeping out the deployed server's 120-second deadline (P0-1c). Simulated time is
   never real time.

3. **The wire stays testable.** `wire_in_the_loop` (0.0–1.0) routes that fraction of client updates
   through the real deterministic safetensors encode/decode and accounts the bytes.
   `tests/test_simulation_federation.py::test_full_wire_matches_no_wire_bitwise` asserts that off
   and on agree bit-for-bit — which is the only reason running experiments with it off is
   defensible. Only floating-point tensors traverse the codec (it is float32-only by design);
   integer buffers such as BatchNorm's `num_batches_tracked` are passed through rather than
   silently coerced.

### Memory and the client factory

Clients are constructed per round and released (`del client` after each submit), so peak memory
scales with `clients_per_round`, not `num_clients`. The corollary is the one real constraint on
`client_factory`: it is called **once per participation**, so it must be cheap. Clients are assumed
stateless between rounds, which holds for the FedAvg/FedProx/FedOpt family — each round begins by
loading the global parameters.

### What a run records

`SimulationResult.to_json()` emits `{"meta": …, "per_round": […]}`. Each `RoundRecord` carries
`round`, `selected`, `reported`, `dropped`, **`forced`** (a round force-aggregated with a partial
cohort — kept distinct so a dropout study cannot read as a clean run), `num_examples`, `loss`,
`metrics`, `wire_clients`, `wire_bytes`, `wall_seconds`. `meta` carries the seed, cohort sizing,
strategy name, client config, wire/dropout settings, torch/numpy/python versions, platform, total
wall-clock, total wire bytes, and a `final_digest` — the sha256 **of the canonical safetensors
encoding** of the final parameters, deliberately the same function the wire uses so the digest is
comparable against a C++/mobile encoding of the same state rather than a Python-only hash.

### Measured scale

`research/benchmarks/simulation_scale.py` (seeded, re-runnable) sweeps `10 → 5000` clients at 10%
participation over 20 rounds × 3 seeds. All 21 cells ran; the record lives in
`research/results/simulation/scale_m4max.json`. Read it with its own caveat, which the meta block
states: it measures *simulator* scaling (driver overhead, coordinator throughput, memory per client)
on a deliberately tiny model, so a cell's wall-clock is a **floor** on what a real architecture would
cost, never an estimate of it. (`research/` is gitignored — it is a local working area, not a
backup.)

---

## Integration with the Rest of the Platform

The Python framework is invoked as a **child process** by several parts of the platform:

### 1. Spring Boot Backend

The `FlServerManager` Java service launches the FL server as a **local process** on the backend host, via the `FlServerProcessRunner` seam (DA-8) — it no longer calls `ProcessBuilder` directly. It shells out to `fl-runtime/run_fl_server.sh` (the path is resolved from the `python.script.fl-server.path` property; there is a `.bat` sibling for Windows), binds it to a free port in the `50000-50010` range, and tracks the resulting `ProcessHandle` in a `ConcurrentHashMap<UUID, ProcessHandle>` so the project can be stopped later. The process's merged stdout+stderr is streamed as log lines to the frontend via STOMP WebSocket.

```
Spring Boot (FlServerManager)
  └── FlServerProcessRunner → bash fl-runtime/run_fl_server.sh --project_id xyz --port 50000 ...
         └── fl-runtime/fl_server.py
                └── import fedlearn; fedlearn.server.start_server(...)
```

> **Local processes are the only supported deployed orchestration mode.** The hardened single-VM
> topology — FL servers as local Python processes next to the backend — is what the `production`
> profile describes. Managed-task orchestration is deferred to `OP-12`: an earlier AWS
> ECS/Fargate implementation existed but was removed along with the AWS SDK, and the leftover
> `ecs.cluster-name` property is now fail-closed. Setting it to a non-blank value makes the
> backend throw at boot (`FlOrchestrationModeValidator`, in **every** profile), and the
> corresponding `FlServerManager` branch throws `UnsupportedOperationException`. Leave it blank.

### 2. Electron Desktop App

The `fedlearn-desktop` Electron app spawns the **client only** — it does *not* run an FL server. `DockerService` (`fedlearn-desktop/src/main/docker.service.ts`) resolves the native client invocation for the current runtime and spawns it as a child process; the server it connects to is the one the Spring backend started.

```
Electron main process (docker.service.ts → resolveNativeInvocation)
  ├── packaged:  spawn(<resources>/<bundle>/fedlearn-client)   ← PyInstaller bundle, no system python
  └── dev:       spawn("python3 -u fl-runtime/client.py ...")  ← PYTHONPATH=framework/src
                    └── --project-id … --server-address … --partition-id …
```

On Jetson, the same service takes the Docker path instead of the native one.

### 3. Docker Containers

`client-docker/` contains a `Dockerfile` that installs the framework **and** `fl-runtime/` — its
build context is the **repo root**, not `client-docker/`. The container is configured entirely by
environment variables (`PROJECT_ID`, `SERVER_ADDRESS`, `PARTITION_ID`, and
`FEDLEARN_CONNECTION_TOKEN` when the server has client auth on); `entrypoint.sh` hard-fails if the
first three are unset and builds the CLI flags itself. The desktop app takes this path on Jetson.

---

## Public API Surface

The `fedlearn/__init__.py` exports the minimal surface that application code needs:

```python
import fedlearn as fl

# ─── Server ────────────────────────────────────────────────────────────────
fl.server.start_server(server_address, config, strategy)  # blocking call
fl.server.ServerConfig(num_rounds=10)

# ─── Strategies ────────────────────────────────────────────────────────────
fl.FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=my_eval_fn,
    min_fit_clients=2,
    clients_per_round=3,
)
fl.DeComFL(
    initial_parameters=...,
    num_local_steps=5,
    num_perturbations=10,
    learning_rate=0.001,
    smoothing_param=0.001,
)

# ─── Client (standard FL) ───────────────────────────────────────────────────
class MyClient(fl.Client):
    def get_parameters(self): ...
    def fit(self, parameters, config): ...

fl.client.start_client(server_address, client=MyClient(), client_id="c0")

# ─── Client (DeComFL) ──────────────────────────────────────────────────────
from fedlearn.client.decomfl_start import start_decomfl_client

class MyDeComFLClient(fl.DeComFLClient): ...
start_decomfl_client(server_address, client=MyDeComFLClient(...), client_id="c0")
```

### What each namespace actually exports

`fedlearn/__init__.py` is deliberately small — `server`, `client`, `Client`, `Strategy`, `FedAvg`,
`FedProx`, `FedOpt`, `DeComFLClient`, `DeComFL`, `LocalTrainer`, `create_strategy`.

`FedLoRA`, `RobustAggregator` and `STRATEGY_REGISTRY` are exported from **`fedlearn.server`**, not
from the top level:

```python
from fedlearn.server import FedLoRA, RobustAggregator, STRATEGY_REGISTRY, create_strategy
```

The remaining subsystems are imported by module path rather than re-exported:

```python
from fedlearn.simulation.federation import SimulatedFederation
from fedlearn.simulation.partition import dirichlet_partition, partition_report
from fedlearn.privacy.dp_accountant import RDPAccountant, compute_rdp, get_epsilon
from fedlearn.estimators.params import trainable_state, frozen_state, federable_state
from fedlearn.backbone.distribution import serialize_backbone, BackboneCache
from fedlearn.bundle.manifest import build_manifest, adapter_to_safetensors
```

> **Key design principle:** Users only implement two methods (`get_parameters` and `fit`) — and
> for the first-order family they need not even do that, since `LocalTrainer` is a ready-made
> `Client`. All network communication, retry logic, heartbeating, and round synchronisation is
> handled by the framework.
