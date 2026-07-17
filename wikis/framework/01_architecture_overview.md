# 01 — Architecture & Package Overview

## Table of Contents
- [What Is the Framework?](#what-is-the-framework)
- [Technology Stack](#technology-stack)
- [Package Layout](#package-layout)
- [Module Dependency Graph](#module-dependency-graph)
- [How the Components Fit Together](#how-the-components-fit-together)
- [Federated Learning Lifecycle at a Glance](#federated-learning-lifecycle-at-a-glance)
- [Integration with the Rest of the Platform](#integration-with-the-rest-of-the-platform)
- [Public API Surface](#public-api-surface)

---

## What Is the Framework?

The `framework/` directory contains `fedlearn` — a self-contained Python library that implements the core federated learning logic. It is the component that actually performs model training, parameter aggregation, and client-server communication. Everything else in the platform (the Spring Boot backend, the React frontend, the Electron desktop app) ultimately delegates to this library.

The library is intentionally decoupled from the platform orchestration layer. You can run it standalone on a laptop with three terminals, inside Docker containers, or as processes spawned by the Java backend on a deployed VM — the library itself doesn't know or care.

---

## Technology Stack

| Layer | Technology | Version / Notes |
|-------|------------|-----------------|
| Language | Python | 3.10+ (uses `list[X]` built-in generics) |
| Deep Learning | PyTorch | Pinned to `torch==2.12.0` (used for parameter tensors, autograd) — see the pin rationale at the top of `requirements.txt` |
| Communication | gRPC + Protobuf | `grpcio>=1.75.1`, `protobuf>=4.21.6,<5.0.0` |
| Wire format | safetensors (own codec) | `communication/safetensors_codec.py` — deterministic, float32-only, no pickle |
| Data Science | NumPy | Array manipulation, Dirichlet sampling |
| LLM Support | HuggingFace Transformers | OPT, GPT-2, etc. |
| Compression | lz4 (optional) | Activated by `FEDLEARN_USE_COMPRESSION=1` |
| Packaging | setuptools | `setup.py` + `pyproject.toml` |

The library has **zero dependency on the Spring Boot backend**. Its only external communication is gRPC between the Python server and Python clients.

> **No Flower FL semantics.** Despite the legacy `flower` package name on the Java side (renamed to `orchestration` / `FlServerManager` — DA-12), the FL framework is entirely custom — its own protobuf contract (`fedlearn.v2`) and its own FedAvg / DeComFL strategies. `framework/requirements.txt` has **no `flwr` / `flwr-datasets`** entry (they carried zero imports and were removed).
>
> Be precise about the scope of that claim, though: `flwr-datasets` *is* still a dependency **elsewhere in the platform** — `backend/fl-platform-api/requirements.txt` pins `flwr==1.20.0` + `flwr-datasets==0.5.0` and `client-docker/requirements.txt` pins `flwr-datasets>=0.3.0`, because `fl-runtime/client.py` and `fl-runtime/fl_server.py` use `flwr_datasets.FederatedDataset` **for dataset partitioning only** — never for FL server/client/strategy semantics. Known wart: that pin drags in `cryptography<45.0.0`, which sits below the framework's own `>=46.0.6` floor (the SE-22 residual, documented at `backend/fl-platform-api/requirements.txt:16`).
>
> `transformers` *is* pinned in the framework — it backs the FoT (Federation over Text) path and the HuggingFace model loaders.

---

## Package Layout

```
framework/
├── src/
│   └── fedlearn/                   ← Installable Python package
│       ├── __init__.py             ← Public API (re-exports key symbols)
│       │
│       ├── server/                 ← Server-side logic
│       │   ├── __init__.py
│       │   ├── server.py           ← start_server() entry point
│       │   ├── coordinator.py      ← FLCoordinator (round management)
│       │   ├── strategy.py         ← Strategy ABC + FedAvg / FedProx / FedOpt
│       │   ├── strategy_factory.py ← create_strategy() dispatch
│       │   ├── decomfl_strategy.py ← DeComFL strategy (Algorithm 3)
│       │   ├── robust_aggregation.py ← Byzantine-robust aggregators (FR-12)
│       │   ├── subset_federation.py← Subset / cohort selection
│       │   ├── _update_normalize.py← Update normalisation helpers
│       │   └── grpc_servicer.py    ← FederatedLearningServiceServicer (RPC handlers)
│       │
│       ├── client/                 ← Client-side logic
│       │   ├── __init__.py
│       │   ├── client.py           ← Client ABC + start_client() entry point
│       │   ├── grpc_client.py      ← GrpcClient (transport layer)
│       │   ├── local_trainer.py    ← LocalTrainer (first-order FedAvg/FedProx/FedOpt client)
│       │   ├── decomfl_client.py   ← DeComFLClient (Algorithm 4)
│       │   └── decomfl_start.py    ← start_decomfl_client() entry point
│       │
│       ├── communication/          ← Protocol layer
│       │   ├── __init__.py
│       │   ├── serializer.py       ← Proto ↔ PyTorch tensor conversion + chunking
│       │   ├── safetensors_codec.py← Deterministic float32 safetensors wire codec
│       │   ├── protos/
│       │   │   ├── fedlearn.proto  ← Source of truth for all RPC messages (fedlearn.v2)
│       │   │   └── fot.proto       ← FoT service contract (fedlearn.fot.v1)
│       │   └── generated/          ← Code generated by protoc (committed)
│       │       ├── fedlearn_pb2.py / fedlearn_pb2.pyi / fedlearn_pb2_grpc.py
│       │       └── fot_pb2.py / fot_pb2_grpc.py
│       │
│       ├── estimators/             ← Gradient estimation algorithms
│       │   ├── __init__.py
│       │   ├── params.py           ← Parameter flatten/unflatten helpers
│       │   ├── perturbation.py     ← Seeded perturbation generation
│       │   └── zeroth_order.py     ← ZerothOrderEstimator (DeComFL forward-diff)
│       │
│       ├── privacy/                ← Central DP (FR-13)
│       │   ├── dp_mechanism.py     ← Clipping + Gaussian noise
│       │   └── dp_accountant.py    ← From-scratch RDP accountant
│       │
│       ├── security/               ← FL-boundary security
│       │   ├── tls.py              ← SE-2 fail-closed TLS policy
│       │   ├── token_verify.py     ← SE-1 connection-token verification
│       │   ├── identity.py         ← SE-15 partition binding
│       │   └── interceptor.py / client_interceptor.py
│       │
│       ├── backbone/               ← Shared backbone/distribution helpers
│       ├── bundle/                 ← Adapter-bundle manifest + JSON schema
│       ├── fot/                    ← Federation over Text (separate research mode)
│       │
│       └── data/                   ← Dataset utilities
│           └── MNIST/              ← MNIST download + Dirichlet partition helpers
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
├── benchmarks/                     ← DP / robust-aggregation experiment harnesses
│
├── requirements.txt                ← Python dependencies
├── setup.py                        ← Package metadata + install configuration
├── pyproject.toml                  ← Minimal build-system config
├── run_full_test_suite.py          ← Comprehensive e2e test runner
├── run_local_test.py               ← Quick local sanity test
└── run_platform_e2e_test.py        ← Integration test (talks to Spring backend)
```

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
  strategy.py ──► FedAvgAggregator
  decomfl_strategy.py ──► zeroth_order.ZerothOrderEstimator
  serializer.py ──► communication/generated/fedlearn_pb2.py
```

---

## How the Components Fit Together

### The Server Side

1. **`server.py`** — the single public entry point. It creates a `FLCoordinator`, wraps it in a gRPC server, and runs the training loop.
2. **`FLCoordinator`** — the brain of the server. It tracks round state, client registrations, heartbeats, update submissions, and triggers aggregation when all expected clients have submitted.
3. **`FederatedLearningServiceServicer`** — the gRPC I/O layer. Each incoming RPC call is a thin dispatcher that validates the request and calls a method on `FLCoordinator`.
4. **`Strategy`** — pluggable aggregation and evaluation logic. The coordinator delegates `aggregate_fit()` and `evaluate()` to the active strategy.

### The Client Side

1. **`client.py`** — the single public entry point for standard FL clients. `start_client()` wraps the entire training loop: register → fetch model → `client.fit()` → submit update → poll for next round.
2. **`GrpcClient`** — handles all network I/O. It manages the gRPC channel, heartbeat thread, retry logic, and automatically selects between unary and streaming RPC for model uploads.
3. **`Client` ABC** — the interface that application code must implement. Only two methods: `get_parameters()` and `fit()`.
4. **`DeComFLClient`** — a specialised subclass for the DeComFL protocol. Instead of returning updated model weights, it returns gradient scalars, making communication O(1) in model size.

---

## Federated Learning Lifecycle at a Glance

```
Server starts
    │
    ▼
coordinator.set_initial_parameters(strategy.initialize_parameters())
    │
    └─► For each round r = 1 … N:
            │
            ├── coordinator.start_round()   ← clears stale state
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
            │       (blocks until clients_per_round updates received)
            │
            ├── strategy.aggregate_fit(round, results)
            │       → new global model params
            │
            ├── strategy.evaluate(round, params)
            │       → loss + metrics dict
            │
            └── coordinator.current_round += 1
                coordinator._round_complete_event.set()

Server stops gRPC server, returns (history, final_parameters)
```

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

`client-docker/` contains a `Dockerfile` that installs the framework and exposes a client entrypoint. The backend can instruct clients to spin up Docker containers for isolated training environments.

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

Also re-exported from `fedlearn/__init__.py`: the `Strategy` ABC, the `FedProx` / `FedOpt`
strategies alongside `FedAvg`, `LocalTrainer` (the first-order client trainer), and
`create_strategy` (the name → `Strategy` factory). `fedlearn.server` additionally exports
`FedLoRA`, `RobustAggregator` (FR-12), and `STRATEGY_REGISTRY`.

> **Key design principle:** Users only implement two methods (`get_parameters` and `fit`). All network communication, retry logic, heartbeating, and round synchronisation is handled by the framework.
