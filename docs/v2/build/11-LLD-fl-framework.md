# 11 — Low-Level Design (LLD): Python FL (Federated Learning) Framework

**Unit:** the custom Python FL (Federated Learning) framework — server, client, FedAvg (Federated Averaging) + DeComFL (Dimension-Free Communication Federated Learning) strategies, the ZO (Zeroth-Order) gradient estimator, parameter chunking, the dual-heartbeat transport, and the gRPC (Google Remote Procedure Call) layer (proto package `fedlearn.v2`).

**Document type:** Production build specification — the implementing-model's complete contract for `framework/`.
**Audience:** a mid-sized (~30 billion-parameter) local LLM (Large Language Model). Every interface, version, file path, environment variable, and command below is **pre-decided**. Do not choose alternatives. Implement the bodies; do not redesign the contracts.
**Date authored:** 2026-05-29.

**Authoritative inputs this document conforms to (do not contradict):**
- `docs/v2/build/02-TECH-STACK.md` — pinned versions (§3 below cites the exact lines).
- `docs/v2/build/03-DATA-MODEL.md` — the schema this unit writes telemetry into (it does **not** own the schema; it emits `RoundResultDto`-shaped rows the backend persists into `round_results`).
- `docs/v2/build/04-API-CONTRACTS.md` — the gRPC contract (`fedlearn.v2`, §10), the internal-callback REST contract (§5, §13), and the W3C (World Wide Web Consortium) `traceparent` contract (§14) this unit consumes.
- `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` — the three DeComFL correctness bugs and their locked fixes (cited as **SPEC §n**).
- `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md` — the TDD (Test-Driven Development) implementation plan with exact code (cited as **PLAN Task n**).

**Audit reports driving this design (cited as `A3-Fn`, `B1-Cn`, `C3-Rn`):**
- `docs/audit/2026-05-29/A3-framework.md` — framework verdict (REFACTOR: keep the core, rebuild transport/serializer/deps).
- `docs/audit/2026-05-29/B1-paper-alignment.md` — DeComFL fidelity to arXiv 2405.15861.
- `docs/audit/2026-05-29/C3-reproducibility.md` — determinism manifest + CPU-canonical RNG.

> **Federated context (stated up front, per project convention).**
> - **Aggregation strategies:** FedAvg (sample-count-weighted mean of full model parameters) and DeComFL (uniform/unweighted mean of ZO gradient *scalars*, applied to a server-regenerated perturbation). Both are implemented as pure-math `Strategy` subclasses with no I/O.
> - **Client heterogeneity:** the fleet mixes Apple M4 Max (MPS — Metal Performance Shaders), Nvidia Jetson ARM64 (CUDA — Compute Unified Device Architecture), and CPU-only Docker clients. The server may run on a GPU host. This heterogeneity is the precise reason the RNG path is CPU-canonical (C3-R1).
> - **Communication-round-bounded:** every round has a hard wall-clock deadline (`round_deadline_seconds`) and a minimum quorum (`min_clients`). The round loop **never** hangs on a straggler (`04-API-CONTRACTS.md §10.3`).

---

## 0. Abbreviations (first-use expansions; short forms thereafter)

| Short form | Full form |
|---|---|
| FL | Federated Learning |
| LLM | Large Language Model |
| DeComFL | Dimension-Free Communication Federated Learning (the v1 wiki's "Decomposed" expansion is wrong per the paper, `B1-paper-alignment.md:33`) |
| FedAvg | Federated Averaging |
| ZO | Zeroth-Order (optimization) |
| RNG | Random Number Generator |
| gRPC | Google Remote Procedure Call |
| RPC | Remote Procedure Call |
| REST | Representational State Transfer |
| STOMP | Simple Text Oriented Messaging Protocol |
| TLS | Transport Layer Security |
| mTLS | mutual TLS |
| CN | Common Name (of a TLS certificate) |
| API | Application Programming Interface |
| DTO | Data Transfer Object |
| CPU | Central Processing Unit |
| GPU | Graphics Processing Unit |
| MPS | Metal Performance Shaders (Apple GPU backend) |
| CUDA | Compute Unified Device Architecture (Nvidia GPU backend) |
| sha256 | Secure Hash Algorithm 256-bit |
| SPSA | Simultaneous Perturbation Stochastic Approximation |
| ABC | Abstract Base Class |
| TDD | Test-Driven Development |
| CI | Continuous Integration |
| OTel | OpenTelemetry |
| W3C | World Wide Web Consortium |
| DLG | Deep Leakage from Gradients |
| DP | Differential Privacy |
| K | number of local steps per round (DeComFL) |
| P | number of perturbations per local step (DeComFL) |
| η (eta) | learning rate |
| μ (mu) | ZO smoothing radius |
| N | number of reporting clients in a round |
| d | model dimension (flat parameter count) |
| GIL | Global Interpreter Lock |
| UUID | Universally Unique Identifier |
| HMAC | Hash-based Message Authentication Code |

---

## 1. Purpose & single responsibility

**Single responsibility:** run the FL training loop. Given a launched run (a `fl_runs` row materialized by the control plane), this unit (a) accepts client registrations bound to a `run_id`, (b) drives bounded rounds under FedAvg or DeComFL, (c) regenerates DeComFL perturbations deterministically and aggregates the scalar gradients, (d) serializes/transports model parameters (typed `Tensor` framing; chunked for models that exceed the chunk threshold), (e) emits per-round telemetry (including the DeComFL communication-cost wedge) to the control plane, and (f) emits a determinism manifest at startup so the run is reproducible.

**Explicitly NOT this unit's responsibility (owned elsewhere — do not implement here):**

| Concern | Owner |
|---|---|
| Run lifecycle / lease / reconciler / quotas / port assignment | Spring Boot control plane (`fl_runs` lease, `02-TECH-STACK.md §18.2`). |
| Schema / migrations / persistence | Flyway + JPA (Jakarta Persistence API) in the backend (`03-DATA-MODEL.md`). This unit POSTs DTOs; the backend writes rows. |
| Minting the per-run scoped token | Backend security layer (`04-API-CONTRACTS.md §13`). This unit only **reads** `FEDLEARN_RUN_TOKEN` from its env and presents it. |
| Pre-signed S3/MinIO upload URLs, artifact-registry rows | Backend `/api/artifacts/*` (`04-API-CONTRACTS.md §9`). This unit computes the sha256 and PUTs bytes to the pre-signed URL. |
| Choosing the launcher (Kubernetes Job / ECS RunTask / LocalProcess) | Backend `FlServerLauncher` (`02-TECH-STACK.md §18.1`). This unit is launched **by** one of these; it is launcher-agnostic. |
| The C++ mobile ZO core | `mobile_client/` (deferred; this unit only emits the golden-vector fixtures the C++ port must later pass — SPEC §6). |

> **Why this scoping (A3 §6 verdict "keep the core, rebuild the transport/lifecycle layer"):** v1 mixed transport, math, and lifecycle inside `serializer.py`/`grpc_client.py`/`coordinator.py`, which made the aggregation math un-unit-testable and produced the RPC-ordering coupling A3-N1 flagged. v2 separates **pure-math `strategies/`** (no I/O) from **`transport/`** (the only place that imports grpc) so the math is testable without a server.

---

## 2. Position in the system

### 2.1 Depends-on (this unit calls these)

| Dependency | Direction | Contract (exact name) |
|---|---|---|
| Control plane internal callbacks | this unit → backend (REST/JSON) | `POST /api/internal/runs/{runId}/results`, `/finished`, `/checkpoint`, `/status` (`04-API-CONTRACTS.md §5`). Auth = `Authorization: Bearer flrun_<...>` from `FEDLEARN_RUN_TOKEN` (`§13`). |
| Artifact store | this unit → S3/MinIO (pre-signed PUT) + backend register | `POST /api/artifacts/upload-url` → PUT bytes → `POST /api/artifacts` (`04-API-CONTRACTS.md §9`). The S3 object key is the model `sha256`. |
| Environment / launch context | backend → this unit (process env) | `FEDLEARN_RUN_ID`, `FEDLEARN_RUN_TOKEN`, `FEDLEARN_BACKEND_URL`, `FEDLEARN_PROJECT_ID`, `TRACEPARENT` (`04-API-CONTRACTS.md §13`, §14; full table in §8 below). |

### 2.2 Depended-by (these call this unit)

| Consumer | Direction | Contract (exact name) |
|---|---|---|
| FL clients (Python desktop sidecar, Docker, C++ mobile) | client → this unit (gRPC) | `service FederatedLearningService` in `fedlearn.v2` (`04-API-CONTRACTS.md §10.2`). All client-initiated RPCs carry `run_id`. |
| OTel collector / Tempo | this unit → collector (traces) | The Python server roots its span at `TRACEPARENT` extracted from env and continues `traceparent` over gRPC metadata (`04-API-CONTRACTS.md §14`). |

### 2.3 Interfaces EXPOSED (gRPC, `fedlearn.v2`) — by exact RPC name

The authoritative `.proto` is `04-API-CONTRACTS.md §10.2`. This unit **implements the server side** of `service FederatedLearningService` and **implements the client side** in `client/`:

```
RegisterClient        (RegisterClientRequest)        -> RegisterClientResponse
GetServerStatus       (GetServerStatusRequest)       -> GetServerStatusResponse
Heartbeat             (HeartbeatRequest)             -> HeartbeatResponse        # parallel stub
GetGlobalModel        (GetGlobalModelRequest)        -> GetGlobalModelResponse   # FedAvg unary
GetGlobalModelStream  (GetGlobalModelRequest)        -> stream ModelChunk        # FedAvg >threshold
SubmitModelUpdate     (SubmitModelUpdateRequest)     -> SubmitModelUpdateResponse
SubmitModelUpdateStream(stream ModelUpdateChunk)     -> SubmitModelUpdateResponse
GetDeComFLConfig      (GetDeComFLConfigRequest)      -> GetDeComFLConfigResponse # seeds + rebuild history
SubmitGradientScalars (SubmitGradientScalarsRequest) -> SubmitGradientScalarsResponse
ReportClientMetrics   (ReportClientMetricsRequest)   -> ReportClientMetricsResponse
```

### 2.4 Interfaces CONSUMED (REST callbacks) — by exact path

| Method | Path | Body shape (`04-API-CONTRACTS.md §5.1`) | When this unit fires it |
|---|---|---|---|
| `POST` | `/api/internal/runs/{runId}/status` | `RunStatusReportDto` | First reachable gRPC endpoint; between rounds. |
| `POST` | `/api/internal/runs/{runId}/results` | `RoundResultDto` | **Once per round, during the round loop**, immediately after aggregation/eval (incremental — `B3` fix). Best-effort. |
| `POST` | `/api/internal/runs/{runId}/checkpoint` | `CheckpointReportDto` | After a per-round content-addressed checkpoint is uploaded to S3. |
| `POST` | `/api/internal/runs/{runId}/finished` | `RunFinishedDto` | On terminal completion (`SUCCEEDED`/`FAILED`). |

> **Topic note:** this unit does **not** publish to STOMP. STOMP topics (`/topic/results/{projectId}` etc., `04-API-CONTRACTS.md §11`) are the backend's responsibility — the backend re-broadcasts the `RoundResultDto` it receives on `/api/internal/runs/{runId}/results`. This unit's only telemetry sink is the four internal REST callbacks above.

### 2.5 ASCII position diagram

```
            (browser/desktop)            <-- not this unit
                  |
            Spring Boot control plane    <-- not this unit (owns fl_runs lease, mints run token)
              |          ^   |
   launch env |   REST   |   | pre-signed URL broker
   (§8 vars)  |callbacks |   |
              v   (§2.4) |   v
   +======================================================+
   |   THIS UNIT: framework/  (fl_server.py entrypoint)   |
   |   transport/grpc  <-- gRPC fedlearn.v2 (§2.3) -->    |
   |   server/coordinator  server/strategies  estimators  |
   +======================================================+
              ^                       |
        gRPC  | (run_id-bound)        | traceparent over gRPC metadata
              |                       v
   FL clients (Python sidecar / Docker / C++ mobile)   <-- separate units
```

---

## 3. Tech stack for this unit (pinned versions from `02-TECH-STACK.md`)

| Technology | Pinned version | One-line reasoning |
|---|---|---|
| CPython | `3.12.x` (e.g. `3.12.9`, `verify-before-use`) | Framework runtime; faster startup matters because the substrate spawns server processes; PyTorch 2.12 ships cp312 wheels for x86-64 and ARM64 (`02-TECH-STACK.md §1.2`). |
| PyTorch (`torch`) | `2.12.0` (`verify-before-use`, cp312 x86-64 + ARM64) | Tensors, FedAvg averaging, DeComFL perturbations; pinned exact so the CPU-canonical RNG golden vectors stay reproducible (`02-TECH-STACK.md §4.1`). |
| `numpy` | `verify-before-use` (1.26+/2.x consistent with the torch build; PLAN observed `2.1.2`) | Data-seed RNG (`np.random.default_rng(PCG64)`) for partitioning + DeComFL seed generation (`02-TECH-STACK.md §4.3`). |
| `safetensors` (Hugging Face) | `verify-before-use` (latest stable, e.g. `0.4.x`) | The on-the-wire codec; no `pickle`/`torch.save` on the wire — kills the v1 `weights_only` foot-gun and the C1 `KeyError` (`02-TECH-STACK.md §4.2`, `04-API-CONTRACTS.md §10.3`). |
| `grpcio` + `grpcio-tools` | `verify-before-use` (exact pair, matched to the protobuf runtime, e.g. `1.6x.x`) | gRPC runtime for `fedlearn.v2`; default TLS+mTLS, plaintext only in `dev` (`02-TECH-STACK.md §3.2`). |
| `protobuf` (Python runtime) | `verify-before-use` (the exact wheel the buf-generated stubs require) | Generated message classes; pin to what `buf generate` produces (do **not** carry v1's phantom `protobuf>=4.21.6,<5.0.0` range — `02-TECH-STACK.md §3.1`). |
| buf CLI | `1.70.0` (`verify-before-use`) | Single source of truth for the proto + breaking-change gate; one `buf.gen.yaml` generates Python/Java/TS/C++ from `fedlearn/v2/fedlearn.proto` (`02-TECH-STACK.md §3.3`). |
| `lz4` | `verify-before-use` (single-digit dep set, A3-N3) | Optional on-the-wire compression for the FedAvg chunk path (`codec="lz4+safetensors"`, `04-API-CONTRACTS.md §10.2`). |
| `pydantic` | `verify-before-use` (latest 2.x) | Typed `RoundConfig`/`ClientUpdate`/manifest models replacing v1's `Dict`/tuple soup (A3 §6.2); mypy (already `strict`) gets real types. |
| `opentelemetry-sdk` + `opentelemetry-exporter-otlp` | `verify-before-use` (matched to OTel Collector `0.153.0`) | Roots the run span at `TRACEPARENT`, continues `traceparent` over gRPC metadata (`02-TECH-STACK.md §20`, `04-API-CONTRACTS.md §14`). |
| `structlog` | `verify-before-use` (latest stable) | Structured Python logs binding `trace_id`/`project_id`/`round_idx` (`02-TECH-STACK.md §20`). |
| `requests` (or `httpx`) | `verify-before-use` | The internal REST callback client (`§2.4`). Short timeouts, best-effort (`04-API-CONTRACTS.md §5` reasoning). |
| `pytest` | `verify-before-use` | Test runner; `addopts = "-v --tb=short"`; CPU-forcing autouse fixtures (PLAN Task 0). |

**Hard exclusions (A3-N2/N3, `02-TECH-STACK.md §25.1`):** no `flwr`, no `flwr-datasets`, no `ray`, no `pika`/RabbitMQ in this unit, no `matplotlib`/`seaborn`, no `opencensus`/`google-*`. The dependency set is single digits. The `pyproject.toml` MUST gain a `[project]` table with these exact pinned deps (v1 had none — A3 §6.2 / C3 §4.3).

---

## 4. Module / file structure

Target tree under `framework/src/fedlearn/` (the v2 layout from A3 §6.1, made concrete). Files marked **NEW** do not exist in v1; **REBUILD** replaces a v1 file of the same path; **SALVAGE** keeps v1 logic with the fixes named here.

```
framework/
  pyproject.toml                       # REBUILD: add [project] table, pinned single-digit deps, pytest+ruff+mypy config
  fl_server.py                         # REBUILD: server entrypoint; reads §8 env, builds manifest, runs the coordinator
  proto/
    fedlearn/v2/fedlearn.proto         # NEW location: the §10.2 contract (buf-governed; package fedlearn.v2)
  buf.yaml                             # NEW: buf module + lint + breaking config
  buf.gen.yaml                         # NEW: managed mode; python plugin output -> src/fedlearn/transport/generated/
  src/fedlearn/
    __init__.py
    core/                              # NEW: framework-agnostic, no torch-device, no grpc imports
      types.py                         #   typed records: Parameters, RoundConfig, ClientUpdate, GradientScalars, AggregationResult
      manifest.py                      #   DeterminismManifest builder + sha256 helpers (C3 §5.2)
      hashing.py                       #   sha256 of safetensors bytes / dataset-split index arrays
      errors.py                        #   typed exceptions -> gRPC status mapping (§9)
    strategies/                        # pure math, NO I/O, NO grpc — unit-testable in isolation
      strategy.py                      # REBUILD: Strategy ABC (abstract base class); §5.1 signature
      fedavg.py                        # SALVAGE: sample-weighted mean of full params (math correct in v1)
      decomfl.py                       # REFACTOR: DeComFL strategy; 1/P fix, local RNG, O(K*P) hoist, bounded history
    estimators/
      perturbation.py                  # NEW (SPEC §3 Bug2 / PLAN Task2): canonical_perturbation() — CPU-canonical
      zeroth_order.py                  # SALVAGE: ZerothOrderEstimator; delegates to canonical_perturbation
    server/
      coordinator.py                   # SALVAGE: FLCoordinator round state machine; deadline+quorum; eviction wiring
      lifecycle.py                     # NEW: status/finished/checkpoint emission orchestration
    client/
      base_client.py                   # REBUILD: gRPC client base (channel factory, retry, TLS)
      decomfl_client.py                # SALVAGE: DeComFLClient.fit + rebuild_model (client math is correct; untouched)
      heartbeat.py                     # REBUILD: heartbeat supervisor (parallel stub, threading.Event abort latch)
    transport/
      generated/                       # buf output: fedlearn_pb2.py, fedlearn_pb2_grpc.py (do NOT hand-edit)
      channel.py                       # NEW: secure_channel / server_credentials factory (TLS+mTLS; plaintext dev only)
      servicer.py                      # REBUILD: FederatedLearningServicer (server side of all RPCs)
      codec.py                         # REBUILD (replaces serializer.py): safetensors typed framing; chunking; symmetry
    telemetry/
      callbacks.py                     # NEW: internal-REST client (§2.4); best-effort POSTs
      tracing.py                       # NEW: OTel root span from TRACEPARENT; gRPC metadata propagation
      metrics.py                       # NEW: per-round comm-cost computation (uplink/downlink/scalars bytes)
  tests/
    conftest.py                        # SALVAGE: CPU-forcing autouse fixtures (monkeypatch cuda.is_available -> False)
    fixtures/decomfl_golden/           # NEW (SPEC §6 / PLAN Task2): generate.py, manifest.json, *.npy golden vectors
    test_perturbation.py               # NEW: T2 golden + cross-device parity + server/client agree
    test_decomfl_strategy.py           # T1, T4, T5, B-1, B-2 (PLAN Tasks 4-8)
    test_serializer.py                 # -> test_codec.py: T3 chunk roundtrip (multi-chunk + transformer-shaped)
    test_fedavg_aggregator.py          # SALVAGE
    test_coordinator.py                # SALVAGE + deadline/quorum tests
    test_zeroth_order.py               # SALVAGE
```

**One-line responsibility per key file:**

| File | Responsibility |
|---|---|
| `fl_server.py` | Entrypoint: parse `§8` env + run config, build `DeterminismManifest`, construct strategy + coordinator + servicer, serve until done/stopped. |
| `core/types.py` | The typed data records crossing module boundaries (no `Dict`/tuple soup). |
| `core/manifest.py` | Compute the C3 §5.2 determinism manifest (torch/numpy/git/proto versions, CPU rng declaration, seed, model+split hashes). |
| `strategies/strategy.py` | The `Strategy` ABC (interface in §5.1). |
| `strategies/fedavg.py` | `FedAvg.aggregate_fit` = sample-count-weighted mean of `ModelParameters`. |
| `strategies/decomfl.py` | `DeComFL`: seed schedule, `aggregate_fit` (corrected `1/P`, hoisted O(K·P)), `_generate_perturbation` (delegates), bounded history. |
| `estimators/perturbation.py` | `canonical_perturbation(seed, num_params, dtype)` — the single source of truth for `z` (CPU). |
| `estimators/zeroth_order.py` | `ZerothOrderEstimator.compute_gradient_scalar` (forward/central SPSA) + `generate_perturbation` (delegates). |
| `server/coordinator.py` | Round state machine; deadline + quorum; calls `strategy.aggregate_fit`; stores averaged gradients; calls `evict_old_history`. |
| `transport/codec.py` | safetensors typed (de)serialization, chunk framing (sha256/codec/total_bytes), symmetric save/load. |
| `transport/servicer.py` | The gRPC server methods; binds every call to `run_id`; enforces framing rules (`§10.3`). |
| `client/heartbeat.py` | Parallel heartbeat stub; `threading.Event` abort latch when `should_stop` fires. |
| `telemetry/callbacks.py` | The four internal REST POSTs, best-effort, short-timeout, with the run token. |
| `telemetry/metrics.py` | Compute `uplink_bytes`/`downlink_bytes`/`scalars_transmitted`/`modelParamCount` per round. |

---

## 5. Key interfaces & type signatures (FULL)

> All signatures are Python 3.12 with full type hints. The implementing model writes the bodies. Field names exactly match `04-API-CONTRACTS.md` / `03-DATA-MODEL.md` where they cross a boundary.

### 5.1 The `Strategy` ABC (`strategies/strategy.py`)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Callable
import torch

# Parameters = state-dict-shaped mapping name -> tensor. Matches proto ModelParameters.tensors.
Parameters = "OrderedDict[str, torch.Tensor]"

class Strategy(ABC):
    """Pure-math FL aggregation. NO grpc, NO file/network I/O, NO process-global RNG.

    A Strategy holds the global model and reduces a round's client updates into a
    new global model. It is unit-testable without a server (A3 N1/M6 fix).
    """

    @abstractmethod
    def initial_parameters(self) -> Parameters:
        """Return the starting global parameters (state-dict shape)."""

    @abstractmethod
    def configure_round(self, server_round: int) -> dict[str, str]:
        """Return the per-round config map sent to clients (string->string, matches proto config maps)."""

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[str, object, int]],
        # FedAvg: results = [(client_id, Parameters, num_examples), ...]
        # DeComFL: results = [(client_id, GradientScalars (List[List[float]] [K][P]), num_examples), ...]
    ) -> Optional[Parameters]:
        """Reduce one round's client updates into the new global parameters. None on empty/failed round."""

    @abstractmethod
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, float]]]:
        """Optional server-side eval. Return (loss, metrics) or None when no evaluate_fn is set.
        NOTE: callers MUST guard: r = strategy.evaluate(...); if r is not None: loss, m = r   (A3-M2)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """'FedAvg' | 'DeComFL' — matches fl_runs.strategy CHECK set (03-DATA-MODEL.md §5.2)."""
```

### 5.2 `DeComFL` strategy (`strategies/decomfl.py`) — full constructor + methods

```python
class DeComFL(Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        evaluate_fn: Optional[Callable[[Parameters], tuple[float, dict[str, float]]]],
        min_fit_clients: int,
        clients_per_round: int,
        num_local_steps: int,          # K
        num_perturbations: int,        # P
        learning_rate: float,          # eta
        smoothing_param: float,        # mu
        seed: int,
        grad_estimate_method: str = "forward",   # "forward" | "central"  (B1-H2)
        max_retained_rounds: int = 100,          # bounded history cap (SPEC C-2 / PLAN Task7)
    ) -> None: ...

    # --- determinism / state (instance-local; NEVER process-global, SPEC B-2) ---
    self._np_rng: "numpy.random.Generator"          # np.random.default_rng(seed) — local, not np.random.seed
    self.global_params_flat: torch.Tensor           # the flat global model (CPU canonical)
    self.seed_history: dict[int, list[list[int]]]    # round -> [K][P] seeds
    self.gradient_history: dict[int, list[list[float]]]  # round -> [K][P] AVERAGED (1/N) scalars
    self.client_last_round: dict[str, int]           # client_id -> last round it participated in

    def generate_seeds(self, round_idx: int) -> list[list[int]]:
        """[K][P] int seeds drawn from self._np_rng.integers(0, 2**31-1). Pure; does not cache."""

    def get_or_generate_seeds(self, round_idx: int) -> list[list[int]]:
        """Lock-guarded: return seed_history[round_idx], generating+caching on first call (A3-N1 fix:
        aggregate_fit must never KeyError on an unseeded round)."""

    def aggregate_fit(self, server_round: int, results) -> Optional[Parameters]:
        """Corrected DeComFL server update (SPEC Bug1 + Cleanup C-1):
        for k in range(K):
            delta = 0
            for p in range(P):
                z = self._generate_perturbation(seed_history[server_round][k][p])   # O(K*P) hoist
                g_sum = sum(client_grads[k][p] for each client)
                delta += g_sum * z
            delta = delta / (num_clients * P)        # the 1/(N*P) average
            x_current = x_current - eta * delta      # NO '* P' (Bug1 fix)
        Returns new global Parameters (unflattened)."""

    def _generate_perturbation(self, seed: int) -> torch.Tensor:
        """z for the full model dimension. Delegates to canonical_perturbation then .to(self.device).
        MUST NOT construct torch.Generator(device=non-cpu) (SPEC Bug2)."""

    def evict_old_history(self) -> None:
        """Bounded eviction (SPEC C-2): keep rounds >= max(min(client_last_round), newest - cap + 1)."""
```

> **DeComFL `results` element shape (locked):** `(client_id: str, gradients: list[list[float]], num_examples: int)` where `gradients[k][p]` is the ZO scalar `g` for local step `k`, perturbation `p`. `num_examples` is **collected but ignored** — DeComFL aggregation is **unweighted** (B1 Low note, `04-API-CONTRACTS.md §10.2` `SubmitGradientScalarsRequest`).

### 5.3 `FedAvg` strategy (`strategies/fedavg.py`)

```python
class FedAvg(Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        evaluate_fn: Optional[Callable[[Parameters], tuple[float, dict[str, float]]]],
        min_fit_clients: int,
        max_num_examples: int = 100_000,   # weighting cap; WARN when it fires (A3-H2)
    ) -> None: ...

    def aggregate_fit(self, server_round, results) -> Optional[Parameters]:
        """Sample-count-weighted mean: sum(n_i * params_i) / sum(n_i), per tensor key.
        WARN(client_id, requested, capped) when n_i is clamped to max_num_examples (A3-H2)."""
```

### 5.4 ZO estimator (`estimators/zeroth_order.py`) + canonical perturbation (`estimators/perturbation.py`)

```python
# estimators/perturbation.py  (SPEC §3 Bug2 — verbatim contract, PLAN Task2)
import torch
CANONICAL_DTYPE = torch.float32   # pinned; never follows the model dtype (breaks golden parity)

def canonical_perturbation(
    seed: int,
    num_params: int,
    dtype: torch.dtype = CANONICAL_DTYPE,
) -> torch.Tensor:
    """Device-independent N(0, I_d) of shape (num_params,). Generated on CPU with a LOCAL
    torch.Generator(device='cpu') for bit-stable output across CPU/CUDA/MPS. Callers .to(device)
    at the use site. THE single source of truth; server + client both delegate here."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(num_params, generator=g, dtype=dtype, device="cpu")
```

```python
# estimators/zeroth_order.py
class ZerothOrderEstimator:
    def __init__(self, smoothing_param: float, device: str = "cpu",
                 grad_estimate_method: str = "forward") -> None: ...

    def generate_perturbation(self, seed: int, num_params: int) -> torch.Tensor:
        """Delegate to canonical_perturbation(seed, num_params).to(self.device). (SPEC Bug2 / PLAN Task3)"""

    def compute_gradient_scalar(
        self, model: torch.nn.Module, loss_fn, batch, z: torch.Tensor
    ) -> float:
        """forward:  g = (f(x+mu*z) - f(x)) / mu
           central:  g = (f(x+mu*z) - f(x-mu*z)) / (2*mu)   (B1-H2; default forward)
        Two/three forward passes under no_grad. Returns the scalar g."""
```

### 5.5 Codec (`transport/codec.py`) — symmetric chunking (SPEC Bug3)

```python
from collections import OrderedDict
import torch

CHUNK_THRESHOLD_BYTES: int = 300 * 1024 * 1024   # 300 MB — reconcile the project conventions vs v1's 100 MB (A3-N5)
DEFAULT_CHUNK_SIZE: int = 4 * 1024 * 1024         # 4 MB per gRPC ModelChunk

def parameters_to_safetensors(params: "OrderedDict[str, torch.Tensor]") -> bytes:
    """Serialize a state-dict to safetensors bytes (typed; NO pickle/torch.save). codec='safetensors'."""

def safetensors_to_parameters(blob: bytes) -> "OrderedDict[str, torch.Tensor]":
    """Inverse; dtype-whitelist + shape validation on load."""

def parameters_to_chunks(
    params: "OrderedDict[str, torch.Tensor]",
    num_examples: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compress: bool = False,
) -> "Iterator[dict]":
    """Yield ModelChunk-shaped dicts. The reassembled blob is the FULL safetensors payload of
    {'parameters': params, 'num_examples': num_examples} so chunks_to_parameters reads it SYMMETRICALLY
    (SPEC Bug3 — v1 saved a bare OrderedDict and load expected a wrapped dict -> KeyError).
    Each dict carries: chunk_index, total_chunks, chunk_data, is_final_chunk,
    codec ('safetensors' | 'lz4+safetensors'), compressed (bool, ON THE WIRE not env A3-C3),
    total_bytes (full reassembled size), sha256 (of full blob)."""

def chunks_to_parameters(
    full_data: bytes, *, compressed: bool, codec: str, expected_sha256: str | None = None
) -> tuple["OrderedDict[str, torch.Tensor]", int]:
    """Reassemble -> (parameters, num_examples). Verify sha256 if provided. Reject codec not in
    {'safetensors','lz4+safetensors'} with a typed error -> gRPC INVALID_ARGUMENT (§10.3)."""
```

### 5.6 Determinism manifest (`core/manifest.py`) — matches `03-DATA-MODEL.md §5.3` + `04-API-CONTRACTS.md §4.4`

```python
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class DeterminismManifest:
    run_id: str
    framework_git_sha: str
    proto_version: str = "fedlearn.v2"
    torch_version: str = ""           # torch.__version__   (e.g. "2.12.0")
    numpy_version: str = ""
    torch_cuda_version: str | None = None   # None when CPU-only
    rng_device: str = "cpu"           # INVARIANT: must be 'cpu' (DB CHECK rng_device='cpu', C3 §5.1)
    rng_engine: str = "torch.Generator(cpu)"
    use_deterministic_algorithms: bool = False
    seed: int = 0                     # optimizer/perturbation seed
    strategy: str = "DeComFL"
    initial_model_sha256: str | None = None
    dataset_split_sha256: str | None = None
    golden_vector_sha256: str | None = None    # the committed RNG fixture hash (SPEC §6)
    platform_os: str = ""             # "linux"
    platform_arch: str = ""           # "x86_64" | "arm64"
    hyperparameters: dict | None = None        # exact StartRunRequest.hyperparameters echo

    def to_json(self) -> dict: return asdict(self)   # -> determinism_manifests.manifest_json (JSONB)

def build_manifest(run_id: str, config: "RoundConfig", initial_model_sha256: str,
                   dataset_split_sha256: str | None) -> DeterminismManifest: ...
```

### 5.7 Telemetry DTO this unit POSTs (`telemetry/callbacks.py`) — matches `04-API-CONTRACTS.md §5.1`

```python
@dataclass
class RoundResultDto:           # POST /api/internal/runs/{runId}/results
    serverRound: int
    loss: float | None
    accuracy: float | None
    gpuUtilization: float | None
    uplinkBytes: int | None         # bytes clients -> server this round
    downlinkBytes: int | None       # bytes server -> clients this round
    scalarsTransmitted: int | None  # K*P scalars this round (the O(K*P) proof; null for FedAvg)
    modelParamCount: int | None     # model dimension d
    roundDurationSeconds: float | None
    aggregationSeconds: float | None
    activeClients: int | None
```

> **Wire-mapping note (do not drift):** the on-the-wire field is `serverRound`; the backend persists it into `round_results.round_idx` (`03-DATA-MODEL.md §5.2`, `04-API-CONTRACTS.md §5.1` reasoning). `uplinkBytes`/`downlinkBytes`/`scalarsTransmitted` map to `round_results.uplink_bytes`/`downlink_bytes`/`scalars_transmitted` (the DeComFL communication-cost wedge).

---

## 6. Core algorithms & flows

### 6.1 DeComFL server update — the corrected `aggregate_fit` (SPEC Bug1 + C-1)

This is the canonical body. **The `* self.P` of v1 is deleted** (Bug1, B1-C1); `z` is generated **once per `(k, p)`** not per client (C-1, B1-M5/A3-H4); `delta` carries the `1/(N·P)` average.

```python
def aggregate_fit(self, server_round, results):
    # results: list[(client_id, grads[K][P], num_examples)]  -- num_examples ignored (unweighted)
    client_grads = {cid: g for (cid, g, _n) in results}
    num_clients = len(client_grads)
    if num_clients == 0:
        return None
    for cid in client_grads:
        self.client_last_round[cid] = server_round

    x_current = self.global_params_flat                       # CPU-canonical flat tensor
    avg_grads = [[0.0] * self.P for _ in range(self.K)]       # store averaged scalars for rebuild history

    for k in range(self.K):
        delta = torch.zeros_like(x_current)
        for p in range(self.P):
            # z depends ONLY on (k, p) -> generate once (O(K*P) not O(K*P*N))
            z = self._generate_perturbation(self.seed_history[server_round][k][p])
            g_sum = 0.0
            for cid, grads in client_grads.items():
                g_sum += grads[k][p]
            delta += g_sum * z
            avg_grads[k][p] = g_sum / num_clients             # the 1/N averaged scalar
        delta = delta / (num_clients * self.P)                # 1/(N*P)
        x_current = x_current - self.eta * delta              # NO '* self.P'  (Bug1 fix)

    self.global_params_flat = x_current
    self.gradient_history[server_round] = avg_grads           # for client rebuild (Alg.2 replay)
    return self._unflatten(x_current)
```

**Why this is correct (B1-C1):** the reference `cezo_fl` averages over P (`random_gradient_estimator.py:176` `grad.div_(self.num_pert)`); the client applies `(eta/P)·delta` (`decomfl_client.py:208`); the rebuild replays the `(1/P)` step. v1's server applied `eta·delta·P`, diverging from every client by a factor of P (10× at the `P=10` default). The fix makes `global_params_flat` and the rebuild trajectory **bit-equal** (test T1, §10).

### 6.2 CPU-canonical perturbation flow (Bug2 / C3-R1)

```
server (maybe GPU host)                  client (CPU / MPS / CUDA)
  seed s for (round,k,p)                   seed s for (round,k,p)   [identical seed on the wire]
       |                                          |
  canonical_perturbation(s, d)            canonical_perturbation(s, d)
  = torch.Generator(device='cpu')         = torch.Generator(device='cpu')
    .manual_seed(s); randn(...,cpu)          .manual_seed(s); randn(...,cpu)
       |  z_cpu  (bit-identical) <----- SAME BYTES -----> z_cpu  (bit-identical)
       v                                          v
  z.to(self.device) only for the math      z.to(self.device) only for forward pass
```

**Why CPU-canonical (C3 §2.3):** PyTorch does not guarantee `torch.randn(seed)` parity across CPU/CUDA/MPS or across machines. Generating `z` on the working device made a GPU server reconstruct a *different* `z` than a CPU client for the same seed, so `Σ g·z` aggregated along the wrong direction and the model "trained" while silently not learning (B1-C2). Generating on CPU then moving is ~free (the RNG draw is negligible vs two forward passes; C3 risk #3) and removes the entire divergence class. The dtype is pinned `float32` so it never follows a model's dtype and breaks golden parity (SPEC §3 dtype note).

### 6.3 DeComFL one round (sequence diagram)

```
Client(s)                         Server (coordinator + DeComFL strategy)        Backend (REST)
   |  RegisterClient(run_id, proto_ver, enrollment_token) ->                       |
   |  <- ACCEPTED(assigned_round)                                                  |
   |                                                                               |
   |  GetDeComFLConfig(run_id) -> current_seeds[K][P], rebuild_history,            |
   |     config{lr,mu,P,K}, torch_version, grad_estimate_method, golden_sha256     |
   |  <-------------------------------------------------------------               |
   |  (client validates torch_version + golden_vector_sha256; rejects on mismatch) |
   |                                                                               |
   |  -- local ZO fit: for k in K, for p in P:                                     |
   |        z = canonical_perturbation(seed[k][p], d).to(device)                   |
   |        g = (f(x+mu z) - f(x))/mu        # 2 forward passes, no_grad           |
   |     accumulate scalars; x reverts to start (ships only scalars)               |
   |                                                                               |
   |  SubmitGradientScalars(run_id, round, grads[K][P], num_examples) ->           |
   |    <- received, bytes_received (= K*P*8)                                       |
   |                          [parallel Heartbeat(run_id) on the OTHER stub]        |
   |                                                                               |
   |          server: wait until quorum>=min_clients OR deadline                   |
   |          server: strategy.aggregate_fit(round, results)  (§6.1)               |
   |          server: coordinator stores gradient_history[round]; evict_old_history|
   |          server: compute comm-cost metrics (uplink/downlink/scalars bytes)    |
   |                          POST /api/internal/runs/{runId}/results (RoundResultDto) -->
   |                                                       (best-effort, short timeout)
   |  (next round: GetDeComFLConfig returns the new round's seeds; absent clients   |
   |   receive rebuild_history to replay missed rounds via rebuild_model)          |
```

### 6.4 Round loop: deadline + minimum quorum (never hang — `04-API-CONTRACTS.md §10.3`, `02-TECH-STACK.md §18.2`)

```python
def run_round(self, round_idx: int) -> RoundOutcome:
    deadline = monotonic() + self.round_deadline_seconds
    received: list = []
    while monotonic() < deadline and len(received) < self.expected_clients:
        r = self.wait_for_next_result(timeout=deadline - monotonic())
        if r is not None:
            received.append(r)
        # consume should_stop: if a heartbeat marks a client dead, drop it from expected (A3-H1)
        self.prune_dead_clients()
    if len(received) < self.min_clients:
        return RoundOutcome.QUORUM_NOT_MET     # mark FAILED; do NOT hang forever (v1 hung, C1/R9)
    new_params = self.strategy.aggregate_fit(round_idx, received)
    return RoundOutcome.ok(new_params)
```

**Why (A3-H1, `02-TECH-STACK.md §18.2`):** v1's `should_stop` was hard-coded `False` and `is_client_alive` was never called, so a client that died mid-round hung the round until `num_rounds` exhausted. v2 wires `HeartbeatResponse.should_stop` (now a real field, `04-API-CONTRACTS.md §10.1.6`) and the deadline so a dead/straggling client cannot deadlock the federation.

### 6.5 Chunked upload symmetry (SPEC Bug3, `04-API-CONTRACTS.md §10.3`)

```
SENDER (client or server)                    RECEIVER
  blob = safetensors({'parameters': p,         accumulate chunk_data into BytesIO
                      'num_examples': n})       bounds-check cumulative <= max_payload_bytes (H5)
  sha = sha256(blob)                            on is_final_chunk:
  for chunk in split(blob, chunk_size):           assert codec in {'safetensors','lz4+safetensors'}
     yield ModelChunk(chunk_data=..., codec,        (else INVALID_ARGUMENT)
        compressed, total_bytes=len(blob),          if compressed: lz4.decompress
        sha256=sha, is_final_chunk=...)             assert sha256(reassembled) == sha (else INVALID_ARGUMENT)
                                                  params, n = chunks_to_parameters(reassembled, ...)
```

**Why (SPEC §3 Bug3, A3-C1):** v1 `parameters_to_chunks` saved a **bare** `OrderedDict` while `chunks_to_parameters` read `model_data['parameters']`, so every model without a tensor literally named `parameters` (i.e. every transformer/LLM) raised `KeyError: 'parameters'` — the LLM federation could not complete one round. v2 wraps symmetrically and the framing (`sha256`/`codec`/`compressed`/`total_bytes`) is **on the wire**, never inferred from `FEDLEARN_USE_COMPRESSION` (A3-C3). The payload is `safetensors`, not `torch.save` — no pickle on the wire (`04-API-CONTRACTS.md §10.3` codec whitelist).

### 6.6 Startup: manifest emission + determinism hooks (C3 §5.2)

```
fl_server.py:
  1. read env (§8): FEDLEARN_RUN_ID, FEDLEARN_RUN_TOKEN, FEDLEARN_BACKEND_URL, TRACEPARENT, run config (--seed/--K/--P/...)
  2. tracing.root_span = OTel extract(os.environ['TRACEPARENT'])  -> span "fl-run {run_id}"
  3. build initial model; initial_model_sha256 = sha256(safetensors(state_dict))
  4. manifest = build_manifest(run_id, config, initial_model_sha256, dataset_split_sha256)
        - assert manifest.rng_device == 'cpu'   (HARD INVARIANT; DB CHECK rng_device='cpu')
        - golden_vector_sha256 = read tests/fixtures/decomfl_golden/manifest.json sha
  5. POST /api/internal/runs/{run_id}/status (RunStatusReportDto: STARTING, grpcEndpoint)
        the backend persists the manifest into determinism_manifests on first status (03-DATA-MODEL.md §5.3)
  6. construct strategy (FedAvg|DeComFL) + coordinator + servicer; serve gRPC (TLS+mTLS unless dev)
  7. per round: aggregate -> POST .../results -> checkpoint upload -> POST .../checkpoint
  8. on done: POST .../finished (RunFinishedDto: SUCCEEDED, finalModelArtifactId, finalModelSha256)
```

### 6.7 Per-round communication-cost metrics (`telemetry/metrics.py`, B3 §6.2)

```python
def compute_round_comm_cost(strategy_name, K, P, num_clients, model_param_count, dtype_bytes=8):
    if strategy_name == "DeComFL":
        scalars_transmitted = K * P                    # per client; the O(K*P) proof
        uplink_bytes   = num_clients * K * P * dtype_bytes      # scalars client->server
        downlink_bytes = num_clients * K * P * 4               # seeds (int32-range) server->client
    else:  # FedAvg
        scalars_transmitted = None
        uplink_bytes   = num_clients * model_param_count * 4   # full model up (float32)
        downlink_bytes = num_clients * model_param_count * 4   # full model down
    return uplink_bytes, downlink_bytes, scalars_transmitted, model_param_count
```

**Why (B3 §6.2):** DeComFL's entire thesis is O(K·P) communication independent of `d`. v1 had no comm-cost column, so the platform could not demonstrate its own differentiator. These four numbers back the "bytes-per-round vs equivalent FedAvg full-model bytes" panel and are emitted on every `RoundResultDto`.

---

## 7. Data it owns

This unit **owns no database table**. Per `03-DATA-MODEL.md §1` (the two-plane model), raw training features/labels live only on the client and never enter any control-plane table; this unit holds run state in memory and emits DTOs the backend persists.

### 7.1 Tables this unit writes (indirectly, via REST DTOs — never direct SQL)

| Table (`03-DATA-MODEL.md`) | How this unit affects it | Columns it supplies |
|---|---|---|
| `round_results` (`§5.2`) | via `POST /api/internal/runs/{runId}/results` | `round_idx` (← `serverRound`), `loss`, `accuracy`, `uplink_bytes`, `downlink_bytes`, `scalars_transmitted`, `gpu_utilization`, `round_started_at`, `round_ended_at`. |
| `determinism_manifests` (`§5.3`) | via the manifest in the first `…/status` callback | all manifest fields; `rng_device='cpu'` (hard invariant, DB `CHECK`). |
| `model_artifacts` (`§5.2`) | via `POST /api/artifacts` after PUTting bytes | `sha256`, `size_bytes`, `kind` (`INITIAL`/`CHECKPOINT`/`FINAL`), `round_idx`. |
| `fl_runs` (`§5.2`) | indirectly: backend updates `status`/`round_idx`/`final_model_artifact_id` from this unit's callbacks | (read-only from this unit's perspective). |

### 7.2 In-memory structures (the run state machine)

| Structure | Type | Purpose |
|---|---|---|
| `DeComFL.global_params_flat` | `torch.Tensor` (CPU, float32) | the authoritative global model (flat). |
| `DeComFL.seed_history` | `dict[int, list[list[int]]]` | round → `[K][P]` seeds; bounded by `evict_old_history`. |
| `DeComFL.gradient_history` | `dict[int, list[list[float]]]` | round → `[K][P]` averaged (1/N) scalars; for client rebuild. |
| `DeComFL.client_last_round` | `dict[str, int]` | client_id → last participated round; the eviction floor. |
| `Coordinator.received_this_round` | `list[ClientUpdate]` | accumulates results until quorum/deadline. |
| `Coordinator.active_clients` | `dict[str, ClientHandle]` | registered clients + liveness. |
| Heartbeat abort latch | `threading.Event` | set when `should_stop` fires; training loop checks between local steps (A3-H1). |

---

## 8. Configuration & environment variables

### 8.1 Launch environment (set by the backend launcher; `04-API-CONTRACTS.md §13`, §14)

| Env var | Type | Default | Profile/mode | Meaning |
|---|---|---|---|---|
| `FEDLEARN_RUN_ID` | UUID string | (required) | all | the `fl_runs.id` this server serves; bound into every gRPC call. |
| `FEDLEARN_RUN_TOKEN` | `flrun_<...>` signed token | (required) | all | the per-run scoped token; set as `Authorization: Bearer ${FEDLEARN_RUN_TOKEN}` on every callback. |
| `FEDLEARN_BACKEND_URL` | URL | (required) | all | base for `/api/internal/...`; HTTPS/VPC-internal outside `dev`. |
| `FEDLEARN_PROJECT_ID` | UUID string | (required) | all | display/log convenience only; never asserted for authz. |
| `TRACEPARENT` | W3C header value | (optional) | all | OTel root-span parent (`00-traceid-spanid-flags`); extracted at startup (`§14`). |

### 8.2 Run config (passed as CLI flags by the launcher; sourced from `fl_runs.config` JSONB, `03-DATA-MODEL.md §5.4`)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--seed` | int | (required) | optimizer/perturbation seed → CPU-canonical RNG + manifest. **Distinct** from the data seed (C2 §2.3). |
| `--strategy` | `DeComFL`\|`FedAvg` | (required) | which `Strategy` to construct. |
| `--num-rounds` | int 1..1000 | (required) | total FL rounds. |
| `--min-clients` | int >=1 | (required) | minimum quorum to start/continue a round. |
| `--round-deadline-seconds` | int >=1 | `600` | per-round wall-clock deadline (no infinite hang). |
| `--decomfl-K` | int 1..1000 | (DeComFL) | local steps per round. |
| `--decomfl-P` | int 1..256 | (DeComFL) | perturbations per local step. |
| `--decomfl-eta` | float >0 | (DeComFL) | learning rate η. |
| `--decomfl-mu` | float >0 | (DeComFL) | smoothing radius μ. |
| `--grad-estimate-method` | `forward`\|`central` | `forward` | ZO estimator variant (B1-H2). |
| `--max-retained-rounds` | int | `100` | bounded-history cap (SPEC C-2). |
| `--model-path` | path | (required) | initial model artifact (safetensors). |

### 8.3 gRPC transport config

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `FEDLEARN_GRPC_TLS` | bool | `true` (false only when profile=`dev`) | TLS+mTLS default; plaintext only in `dev` (`02-TECH-STACK.md §3.2`, A3-C4). Refuse to boot insecure outside `dev`. |
| `FEDLEARN_GRPC_PORT` | int | (assigned by launcher) | listen port; clients dial via `fl_runs.grpc_endpoint`. |
| `max_payload_bytes` | int | `2 * 1024**3` (2 GB) | server-enforced cumulative cap on chunked uploads (A3-H5; `04-API-CONTRACTS.md §10.3`). |
| `protocol_version` | int | (constant, matches proto) | rejected at `RegisterClient` on mismatch (B4/R6). |
| `CHUNK_THRESHOLD_BYTES` | int | `300 * 1024**2` (300 MB) | FedAvg models above this stream; below go unary (reconcile A3-N5). |

> **Removed v1 env knob (A3-C3):** `FEDLEARN_USE_COMPRESSION` is **deleted**. Compression is carried on the wire (`ModelChunk.compressed` + `codec`), never inferred from env.

---

## 9. Error handling & edge cases

| # | Failure mode | Detection | Exact handling |
|---|---|---|---|
| E1 | Client registers with mismatched `protocol_version` | `RegisterClient` compares to server constant | Return `RegisterClientResponse.status=REJECTED` + the server's version; gRPC status `INVALID_ARGUMENT` if the field is absent. |
| E2 | Client `torch_version` ≠ manifest / golden vector mismatch | `GetDeComFLConfig` returns `torch_version` + `golden_vector_sha256`; client validates | Client WARN/reject (federation version gate, C3 §6.3). Server still serves; the determinism manifest records the discrepancy. |
| E3 | Chunked upload exceeds `max_payload_bytes` | cumulative byte counter in `SubmitModelUpdateStream` | Abort with gRPC `RESOURCE_EXHAUSTED` (A3-H5). |
| E4 | `codec` not in `{safetensors, lz4+safetensors}` | `chunks_to_parameters` validates | gRPC `INVALID_ARGUMENT` (`04-API-CONTRACTS.md §10.3`). |
| E5 | Reassembled `sha256` ≠ framed `sha256` | hash check on final chunk | gRPC `INVALID_ARGUMENT` (integrity); do not aggregate the corrupt update. |
| E6 | Round deadline hit with `< min_clients` | `run_round` quorum check (§6.4) | mark run `FAILED`, reason `quorum_not_met`; `POST .../finished` with `finalStatus=FAILED`, `errorMessage`. Never hang (C1/R9). |
| E7 | A client dies mid-round | heartbeat stub stops; `should_stop`/liveness | prune from `expected_clients`; if quorum still met, proceed; else E6 (A3-H1). |
| E8 | `aggregate_fit` looks up an unseeded round | `seed_history[server_round]` via `get_or_generate_seeds` | always go through `get_or_generate_seeds` (never raw index) so aggregation never depends on RPC ordering (A3-N1). |
| E9 | `evaluate()` returns `None` (no eval fn) | guard at every call site | `r = strategy.evaluate(...); if r is not None: loss, m = r` (A3-M2); never unpack `None`. |
| E10 | Telemetry callback fails (backend down / 5xx) | `requests` exception / non-2xx | best-effort: log WARN, continue the run. A telemetry failure NEVER crashes the run (`04-API-CONTRACTS.md §5` reasoning; B3 risk #9). Short timeout (e.g. 5 s). |
| E11 | Callback to a terminal run | backend returns `409 RUN_TERMINAL` | stop emitting; the run is being reconciled as terminal; log and exit cleanly. |
| E12 | Run token expired / mismatched | backend returns `401 RUN_TOKEN_INVALID` / `403 RUN_TOKEN_MISMATCH` | log ERROR; the run cannot report; the reconciler will eventually mark it FAILED on lease expiry. Do not retry indefinitely. |
| E13 | DeComFL `num_clients == 0` for a round | `aggregate_fit` head check | return `None`; coordinator treats as quorum-not-met (E6). |
| E14 | Process-global RNG mutation attempted | code review / test `TestNoGlobalRNGMutation` | forbidden: never call `np.random.seed` / `torch.manual_seed`; only instance-local generators (SPEC B-2). |
| E15 | `rng_device != 'cpu'` in manifest | startup assert | refuse to boot (the DB `CHECK (rng_device='cpu')` would also reject the row; fail fast here). |
| E16 | Bounded history evicts a round a slow client still needs | `evict_old_history` floor = `min(client_last_round)` | clients absent beyond `max_retained_rounds` must resync from a checkpoint (out of scope here; owned by the C1 reliability item — SPEC §5 C-2). |

---

## 10. Testing strategy

**Framework:** `pytest` (`framework/pyproject.toml`, `addopts = "-v --tb=short"`). Autouse `conftest.py` fixtures force CPU (`torch.cuda.is_available` monkeypatched `False`) and seed `torch`/`numpy`/`random` to `0` before each test (PLAN Task 0). Tests run **GPU-free** in CI; cross-device assertions are `skipif`-guarded.

| Test (name) | Pins | What it asserts |
|---|---|---|
| `TestRebuildTrajectoryEquivalence::test_server_trajectory_matches_client_rebuild` (T1) | Bug1 | A client that reconstructs every round via `rebuild_model` lands `torch.allclose(atol=1e-6)` on the server's aggregated trajectory. Fails while the server step is P× too large. **The canary.** |
| `TestGoldenVectors::test_canonical_perturbation_matches_committed_golden` (T2a) | Bug2 | `canonical_perturbation` is bit-exact against the committed `.npy`/sha256 fixture (always runs, CPU). |
| `TestGoldenVectors::test_canonical_perturbation_is_float32_on_cpu` | Bug2 | dtype is `float32`, device `cpu`, shape `(num_params,)`. |
| `TestServerClientPerturbationAgree::test_server_and_client_agree_for_same_seed` (T2b) | Bug2 | `DeComFL._generate_perturbation(seed)` == `ZerothOrderEstimator.generate_perturbation(seed)` for the same seed (structural single-path agreement). |
| `TestCrossDeviceParity::test_cuda_move_preserves_values` / `…_mps_…` | Bug2 | `z.to('cuda')`/`.to('mps')` round-trips equal; `skipif` no GPU (documentation of intent). |
| `TestChunkedRoundtrip::test_chunks_roundtrip_forced_multichunk_large_model` (T3a) | Bug3 | a >chunk-size model spans multiple chunks and reassembles `allclose`. |
| `TestChunkedRoundtrip::test_chunks_roundtrip_transformer_shaped_state_dict` (T3b) | Bug3 | a transformer-shaped state-dict (no tensor named `parameters`) roundtrips with `num_examples` intact — the exact case that `KeyError`'d in v1. |
| `TestOptimizedEqualsNaiveAggregate::test_aggregate_fit_matches_corrected_naive` (T4) | C-1 | the hoisted O(K·P) `aggregate_fit` equals a reference *corrected-naive* O(K·P·N) loop (the one that includes the 1/P fix) `allclose`. |
| `TestBoundedHistory::test_history_stays_bounded_across_many_rounds` (T5a) | C-2 | over 20 rounds with `max_retained_rounds=3`, `seed_history`/`gradient_history` hold ≤ 3 entries. |
| `TestBoundedHistory::test_rebuild_within_window_still_works` (T5b) | C-2 | a client missing N ≤ window rounds rebuilds correctly and matches the server. |
| `TestNoGlobalRNGMutation::test_constructor_leaves_global_rng_untouched` (B-2) | B-2 | constructing `DeComFL` does not move process-global torch/numpy RNG state. |
| `TestDeComFLStrategy::test_aggregate_fit_updates_global_params` (B-1) | B-1 | `seed_history` is round-keyed; use `get_or_generate_seeds`, not `.append`. |
| `test_codec` (codec) | A3-C4/H5 | bad `codec` → typed error; cumulative cap → error; sha mismatch → error. |
| `test_coordinator::test_round_deadline_no_quorum` | E6 | a round that misses quorum by the deadline returns `QUORUM_NOT_MET`, never hangs. |

**What is NOT tested here (deferred per SPEC §9):** C++ mobile parity (contract-gated by T2 fixtures only), checkpoint/resume + long-absence resync (C1 item), DP/robust-aggregation (B4/B1 item).

---

## 11. Build & run (this unit in isolation)

```bash
# --- install (editable) ---
cd /home/anurag/codebase/FedLearn-Platform/framework
python3 -c "import torch, numpy, grpc, safetensors"   # confirm deps present (3.12, torch 2.12.0)
pip install -e .

# --- generate proto stubs (buf is the single source of truth) ---
buf lint                       # lint fedlearn/v2/fedlearn.proto
buf breaking --against '.git#branch=main'   # breaking-change gate
buf generate                   # writes src/fedlearn/transport/generated/fedlearn_pb2*.py

# --- run the full test suite (GPU-free) ---
pytest                         # expect green; CUDA/MPS parity tests SKIPPED, not failed

# --- (re)freeze the RNG golden vectors (ONLY on an intentional torch bump) ---
python tests/fixtures/decomfl_golden/generate.py   # prints "Froze N golden vectors for torch <ver>"

# --- lint + type-check (repo gates framework/ with ruff + mypy strict) ---
ruff check src tests
mypy src

# --- run the server in isolation (dev profile; LOCAL_PROCESS launcher) ---
FEDLEARN_RUN_ID=$(uuidgen) \
FEDLEARN_RUN_TOKEN=flrun_dev \
FEDLEARN_BACKEND_URL=http://localhost:8081 \
FEDLEARN_PROJECT_ID=$(uuidgen) \
FEDLEARN_GRPC_TLS=false \
python fl_server.py --strategy DeComFL --seed 42 --num-rounds 3 --min-clients 1 \
  --decomfl-K 1 --decomfl-P 10 --decomfl-eta 0.001 --decomfl-mu 0.001 \
  --model-path ./examples/initial_model.safetensors
```

**Verify-in-isolation checklist:**
1. `pytest` is fully green; the three v1-red tests (2 serializer + `test_aggregate_fit_updates_global_params`) pass.
2. `grep -rn "torch.Generator(device=" src` returns only `estimators/perturbation.py` (CPU).
3. `grep -n "np.random.seed\|torch.manual_seed" src/fedlearn/strategies/decomfl.py` returns nothing.
4. `grep -rn "torch.save\|import flwr\|flwr_datasets\|import pika\|import ray" src` returns nothing.
5. The server emits `POST /api/internal/runs/{runId}/status` with the manifest within the first second of startup.

---

## 12. Reasoning & alternatives

| Decision | Why this | Rejected alternative & why (audit) |
|---|---|---|
| Keep the FL core custom (no `flwr`) | DeComFL + dual-heartbeat + chunking are genuine differentiators; the native C++ mobile client and DeComFL's scalar protocol do not fit Flower's Python `Parameters` model. | **Adopt Flower/FLARE substrate** — neither ships DeComFL nor a native C++ on-device client (A3 §6, B2-tech-stack §136-148). |
| Split `strategies/` (pure math) from `transport/` (grpc) | Makes aggregation unit-testable without a gRPC server; removes the RPC-ordering coupling. | **v1's mixed serializer/client/coordinator** — math un-testable; produced A3-N1 (RPC-ordered seed coupling) and the RED suite. |
| safetensors typed framing on the wire | Deletes the C1 `KeyError`, C3 env-inferred compression, and the `weights_only` pickle foot-gun in one move. | **Keep `torch.save` blob on the wire** — opaque, asymmetric save/load, RCE surface (A3-C1/C3, B1 Bug3). |
| CPU-canonical RNG (Approach A) | Bit-stable `z` across CPU/CUDA/MPS for a pinned torch; the only defensible design given PyTorch's no-cross-device-parity disclaimer; ~free cost. | **Counter-based Philox/Threefry (Approach B)** — bigger lift, changes perturbation values, needs B1 review; deferred (SPEC §2, C3 P3.12). **Generate on working device (v1)** — silently corrupts heterogeneous fleets (B1-C2, C3-R1). |
| Drop `* self.P` in `aggregate_fit` | Restores the paper's `1/P` averaging; makes the global trajectory consistent with every client's rebuild trajectory (the central correctness guarantee). | **Keep the `*P` "cancels in derivation"** — the bug rationalized as intent; 10× LR inflation; rebuild divergence (B1-C1, SPEC Bug1). |
| Hoist `z` to O(K·P) | `z` depends only on `(k,p)`, not the client; N× fewer `randn`/`mul` over the d-vector; numerically identical (T4). | **Per-(client,k,p) regeneration (v1)** — O(K·P·N·d) — the single biggest server-side scaling cliff for large models (A3-H4, B1-M5). |
| Instance-local RNG | Two strategies in one process no longer clobber each other; nothing reads process-global RNG. | **`np.random.seed`/`torch.manual_seed` (v1)** — process-global mutation corrupts in-process reproducibility (A3-M5, SPEC B-2). |
| Bounded seed/gradient history | Unbounded dicts grew forever; eviction floor = oldest round any known client could still rebuild from, capped at `max_retained_rounds`. | **Unbounded growth (v1)** — memory leak over long runs (SPEC C-2, B1 §47). |
| Determinism manifest emitted at startup, `rng_device='cpu'` invariant | Makes a run reproducible & auditable within a pinned environment (the honest product claim); DB `CHECK` enforces the CPU invariant. | **No run/lineage capture (v1)** — could not reproduce your own run (C3 §4). |
| Deadline + min-quorum round loop; wire `should_stop` | A dead/straggling client can never deadlock the federation. | **v1 hang-on-straggler** with dead `should_stop`/`is_client_alive` (A3-H1, C1/R9). |
| Per-round incremental telemetry (best-effort) | Live chart populates during training; comm-cost wedge measured per round; a telemetry failure never crashes the run. | **Batch-after-run POST (v1)** — chart empty until the end (B3, `04-API-CONTRACTS.md §5`). |
| Kill `async_coordinator.py`, prune deps to single digits, add `[project]` table | Dead RabbitMQ code imports uninstalled `pika` and shadows the real coordinator; ~40 v1 deps (incl. dead `ray`/`flwr`) bloat every client image and the Jetson ARM64 wheel surface. | **Keep v1 deps/`async_coordinator`** — A3-N3/N4, maintenance trap, ARM64 wheel risk. |

---

## 13. Build task checklist for the ~30B local model (ordered, dependency-respecting)

Each task is one file/feature with a done-condition. Order respects dependencies; TDD where a test exists in PLAN.

1. **`pyproject.toml` `[project]` table.** Add pinned single-digit deps (`torch==2.12.0`, `numpy`, `safetensors`, `grpcio`, `grpcio-tools`, `protobuf`, `lz4`, `pydantic`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `structlog`, `requests`) + `pytest`/`ruff`/`mypy` config. **Done:** `pip install -e .` resolves; no `flwr`/`ray`/`pika`.
2. **`proto/fedlearn/v2/fedlearn.proto` + `buf.yaml` + `buf.gen.yaml`.** Transcribe `04-API-CONTRACTS.md §10.2` verbatim; managed mode; python output to `src/fedlearn/transport/generated/`. **Done:** `buf lint` clean; `buf generate` writes `fedlearn_pb2*.py`.
3. **`estimators/perturbation.py`.** Implement `canonical_perturbation` (§5.4, verbatim from SPEC/PLAN Task2). **Done:** importable; returns CPU float32 `(num_params,)`.
4. **`tests/fixtures/decomfl_golden/generate.py` + freeze fixtures.** Generate `manifest.json` + `*.npy` (PLAN Task2). **Done:** `manifest.json` records `torch_version`; 3 `.npy` committed.
5. **`tests/test_perturbation.py`.** T2 golden + cross-device parity (skip-guarded). **Done:** non-skipped tests green.
6. **`transport/codec.py`** (replaces `serializer.py`). Implement safetensors framing + symmetric `parameters_to_chunks`/`chunks_to_parameters` (§5.5, SPEC Bug3). **Done:** `tests/test_codec.py` T3 multi-chunk + transformer-shaped roundtrip green; no `torch.save`.
7. **`core/types.py`.** Typed `Parameters`, `RoundConfig`, `ClientUpdate`, `GradientScalars`, `AggregationResult`, `RoundResultDto`. **Done:** mypy-strict clean.
8. **`strategies/strategy.py`.** The `Strategy` ABC (§5.1). **Done:** ABC importable; subclasses must implement all abstractmethods.
9. **`strategies/fedavg.py`.** Sample-weighted mean + cap-WARN (§5.3). **Done:** `test_fedavg_aggregator.py` green; WARN fires on cap.
10. **`strategies/decomfl.py`.** Implement constructor + `generate_seeds`/`get_or_generate_seeds`, `_generate_perturbation` (delegates to `canonical_perturbation`), `aggregate_fit` (corrected `1/P`, hoisted O(K·P)), instance-local RNG, `evict_old_history` (§5.2, §6.1). **Done:** T1, T4, T5, B-1, B-2 green; grep confirms no `*P`, no global RNG, only CPU generator.
11. **`estimators/zeroth_order.py`.** `generate_perturbation` delegates; `compute_gradient_scalar` forward+central (§5.4). **Done:** `test_zeroth_order.py` green; `TestServerClientPerturbationAgree` green.
12. **`client/decomfl_client.py`.** Salvage v1 `fit` + `rebuild_model` (client math is correct — **do not touch the `(eta/P)` step**). **Done:** rebuild matches server in T1.
13. **`client/heartbeat.py`.** Parallel heartbeat stub + `threading.Event` abort latch consuming `should_stop` (§6.4, A3-H1). **Done:** dead-client prune unit test green.
14. **`transport/channel.py`.** Secure channel/credentials factory; TLS+mTLS default, plaintext only `dev` (§8.3). **Done:** refuses insecure boot outside `dev`.
15. **`transport/servicer.py`.** Implement all 10 RPCs; bind every call to `run_id`; enforce framing rules (§10.3); wire `should_stop`. **Done:** `RegisterClient` rejects bad `protocol_version`; chunk caps/codec/sha enforced.
16. **`server/coordinator.py`.** Round state machine + deadline + quorum (§6.4); store `gradient_history`; call `evict_old_history`; guard `evaluate()==None` (A3-M2). **Done:** `test_coordinator` deadline/quorum green; no unpack-None.
17. **`core/manifest.py` + `core/hashing.py`.** Build the C3 §5.2 manifest; assert `rng_device='cpu'`; sha256 of safetensors + split arrays (§5.6). **Done:** manifest serializes to the `determinism_manifests` JSONB shape.
18. **`telemetry/metrics.py`.** Per-round comm-cost computation (§6.7). **Done:** DeComFL `scalars_transmitted = K*P`; FedAvg `None`.
19. **`telemetry/callbacks.py`.** Best-effort `requests` client for the four `/api/internal/runs/{runId}/*` POSTs with the run token; short timeout; never crash the run (E10). **Done:** non-2xx logs WARN and continues.
20. **`telemetry/tracing.py`.** OTel root span from `TRACEPARENT`; `traceparent` over gRPC metadata (§14). **Done:** span "fl-run {run_id}" parents to the JVM span when `TRACEPARENT` is set.
21. **`server/lifecycle.py` + `fl_server.py`.** Wire startup (§6.6): env → manifest → status callback → strategy+coordinator+servicer → per-round results/checkpoint → finished. **Done:** the §11 isolation run completes 3 rounds and POSTs `…/finished` with `SUCCEEDED`.
22. **Delete `server/async_coordinator.py`** (A3-N4). **Done:** file removed; no references; suite still green.
23. **Final gate.** `pytest` fully green (CUDA/MPS SKIPPED), `ruff check`/`mypy` clean, the §11 verify-checklist greps all pass. **Done:** all of §11's five checks satisfied.

---

*End of 11-LLD-fl-framework.md. All claims about existing v1 code cite `file:line` against `main-clean` via the A3/B1/C3 audit reports; all contract claims cite `03-DATA-MODEL.md` / `04-API-CONTRACTS.md` / `02-TECH-STACK.md` by section; all DeComFL correctness behaviour is the canonical fix specified in `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` and `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md`. Uncertainty (cross-language FP parity; `verify-before-use` pins) is flagged inline, never asserted.*
