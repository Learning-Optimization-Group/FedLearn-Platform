# FedLearn Platform — Master Wiki

> **Repository:** `FedLearn-Platform`  
> **Last Updated:** 2026-08-13

This is the top-level technical wiki for the **FedLearn Platform** — a privacy-preserving, distributed machine learning system that enables federated learning across heterogeneous hardware. Use this page as your starting point and navigate into any subsection for in-depth documentation.

> **`flwr` is gone from the whole repo (2026-08-12, `65048b6`).** It was never used for FL semantics — only to cut one CIFAR-10 IID shard — but `flwr-datasets` pulled in `flwr` 1.20.0, which capped `cryptography<45.0.0` (the **SE-22** residual, blocking the framework's own `>=46.0.6` security floor) and `protobuf<5.0.0` (which made the FoT path uninstallable in the backend/client lockfiles, since `fot_pb2.py` is 5.29.0 gencode). Dropping it cleared both caps: `backend/fl-platform-api/requirements.txt` now pins `cryptography==46.0.7` and `protobuf==5.29.5`. The shard is now implemented natively in [`fl-runtime/recipes.py`](#fl-runtime) (`_cnn_iid_shard`, `CNN_SHUFFLE_SEED = 42` — the `FederatedDataset` default it replaced), and the replacement was verified **byte-identical per partition** before the swap, so CIFAR-10 results recorded before and after remain comparable. `fl-runtime/tests/test_requirements_security_floors.py::test_flwr_stays_out_of_the_lockfiles` is the regression guard. Residual `flwr` *mentions* survive as comments and as a `flwr_datasets` hidden-import in `client-docker/packaging/fedlearn-client.spec` — neither is an installed dependency. (A dead `framework/requirements.txt.bak` still carrying the old `flwr==1.20.0` pin was deleted on 2026-08-13; the guard scans the four live lockfiles only.)
>
> **Ledger design-system redesign (2026-07-17, `2c50672` + `fdd8a79`):** the frontend, the desktop renderer, and the mobile client adopted the **Ledger** design system — navy structural ink (`#1C314D`, hover `#14243A`) on quiet paper surfaces (`#F6F3EE` canvas, `#FFFFFF` cards, `#191A1C` ink, `#6B6760` muted), with a single Hanken Grotesk type family for both sans and display (JetBrains Mono for logs/ids), light-first; the dark family is navy-dark (`#0B1622` canvas, `#4F8AC9` accent) and stays wired. Ledger superseded **Ember** (warm canvas + burnt orange + Bricolage Grotesque, 2026-06), which had superseded "Instrument" — see [`frontend/UI_and_Components.md`](./frontend/UI_and_Components.md). `design/tokens.json` is the single source of truth; `design/build-tokens.mjs` generates the per-platform outputs (web CSS vars, desktop `tokens.css`, `mobile_client/src/theme/tokens.generated.ts`) and CI has a "Design tokens in sync with source of truth" step. Bricolage Grotesque is Ember-era and survives only in `design/brand/*.html` comparison assets. "Ledger" is the name of the **design system / theme**; the platform itself is still **FedLearn** (a product-domain rename is a separate, in-progress effort). This wiki ships an HTML rendering under [`html/`](./html/index.html), **regenerated on 2026-08-13 from the current Markdown and rethemed onto Ledger** (`wikis/html/build.py` — its palette is mirrored from `design/tokens.json`, Bricolage is dropped, and the `fl-runtime/` section is in the nav). It is generated output: re-run `python3 wikis/html/build.py` after editing any page here rather than hand-editing the HTML.

---

## Platform Overview

FedLearn is built around four core components — backend, frontend, framework, desktop — plus the **FL runtime**, a **mobile client** and a **containerised client**, for **seven deployable units** in total. They work together to provide an end-to-end federated learning experience:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FedLearn Platform                                 │
│                                                                          │
│  ┌──────────────┐    REST / WebSocket    ┌───────────────────────────┐   │
│  │   Frontend   │◄──────────────────────►│      Backend (API)        │   │
│  │  React SPA   │                        │  Spring Boot 3 / Postgres │   │
│  └──────────────┘                        └──────────┬────────────────┘   │
│                                                     │ shells out to      │
│  ┌──────────────┐    gRPC (Python)                  │ run_fl_server.sh   │
│  │   Desktop    │◄──────────────────────►┌──────────▼────────────────┐   │
│  │  Electron 42 │   PyInstaller / Docker │  fl-runtime (executables) │   │
│  └──────────────┘                        │  fl_server.py / client.py │   │
│                                          │  recipes.py / init_model  │   │
│                                          └──────────┬────────────────┘   │
│                                                     │ import fedlearn    │
│                                          ┌──────────▼────────────────┐   │
│                                          │  Framework (Python lib)   │   │
│                                          │  fedlearn + PyTorch gRPC  │   │
│                                          └───────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

| Component | Primary Language | Responsibility |
|---|---|---|
| [**Backend**](#backend) | Java (Spring Boot 3) | REST API, JWT auth, project lifecycle, FL orchestration, WebSocket log streaming |
| [**Frontend**](#frontend) | TypeScript (React 19 + Vite) | Web dashboard, real-time monitoring, project management, auth flows |
| [**Framework**](#framework) | Python (PyTorch + gRPC) | Core FL **library** (`import fedlearn`) — coordinator, strategies (FedAvg / FedProx / FedOpt / FedLoRA / DeComFL / robust), the safetensors wire, the in-process simulator |
| [**FL runtime**](#fl-runtime) | Python (scripts) | The **executable layer** the backend actually runs — `fl_server.py`, `client.py`, `init_model.py`, `infer.py`, `recipes.py`, `data.py` + their `run_*.sh` wrappers. Owns the model-recipe catalog and the training arms. |
| [**Desktop**](#desktop) | TypeScript (Electron 42) | Local training orchestrator, hardware detection, Docker/PyInstaller execution |
| [**Mobile**](#mobile) | React Native 0.80 + native C++ (ExecuTorch) | On-device FL client; runs the DeComFL zeroth-order path natively in C++ via a TurboModule bridge. **Android only today** — the iOS native wiring is still a libtorch scaffold (MO-14) |
| [**Client (Docker)**](#client-docker) | Docker (multi-arch) | Containerised FL client; packages the framework **and** the FL runtime for Jetson / CUDA / CPU deployments |

> **The split that trips people up:** `framework/` is the pip-installable *library*; `fl-runtime/` is the *executable layer* that imports it (`import fedlearn as fl`) and is what the backend shells out to. The backend never runs `framework/` directly. Each has its own pytest suite and its own CI job.

---

## How the Components Connect

### Training Round — End-to-End Flow

```
User (Browser / Desktop)
   │
   │  1. Create / configure a training project
   ▼
Backend (Spring Boot API)
   │
   │  2. Persist project config (recipe, strategy, training arm) to PostgreSQL
   │  3. Spawn the Python FL server as a local process (`fl-runtime/run_fl_server.sh`)
   ▼
fl-runtime — `fl_server.py` (drives the framework's coordinator over gRPC)
   │
   │  4. Broadcast global model parameters to all connected clients
   │     (only the arm's federated subset; non-float32 buffers are withheld)
   ▼
fl-runtime — `client.py` (Python / gRPC)   ◄── invoked by Desktop or client-docker
   │
   │  5. Train locally on private data under the project's training arm
   │  6. Upload the update — safetensors-encoded, chunked when streaming
   ▼
fl-runtime — `fl_server.py` → framework `FLCoordinator` + `Strategy`
   │
   │  7. Aggregate updates (FedAvg / FedProx / FedOpt / FedLoRA / DeComFL /
   │     robust) → new global model
   │  8. Emit training logs to stdout
   ▼
Backend (WebSocket bridge)
   │
   │  9. Capture Python stdout; route via STOMP topics
   ▼
Frontend / Desktop (real-time log panel)
   │
   │  10. Render live training metrics to the user
   ▼
Backend
      11. Persist final model checkpoint & round results to the database (PostgreSQL)
```

### Authentication Flow

```
Browser / Desktop / Mobile
   │  POST /api/auth/login  (username + password)
   ▼
Backend  →  validates credentials  →  issues an audience-scoped JWT (SE-20)
   │
   ├── Frontend: set as the HttpOnly `jwtToken` cookie. The browser sends it; Axios only
   │             sets withCredentials: true. NO Bearer header, NO localStorage, no
   │             JS-readable token
   ├── Desktop:  JWT stored via Electron safeStorage (OS keychain); never exposed to the renderer
   └── Mobile:   Authorization: Bearer, accepted only alongside X-FedLearn-Client (SE-9)
```

The cookie's flags are profile-driven, not fixed: `app.auth.cookie.secure` defaults to `false`
(`dev`, `ec2demo`, and the base profile — the demo runs plain HTTP) and to `true` under
`production`, and `app.auth.cookie.same-site` is `Lax` everywhere **except `production`, which
defaults to `Strict`**.

`GET /api/auth/me` is a deliberate silent 401 probe — `axiosConfig.ts` lists it in
`SILENT_401_ENDPOINTS` so bootstrap does not redirect-loop; a 401 anywhere else dispatches an
`authError` window event and sends the user to `/login`. The STOMP WebSocket at `/ws-logs` reuses
the same cookie through `JwtHandshakeInterceptor`. Authorities are reloaded from the database on
every request (`JwtAuthenticationFilter` → `CustomUserDetailsService`), so a role change takes
effect immediately without re-login.

### FL Server Provisioning — Local Process

```
Backend (FlServerManager)
   │
   └── LOCAL PROCESS  (the only supported path)
         reserves a free port from the 50000-50010 range
         FlServerProcessRunner  →  bash fl-runtime/run_fl_server.sh --port <N>
         stdout + stderr piped  →  WebSocket log streaming
         ProcessHandle tracked per project id  →  /stop terminates it
```

> **Managed cloud tasks are not available.** An AWS ECS/Fargate orchestration mode existed
> historically, but the AWS SDK — and with it the implementation — was removed; the backend
> carries no AWS dependency and no task-orchestration code. The single remaining setting,
> `ecs.cluster-name` (blank by default), exists only to be **rejected**: `FlOrchestrationModeValidator`
> throws at boot, in **every** profile, if it is set to a non-blank value. Managed-task orchestration
> is deferred to **OP-12**. The only supported deployed architecture — including under the
> `production` profile — is the hardened single VM running FL servers as local processes.

### In-Process Simulation — the experiment path that has no ports

The port pool above caps the deployed platform at **11 concurrent federations**, which makes a
thousand-client experiment inexpressible — a deployment constraint leaking into the science.
`framework/src/fedlearn/simulation/` (`federation.py`, `partition.py`, `rng.py`, landed
2026-08-12 in `d4e91f3`) is the answer: `SimulatedFederation` drives the **production**
`FLCoordinator` and the **production** `Strategy` objects by direct method call — no gRPC
channel, no TCP port, no subprocess.

```
SimulatedFederation(strategy, client_factory, num_clients, clients_per_round, seed,
                    wire_in_the_loop=0.0, dropout_rate=0.0, device="cpu", …)
   │
   ├── partition.py   iid_partition / dirichlet_partition / shard_partition /
   │                  pathological_partition  (+ partition_report)
   ├── rng.py         ClientRng / RunRng, derived from (seed, client_id, round)
   └── direct calls → FLCoordinator.start_round / get_global_model_for_client /
                      submit_client_update / resolve_round_incomplete  ← production code, unmodified
```

What it buys, and what it costs:

- **Same aggregation, same defenses.** Only the transport is elided; nothing here re-derives
  FedAvg. The coordinator was already transport-free (gRPC lives entirely in the servicer that
  wraps it), so this module is a driver loop.
- **The wire can be put back.** `wire_in_the_loop` routes a configurable fraction of client
  updates through the real deterministic safetensors encode/decode. Running with it off is only
  defensible because the suite asserts off and on agree bit-for-bit.
- **Determinism is a property, not an accident.** Client selection, dropout, wire routing and
  local training all derive from `(seed, client_id, round)` alone, and torch's global RNG is
  scoped and restored around every client — so a client's trajectory does not depend on how many
  peers exist.
- **No wall-clock dependence.** A round with modelled dropout resolves immediately via
  `FLCoordinator.resolve_round_incomplete` rather than sleeping out the deployed server's
  120-second deadline.
- **Memory scales with `clients_per_round`, not `num_clients`** — clients are built per round and
  released. The corollary is that `client_factory` is called once per participation, so it must
  close over pre-partitioned indices rather than re-reading a dataset.
- **Measured to 5,000 clients.** `research/results/simulation/scale_m4max.json` records the grid
  10 / 50 / 100 / 500 / 1,000 / 2,000 / 5,000 clients × seeds 0, 1, 2 — 21 cells, all run, none
  skipped — at 20 rounds, 10% participation, on macOS arm64 (14 CPUs, torch 2.9.1, CPU device).
  The run's own recorded caveat is the honest reading: it measures **simulator** scaling on a
  deliberately tiny model (linear 8 → 4, synthetic data), so a cell's wall-clock is a **floor**
  on what a real architecture would cost, never an estimate of it.

This is an experiment/research path. It does not replace the deployed gRPC server, and no backend
orchestration path uses it.

> **A note on the `research/` paths cited in this wiki.** `research/` is **gitignored and
> untracked** (`.gitignore:166`), so it exists only in a working copy where those runs were
> executed — a fresh clone will not have it. The citations are kept because they name the exact
> file that backs a number and the harness that regenerates it; treat them as provenance, not as
> paths you can expect to `ls` after cloning.

---

## Component Wikis

### Backend

> **Path:** [`wikis/backend/`](./backend/README.md)  
> **Stack:** Java 21, Spring Boot 3, Spring Security 6, PostgreSQL 16 (every profile), STOMP WebSocket

The backend is the central control plane. It owns the REST API, user authentication, project management, and acts as the bridge between the web clients and the Python FL processes.

| Document | Description |
|---|---|
| [Architecture & Core Concepts](./backend/01_architecture_overview.md) | Directory structure, domain models (Projects, Results, Logs), technology stack |
| [Security & Authentication](./backend/02_security_and_auth.md) | Stateless JWT filter chain, WebSocket handshake security, internal API key mechanism |
| [Project Management Lifecycle](./backend/03_project_management.md) | `ProjectService`, `ProjectController`, round configuration, model initialization |
| [Federated Orchestration](./backend/04_federated_orchestration.md) | `FlServerManager` — port reservation, local FL-server process spawn via the `FlServerProcessRunner` seam, `ProcessHandle` tracking, and the fail-closed `ecs.cluster-name` boot guard |
| [WebSocket Log Streaming](./backend/05_websocket_logs_streaming.md) | Stdout capture → STOMP topics → frontend real-time observability |
| [Identity, Multi-Tenancy & Audit](./backend/06_identity_multitenancy_and_audit.md) | **Present on this branch** (`V4`–`V7` migrations): organizations + org/project memberships, platform/org/project role model (`PlatformRole` enum), org-scoped data isolation (`OrgScopeFilter`), `@Auditable` audit trail. Supersedes the original coarse `users.role IN (USER, ADMIN)` model. |
| [Content-Addressed Model Artifact Registry](./backend/07_artifact_registry.md) | The versioned, content-addressed registry (`artifact_blobs` / `model_artifacts` / `artifact_lineage`) that superseded the single overwritable `.npz`; write path, registry-first inference/warm-start read path, HTTP surface, `V12`/`V18` migrations |

**Key cross-component interfaces:**
- Exposes `POST /api/projects/{id}/start` → spawns `fl-runtime/run_fl_server.sh` as a local process.
- Streams logs to Frontend via STOMP topic `/topic/logs/{projectId}`.
- Desktop authenticates against `POST /api/auth/login` before initiating training.
- Authorization is the layered identity model — `PlatformRole` (platform), `OrgRole` (organization), `MembershipRole` (project) — committed in the `V4`–`V7` migrations. Organizations, org/project memberships, org-scoped isolation, and the `@Auditable` audit trail are all present; see [Identity, Multi-Tenancy & Audit](./backend/06_identity_multitenancy_and_audit.md). The original coarse `users.role IN (USER, ADMIN)` column (`V2`) has been superseded.
- The schema is owned by **Flyway**, not JPA (`ddl-auto=validate` in every profile but `test`). The **highest committed migration is `V23`** — `V22__project_training_arm.sql` adds `projects.training_arm VARCHAR(32) NOT NULL DEFAULT 'FULL'` plus the `chk_projects_training_arm` CHECK, and `V23__training_arm_ova_lp.sql` widens that CHECK to `('FULL','FROZEN_HEAD','OVA_LP')`. Note `ls` sorts the migration directory lexicographically, so `V5`–`V9` appear *after* `V21` — the last line is not the highest version.

---

### Frontend

> **Path:** [`wikis/frontend/`](./frontend/README.md)  
> **Stack:** React 19, TypeScript, Vite, Tailwind CSS, Axios, STOMP over WebSocket (`@stomp/stompjs`)

The frontend is a single-page application providing the primary web-based control plane. It handles project management, live training monitoring, and user authentication entirely in the browser.

| Document | Description |
|---|---|
| [Architecture & State Management](./frontend/Architecture.md) | Tech stack overview, project structure, global contexts |
| [Routing & Authentication](./frontend/Routing_and_Auth.md) | React Router config, protected routes, HttpOnly cookie auth |
| [API & Services](./frontend/API_and_Services.md) | Axios configuration, interceptors, WebSocket integration, log store |
| [UI & Components](./frontend/UI_and_Components.md) | Design system, legacy vs. V2 components, reusable patterns |

**Key cross-component interfaces:**
- Connects to Backend REST API via Axios (`/api/**`).
- Subscribes to `STOMP /topic/logs/{projectId}` for the FL server's live stdout.
- Renders the project-creation picker from `GET /api/model-recipes` — including the training-arm choice and its measured trade-off, where a recipe offers one.
- No direct connection to the FL processes or Desktop — all routing via Backend.

---

### Framework

> **Path:** [`wikis/framework/`](./framework/README.md)  
> **Stack:** Python (declared floor 3.10+, CI tests 3.12), PyTorch, gRPC / Protocol Buffers — **custom FL engine, no Flower anywhere**

The framework is the heart of the platform — a standalone Python library (`fedlearn`) that implements the full federated learning lifecycle using gRPC for communication and PyTorch for model training. It is a *library*: the backend does not run it directly, it runs [`fl-runtime/`](#fl-runtime), which imports it. The Java-side orchestration package (`orchestration/`, class `FlServerManager`) was renamed from the legacy `flower` / `FlowerServerManager` name (DA-12).

**No Flower, in any sense, anywhere.** No Flower server/client/strategy semantics are used — the protobuf contract is entirely custom — and as of `65048b6` (2026-08-12) `flwr` / `flwr-datasets` are not dependencies of *any* unit either. The last touchpoint was one CIFAR-10 IID shard, which now lives natively in `fl-runtime/recipes.py`:

```python
CNN_SHUFFLE_SEED = 42     # == the flwr FederatedDataset default this replaced

def _cnn_iid_shard(partition_id, num_shards=CNN_NUM_PARTITIONS, seed=CNN_SHUFFLE_SEED):
    train = hf_datasets.load_dataset("cifar10")["train"].shuffle(seed=seed)
    return train.shard(num_shards=num_shards, index=partition_id, contiguous=True)
```

That is not a re-implementation by guesswork: `FederatedDataset` defaults to `shuffle=True, seed=42` and shuffles each split *before* partitioning, and `IidPartitioner.load_partition(i)` is precisely `shard(num_shards=N, index=i, contiguous=True)` — both read off the installed `flwr-datasets` 0.5.0 source, then verified empirically. The equivalence check ran the two partitioners side by side and compared **per partition**, verdict `EQUIVALENT`, recorded at `research/results/reproducibility/flwr_shard_equivalence.json` from the committed harness `research/benchmarks/verify_flwr_shard_equivalence.py`. Honest scope: the recorded run covers the *mechanism* level (synthetic 1,000-row dataset, 10 partitions, seed 42, all identical); the end-to-end CIFAR-10 level is marked **not run** in that file (`--full` was not requested; it needs the ~170MB download). So the shard logic is proven identical and the full-dataset instantiation of it is argued, not measured.

**Registered aggregation strategies** (`framework/src/fedlearn/server/strategy_factory.py`; names matched case-insensitively with hyphens/underscores ignored, so `"fed_avg" == "FedAvg"`):

| Name | Class | Module |
|---|---|---|
| `fedavg` | `FedAvg` | `server/strategy.py` |
| `fedprox` | `FedProx` | `server/strategy.py` |
| `fedopt` | `FedOpt` | `server/strategy.py` |
| `fedlora` | `FedLoRA` | `server/strategy.py` |
| `decomfl` | `DeComFL` | `server/decomfl_strategy.py` |
| `robust` | `RobustAggregator` | `server/robust_aggregation.py` |

Adding a strategy is a one-line registry entry, not another `if`/`elif` branch. The factory lives in its own module to break a circular import (`strategy.py` must not import `decomfl_strategy`).

| Document | Description |
|---|---|
| [Architecture & Package Overview](./framework/01_architecture_overview.md) | Module map, package layout, component interaction |
| [gRPC Communication Layer](./framework/02_grpc_communication.md) | Proto definitions, serialization, streaming, TLS, retry logic |
| [Server Internals](./framework/03_server_internals.md) | `start_server`, `FLCoordinator`, round lifecycle, heartbeat management |
| [Client Internals](./framework/04_client_internals.md) | `Client` ABC, `GrpcClient`, polling loop, large model streaming |
| [Aggregation Strategies](./framework/05_strategies.md) | `Strategy` ABC, `FedAvg` deep-dive, weighted aggregation, extensibility |
| [DeComFL — Dimension-Free FL](./framework/06_decomfl.md) | Zeroth-order gradient estimation, seed/gradient protocol, Algorithms 3 & 4 |
| [Data Partitioning & Non-IID](./framework/07_data_partitioning.md) | Dirichlet distribution, heterogeneous splits, practical setup |
| [Examples Walkthrough](./framework/08_examples.md) | End-to-end traces for MNIST, LLM, and ECG federation |
| [Developer Guide](./framework/09_developer_guide.md) | Custom strategies, custom clients, testing, contributing |

**Key cross-component interfaces:**
- Consumed by **`fl-runtime/`** as a library (`import fedlearn as fl`); the **Backend** reaches it only through that layer — `FlServerManager` shells out to `fl-runtime/run_fl_server.sh` via the `FlServerProcessRunner` seam.
- Consumed by **Desktop** as a PyInstaller-bundled native client (entry binary `fedlearn-client` / `fedlearn-client.exe`, per `fedlearn-desktop/src/shared/bundleVariants.ts`); in dev mode the desktop falls back to system `python3` running `fl-runtime/client.py`.
- Consumed by **`client-docker`** image for containerised Jetson / CUDA deployments.
- Ships `fedlearn.simulation` (see [In-Process Simulation](#in-process-simulation--the-experiment-path-that-has-no-ports)) — it reuses the production coordinator and strategies and has **no dedicated wiki page yet**; the module docstrings in `framework/src/fedlearn/simulation/` are currently the reference.

---

### fl-runtime

> **Path:** [`wikis/fl-runtime/`](./fl-runtime/README.md)  
> **Stack:** Python scripts + `run_*.sh` / `.bat` wrappers; imports `framework/` as a library

`fl-runtime/` is **the executable FL layer** — the unit the backend actually shells out to, and the one most easily missed because it is neither the Java control plane nor the pip-installable library. `fl-runtime/client.py` does `import fedlearn as fl`; the backend never invokes `framework/` directly.

| Document | Description |
|---|---|
| [FL Runtime Overview](./fl-runtime/README.md) | What the unit is, how it relates to `framework/`, how the backend resolves it |
| [Entry Points](./fl-runtime/01_entry_points.md) | `fl_server.py`, `client.py`, `init_model.py`, `infer.py` and the `run_*.sh` wrappers |
| [Recipe Catalog](./fl-runtime/02_recipe_catalog.md) | `recipes.py` — the seven catalog recipes, the training arms, `--describe` |
| [Training Arms](./fl-runtime/03_training_arms.md) | `FULL` / `FROZEN_HEAD` / `OVA_LP` end to end — objectives, `supported_arms`, `arm_tradeoff.json`, provenance |
| [The Federated Set](./fl-runtime/04_the_federated_set.md) | Which tensors cross the wire: the non-float32 exclusion, subset federation, save-time merge, eval strictness |

| File | Role |
|---|---|
| `fl_server.py` / `run_fl_server.sh` (`.bat`) | FL server entry point — resolves the strategy and the training arm, builds the initial model, runs the round loop, prints the eval card |
| `client.py` / `run_clients.sh` | The **one canonical FL client** (DA-5) — used by the Docker image, the desktop PyInstaller bundle, and dev/local runs alike. Takes `--project-id` / `--server-address` / `--partition-id` (there is no `--client-id`) |
| `recipes.py` / `run_recipes.sh` | The model-recipe catalog and the training arms; `--describe` is deliberately torch-free so the backend can serve `GET /api/model-recipes` cheaply |
| `init_model.py` / `run_init_model.sh` | Builds and persists the initial global model for a project |
| `infer.py` / `run_infer.sh` | Inference entry point behind the Model Playground |
| `data.py`, `config.py`, `device.py`, `models.py` | Dataset loaders, per-type config, device selection, model definitions |
| `fl_fot_server.py` / `run_fot_server.sh` | The FoT (Federation over Text) server — additive and orthogonal to the gradient path |

The backend resolves these paths from `application.properties` (`python.script.fl-server.path`, `python.script.recipes.path`, `python.script.init-model.path`, `python.script.infer.path`, `python.executable.path`); the relative defaults resolve from the backend's working directory, which is why the backend is normally launched from `backend/fl-platform-api/`.

`fl-runtime/` has its **own** pytest suite (`cd fl-runtime && python -m pytest -q`) and its own CI job, run with `FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1` — a *skipped* test fails that job (TE-10). Its `pytest.ini` deselects `-m slow`, which is a deselection and not a skip.

#### Training arms

An **arm** is a first-class, persisted property of a project: it says which parameters a run trains and federates **and under which objective**.

```python
TRAINING_ARMS  = ("FULL", "FROZEN_HEAD", "OVA_LP")
ARM_OBJECTIVES = {"FULL": "cross_entropy", "FROZEN_HEAD": "cross_entropy", "OVA_LP": "one_vs_all"}
DEFAULT_ARM    = "FULL"     # an omitted arm resolves to FULL, so existing projects are unchanged
```

| Arm | Trains | Objective |
|---|---|---|
| `FULL` | every parameter; the whole model rides the wire | cross-entropy |
| `FROZEN_HEAD` | the head only (per-recipe module prefixes); the backbone is frozen, so the wire carries the head alone | cross-entropy |
| `OVA_LP` | the *same* parameters as `FROZEN_HEAD` | one-vs-all — C independent binary classifiers instead of one softmax (arXiv:2511.05028) |

`OVA_LP` is the first arm that differs from another arm in its **objective** rather than in its parameter subset, which is why the objective is part of the arm and part of the provenance stamp: without it, an `OVA_LP` and a `FROZEN_HEAD` result would be indistinguishable. `recipes.arm_stamp()` records `{recipe, arm, objective, trainable_prefixes}` on every eval card — carrying the prefixes as well as the name, because two runs can share an arm name while freezing different modules. Honest caveat recorded in the recipe's `arm_notes`: the OvA-LP paper's two-stage schedule is **not** implemented, so results read as OvA heads on a frozen encoder, not as a reproduction.

Each recipe declares `supported_arms` and, per arm, `trainable_spec[arm]` (the module-name prefixes that stay trainable; `None` means all). The flow is end-to-end: frontend picker → Java `TrainingArm` enum + DTO validation → `projects.training_arm` (migrations `V22`/`V23`, CHECK-constrained) → `--training-arm` on `fl_server.py` and `client.py`. The two sides of that flow validate different things, and it is worth knowing which: the Java layer bounds only the *vocabulary* — a `@Pattern(regexp = "FULL|FROZEN_HEAD|OVA_LP")` on `CreateProjectRequest`/`StartProject`, the `TrainingArm` enum, and the `chk_projects_training_arm` CHECK — while **whether the selected recipe actually supports that arm is checked by `recipes.validate_arm()` on the Python side, at FL-server spawn** (`fl_server.py`) and again in `client.py`. The recipe catalog is the authority (`TrainingArm.java` and `FlServerManager` both say so); project creation does not consult it. `V22TrainingArmMigrationTest` asserts every Java enum constant is accepted by the CHECK so the enum and the constraint cannot drift. `fl-runtime/arm_tradeoff.json` carries the per-recipe measured trade-off the picker displays — keyed **by recipe**, because an earlier version attached one chest X-ray measurement to every dual-arm recipe and so advertised a pneumonia figure on a CIFAR-10 recipe.

#### Model-recipe catalog

A recipe bundles `{architecture + dataset loader + input transform + class labels + input kind + supported arms}` under a `key`. `RECIPE_METADATA` holds **seven** catalog entries, served to the frontend picker via `recipes.py --describe` → `GET /api/model-recipes`:

| Key | Display name | Input | Supported arms |
|---|---|---|---|
| `PNEUMONIA_CNN` | Pneumonia Chest X-ray | image | `FULL`, `FROZEN_HEAD` |
| `CNN` | Image classifier (CIFAR-10) | image | `FULL`, `FROZEN_HEAD` |
| `CIFAR_RESNET18` | Image classifier (CIFAR-10, pretrained ResNet-18) | image | `FULL`, `FROZEN_HEAD`, `OVA_LP` |
| `MLP` | ECG heartbeat (Normal/Abnormal) | vector | `FULL` |
| `TRANSFORMER` | Text classifier (OPT-125M) | text | `FULL` |
| `LLM_LORA` | Text LLM (LoRA fine-tune) | text | `FULL` |
| `TINYNET_GOLDEN` | On-device DeComFL demo (TinyNet) | vector | `FULL` |

`CIFAR_RESNET18` is the first recipe to declare `pretrained` weights explicitly (`torchvision`, `ResNet18_Weights.IMAGENET1K_V1`, 1000-class head discarded) — declared rather than implicit so a result can say *which* backbone produced it, which is the first question anyone asks of a frozen-arm number.

Two further recipes exist in `recipes.py` but are deliberately **outside** the catalog and `--describe`: `BLOOD_CNN` and `FROZEN_DEMO`. They stay dispatchable-but-not-selectable.

> **The registry is now the dispatch path, not only the catalog** (DA-14 Ph3.1). `init_model.py`'s model builder is fully data-driven — it upper-cases the key and returns `recipes.get_recipe(model_type).build_model(device, ...)` with no per-type branch at all — and `infer.py` does the same through `build_for_inference`. `catalog_keys()` derives the accepted `--model-type` values from `RECIPE_METADATA`, so a new catalog recipe is automatically an accepted model type with no argparse edit. Older documentation describing an `if`/`elif` chain in `init_model.py` that falls through to `raise ValueError("Unsupported model architecture")` is stale.
>
> The residue is in **`client.py`**, which still has a `USE_MLP` / `USE_LLM` / `USE_PNEUMONIA` / `USE_LLM_LORA` / `USE_DERIVED` / else chain — but every branch of it already calls `recipes.get_recipe(...)`; the chain selects *which* key and *which* loader kwargs, not how the model is built, and the else branch honours `MODEL_TYPE` whenever it names a registry recipe. The training arm is applied **after** the whole chain (`apply_declared_arm`), deliberately: it used to live inside one branch, so a `FROZEN_HEAD` pneumonia run trained its entire backbone while reporting itself as frozen. Correctness must not depend on the order of a build chain.

**Key cross-component interfaces:**
- Executed by **Backend** through `FlServerProcessRunner` → `run_fl_server.sh` / `run_recipes.sh` / `run_init_model.sh` / `run_infer.sh`.
- Imports **Framework** as a library; packaged together with it into the **client-docker** image and the **Desktop** PyInstaller bundle.
- Serves the recipe catalog and the arm trade-offs that the **Frontend** project-creation picker renders.

---

### Desktop

> **Path:** [`wikis/desktop/`](./desktop/README.md)  
> **Stack:** Electron 42, React 18, TypeScript 5.7, Webpack, PyInstaller, dockerode

The desktop application is the local training orchestrator for FL participants. It provides a secure GUI to configure hardware profiles, launch Framework training processes (native binary or Docker), and stream logs — all while keeping the JWT confined to the OS keychain.

| Document | Description |
|---|---|
| [Overview & Architecture](./desktop/01-overview-and-architecture.md) | Three-process model (main / preload / renderer), execution paths, data flow |
| [Security Model](./desktop/02-security-model.md) | BrowserWindow hardening, CSP, contextBridge, double input validation, JWT confinement |
| [Main Process Deep Dive](./desktop/03-main-process.md) | `main.ts`, `ipc.handlers.ts`, `docker.service.ts`, `auth.service.ts`, `hardware.probe.ts` |
| [Preload & IPC Bridge](./desktop/04-preload-ipc-bridge.md) | contextBridge API design, full `window.fedLearnAPI` reference |
| [Renderer & React Components](./desktop/05-renderer-components.md) | `App.tsx` state machine, `AuthModal`, `HardwareSelector`, `LogPanel`, `StatusIndicator` |
| [Build, Packaging & Distribution](./desktop/06-build-and-packaging.md) | Webpack configs, electron-builder, PyInstaller bundling, code signing |
| [Hardware Profiles & Training Execution](./desktop/07-hardware-profiles.md) | MPS, CUDA, CPU, Jetson profiles; Docker device mounts; lifecycle state machine |
| [Developer Guide & Contributing](./desktop/08-developer-guide.md) | Prerequisites, setup, script reference, adding IPC channels/components/profiles |

**Key cross-component interfaces:**
- Authenticates against **Backend** `POST /api/auth/login`; JWT stored via Electron `safeStorage`.
- Spawns the FL client as a child process — the PyInstaller `fedlearn-client` binary, `fl-runtime/client.py` under dev mode, or a Docker container. All three are the same canonical client (DA-5). It orchestrates *clients* only; FL servers are spawned by the Backend.
- Does not connect to the **Frontend** — both are independent clients of the Backend API.

---

### Mobile

> **Path:** [`wikis/mobile/`](./mobile/README.md)  
> **Stack:** React Native 0.80, TypeScript, native C++ (ExecuTorch on the shared core + Android; the iOS native wiring is still a **libtorch** scaffold pending its ExecuTorch migration — MO-14) via a TurboModule bridge, Android + iOS

The mobile client is an on-device FL participant for phones and tablets. The JS/TS layer handles UI, auth, and orchestration; the heavy lifting — the **DeComFL zeroth-order training path** — runs natively in C++ on ExecuTorch through a TurboModule (JSI) bridge, keeping training data on-device. **That native path is Android-only today.** `ios/FedLearnCore.podspec` vendors a libtorch xcframework while `shared/` targets ExecuTorch — incompatible runtimes, so the podspec's own guard says not to enable `FEDLEARN_NATIVE_IOS`; iOS builds the JS shell, `isNativeCoreAvailable()` returns false, and the training entry point is disabled rather than crashing. See [`mobile/README.md`](./mobile/README.md) for the full status. It picked up the **Ember** design system and brand fonts in `2.1.0` (2026-06-10), and was moved onto **Ledger** with the rest of the platform on 2026-07-17 (`2c50672`, which regenerated `mobile_client/src/theme/tokens.generated.ts` and `global.css` from `design/tokens.json` — they now carry the Ledger canvas `#F6F3EE`, accent `#1C314D`, and Hanken Grotesk for both sans and display). That re-theme shipped without a mobile version bump, so `2.1.0` is still the current version and no longer implies Ember.

| Document | Description |
|---|---|
| [Mobile Client Overview](./mobile/README.md) | Architecture (RN + native C++), the TurboModule bridge, DeComFL on-device path, iOS/Android project layout, current build status |

**Key cross-component interfaces:**
- Authenticates against **Backend** `POST /api/auth/login` — but **not** on the web client's cookie contract. Mobile uses `Authorization: Bearer` (`mobile_client/src/lib/restClient.ts`), which the backend accepts **only** when the request also carries the native-client marker header `X-FedLearn-Client` (mobile sends `fedlearn-mobile`; `JwtAuthenticationFilter.isNativeClient` gates on the header being present and non-blank, not on its value — SE-9); browsers never set it, so they stay strictly cookie-only. The marker is an intent signal, not a secret. As with the web JWT, the token is audience-scoped (SE-20) so it cannot be replayed against another surface.
- Connects to an FL server (`fl-runtime/fl_server.py`) over gRPC and runs the native DeComFL client path — on Android; on iOS the native core is unavailable and the entry point is disabled (MO-14).
- Shares the canonical `proto/` contract (byte-mirrored into `mobile_client/proto/fedlearn/v2/fedlearn.proto`, CI-enforced byte-identical).

---

### Client (Docker)

> **Path:** [`wikis/client-docker/`](./client-docker/README.md)  
> **Stack:** Docker, multi-arch base images; packages `framework/` + `fl-runtime/`

`client-docker` is the containerised FL client — it packages **both** `framework/` and `fl-runtime/` (the build context is the **repo root**, not `client-docker/`) and, via `entrypoint.sh`, execs `python3 -u client.py` (the canonical `fl-runtime/client.py`). It exists so a client can be deployed without a local Python toolchain, and it is the execution path the desktop app uses for the **Jetson** profile. The container is configured by **environment variables**, not CLI flags: `entrypoint.sh` hard-fails if `PROJECT_ID` / `SERVER_ADDRESS` / `PARTITION_ID` are unset, builds the matching `--project-id` / `--server-address` / `--partition-id` flags itself, and forwards `"$@"` for extras.

| Document | Description |
|---|---|
| [Client (Docker) Overview](./client-docker/README.md) | Image build (x86 + ARM64), Jetson L4T base image, device-mount notes, run flags, relationship to the framework |

**Key cross-component interfaces:**
- Bundles the **Framework** (`pip install -e framework`) and the **FL runtime** — no duplicated FL logic, and no client fork (DA-5).
- Connects to an FL server over gRPC like any other client.
- On **Jetson**, build with an L4T base image (`--build-arg BASE_IMAGE=...`). **The old blanket "never `--runtime nvidia`" rule is withdrawn.** On a JetPack 6.2 / L4T R36.5 AGX Orin, `docker run --runtime nvidia` is what *works* (`torch.cuda.is_available()` True, device "Orin"), while the hand-rolled device-mount path fails with `cuInit → 801` because the in-container `libcuda.so.1` is a stub. Two related snags on that generation: `docker.service.ts`'s device list includes `/dev/nvhost-ctrl`, which does not exist on R36.5 (Docker hard-errors on a missing device node), and the `r35.2.1-pth2.0-py3` base tag documented in the Dockerfile is JetPack-5-era, two L4T generations behind. **Scope of this correction:** the failure the old rule described was plausibly real on the JetPack 5 / `nvidia-container-runtime` it was written against — that is an inference, not a re-test, since no JetPack 5 hardware was available. Treat `--runtime nvidia` as the default to try on JetPack 6+, keep device mounts as the fallback, and re-verify on whatever L4T the target device actually runs. The desktop `DockerService` Jetson flow itself has not been re-run end to end against this finding.

---

## Developer Quick Start

### Run the Full Stack Locally

> macOS shortcut: `./launch_all.sh` opens four Terminal windows (backend, frontend, server, clients). On Linux, start each component manually as below.

```bash
# 1. Start PostgreSQL (required — H2 is retired; dev runs against a local Postgres)
cd backend/fl-platform-api
docker compose up -d                                # postgres:16.6-alpine on :5432

# 2. Start the Backend (Spring Boot, Gradle wrapper, Java 21)
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun        # :8081

# 3. Start the Frontend (Vite dev server)
cd frontend
npm install && npm run dev                          # :5173 → backend on :8081

# 4. Start the Desktop app (Electron dev mode)
cd fedlearn-desktop
npm install && npm run dev

# 5. (Optional) Run a standalone FL training round manually
cd framework
pip install -e .            # installs the `fedlearn` library that fl-runtime imports
python run_local_test.py

# 6. (Optional) Inspect the recipe catalog the picker is built from — torch-free, so it's cheap
cd fl-runtime
python recipes.py --describe
```

> **Pinned toolchain — don't guess versions.** `.tool-versions` (asdf/mise) and `.nvmrc` are the
> pins and CI matches them: **Node 24** (`nodejs 24.4.0`, `.nvmrc` → `24`), **Python 3.12.9** (the
> framework's `setup.py` declares a `python_requires='>=3.10'` floor, but 3.12 is what CI tests),
> **Java 21** (`temurin-21.0.7+6`). `rust 1.87.0` is pinned but unused — there is no crate in this
> repo. `.editorconfig` is authoritative for formatting: 2 spaces by default, **4 for Python and
> Java**.

### Per-Unit Test Commands

```bash
cd framework    && PYTHONPATH=src python -m pytest -q   # coverage-enforced (--cov=fedlearn)
cd fl-runtime   && python -m pytest -q                  # its own suite; pytest.ini deselects -m slow
cd backend/fl-platform-api && SPRING_PROFILES_ACTIVE=test ./gradlew test   # needs a live Docker daemon
cd frontend     && npm run lint && npx tsc --noEmit && npm run test:coverage
cd fedlearn-desktop && npm run lint && npm test
cd mobile_client    && npm run lint && npx tsc --noEmit && npm test
./scripts/check_proto_mirror.sh                         # proto mirrors must stay byte-identical
```

CI (`.github/workflows/`: `ci.yml`, `mobile.yml`, `proto.yml`, `security.yml`, `release-desktop.yml`, `release-mobile.yml`) is **path-filtered** — a unit's job runs only when that unit changes. Three gates are easy to trip over: `framework/` enforces coverage, so a bare pytest pass is not enough; `fl-runtime/` runs with `FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1`, so a *skipped* test fails the job (TE-10; `-m "not slow"` deselection is not a skip); and the backend gate is `./gradlew test jacocoTestCoverageVerification` with a 0.70 bundle line-coverage floor (TE-11), not `test` alone. Unlike `frontend/` and `mobile_client/`, `fedlearn-desktop/` has **no** standalone `tsc --noEmit` gate — its types are checked only incidentally by ts-jest on test-reachable sources. The proto-mirror check lives in `proto.yml` / `mobile.yml`, not `ci.yml`.

### Useful Cross-Component Entry Points

| Task | Where to Look |
|---|---|
| Add a new FL aggregation strategy | [Framework: Strategies](./framework/05_strategies.md) + [Developer Guide](./framework/09_developer_guide.md) — then register it in `framework/src/fedlearn/server/strategy_factory.py` |
| Add a new model type | [`fl-runtime`](#fl-runtime) — a `RECIPE_METADATA` entry in `fl-runtime/recipes.py`, with `supported_arms` + `trainable_spec`. No Java or TypeScript edit |
| Add a training arm | `fl-runtime/recipes.py` (`TRAINING_ARMS` + `ARM_OBJECTIVES`), the Java `TrainingArm` enum, **and** a new migration widening `chk_projects_training_arm` |
| Run a large-scale (100–5,000 client) experiment | `framework/src/fedlearn/simulation/` — [In-Process Simulation](#in-process-simulation--the-experiment-path-that-has-no-ports) |
| Secure a new API endpoint | [Backend: Security & Auth](./backend/02_security_and_auth.md) |
| Add a new IPC channel to Desktop | [Desktop: Developer Guide](./desktop/08-developer-guide.md) |
| Build a new React page / component | [Frontend: UI & Components](./frontend/UI_and_Components.md) |
| Stream new data to the log panel | [Backend: WebSocket Streaming](./backend/05_websocket_logs_streaming.md) + [Frontend: API & Services](./frontend/API_and_Services.md) |
| Package Desktop for a new platform | [Desktop: Build & Packaging](./desktop/06-build-and-packaging.md) |
| Implement a custom FL client | [Framework: Client Internals](./framework/04_client_internals.md) + [Developer Guide](./framework/09_developer_guide.md) |
| Change the gRPC contract | Edit `proto/fedlearn/v2/fedlearn.proto` (or `proto/fedlearn/fot/v1/fot.proto`), then `cp` to the three mirrors — never hand-edit a mirror; `scripts/check_proto_mirror.sh` prints the exact command |

---

## Directory Structure

### The seven deployable units, at the repo root

```
FedLearn-Platform/
├── backend/fl-platform-api/     ← Spring Boot 3 / Java 21 / Gradle — REST + STOMP control plane
├── framework/                   ← the pip-installable `fedlearn` library (server, client,
│                                  strategies, safetensors wire, simulation, FoT)
├── fl-runtime/                  ← THE EXECUTABLE LAYER — fl_server.py, client.py, init_model.py,
│                                  infer.py, recipes.py, data.py + run_*.sh wrappers
├── frontend/                    ← React 19 / Vite 6 / TS / Tailwind v4 dashboard SPA
├── fedlearn-desktop/            ← Electron + dockerode end-user orchestrator
├── client-docker/               ← containerised FL client (build context is the REPO ROOT)
├── mobile_client/               ← React Native 0.80 + native C++ (ExecuTorch) via a TurboModule
│
├── proto/                       ← THE canonical gRPC contract, governed by buf
│   ├── fedlearn/v2/fedlearn.proto        package fedlearn.v2       (the gradient path)
│   └── fedlearn/fot/v1/fot.proto         package fedlearn.fot.v1   (the FoT path)
├── design/                      ← tokens.json (single source of truth) + build-tokens.mjs
├── scripts/                     ← check_proto_mirror.sh, check_no_skipped_tests.sh, deploy, …
└── wikis/                       ← this wiki
```

The canonical contract has **one** home (`proto/`) and **three** byte-identical in-tree mirrors,
enforced by `scripts/check_proto_mirror.sh` (the failure message prints the exact `cp` to fix it):
`framework/src/fedlearn/communication/protos/fedlearn.proto`,
`mobile_client/proto/fedlearn/v2/fedlearn.proto`, and
`framework/src/fedlearn/communication/protos/fot.proto`. There is no `fedlearn.v1` anywhere —
the framework is on v2 and byte-identical to canonical.

### This wiki

```
wikis/                              ← repo-root docs (promoted out of docs/)
├── README.md                       ← You are here (master wiki)
├── VERSIONS.md                     ← per-unit release versions (source of truth)
├── assets/                         ← shared diagrams (architecture*.png)
├── html/                           ← generated HTML rendering of this wiki (Ledger-themed).
│   ├── index.html                    Regenerate with `python3 wikis/html/build.py` after editing
│   └── build.py                      any page here — never hand-edit the HTML. The generator's
│                                     palette mirrors design/tokens.json.
│
├── backend/                        ← Spring Boot 3 API
│   ├── README.md
│   ├── 01_architecture_overview.md
│   ├── 02_security_and_auth.md
│   ├── 03_project_management.md
│   ├── 04_federated_orchestration.md
│   ├── 05_websocket_logs_streaming.md
│   ├── 06_identity_multitenancy_and_audit.md   ← present on this branch (V4–V7 migrations)
│   └── 07_artifact_registry.md
│
├── frontend/                       ← React 19 SPA
│   ├── README.md
│   ├── Architecture.md
│   ├── Routing_and_Auth.md
│   ├── API_and_Services.md
│   └── UI_and_Components.md
│
├── framework/                      ← Python FL library (gRPC + PyTorch)
│   ├── README.md
│   ├── 01_architecture_overview.md
│   ├── 02_grpc_communication.md
│   ├── 03_server_internals.md
│   ├── 04_client_internals.md
│   ├── 05_strategies.md
│   ├── 06_decomfl.md
│   ├── 07_data_partitioning.md
│   ├── 08_examples.md
│   └── 09_developer_guide.md
│
├── fl-runtime/                     ← the executable FL layer the backend shells out to
│   ├── README.md
│   ├── 01_entry_points.md
│   ├── 02_recipe_catalog.md
│   ├── 03_training_arms.md
│   └── 04_the_federated_set.md
│
├── desktop/                        ← Electron 42 local training client
│   ├── README.md
│   ├── 01-overview-and-architecture.md
│   ├── 02-security-model.md
│   ├── 03-main-process.md
│   ├── 04-preload-ipc-bridge.md
│   ├── 05-renderer-components.md
│   ├── 06-build-and-packaging.md
│   ├── 07-hardware-profiles.md
│   └── 08-developer-guide.md
│
├── mobile/                         ← React Native 0.80 + native C++ FL client
│   └── README.md
│
└── client-docker/                  ← containerised FL client (multi-arch)
    └── README.md
```
