# FedLearn Platform — Master Wiki

> **Repository:** `FedLearn-Platform`  
> **Last Updated:** 2026-07-20

This is the top-level technical wiki for the **FedLearn Platform** — a privacy-preserving, distributed machine learning system that enables federated learning across heterogeneous hardware. Use this page as your starting point and navigate into any subsection for in-depth documentation.

> **Recent hardening (2026-06):** dependency/security pass — npm and pip audits cleared, Electron upgraded 34 → 42, gitleaks secret scanning added via the free CLI, and the unused `flwr` dependency dropped from the **framework** (`framework/pyproject.toml`). Note that `flwr` / `flwr-datasets` are still pinned in `backend/fl-platform-api/requirements.txt` and `client-docker/requirements.txt` — they are used for **dataset partitioning only** (`FederatedDataset`), never for FL semantics.
>
> **Ledger design-system redesign (2026-07):** the frontend, the desktop renderer, and the mobile client adopted the **Ledger** design system — navy structural ink (`#1C314D`) on quiet paper surfaces (`#F6F3EE` canvas, white cards), with a single Hanken Grotesk type family (JetBrains Mono for logs/ids). Ledger superseded **Ember** (warm canvas + burnt orange + Bricolage Grotesque, 2026-06), which had superseded "Instrument" — see [`frontend/UI_and_Components.md`](./frontend/UI_and_Components.md). "Ledger" is the name of the **design system / theme**; the platform itself is still **FedLearn** (a product-domain rename is a separate, in-progress effort). This wiki ships an HTML rendering under [`html/`](./html/index.html) (generated during the Ember cycle; re-theming is pending).

---

## Platform Overview

FedLearn is built around four core components — backend, frontend, framework, desktop — plus a **mobile client** and a **containerised client**, for **six deployable units** in total. They work together to provide an end-to-end federated learning experience:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FedLearn Platform                                │
│                                                                         │
│  ┌──────────────┐    REST / WebSocket    ┌──────────────────────────┐   │
│  │   Frontend   │◄──────────────────────►│      Backend (API)       │   │
│  │  React SPA   │                        │  Spring Boot 3 / Postgres│   │
│  └──────────────┘                        └──────────┬───────────────┘   │
│                                                     │ spawns FL server  │
│  ┌──────────────┐    gRPC (Python)                  │ as local process  │
│  │   Desktop    │◄──────────────────────►┌──────────▼───────────────┐   │
│  │  Electron 42 │   PyInstaller / Docker │  Framework (Python FL)   │   │
│  └──────────────┘                        │  fedlearn + PyTorch gRPC │   │
│                                          └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Component | Primary Language | Responsibility |
|---|---|---|
| [**Backend**](#backend) | Java (Spring Boot 3) | REST API, JWT auth, project lifecycle, FL orchestration, WebSocket log streaming |
| [**Frontend**](#frontend) | TypeScript (React 19 + Vite) | Web dashboard, real-time monitoring, project management, auth flows |
| [**Framework**](#framework) | Python (PyTorch + gRPC) | Core FL engine — FedAvg, DeComFL, data partitioning, gRPC client/server |
| [**Desktop**](#desktop) | TypeScript (Electron 42) | Local training orchestrator, hardware detection, Docker/PyInstaller execution |
| [**Mobile**](#mobile) | React Native 0.80 + native C++ (ExecuTorch) | On-device FL client; runs the DeComFL zeroth-order path natively in C++ via a TurboModule bridge (iOS + Android) |
| [**Client (Docker)**](#client-docker) | Docker (multi-arch) | Containerised FL client; thin wrapper around the framework for Jetson / CUDA / CPU deployments |

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
   │  2. Persist project config to the database (PostgreSQL)
   │  3. Spawn the Python FL server as a local process (`fl-runtime/run_fl_server.sh`)
   ▼
Framework — FL Server (Python / gRPC)
   │
   │  4. Broadcast global model parameters to all connected clients
   ▼
Framework — FL Client (Python / gRPC)   ◄── invoked by Desktop or client-docker
   │
   │  5. Train locally on private data; compute gradients / updates
   │  6. Upload aggregated gradient or delta to server
   ▼
Framework — FL Server (aggregation)
   │
   │  7. Aggregate updates (FedAvg / DeComFL) → new global model
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
Browser / Desktop
   │  POST /api/auth/login  (username + password)
   ▼
Backend  →  validates credentials  →  issues JWT (HttpOnly cookie or JSON)
   │
   ├── Frontend: stores JWT in HttpOnly cookie; axios interceptor attaches on every request
   └── Desktop: stores JWT via Electron safeStorage (OS keychain); never exposed to renderer
```

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
- Exposes `POST /api/projects/{id}/start` → triggers Framework server spawn.
- Streams logs to Frontend via STOMP topic `/topic/logs/{projectId}`.
- Desktop authenticates against `POST /api/auth/login` before initiating training.
- Authorization is the layered identity model — `PlatformRole` (platform), `OrgRole` (organization), `MembershipRole` (project) — committed in the `V4`–`V7` migrations (highest committed migration is `V19`). Organizations, org/project memberships, org-scoped isolation, and the `@Auditable` audit trail are all present; see [Identity, Multi-Tenancy & Audit](./backend/06_identity_multitenancy_and_audit.md). The original coarse `users.role IN (USER, ADMIN)` column (`V2`) has been superseded.

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
- Subscribes to `STOMP /topic/logs/{projectId}` for live Framework training output.
- No direct connection to Framework or Desktop — all routing via Backend.

---

### Framework

> **Path:** [`wikis/framework/`](./framework/README.md)  
> **Stack:** Python 3.10+, PyTorch, gRPC / Protocol Buffers — **custom FL engine (no Flower runtime)**

The framework is the heart of the platform — a standalone Python library (`fedlearn`) that implements the full federated learning lifecycle using gRPC for communication and PyTorch for model training. The Java-side orchestration package (`orchestration/`, class `FlServerManager`) was renamed from the legacy `flower` / `FlowerServerManager` name (DA-12). No Flower server, client, or strategy semantics are used anywhere, and `framework/pyproject.toml` declares no `flwr` dependency — the protobuf contract is entirely custom. The one remaining touchpoint is `flwr-datasets`, pinned in `backend/fl-platform-api/requirements.txt` and `client-docker/requirements.txt` purely to partition datasets (`from flwr_datasets import FederatedDataset` in `fl-runtime/client.py`).

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
- Consumed by **Backend** as a local process — `FlServerManager` shells out to `fl-runtime/run_fl_server.sh` through the `FlServerProcessRunner` seam.
- Consumed by **Desktop** as a PyInstaller-bundled native client (entry binary `fedlearn-client` / `fedlearn-client.exe`, per `fedlearn-desktop/src/shared/bundleVariants.ts`); in dev mode the desktop falls back to system `python3` running `fl-runtime/client.py`.
- Consumed by **`client-docker`** image for containerised Jetson / CUDA deployments.

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
- Spawns the **Framework** FL client as a child process — the PyInstaller `fedlearn-client` binary, `fl-runtime/client.py` under dev mode, or a Docker container. It orchestrates *clients* only; FL servers are spawned by the Backend.
- Does not connect to the **Frontend** — both are independent clients of the Backend API.

---

### Mobile

> **Path:** [`wikis/mobile/`](./mobile/README.md)  
> **Stack:** React Native 0.80, TypeScript, native C++ (ExecuTorch) via a TurboModule bridge, Android + iOS

The mobile client is an on-device FL participant for phones and tablets. The JS/TS layer handles UI, auth, and orchestration; the heavy lifting — the **DeComFL zeroth-order training path** — runs natively in C++ on ExecuTorch through a TurboModule (JSI) bridge, keeping training data on-device. It adopted the **Ember** design system and brand fonts in `2.1.0`.

| Document | Description |
|---|---|
| [Mobile Client Overview](./mobile/README.md) | Architecture (RN + native C++), the TurboModule bridge, DeComFL on-device path, iOS/Android project layout, current build status |

**Key cross-component interfaces:**
- Authenticates against **Backend** `POST /api/auth/login` (same cookie/JWT contract as the web client).
- Connects to a **Framework** FL server over gRPC and runs the native DeComFL client path.
- Shares the canonical `proto/` contract (byte-mirrored into `mobile_client/proto/`).

---

### Client (Docker)

> **Path:** [`wikis/client-docker/`](./client-docker/README.md)  
> **Stack:** Docker, multi-arch base images, thin wrapper around `framework/`

`client-docker` is the containerised FL client — a thin wrapper that `pip install -e`'s the framework and, via `entrypoint.sh`, execs `python3 -u client.py` (the canonical `fl-runtime/client.py`). It exists so a client can be deployed without a local Python toolchain, and it is the execution path the desktop app uses for the **Jetson** profile.

| Document | Description |
|---|---|
| [Client (Docker) Overview](./client-docker/README.md) | Image build (x86 + ARM64), Jetson L4T base image, device-mount notes, run flags, relationship to the framework |

**Key cross-component interfaces:**
- Bundles the **Framework** (`pip install -e framework`) — no duplicated FL logic.
- Connects to a **Framework** FL server over gRPC like any other client.
- On **Jetson**, must use the L4T base image and direct `/dev/nvhost-*` device mounts (never `--runtime nvidia`).

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
pip install -e .
python run_local_test.py
```

### Useful Cross-Component Entry Points

| Task | Where to Look |
|---|---|
| Add a new FL aggregation strategy | [Framework: Strategies](./framework/05_strategies.md) + [Developer Guide](./framework/09_developer_guide.md) |
| Secure a new API endpoint | [Backend: Security & Auth](./backend/02_security_and_auth.md) |
| Add a new IPC channel to Desktop | [Desktop: Developer Guide](./desktop/08-developer-guide.md) |
| Build a new React page / component | [Frontend: UI & Components](./frontend/UI_and_Components.md) |
| Stream new data to the log panel | [Backend: WebSocket Streaming](./backend/05_websocket_logs_streaming.md) + [Frontend: API & Services](./frontend/API_and_Services.md) |
| Package Desktop for a new platform | [Desktop: Build & Packaging](./desktop/06-build-and-packaging.md) |
| Implement a custom FL client | [Framework: Client Internals](./framework/04_client_internals.md) + [Developer Guide](./framework/09_developer_guide.md) |

---

## Directory Structure

```
wikis/                              ← repo-root docs (promoted out of docs/)
├── README.md                       ← You are here (master wiki)
├── VERSIONS.md                     ← per-unit release versions (source of truth)
├── assets/                         ← shared diagrams (architecture*.png)
├── html/                           ← Ember-themed HTML rendering (generated)
│   └── index.html
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
├── framework/                      ← Python FL engine (gRPC + PyTorch)
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
