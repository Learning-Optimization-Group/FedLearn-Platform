# FedLearn Platform — Master Wiki

> **Repository:** `FedLearn-Platform`  
> **Last Updated:** 2026-04-28

This is the top-level technical wiki for the **FedLearn Platform** — a privacy-preserving, distributed machine learning system that enables federated learning across heterogeneous hardware. Use this page as your starting point and navigate into any subsection for in-depth documentation.

---

## Platform Overview

FedLearn is built around four major components that work together to provide an end-to-end federated learning experience:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FedLearn Platform                                │
│                                                                         │
│  ┌──────────────┐    REST / WebSocket    ┌──────────────────────────┐   │
│  │   Frontend   │◄──────────────────────►│      Backend (API)       │   │
│  │  React SPA   │                        │  Spring Boot 3 / PgSQL   │   │
│  └──────────────┘                        └──────────┬───────────────┘   │
│                                                     │ ProcessBuilder /  │
│  ┌──────────────┐    gRPC (Python)                  │ AWS ECS Fargate   │
│  │   Desktop    │◄──────────────────────►┌──────────▼───────────────┐   │
│  │  Electron 34 │   PyInstaller / Docker │  Framework (Python FL)   │   │
│  └──────────────┘                        │  fedlearn + PyTorch gRPC │   │
│                                          └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Component | Primary Language | Responsibility |
|---|---|---|
| [**Backend**](#backend) | Java (Spring Boot 3) | REST API, JWT auth, project lifecycle, FL orchestration, WebSocket log streaming |
| [**Frontend**](#frontend) | TypeScript (React 19 + Vite) | Web dashboard, real-time monitoring, project management, auth flows |
| [**Framework**](#framework) | Python (PyTorch + gRPC) | Core FL engine — FedAvg, DeComFL, data partitioning, gRPC client/server |
| [**Desktop**](#desktop) | TypeScript (Electron 34) | Local training orchestrator, hardware detection, Docker/PyInstaller execution |

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
   │  2. Persist project config to PostgreSQL
   │  3. Spawn Python FL Server via ProcessBuilder or AWS ECS Fargate
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
      11. Persist final model checkpoint & round results to PostgreSQL
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

### FL Server Provisioning — Local vs. Cloud

```
Backend (FlowerServerManager)
   │
   ├── LOCAL MODE
   │     ProcessBuilder.start()  →  python run_server.py --port <N>
   │     stdout piped  →  WebSocket log streaming
   │
   └── CLOUD MODE (AWS ECS Fargate)
         RunTask API  →  framework Docker image on Fargate
         CloudWatch Logs  →  polled and re-streamed to WebSocket clients
```

---

## Component Wikis

### Backend

> **Path:** [`docs/wikis/backend/`](./backend/README.md)  
> **Stack:** Java 21, Spring Boot 3, Spring Security 6, PostgreSQL, STOMP WebSocket

The backend is the central control plane. It owns the REST API, user authentication, project management, and acts as the bridge between the web clients and the Python FL processes.

| Document | Description |
|---|---|
| [Architecture & Core Concepts](./backend/01_architecture_overview.md) | Directory structure, domain models (Projects, Results, Logs), technology stack |
| [Security & Authentication](./backend/02_security_and_auth.md) | Stateless JWT filter chain, WebSocket handshake security, internal API key mechanism |
| [Project Management Lifecycle](./backend/03_project_management.md) | `ProjectService`, `ProjectController`, round configuration, model initialization |
| [Federated Orchestration](./backend/04_federated_orchestration.md) | `FlowerServerManager` — local `ProcessBuilder` vs. AWS ECS Fargate provisioning |
| [WebSocket Log Streaming](./backend/05_websocket_logs_streaming.md) | Stdout capture → STOMP topics → frontend real-time observability |

**Key cross-component interfaces:**
- Exposes `POST /api/projects/{id}/start` → triggers Framework server spawn.
- Streams logs to Frontend via STOMP topic `/topic/logs/{projectId}`.
- Desktop authenticates against `POST /api/auth/login` before initiating training.

---

### Frontend

> **Path:** [`docs/wikis/frontend/`](./frontend/README.md)  
> **Stack:** React 19, TypeScript, Vite, Tailwind CSS, Axios, SockJS / STOMP

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

> **Path:** [`docs/wikis/framework/`](./framework/README.md)  
> **Stack:** Python 3.10+, PyTorch, gRPC / Protocol Buffers, Flower (custom fork)

The framework is the heart of the platform — a standalone Python library (`fedlearn`) that implements the full federated learning lifecycle using gRPC for communication and PyTorch for model training.

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
- Consumed by **Backend** via `ProcessBuilder` (`run_server.py`) or AWS ECS Fargate Docker image.
- Consumed by **Desktop** via PyInstaller-bundled binaries (`run_server.py`, `run_client.py`).
- Consumed by **`client-docker`** image for containerised Jetson / CUDA deployments.

---

### Desktop

> **Path:** [`docs/wikis/desktop/`](./desktop/README.md)  
> **Stack:** Electron 34, React 18, TypeScript 5.7, Webpack, PyInstaller, dockerode

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
- Spawns **Framework** `run_server.py` and `run_client.py` as child processes (PyInstaller binaries) or Docker containers.
- Does not connect to the **Frontend** — both are independent clients of the Backend API.

---

## Developer Quick Start

### Run the Full Stack Locally

```bash
# 1. Start the Backend (Spring Boot)
cd fedlearn-backend
./mvnw spring-boot:run

# 2. Start the Frontend (Vite dev server)
cd fedlearn-frontend
npm install && npm run dev

# 3. Start the Desktop app (Electron dev mode)
cd fedlearn-desktop
npm install && npm run dev

# 4. (Optional) Run a standalone FL training round manually
cd framework
python examples/mnist_example.py
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
docs/
└── wikis/
    ├── README.md                   ← You are here (master wiki)
    │
    ├── backend/                    ← Spring Boot 3 API
    │   ├── README.md
    │   ├── 01_architecture_overview.md
    │   ├── 02_security_and_auth.md
    │   ├── 03_project_management.md
    │   ├── 04_federated_orchestration.md
    │   └── 05_websocket_logs_streaming.md
    │
    ├── frontend/                   ← React 19 SPA
    │   ├── README.md
    │   ├── Architecture.md
    │   ├── Routing_and_Auth.md
    │   ├── API_and_Services.md
    │   └── UI_and_Components.md
    │
    ├── framework/                  ← Python FL engine (gRPC + PyTorch)
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
    └── desktop/                    ← Electron 34 local training client
        ├── README.md
        ├── 01-overview-and-architecture.md
        ├── 02-security-model.md
        ├── 03-main-process.md
        ├── 04-preload-ipc-bridge.md
        ├── 05-renderer-components.md
        ├── 06-build-and-packaging.md
        ├── 07-hardware-profiles.md
        └── 08-developer-guide.md
```
