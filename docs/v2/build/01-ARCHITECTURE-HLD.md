# 01 — Architecture High-Level Design (HLD)

**Document:** `docs/v2/build/01-ARCHITECTURE-HLD.md`
**Status:** Build-authoritative. This is the **entry HLD (High-Level Design)** for FedLearn Platform v2 (version 2). Every other build document (10- through 18-) hangs off this one.
**Date:** 2026-05-29
**Sourced from:** the v2 audit synthesis at [`docs/audit/2026-05-29/README.md`](../../audit/2026-05-29/README.md), plus the per-unit reports it indexes — chiefly [`B2-tech-stack.md`](../../audit/2026-05-29/B2-tech-stack.md), [`B6-scale-cost.md`](../../audit/2026-05-29/B6-scale-cost.md), and [`C1-reliability-sre.md`](../../audit/2026-05-29/C1-reliability-sre.md).

---

## 0. How to read this document (instructions for the implementer)

This document is written for an implementer who **cannot infer missing context**. Therefore:

- Every acronym is expanded in parentheses the **first** time it appears. After that, the short form is used. The full glossary is in §9.
- Diagrams are ASCII (American Standard Code for Information Interchange) so they render in any editor. They are normative, not decorative.
- Where a claim is about **existing v1 code**, it is cited as `path:line`. Where a claim is about the **external market or a vendor**, a source Uniform Resource Locator (URL) is given.
- This is an **HLD**. It defines *what* the units are, *how* they talk, and *why* the boundaries fall where they do. It does **not** define interface signatures, database columns, or function bodies — those live in the Low-Level Design (LLD) documents 10- through 18-, named in §3.
- When this document says "MUST", it is a build constraint that downstream LLDs may not contradict. When it says "SHOULD", it is a strong default that an LLD may override with stated reasoning.

The acronyms you need immediately: **FL (Federated Learning)** — training a shared model across many devices that never share their raw data; **DeComFL (Dimension-Free Communication Federated Learning)** — the platform's zeroth-order FL algorithm that sends scalars instead of model weights (note: the v1 wiki mis-expanded this as "Decomposed", which is wrong per the paper — see [`B1-paper-alignment.md:33`](../../audit/2026-05-29/B1-paper-alignment.md)); **API (Application Programming Interface)**; **gRPC (Google Remote Procedure Call)** — the binary Remote Procedure Call protocol the FL server and clients speak; **JWT (JSON Web Token)** — the signed auth token; **STOMP (Simple Text Oriented Messaging Protocol)** — the message protocol carried over WebSocket for live logs.

---

## 1. Product vision & what v2 is (plain language)

### 1.1 The one-sentence product

FedLearn is a **managed control plane for federated learning** whose differentiator is **DeComFL**: it can fine-tune a billion-parameter model across many organizations' private devices while sending **roughly one megabyte of data total over the whole training run**, instead of the tens of terabytes a conventional approach would move ([DeComFL paper, arXiv 2405.15861](https://arxiv.org/abs/2405.15861); quantified in [`B6-scale-cost.md:20`](../../audit/2026-05-29/B6-scale-cost.md) as a six-to-seven-order-of-magnitude egress reduction).

### 1.2 What "federated learning" means here, concretely

A hospital, a bank, or a phone owner has private data they will not upload. FL trains a shared model **by sending the model to the data**, not the data to the model:

1. A central **FL server** holds the current shared model.
2. Each **FL client** (a hospital's machine, a Jetson edge box, a phone) trains locally on its own private data for a short burst.
3. The client sends back **only an update** (in DeComFL: a handful of scalar numbers; in FedAvg: the changed weights), never the raw data.
4. The server **aggregates** all clients' updates into a new shared model.
5. Repeat for many **rounds**.

The privacy claim is structural: raw data never leaves the device. DeComFL strengthens it — because clients upload only scalars (a loss value and a perturbation seed), there are no gradients to invert, which **structurally eliminates the Deep-Leakage-from-Gradients (DLG) reconstruction attack family** ([`README.md:14`](../../audit/2026-05-29/README.md)).

### 1.3 What v2 *is* (the rebuild, in plain language)

v1 is a competent proof-of-concept with one genuine differentiator (DeComFL) and four classes of production blocker ([`README.md:12`](../../audit/2026-05-29/README.md)). v2 is a **greenfield rebuild that keeps the parts that work and replaces the orchestration substrate that does not.**

| v2 keeps (salvage) | v2 rebuilds | v2 kills |
|---|---|---|
| The Spring Boot control plane (auth, organizations, projects, results). | The FL orchestration substrate: one long-running multi-run server keyed on `run_id`, with a durable Postgres lease table and a reconciler loop. | The "one OS (Operating System) process per project via `ProcessBuilder`, capped at 11 Transmission Control Protocol (TCP) ports" model, for everything except local dev. |
| The React frontend shell and cookie-only JWT auth. | The artifact store, dataset registry, run-lineage stack, and the whole observability layer (none of which exist today). | The unsigned Electron auto-updater (replaced by a signed Tauri updater). |
| The DeComFL algorithm (the idea), with three correctness fixes already specified. | The serializer (chunked upload codec), the round loop (add deadline + quorum), and the datastore (H2 → managed Postgres). | The false "Byzantine-robust" README claim ([`README.md:106`](../../audit/2026-05-29/README.md)). |

### 1.4 The non-technical gate (state it, do not hide it)

DeComFL is research from RIT (Rochester Institute of Technology). Under RIT Intellectual-Property (IP) policy C03.0, **RIT — not the founder — likely owns it**. No defensible moat claim or fundable raise is possible until an RIT Intellectual Property Management Office (IPMO) license or spin-out is executed ([`README.md:27`](../../audit/2026-05-29/README.md), risk **R1**). This is a go/no-go gate upstream of the entire product. It is named here because the HLD's whole reason to keep DeComFL custom depends on resolving it; the build can proceed in parallel, but no public launch or moat claim can.

---

## 2. System context diagram

This is the outermost boundary: who uses the platform, and where the platform ends.

```
                              THE OUTSIDE WORLD
  ┌────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐
  │   RESEARCHER    │   │   ORG ADMIN     │   │   FL CLIENT OPERATOR     │
  │ (data          │   │ (tenant owner;  │   │ (runs a client on a      │
  │  scientist;    │   │  invites users, │   │  hospital box / Jetson / │
  │  creates       │   │  sets quotas,   │   │  desktop / phone; holds  │
  │  projects &    │   │  reads audit    │   │  the private data)       │
  │  runs)         │   │  log)           │   │                          │
  └───────┬────────┘   └────────┬────────┘   └────────────┬─────────────┘
          │ HTTPS (browser)     │ HTTPS (browser)         │
          │                     │                         │  desktop app (Tauri) or
          ▼                     ▼                         │  mobile app (native) or
  ┌───────────────────────────────────────────┐          │  docker client
  │            WEB FRONTEND (React SPA)         │          │
  └───────────────────┬─────────────────────────┘         │
                      │ HTTPS + WebSocket (cookie JWT)     │
══════════════════════│════════════════════════════════════│══════════════════════
        PLATFORM       ▼  (the boundary this project owns)   │
  ┌───────────────────────────────────────────────────────┐ │
  │  CONTROL PLANE (Spring Boot)  +  FL SUBSTRATE (k8s/ECS) │ │
  │  + Postgres + S3/MinIO + MLflow + Observability stack   │◀┘ gRPC (TLS + mTLS),
  └───────────────────────────────────────────────────────┘    scalars/seeds only
══════════════════════════════════════════════════════════════════════════════════
        EXTERNAL DEPENDENCIES (not owned, but depended on)
  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐
  │ AWS (RDS,   │  │ Container    │  │ SMTP email  │  │ minisign update    │
  │ S3, EKS/    │  │ registry     │  │ provider    │  │ feed (signed       │
  │ Fargate)    │  │ (images)     │  │             │  │ desktop releases)  │
  └─────────────┘  └──────────────┘  └─────────────┘  └────────────────────┘
```

**Actors (the three the assignment names, plus implicit ones):**

| Actor | Who they are | What they do | Auth surface |
|---|---|---|---|
| **Researcher** | A data scientist inside a tenant org | Creates projects, configures and launches runs, watches live metrics, downloads model checkpoints | Browser → frontend → control plane (cookie JWT) |
| **Org Admin** | The `OWNER`/`ADMIN` of a tenant organization | Invites/removes members, sets per-org concurrency quotas, reads the audit log, manages billing tier | Browser → frontend → control plane (cookie JWT) |
| **FL Client Operator** | The person/process running a client on the device that holds private data | Enrolls a client, points it at a run, contributes local training; raw data never leaves their device | Desktop/mobile/docker client → FL server (gRPC + mTLS); enrollment via control plane |
| Platform Admin (implicit) | The platform operator (you) | Bootstrap admin; bypasses org-membership checks; operates the fleet | Browser/console |

**The platform boundary** (double line above) is everything between the frontend's HTTPS/WebSocket calls inward and the gRPC the clients dial. Raw training data is **always** outside the boundary — it lives only on the FL Client Operator's device. This is the load-bearing privacy invariant.

---

## 3. The full unit map

Every deployable unit, its job, its v2 verdict (from the synthesis decision table, [`README.md:93-152`](../../audit/2026-05-29/README.md)), and the LLD document that specifies it in detail.

| # | Unit | Stack (locked) | Job | Verdict | LLD doc |
|---|---|---|---|---|---|
| 1 | **Control plane / API** | Spring Boot 3.5+ Long-Term-Support (LTS), Java 21, Gradle wrapper | Owns users, orgs, projects, `fl_runs`; serves REST (Representational State Transfer) + STOMP; launches runs; ingests per-round results; enforces authorization | Salvage core, refactor | **10-** Control-plane LLD |
| 2 | **Authorization layer** | Inside the control plane | `org_id`-scoped checks + Row-Level-Security (RLS)-style query filters; role enum across platform/org/project layers; per-run scoped result tokens | Refactor / rebuild org-isolation | **10-** (with the control plane) |
| 3 | **FL orchestration substrate** | `FlServerLauncher` abstraction (Java side) + the launched FL server (Python) | Launches and supervises one long-running multi-run FL server keyed on `run_id`; durable `fl_runs` lease table; reconciler loop; round deadline + min-quorum; per-org quotas + scale-to-zero | Rebuild | **11-** Orchestration-substrate LLD |
| 4 | **FL framework** | Python 3.10+, PyTorch, gRPC, package `fedlearn.v2` | The FL server + Python client; FedAvg + DeComFL strategies; parameter chunking (>300 megabyte (MB) models); dual heartbeat (training stub + parallel heartbeat stub); serializer | Salvage algorithm, rebuild serializer | **12-** FL-framework LLD |
| 5 | **Mobile FL core** | Native C++ (libtorch ARM64 (64-bit Advanced RISC Machine)) + gRPC | On-device DeComFL; Central-Processing-Unit-canonical (CPU-canonical) Random-Number-Generator (RNG); golden-vector parity test gating Python↔C++ determinism | Salvage core, rebuild styling/RNG-harden | **13-** Mobile-FL LLD |
| 6 | **Frontend** | React 19 + Vite 6 + TypeScript (TS); TanStack Query; Vitest + Playwright + Mock-Service-Worker (MSW) | Dashboard Single-Page Application (SPA); cookie auth; STOMP-over-WebSocket for live logs; recharts telemetry; Content-Security-Policy (CSP) + HTTP-Strict-Transport-Security (HSTS) | Salvage core, refactor | **14-** Frontend LLD |
| 7 | **Desktop** | Tauri v2 (Rust command layer + React renderer); bollard for Docker; OS keychain | End-user FL-client orchestrator; signed minisign auto-updater; fail-closed Inter-Process-Communication (IPC) bridge | Rebuild shell, salvage subprocess model | **15-** Desktop LLD |
| 8 | **Observability stack** | Micrometer → Prometheus; Grafana + Loki + Tempo + OpenTelemetry (OTel) Collector; structlog | Metrics, logs, distributed traces; W3C `traceparent` propagated JVM (Java Virtual Machine) → Python → client → mobile | Rebuild | **16-** Observability LLD |
| 9 | **Data & artifact stores** | S3 (Simple Storage Service) / MinIO content-addressed by sha256; Flyway dataset registry; MLflow self-hosted | Model/checkpoint artifact store; dataset/partition registry; experiment/run lineage + Model Registry | Rebuild | **17-** Data-and-artifact LLD |
| 10 | **OLTP datastore** | Managed PostgreSQL (AWS RDS (Relational Database Service)); Flyway migrations; JPA validate-only | The single source of truth for control-plane state and the `fl_runs` lease; no Citus, no sharding | Rebuild (H2 → Postgres) | **18-** Datastore LLD |

> **LLD numbering note (read carefully — the file numbers are authoritative over this table's "LLD doc" column where they differ):** the LLD set decomposes units 1–10 above. The numbering above was the original plan; the LLD documents as actually authored use the following numbering, which **supersedes** the "LLD doc" column where it differs:
> - **10-** Control-plane LLD (units 1 + 2: control plane + authorization layer)
> - **11-** FL-framework LLD (unit 4)
> - **12-** Orchestration-substrate LLD (unit 3) — file `12-LLD-orchestration-substrate.md`
> - **13-** Frontend LLD (unit 6) — file `13-LLD-frontend-dashboard.md`
> - **14-** Desktop LLD (unit 7)
> - **15-** Mobile-FL LLD (unit 5)
> - **16-** Observability LLD (unit 8)
> - **17-** Data-and-artifact + datastore LLD (units 9 + 10)
> - **18-** Security & Compliance LLD (cross-cutting; §6 of this HLD) — file `18-LLD-security-and-compliance.md`
>
> This HLD (01-) plus build docs 02-/03-/04- plus the LLDs (10–18) form the complete v2 build set. Every unit in the table above maps to exactly one LLD under this numbering; security/compliance is a numbered cross-cutting LLD (18-) in addition to the ten unit LLDs. All nine LLDs (10- through 18-) are authored on disk.

**Cross-cutting infrastructure that is not its own deployable unit** but appears in the topology: the **STOMP relay** (Redis or RabbitMQ, only once the control plane is multi-replica — a one-line `enableStompBrokerRelay` swap, [`README.md:117`](../../audit/2026-05-29/README.md)); the **Continuous-Integration (CI) pipeline** (`ci.yml` + branch protection + Renovate + per-stack vulnerability scans + Software Bill of Materials (SBOM)); and the **proto toolchain** (`buf` as the single source of truth for the `fedlearn.v2` gRPC contract, with a breaking-change gate). These are specified within the relevant LLDs and the CI/monorepo build doc.

---

## 4. End-to-end data-flow diagrams (the three core flows)

### 4.1 Flow A — Researcher logs in and creates a project/run

This flow is pure control plane. No FL server is launched yet; "create a run" produces a durable `fl_runs` row in state `PENDING`. The auth contract is **cookie-only**: the JWT lives in an HttpOnly cookie, never in `localStorage`, never as an `Authorization: Bearer` header (this is the v1 posture the audit rated "textbook-correct", [`README.md:119`](../../audit/2026-05-29/README.md), preserved verbatim in v2).

```
 Browser (React SPA)                Control Plane (Spring Boot)              Postgres
      │                                       │                                 │
      │ 1. POST /api/auth/login {user,pass}   │                                 │
      │   (withCredentials:true)              │                                 │
      │──────────────────────────────────────▶│ 2. verify password (BCrypt)     │
      │                                       │────────── SELECT user ─────────▶│
      │                                       │◀───────── row ──────────────────│
      │ 3. 200 + Set-Cookie: jwtToken=...     │                                 │
      │    HttpOnly; SameSite=Strict; Secure  │                                 │
      │◀──────────────────────────────────────│                                 │
      │                                       │                                 │
      │ 4. GET /api/auth/me (silent 401 probe)│                                 │
      │──────────────────────────────────────▶│ 5. validate JWT from cookie     │
      │◀───────────────── user + roles ───────│    resolve role enum            │
      │                                       │    (platform/org/project)       │
      │                                       │                                 │
      │ 6. POST /api/projects {name, orgId}   │                                 │
      │──────────────────────────────────────▶│ 7. AuthZ: caller ∈ org? RLS     │
      │                                       │    filter on org_id             │
      │                                       │────── INSERT project ──────────▶│
      │                                       │       (org_id NOT NULL)         │
      │◀────────────── 201 project ───────────│                                 │
      │                                       │                                 │
      │ 8. POST /api/projects/{id}/runs       │                                 │
      │    {strategy:DeComFL, rounds, model,  │                                 │
      │     dataset_version, hyperparams}     │                                 │
      │──────────────────────────────────────▶│ 9. validate + quota check       │
      │                                       │    (per-org concurrency)        │
      │                                       │────── INSERT fl_runs ──────────▶│
      │                                       │   state=PENDING, lease=NULL,    │
      │                                       │   determinism_manifest=...      │
      │◀────────────── 201 run (run_id) ──────│                                 │
      v                                       v                                 v
```

**Key v2 decisions visible here:**

- **Step 7** is the multi-tenant fix (risk **R8**): v1's `AuthorizationService` never checked `org_id` ([`README.md:97`](../../audit/2026-05-29/README.md)). v2 scopes every query by the caller's `org_id` (RLS-style), so a researcher in org A can never read org B's projects.
- **Step 9** adds the per-org concurrency quota the audit demands **before** lifting the 11-port cap, because once the cap is gone nothing else stops one tenant from spawning unbounded runs ([`B6-scale-cost.md:187`](../../audit/2026-05-29/B6-scale-cost.md), risk **R10**).
- **Step 9** writes the **determinism manifest** (seed, hyperparameters, library/dataset/model content hashes) into the `fl_runs` row at creation, so the run is reproducible from birth (risk **R14**).
- A run is **created** here but **not launched**. Launch is Flow B, triggered by an explicit `POST /api/projects/{id}/runs/{run_id}/start` or by the reconciler picking up the `PENDING` lease.

### 4.2 Flow B — An FL run is launched and trains over rounds with clients connecting

This is the rebuilt orchestration substrate (unit 3). The control plane is a **stateless supervisor over a durable DB lease** — the JVM no longer holds run state in a heap map (v1's fatal `ConcurrentHashMap<UUID,Process>` that was lost on restart, [`C1-reliability-sre.md` F4](../../audit/2026-05-29/C1-reliability-sre.md), risk **R9**).

```
Control Plane          FlServerLauncher        FL Server (long-running,        Clients
(Spring Boot)          (k8s/ECS/Local)         keyed on run_id; Python)        (desktop/jetson/mobile)
     │                       │                          │                            │
     │ 1. /runs/{id}/start   │                          │                            │
     │   acquire lease:      │                          │                            │
     │   UPDATE fl_runs SET  │                          │                            │
     │   state=LAUNCHING,    │                          │                            │
     │   lease_owner=<pod>,  │                          │                            │
     │   lease_expires=now+T │                          │                            │
     │   WHERE state=PENDING │                          │                            │
     │   (optimistic, 1 row) │                          │                            │
     │──────────────────────▶│ 2. launch(run_id,config) │                            │
     │                       │   k8s Job  (PRIMARY)      │                            │
     │                       │   ECS RunTask (secondary) │                            │
     │                       │   LocalProcess (dev only) │                            │
     │                       │─────────────────────────▶│ 3. server boots,           │
     │                       │                          │    reads model + dataset   │
     │                       │                          │    version from S3         │
     │                       │                          │    (content-addressed)     │
     │ 4. readiness probe    │                          │                            │
     │   poll gRPC           │                          │                            │
     │   GetServerStatus     │◀─────────────────────────│ (WAITING_FOR_CLIENTS)      │
     │   until READY or T_out│                          │                            │
     │   (NOT a 3s sleep —   │                          │                            │
     │    fixes C1 F3)       │                          │                            │
     │   UPDATE state=RUNNING │                          │                            │
     │                       │                          │ 5. clients connect ────────│◀──┐ enrollment
     │                       │                          │    gRPC (TLS+mTLS),         │   │ token from
     │                       │                          │    cert-CN-bound identity   │   │ control plane
     │                       │                          │◀────────────────────────────│   │
     │                       │                          │                            │   │
     │                       │       ┌──── ROUND LOOP (per round r) ─────────────────────┐
     │                       │       │ 6. server sends seeds (DeComFL) / params (FedAvg) │
     │                       │       │    to clients via training stub                   │
     │                       │       │ 7. clients train locally; raw data stays put;     │
     │                       │       │    parallel heartbeat stub keeps liveness while   │
     │                       │       │    training stub blocks in fit()                  │
     │                       │       │ 8. clients upload scalars (DeComFL) /             │
     │                       │       │    chunked params (FedAvg >300MB)                 │
     │                       │       │ 9. round completes when: all expected clients     │
     │                       │       │    reported  OR  (round_deadline elapsed AND      │
     │                       │       │    received >= min_quorum)  ← fixes C1 F5         │
     │                       │       │ 10. aggregate (DeComFL: 1/P factor; CPU-canonical │
     │                       │       │     RNG); write checkpoint to S3 (sha256);        │
     │                       │       │     append round to durable ledger BEFORE         │
     │                       │       │     advancing round counter (fixes C1 F1)         │
     │                       │       │ 11. POST RoundResult to /api/internal/runs/{id}/results │
     │                       │       │     PER ROUND (incremental, not batched)          │
     │                       │       └────────────────────────────────────────────────┘
     │ 12. on final round: state=COMPLETED; release lease                                │
     v                       v                          v                            v
```

**Reconciler loop (runs continuously in the control plane, in parallel to all of the above):**

```
 every RECONCILE_INTERVAL (e.g. 15s) and on ApplicationReadyEvent (boot):
   for each fl_runs row where state ∈ {LAUNCHING, RUNNING}:
      if lease_expires < now:                 # the owning pod died
         probe the launcher backend for run_id
         if backend reports the job alive:    re-adopt: refresh lease
         else:                                mark state=INTERRUPTED
                                              → eligible for resume from last checkpoint
   for each fl_runs row where state == PENDING and quota available:
      attempt to acquire lease and launch (same as Flow B step 1)
```

**Key v2 decisions visible here:**

- **`FlServerLauncher` abstraction with three backends** ([`README.md:43`](../../audit/2026-05-29/README.md)): **Kubernetes (k8s) Jobs** is primary/production; **AWS Elastic-Container-Service (ECS) RunTask** is the secondary path (salvaged and completed from v1's fire-and-forget `startEcsFargateServer`); **LocalProcessLauncher** keeps v1's `ProcessBuilder` model but **dev-only**. All three sit behind one Java interface so the control-plane code is backend-agnostic.
- **Long-running, run-keyed server** replaces "one process per project." This is the single biggest argument in the tech-stack audit — production FL substrates (Flower's SuperLink, NVIDIA FLARE's Service-and-Control-Process) all multiplex many runs on one long-running server "without requiring extra open ports" ([`B2-tech-stack.md:57`](../../audit/2026-05-29/B2-tech-stack.md)); v1's 11-port cap was a worse reimplementation of exactly that ([`B2-tech-stack.md:60`](../../audit/2026-05-29/B2-tech-stack.md)).
- **Lease + reconciler** make the JVM stateless. A redeploy, OOM (Out-Of-Memory) kill, or `SIGKILL` no longer orphans a run; the reconciler re-adopts or resumes it ([`C1-reliability-sre.md` §3.3](../../audit/2026-05-29/C1-reliability-sre.md)).
- **Round deadline + min-quorum** at step 9 fixes the v1 wedge where "a single client crashing mid-round hangs the entire run indefinitely" ([`C1-reliability-sre.md` F5](../../audit/2026-05-29/C1-reliability-sre.md), risk **R9**). DeComFL averaging already divides by clients actually received, so partial quorum is mathematically fine.
- **Per-round checkpoint + ledger written before advancing the round counter** (step 10) fixes v1's "crash at round 4/5 writes nothing" ([`C1-reliability-sre.md` F1](../../audit/2026-05-29/C1-reliability-sre.md)). DeComFL's reconstructable per-round state is `O(K·P)` scalars — tiny — so this is cheap.
- **mTLS (mutual Transport-Layer-Security) with cert-CN-bound (Common-Name-bound) identity** replaces v1's plaintext-by-default gRPC and self-asserted `client_id` (the Sybil hole, risk **R6**). TLS code already exists in v1 but is unused ([`B6-scale-cost.md:77`](../../audit/2026-05-29/B6-scale-cost.md)); v2 defaults it on.

### 4.3 Flow C — Per-round telemetry + checkpoint + live logs reach the dashboard

This is the FL-run telemetry pipeline (`RoundResult` → `/api/internal/runs/{run_id}/results` → STOMP → recharts; the exact path/shape are in [`04-API-CONTRACTS.md`](04-API-CONTRACTS.md) §5), salvaged from v1 but with the critical change that the per-round POST is **incremental, not batched after the run completes** ([`README.md:167`](../../audit/2026-05-29/README.md); v1's producer existed but POSTed in one batch after the run, so the chart stayed empty during training).

```
 FL Server (run_id)        Control Plane              STOMP relay        Browser (React + recharts)
     │                          │                     (Redis/RabbitMQ)         │
     │ per round r:             │                         │                    │
     │ 1. write checkpoint to   │                         │                    │
     │    S3, content-addressed │                         │                    │
     │    by sha256             │                         │                    │
     │    ──────────▶ [S3/MinIO]│                         │                    │
     │                          │                         │                    │
     │ 2. POST /api/internal/   │                         │                    │
     │    runs/{run_id}/results │                         │                    │
     │    {serverRound, loss,   │                         │                    │
     │     accuracy, uplinkBytes,│                        │                    │
     │     downlinkBytes,        │                        │                    │
     │     scalarsTransmitted}   │                        │                    │
     │    Authorization:        │                         │                    │
     │    per-run scoped token  │                         │                    │
     │    (NOT global key,      │                         │                    │
     │     fixes R-token reuse) │                         │                    │
     │─────────────────────────▶│ 3. validate scoped      │                    │
     │                          │    token → run_id        │                    │
     │                          │────── INSERT round_results ──▶ [Postgres]     │
     │                          │ 4. publish to            │                    │
     │                          │   /topic/results/{projectId}                  │
     │                          │   (payload carries run_id, 04 §11)            │
     │                          │──────────────────────────▶│ 5. fan-out to     │
     │                          │                          │   all replicas/    │
     │                          │                          │   subscribers      │
     │                          │                          │───────────────────▶│ 6. recharts
     │                          │                          │                    │    appends point
     │                          │                          │                    │    LIVE per round:
     │                          │                          │                    │    - loss curve
     │                          │                          │                    │    - accuracy
     │                          │                          │                    │    - COMM-COST
     │                          │                          │                    │      panel (the
     │                          │                          │                    │      DeComFL wedge)
     │                          │                          │                    │
     │ stdout/stderr lines ─────│ (separate channel)       │                    │
     │ → captured by launcher   │                         │                    │
     │ → /topic/logs/{projectId}│─────────────────────────▶│───────────────────▶│ 7. live log
     │                          │                         │                    │    console
     v                          v                         v                    v

 In parallel, EVERY hop above carries a W3C traceparent header:
   FL Server span → /api/internal/runs/{run_id}/results span → STOMP publish span
   → OTel Collector → Tempo (traces) / Loki (logs) / Prometheus (metrics) → Grafana
```

**Key v2 decisions visible here:**

- **Incremental per-round POST** (step 2) so the dashboard populates live during training, the single re-scoped finding from the audit ([`README.md:167`](../../audit/2026-05-29/README.md)).
- **Per-run scoped result token** (step 2) replaces v1's single global internal API key ([`README.md:98`](../../audit/2026-05-29/README.md)). A compromised FL server can only POST results for *its* run, not impersonate any run.
- **Communication-cost panel** (step 6): the dashboard surfaces `uplinkBytes`/`downlinkBytes` and `scalarsTransmitted` per round (the `round_results` comm-cost columns, `04-API-CONTRACTS.md §5.1`). This is the **DeComFL bandwidth wedge made visible** — the product's whole story rendered as a chart ([`README.md:52`](../../audit/2026-05-29/README.md), [`B6-scale-cost.md:20`](../../audit/2026-05-29/B6-scale-cost.md)).
- **Logs are a separate channel from metrics** (step 7) and **never in the FL-progress critical path** ([`C1-reliability-sre.md` F7](../../audit/2026-05-29/C1-reliability-sre.md)): a slow log subscriber must not back-pressure and wedge the FL server's stdout pipe.
- **W3C `traceparent` end to end** so one run's trace stitches across JVM → Python → client → mobile (risk **R14** reproducibility + B3 observability rebuild).

---

## 5. Deployment topology

### 5.1 Production (Kubernetes, primary path)

```
                              Internet
                                 │ HTTPS (443)
                                 ▼
                    ┌────────────────────────┐
                    │  Ingress / Load Balancer│  (TLS termination; HSTS)
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────────────┐
            ▼                   ▼                            ▼
   ┌─────────────────┐  ┌─────────────────┐        ┌──────────────────┐
   │ control-plane    │  │ control-plane    │  ...   │ STOMP relay       │
   │ pod (Spring Boot)│  │ pod (Spring Boot)│        │ (Redis/RabbitMQ)  │
   │ replica 1        │  │ replica N        │        │ fan-out /topic/*  │
   │ mgmt port:       │  │ mgmt port:       │        │ across replicas   │
   │ Micrometer→Prom  │  │ Micrometer→Prom  │        └──────────────────┘
   └───────┬──────────┘  └───────┬──────────┘
           │ acquire fl_runs lease, launch via FlServerLauncher
           └───────────────┬───────────────────────────────┐
                           ▼ (k8s Job per run_id)            │ (managed conn)
   ┌──────────────────────────────────────────────┐         ▼
   │ FL-server Jobs (one long-running pod per run) │  ┌─────────────────────┐
   │  ┌─────────────┐  ┌─────────────┐             │  │ PostgreSQL (RDS,     │
   │  │ run A pod    │  │ run B pod    │  ...        │  │ Multi-AZ)            │
   │  │ DeComFL      │  │ FedAvg       │             │  │ users/orgs/projects/ │
   │  │ Job          │  │ Job          │             │  │ fl_runs lease +      │
   │  └──────┬───────┘  └──────┬───────┘             │  │ round_results        │
   │         │ Pod Failure Policy: OOM/drain→resume  │  └─────────────────────┘
   └─────────┼─────────────────┼─────────────────────┘
             │ gRPC (TLS+mTLS), scalars/seeds only    
             ▼                 ▼                       ┌─────────────────────┐
        FL CLIENTS (outside the cluster, on customer  │ S3 / MinIO           │
        hardware): desktop (Tauri), Jetson (docker),  │ content-addressed    │
        mobile (native C++).  RAW DATA NEVER ENTERS    │ checkpoints (sha256) │
        THE CLUSTER.                                   └─────────────────────┘

   Sidecar / cluster services:
   ┌──────────────────────────────────────────────────────────────────────┐
   │ OTel Collector ──▶ Tempo (traces) / Loki (logs) / Prometheus (metrics) │
   │                    ──▶ Grafana (dashboards)                            │
   │ MLflow (self-hosted, Apache-2.0) — Model Registry + run lineage        │
   └──────────────────────────────────────────────────────────────────────┘
```

**Topology facts (normative):**

- The control plane runs **N replicas behind a load balancer**. This is only safe once the STOMP broker is the relay (Redis/RabbitMQ), because the in-memory broker cannot route `/topic/*` user-destinations across replicas ([`B6-scale-cost.md:42`](../../audit/2026-05-29/B6-scale-cost.md)). The relay swap is one line ([`README.md:117`](../../audit/2026-05-29/README.md)).
- **One k8s Job per `run_id`**, long-running for the run's life, with a **Pod Failure Policy** distinguishing retriable faults (OOM, node drain → resume from checkpoint) from terminal faults (config error → fail the run) ([`C1-reliability-sre.md` §3.3](../../audit/2026-05-29/C1-reliability-sre.md)).
- **Postgres is single-writer managed RDS, Multi-Availability-Zone (Multi-AZ).** No Citus, no sharding — the control-plane tables (orgs/users/projects/memberships) are bounded ([`B6-scale-cost.md:164`](../../audit/2026-05-29/B6-scale-cost.md)). Aurora is considered only at hyperscale. Append-only telemetry (`round_results`, logs) is the thing that grows; it is routed to Loki/object storage, not allowed to bloat the OLTP DB ([`B6-scale-cost.md:162`](../../audit/2026-05-29/B6-scale-cost.md)).
- **Model artifacts live in S3/MinIO, content-addressed by sha256**, never as Postgres blobs ([`B2-tech-stack.md:171`](../../audit/2026-05-29/B2-tech-stack.md)).
- The cost driver at this tier is **FL-server compute task-hours**, not the cloud baseline ([`B6-scale-cost.md` §3.2](../../audit/2026-05-29/B6-scale-cost.md)); per-org quotas + scale-to-zero keep it bounded.

### 5.2 Local-dev topology

```
   Developer laptop (Apple M4 Max, 36GB unified memory)
   ┌────────────────────────────────────────────────────────────┐
   │  Frontend: `npm run dev`  (Vite :5173, strictPort)           │
   │      │ HTTPS/WS proxy → :8081                                │
   │      ▼                                                        │
   │  Control plane: SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun │
   │      :8081, in-memory STOMP broker (single replica, fine)    │
   │      │ FlServerLauncher = LocalProcessLauncher (dev only)    │
   │      ▼                                                        │
   │  FL server: spawned local Python process (run_id-keyed)      │
   │      │ gRPC plaintext OK on loopback (dev only)              │
   │      ▼                                                        │
   │  FL client(s): local Python / docker                          │
   │                                                              │
   │  Postgres: Testcontainers OR local docker postgres           │
   │      (NOT H2 — H2 hid Postgres dialect bugs in v1)           │
   │  S3:  MinIO in docker  (content-addressed, same API as prod) │
   │  Observability: optional local Grafana/Loki/Tempo via compose│
   └────────────────────────────────────────────────────────────┘
```

**Local-dev facts (normative):**

- **`LocalProcessLauncher` is the dev-only backend** of the `FlServerLauncher` abstraction. It keeps v1's `ProcessBuilder` model so a developer needs no Kubernetes. It is **never** used in a deployed environment ([`README.md:43,99`](../../audit/2026-05-29/README.md)).
- **Postgres in dev, not H2.** v1's H2 hid Postgres-dialect bugs; v2 uses Testcontainers-Postgres in CI and local docker Postgres in dev so dev/CI/prod share one dialect ([`B2-tech-stack.md:168`](../../audit/2026-05-29/B2-tech-stack.md)). The only exception is the `test` profile's in-memory H2 with Flyway disabled — that must stay as-is.
- **MinIO in dev** gives the same S3 API locally, so artifact code is identical in dev and prod.
- The whole dev stack fits the M4 Max's 36GB unified-memory ceiling: a small CNN/MLP run, one control-plane JVM, one Postgres, one MinIO, one FL server, two FL clients. LLM-scale runs are a cloud/cluster concern, not a laptop concern.

---

## 6. Cross-cutting concerns map

Four concerns cut across every unit. This table is the contract for *where* each is implemented; the LLDs specify *how*.

| Concern | What it guarantees | Where implemented (units) | Key v2 rule |
|---|---|---|---|
| **Auth** | Only authenticated principals act; tokens are not JS-readable | Control plane (1) issues + validates cookie JWT; Frontend (6) sends `withCredentials:true`; Desktop (7) stores JWT in OS keychain; STOMP handshake re-uses the cookie | Cookie-only HttpOnly JWT. **No `Authorization: Bearer` header anywhere.** No `localStorage` token. |
| **Authorization (multi-tenancy)** | A principal sees only their org's data; runs can only be acted on by their org | Authorization layer (2) inside the control plane; every Postgres (10) query is `org_id`-scoped (RLS-style) | Role collapses to one **enum** across platform/org/project layers. `projects.org_id` is `NOT NULL`. Per-run scoped result tokens, not a global key. (Fixes R7, R8.) |
| **Observability / tracing** | One run's metrics, logs, and trace stitch across all hops | Observability stack (8) collects; control plane (1), FL framework (4), and mobile (5) all **emit** W3C `traceparent` and structured logs with `project_id`/`round_idx`/`trace_id` | `traceparent` propagated JVM → spawned Python → client → mobile. Metrics on an internal management port. Cardinality budget: `client_id` off histograms (cost control, [`B6-scale-cost.md:193`](../../audit/2026-05-29/B6-scale-cost.md)). |
| **Reproducibility** | Any completed run can be re-run bit-for-bit | FL framework (4) + Mobile (5) use CPU-canonical RNG; Data/artifact store (9) content-addresses datasets, models, checkpoints by sha256; control plane (1) writes a determinism manifest into `fl_runs` (10) | Determinism manifest = {seed, hyperparameters, library hash, dataset hash, model hash}. **CPU-canonical RNG everywhere** + a golden-vector parity test gating Python↔C++ determinism. (Fixes R3, R14.) |

**Cross-cutting concern interactions worth stating explicitly:**

- **Auth + multi-tenancy together** are why the control plane is the only unit that talks to Postgres for identity. The FL server never queries the identity tables; it receives a scoped token and a `run_id` and nothing else. This keeps the tenant-isolation surface small.
- **Observability + reproducibility share the `trace_id`/`run_id` keys**, so a run's Grafana trace and its MLflow lineage entry are joinable. This is deliberate: debugging a non-reproducible run starts from the trace.
- **Reproducibility is gated in CI** by the golden-vector parity test — a frozen set of expected scalars that Python and the C++ mobile core must both reproduce from the same seed. If they diverge, mobile aggregation is silently corrupt (risk **R3**), so the test is a merge gate, not a nicety.

---

## 7. Reasoning — the top 8 architecture decisions

Each decision states the choice, the rejected alternative, the reasoning, and the audit finding it ties to. This is the section the implementer must internalize: the *why* constrains every downstream LLD.

### D1 — Rebuild the FL substrate as a long-running, run-keyed server; do not keep per-project process spawn

- **Choice:** One long-running FL server **per `run_id`**, launched via the `FlServerLauncher` abstraction (k8s Jobs primary). A durable `fl_runs` lease in Postgres + a reconciler loop make the JVM a stateless supervisor.
- **Why-not the alternative (keep v1's `ProcessBuilder`-per-project + 11-port map):** v1's model is `1 OS process + 1 of 11 TCP ports per concurrent project`, with the `Process` handle in an in-memory map lost on JVM restart ([`README.md:23`](../../audit/2026-05-29/README.md)). It caps at 11 concurrent runs, races under concurrent `/start`, and orphans every run on redeploy (split-brain, [`C1-reliability-sre.md` F4](../../audit/2026-05-29/C1-reliability-sre.md)).
- **Why-not adopt Flower/FLARE wholesale:** they solve the multiplexing cleanly, but **neither has a first-class native C++ on-device client** ([`B2-tech-stack.md:18`](../../audit/2026-05-29/B2-tech-stack.md)), and DeComFL's scalar-only protocol fits Flower's `Parameters` model awkwardly. The native C++ mobile core (unit 5) and DeComFL being the product are what tip the decision to "custom substrate, rebuilt" rather than "adopt" ([`B2-tech-stack.md:146-150`](../../audit/2026-05-29/B2-tech-stack.md)).
- **Audit tie:** risk **R9**; synthesis verdict "Rebuild" ([`README.md:43,100`](../../audit/2026-05-29/README.md)).

### D2 — `FlServerLauncher` is an abstraction with three backends, not a single hardcoded launcher

- **Choice:** One Java interface, three implementations — **k8s Jobs (primary, production)**, **ECS RunTask (secondary)**, **LocalProcessLauncher (dev only)**.
- **Why-not pick one backend:** the platform must run on a laptop with zero Kubernetes (dev), on the existing AWS investment (ECS, already half-coded in v1's `startEcsFargateServer`), and on production k8s. Hardcoding any one of these blocks the other two.
- **Why this shape:** it lets the conflict between four audit agents (KILL vs REBUILD vs REFACTOR the substrate) resolve cleanly — KILL the local model for non-dev, REBUILD the concept, SALVAGE-and-complete ECS, keep ProcessBuilder behind the dev-only backend ([`README.md:157`](../../audit/2026-05-29/README.md)).
- **Audit tie:** explicit conflict resolution #1 ([`README.md:157`](../../audit/2026-05-29/README.md)).

### D3 — Keep DeComFL custom; keep the `fedlearn.v2` gRPC contract custom (governed by `buf`)

- **Choice:** DeComFL stays a custom PyTorch implementation, with the three correctness fixes already specified (`1/P` factor, CPU-canonical RNG, serializer symmetry; see [`docs/v2/specs/2026-05-29-decomfl-correctness-design.md`](../specs/2026-05-29-decomfl-correctness-design.md)). The gRPC contract stays custom in package `fedlearn.v2`, with `buf` as the single source of truth + a breaking-change gate.
- **Why-not adopt a framework's protocol:** DeComFL is the **only** paper-backed differentiator and **no off-the-shelf framework (Flower, FLARE, PySyft, FedML) ships it** ([`B2-tech-stack.md:75`](../../audit/2026-05-29/B2-tech-stack.md)). The proto is consumed by four languages (Java, Python, TS, C++); v1 had drift between vendored copies. `buf` kills the drift class without abandoning the custom proto ([`B2-tech-stack.md:152-156`](../../audit/2026-05-29/B2-tech-stack.md)).
- **Audit tie:** "SALVAGE DeComFL" ([`README.md:147`](../../audit/2026-05-29/README.md)); "REFACTOR→buf" ([`README.md:144`](../../audit/2026-05-29/README.md)).

### D4 — Round loop MUST have a deadline + minimum-quorum; no infinite hang on a straggler

- **Choice:** A round completes when **all expected clients reported OR (a per-round deadline elapsed AND received ≥ `min_quorum`)**. Wire the dead-client liveness check that v1 left as dead code.
- **Why-not v1's "wait for exactly `clients_per_round`":** in a 100-client federation, one client crashing mid-round hangs the **entire run indefinitely**, pinning a process + port until a human hits `/stop` ([`C1-reliability-sre.md` F5](../../audit/2026-05-29/C1-reliability-sre.md)). v1 even has the liveness machinery (`is_client_alive`, `should_stop`) but never consults it.
- **Why it is safe:** DeComFL aggregation already divides by clients actually received, and FedAvg weights renormalize naturally, so partial quorum is mathematically correct, not a hack ([`C1-reliability-sre.md` §3.2](../../audit/2026-05-29/C1-reliability-sre.md)).
- **Audit tie:** risk **R9**; "Round loop: REBUILD" ([`README.md:102`](../../audit/2026-05-29/README.md)).

### D5 — Per-round durable checkpoint + ledger; artifacts content-addressed in S3/MinIO, never in Postgres

- **Choice:** Write the round ledger (`seeds`, `avg_gradients`, `client_last_round`, `loss`, `model_hash`) **and** the model checkpoint to object storage **before** advancing the round counter. Models are content-addressed by sha256.
- **Why-not v1's terminal-only, destructive in-place save:** v1 writes the model once, destructively, at the very end ([`C1-reliability-sre.md` F1, F6](../../audit/2026-05-29/C1-reliability-sre.md)); a 6-hour run that dies at hour 5 restarts at round 1. There is no off-host copy and no versioning.
- **Why it is cheap:** DeComFL's reconstructable per-round state is `O(K·P)` scalars — kilobytes — so durable per-round checkpointing costs almost nothing ([`C1-reliability-sre.md` §0, §3.1](../../audit/2026-05-29/C1-reliability-sre.md)).
- **Why content-addressing:** it deduplicates identical base models across orgs and is the prerequisite for reproducibility and round-recovery ([`README.md:47`](../../audit/2026-05-29/README.md), risk **R14/R16**).
- **Audit tie:** risks **R9, R14, R16**; "Per-round checkpoint/resume: REBUILD", "Artifact store: REBUILD" ([`README.md:103,109`](../../audit/2026-05-29/README.md)).

### D6 — Managed PostgreSQL, single-writer, no Citus/sharding; H2 dies outside the `test` profile

- **Choice:** Managed PostgreSQL (AWS RDS), Flyway-owned schema, JPA in validate-only mode. No Citus. Aurora only at hyperscale.
- **Why-not Citus/distributed Postgres:** the control-plane tables (orgs/users/projects/memberships) are **bounded** per the V5 identity model; there is no evidence they need horizontal sharding. The thing that grows is append-only telemetry, which belongs in Loki/object storage, not a sharded RDBMS ([`B6-scale-cost.md:164`](../../audit/2026-05-29/B6-scale-cost.md)).
- **Why-not keep H2:** H2-file-mode is a proof-of-concept crutch that hid Postgres-dialect bugs in v1; CI must use Testcontainers-Postgres to surface them ([`B2-tech-stack.md:168`](../../audit/2026-05-29/B2-tech-stack.md)). Must fix `audit_events.metadata` from `CLOB` to `TEXT`/`JSONB` before cutover ([`README.md:169`](../../audit/2026-05-29/README.md)).
- **Why-not Aurora now:** Aurora Serverless v2's idle floor (~$44/month at 0.5 Aurora-Capacity-Units) is *worse* than provisioned RDS for steady load ([`B6-scale-cost.md:106`](../../audit/2026-05-29/B6-scale-cost.md)).
- **Audit tie:** conflict resolution #7 ([`README.md:169`](../../audit/2026-05-29/README.md)); "DB: H2-file KILL → Postgres" ([`B6-scale-cost.md:207`](../../audit/2026-05-29/B6-scale-cost.md)).

### D7 — Default-secure transport (TLS + mTLS) + org-scoped authorization + per-run scoped tokens

- **Choice:** gRPC defaults to TLS with mTLS, client identity bound to the certificate CN plus a backend-issued enrollment token. Every control-plane query is `org_id`-scoped (RLS-style). The FL server authenticates to `/api/internal/runs/{run_id}/results` with a **per-run scoped token**, not a global key.
- **Why-not v1's posture:** v1's gRPC is **plaintext by default** and the `client_id` is self-asserted (a Sybil hole), even though full TLS+mTLS already exists in the code unused ([`B6-scale-cost.md:77`](../../audit/2026-05-29/B6-scale-cost.md), risk **R6**); v1's `AuthorizationService` never checks `org_id`, leaking cross-org PUBLIC metadata (risk **R8**); and a single global internal key means any FL server could impersonate any run.
- **Why this matters now:** the pneumonia/healthcare demo makes HIPAA (Health Insurance Portability and Accountability Act)-readiness the floor, so default-secure + tenant isolation is not optional ([`README.md:191`](../../audit/2026-05-29/README.md), risk **R11**).
- **Audit tie:** risks **R6, R8**; "Multi-tenant org isolation: REBUILD", "Internal result callbacks: REFACTOR" ([`README.md:97,98`](../../audit/2026-05-29/README.md)).

### D8 — Salvage the Spring Boot control plane and React frontend; do not rewrite them

- **Choice:** Keep Spring Boot 3.5+ LTS / Java 21 for the control plane and React 19 + Vite 6 for the frontend. Bump off the EOL (End-Of-Life) Spring Boot 3.4.5; add TanStack Query, CSP/HSTS, and a real test layer.
- **Why-not the startup instinct to rewrite the "slow Java" in Go/FastAPI:** the Java control plane is the **working, valuable, least-broken layer** — the auth/RBAC (Role-Based Access Control)/audit/V5 identity investment is hard to reproduce quickly elsewhere ([`B2-tech-stack.md:178`](../../audit/2026-05-29/B2-tech-stack.md)). The *substrate* (Python) is the problem, and that is already Python. "Don't rewrite the healthy organ."
- **Why-not Next.js / Server-Side Rendering for the frontend:** there is no SSR need for a logged-in dashboard SPA; React + Vite is the right tool ([`README.md:118`](../../audit/2026-05-29/README.md)).
- **Audit tie:** "Control plane: SALVAGE", "Frontend React/Vite: SALVAGE" ([`README.md:95,118`](../../audit/2026-05-29/README.md)); risk **R13** (bump off EOL Spring Boot).

---

## 8. What this HLD does NOT decide (handed to the LLDs)

To keep the boundary clear for the implementer, the following are **explicitly out of scope for this HLD** and owned by the named LLD:

| Deferred decision | Owning LLD |
|---|---|
| Exact REST endpoint signatures, request/response Data-Transfer-Objects, the role enum's exact values, the `fl_runs` lease columns and state machine | 10- Control-plane LLD |
| The `FlServerLauncher` Java interface signature, the k8s Job spec, the reconciler's exact lease-timeout values, per-org quota schema | 11- Orchestration-substrate LLD |
| The `fedlearn.v2` proto messages, the DeComFL `1/P`/RNG/serializer fix code, the chunking threshold and chunk size, the dual-heartbeat `threading.Event` design | 12- FL-framework LLD (DeComFL math already specified in [`docs/v2/specs`](../specs/) and [`docs/v2/plans`](../plans/)) |
| The C++ libtorch class layout, the golden-vector test vectors, the CPU-canonical RNG bridge | 13- Mobile-FL LLD |
| The TanStack Query cache keys, Zod schemas at the wire boundary, the exact CSP header, recharts panel components | 14- Frontend LLD |
| The Tauri Rust command layer, bollard Docker calls, the minisign updater feed format, keychain integration | 15- Desktop LLD |
| The Micrometer meter names, the OTel Collector pipeline config, the structlog field schema, the cardinality budget enforcement | 16- Observability LLD |
| The S3 bucket/key layout, the sha256 content-addressing scheme, the Flyway dataset/partition-registry migration, the MLflow deployment | 17- Data-and-artifact LLD |
| The exact Flyway migration files, the JPA entity mapping, the `CLOB`→`TEXT`/`JSONB` fix, RLS implementation mechanics | 18- Datastore LLD |

---

## 9. Glossary (all acronyms, alphabetical)

| Acronym | Full form |
|---|---|
| API | Application Programming Interface |
| ARM64 | 64-bit Advanced RISC (Reduced Instruction Set Computer) Machine |
| ASCII | American Standard Code for Information Interchange |
| CI | Continuous Integration |
| CN | Common Name (of an X.509 certificate) |
| CPU | Central Processing Unit |
| CSP | Content-Security-Policy |
| DeComFL | Dimension-Free Communication Federated Learning (the v1 wiki's "Decomposed" expansion is wrong per the paper, [`B1-paper-alignment.md:33`](../../audit/2026-05-29/B1-paper-alignment.md)) |
| DLG | Deep Leakage from Gradients |
| DP | Differential Privacy |
| ECS | (AWS) Elastic Container Service |
| EOL | End Of Life |
| FedAvg | Federated Averaging |
| FL | Federated Learning |
| gRPC | Google Remote Procedure Call |
| HIPAA | Health Insurance Portability and Accountability Act |
| HLD | High-Level Design |
| HSTS | HTTP Strict Transport Security |
| HTTPS | HyperText Transfer Protocol Secure |
| IP | Intellectual Property |
| IPC | Inter-Process Communication |
| IPMO | (RIT) Intellectual Property Management Office |
| JPA | Jakarta Persistence API |
| JVM | Java Virtual Machine |
| JWT | JSON (JavaScript Object Notation) Web Token |
| k8s | Kubernetes |
| LLD | Low-Level Design |
| LLM | Large Language Model |
| LTS | Long-Term Support |
| MB | Megabyte |
| MSW | Mock Service Worker |
| mTLS | mutual Transport Layer Security |
| Multi-AZ | Multi Availability Zone |
| OLTP | Online Transaction Processing |
| OOM | Out Of Memory |
| OS | Operating System |
| OTel | OpenTelemetry |
| RBAC | Role-Based Access Control |
| RDS | (AWS) Relational Database Service |
| REST | Representational State Transfer |
| RIT | Rochester Institute of Technology |
| RLS | Row-Level Security |
| RNG | Random Number Generator |
| RPO | Recovery Point Objective |
| RTO | Recovery Time Objective |
| S3 | (AWS) Simple Storage Service |
| SBOM | Software Bill of Materials |
| SOC 2 | System and Organization Controls 2 |
| SPA | Single-Page Application |
| SRE | Site Reliability Engineering |
| SSR | Server-Side Rendering |
| STOMP | Simple Text Oriented Messaging Protocol |
| TCP | Transmission Control Protocol |
| TLS | Transport Layer Security |
| TS | TypeScript |
| URL | Uniform Resource Locator |
| ZO | Zeroth-Order (optimization) |

---

## 10. Source ledger

**Audit reports (this repo, read for this HLD):**
- Master synthesis — [`docs/audit/2026-05-29/README.md`](../../audit/2026-05-29/README.md)
- Tech stack & architecture — [`docs/audit/2026-05-29/B2-tech-stack.md`](../../audit/2026-05-29/B2-tech-stack.md)
- Scale, cost & infra economics — [`docs/audit/2026-05-29/B6-scale-cost.md`](../../audit/2026-05-29/B6-scale-cost.md)
- Reliability / SRE (Site Reliability Engineering) — [`docs/audit/2026-05-29/C1-reliability-sre.md`](../../audit/2026-05-29/C1-reliability-sre.md)

**v2 spec/plan referenced for the DeComFL fix scope (not re-litigated here):**
- [`docs/v2/specs/2026-05-29-decomfl-correctness-design.md`](../specs/2026-05-29-decomfl-correctness-design.md)
- [`docs/v2/plans/2026-05-29-decomfl-correctness-plan.md`](../plans/2026-05-29-decomfl-correctness-plan.md)

**Existing v1 code cited (verified during authoring):**
- gRPC proto package `fedlearn.v1`, `option java_package = "com.fedlearn.v1"` — `framework/src/fedlearn/communication/protos/fedlearn.proto:3,5`
- 11-port range `50000–50010` — `backend/fl-platform-api/src/main/resources/application.properties:120-121`

**External / market sources (URLs):**
- DeComFL paper, ICLR 2025 — https://arxiv.org/abs/2405.15861
- Flower architecture (multi-run, no extra ports) — https://flower.ai/docs/framework/explanation-flower-architecture.html
- NVIDIA FLARE multi-job architecture — https://nvflare.readthedocs.io/en/2.6.0/user_guide/flower_integration/flare_multi_job_architecture.html
- Kubernetes JobSet — https://jobset.sigs.k8s.io/docs/overview/
- Kubernetes Pod Failure Policy (GA 1.31) — https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/
- AWS Fargate pricing — https://aws.amazon.com/fargate/pricing/
- AWS S3 pricing — https://aws.amazon.com/s3/pricing/
- AWS Aurora pricing — https://aws.amazon.com/rds/aurora/pricing/

**Uncertainty flagged:** Whether DeComFL's exact wire-byte count matches the paper's "~1 MB total" for high-churn reconnecting clients is unverified at the byte level ([`B6-scale-cost.md:221`](../../audit/2026-05-29/B6-scale-cost.md)); the HLD treats it as the directional bandwidth wedge, which is well-supported, not as a hard SLA number. Cross-device RNG bit-parity (Python ↔ C++ mobile) is asserted by the DeComFL correctness spec but is gated by a golden-vector test in CI rather than assumed ([`C1-reliability-sre.md` §6](../../audit/2026-05-29/C1-reliability-sre.md), risk **R3**).
