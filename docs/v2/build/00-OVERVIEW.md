# 00 — MASTER OVERVIEW (FedLearn Platform v2)

**Document type:** The single entry point for the entire v2 (Version 2) build documentation set. Read this after the local-model usage guide and before any other build document.
**Audience:** a mid-sized local Language Model (LM, ~30 billion parameters, e.g. Qwen/Llama 32B running on an Apple M4 Max) that will implement the platform one task at a time. You are competent but you CANNOT infer missing context, CANNOT make architecture decisions, and lose the thread on ambiguity. This document gives you the map; it does not give you contracts — those live in the foundation docs (01–04) and the Low-Level Design (LLD) documents.
**Status:** Build-authoritative orientation. This document is a NAVIGATION and ORIENTATION layer; it never redefines a contract. Where it summarizes a contract, the named source document wins.
**Date authored:** 2026-05-29.
**Source of truth:** the v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md` (the reference architecture, the per-unit salvage/refactor/rebuild/kill decision table, the risk register R1–R17). The locked, pinned technology list is `02-TECH-STACK.md`; the frozen interface surface is `04-API-CONTRACTS.md`.

> **Abbreviation rule (house style, applies to everything you read and write):** the first time any acronym appears it is written in full followed by the short form in parentheses, e.g. "Federated Learning (FL)". After first use the short form is fine. The complete master glossary is in §6 of this document; every acronym used across the whole set is defined there so you never have to guess.

---

## 0. What this document is, and how to use it

This is the **table of contents and the map** for the v2 build. It does five things, in this order:

1. **§1 — The vision in plain language.** What FedLearn v2 is, why it exists, and the one differentiator the whole company is built on.
2. **§2 — The target architecture in one paragraph.** The shape of the system, so a single-unit task never contradicts the whole.
3. **§3 — The complete document map.** A table of every build document (00 through 91): what it is, what it owns, and when to read it.
4. **§4 — The reading order for you (the local model).** The exact sequence to read documents in, and the exact order to build units in.
5. **§5 — Shared facts repeated here so this file is self-contained** (the hard invariants, the IDs, the go/no-go gate).
6. **§6 — The master glossary** of EVERY acronym/term used anywhere in the set.

You do not implement anything from this document. You read it once to orient, then you go to `91-LOCAL-MODEL-GUIDE.md` for the operating rules and `90-BUILD-SEQUENCE.md` for the milestone-ordered build plan. You return here only to look something up (a glossary term, which doc owns a topic).

---

## 1. The v2 vision (plain language)

### 1.1 What FedLearn is, in one sentence

FedLearn is a **managed control plane for Federated Learning (FL)** — training a shared machine-learning model across many organizations' private devices **without any raw data ever leaving those devices** — whose differentiator is **DeComFL (Dimension-Free Communication Federated Learning)**: it can fine-tune a billion-parameter model across many private devices while sending **roughly one megabyte (MB) of data total over the whole training run**, instead of the tens of terabytes a conventional approach would move (`01-ARCHITECTURE-HLD.md §1.1`, citing the DeComFL paper, arXiv 2405.15861, and `docs/audit/2026-05-29/B6-scale-cost.md:20` quantifying a six-to-seven-order-of-magnitude egress reduction).

### 1.2 What "federated learning" means here, concretely

A hospital, a bank, or a phone owner has private data they will not upload. FL sends the model to the data, not the data to the model (`01-ARCHITECTURE-HLD.md §1.2`):

1. A central **FL server** holds the current shared model.
2. Each **FL client** (a hospital machine, a Jetson edge box, a phone, a desktop) trains locally on its own private data for a short burst.
3. The client sends back **only an update** — in DeComFL a handful of scalar numbers (a loss value and a perturbation seed); in **FedAvg (Federated Averaging)** the changed weights — never the raw data.
4. The server **aggregates** all clients' updates into a new shared model.
5. Repeat for many **rounds**.

The privacy claim is structural: raw data never leaves the device. DeComFL strengthens it — because clients upload only scalars, there are no gradients to invert, which **structurally eliminates the Deep-Leakage-from-Gradients (DLG) reconstruction attack family** (`docs/audit/2026-05-29/README.md:14`). This is both the bandwidth/edge-Large-Language-Model (LLM) wedge and the privacy story.

### 1.3 Why v2 is a greenfield rebuild (the audit verdict)

v1 (Version 1) is a competent proof-of-concept with **one genuine, paper-backed differentiator (DeComFL) and four classes of production blocker** (`docs/audit/2026-05-29/README.md:12`). v2 keeps the parts that work and rebuilds the orchestration substrate that does not. The four blocker classes the rebuild closes:

| Blocker class | What was broken in v1 | v2 fix (the milestone that closes it) |
|---|---|---|
| **1 — DeComFL is broken three ways** | Server drops the `1/P` averaging factor (global model steps P× too far); perturbation Random-Number-Generator (RNG) is device-dependent (silent aggregation corruption on any Graphics-Processing-Unit server or mixed fleet); the chunked serializer `KeyError`s on every transformer >100 MB upload. | M3 — the DeComFL correctness trifecta (risks R2/R3/R4). |
| **2 — identity/RBAC dead end-to-end** | The bootstrap admin gets `PLATFORM_ADMIN` but every admin route and the admin UI gate on a legacy `ADMIN` string, so the canonical admin is 403'd; a test seeds the literal `"ADMIN"` and masks it. | M4 — role enum + aligned constants (risk R7). |
| **3 — FL orchestration is a scaling cliff** | One Python process per project via `ProcessBuilder`, capped at 11 ports, in-memory process map lost on restart, no run entity, no checkpoint/resume, a round loop that hangs forever on one straggler. | M5 — the long-running run-keyed substrate (risks R9/R10). |
| **4 — supply-chain & security posture** | Unsigned Electron auto-install Remote-Code-Execution (RCE); gRPC (Google Remote Procedure Call) plaintext-by-default; no Pull-Request (PR)-time Continuous Integration (CI); no org-level multi-tenant isolation; End-Of-Life (EOL) Spring Boot. | M0 (CI), M4 (org isolation), M9 (Tauri signed updater), M11 (Differential Privacy + robust guard) — risks R5/R6/R8/R12/R13. |

### 1.4 The one non-technical go/no-go gate (state it, never hide it)

DeComFL is research from Rochester Institute of Technology (RIT). Under RIT Intellectual-Property (IP) policy C03.0, **RIT — not the founder — likely owns it** (`docs/audit/2026-05-29/README.md:27`, risk **R1**, the only Critical/go-no-go item). **The build (milestones M0–M13) may proceed in parallel, but no public launch or moat claim is defensible until an RIT Intellectual Property Management Office (IPMO) license or spin-out is executed.** This is not a build milestone (no code), but the plan never pretends it away. It is named in `01-ARCHITECTURE-HLD.md §1.4` and `90-BUILD-SEQUENCE.md §2`.

---

## 2. The target architecture (one paragraph)

FedLearn v2 is **five deployable units** behind one platform boundary. A **Spring Boot 3.5 Long-Term-Support (LTS) control plane** on **Java 21** (salvaged from v1) owns users, organizations, projects, and the `fl_runs` lease in a **managed PostgreSQL** datastore (schema owned by **Flyway** migrations, Jakarta-Persistence-API (JPA) validate-only); it serves a **React 19 + Vite 6** frontend over **Representational-State-Transfer (REST)** and streams live logs over **Simple-Text-Oriented-Messaging-Protocol (STOMP)-over-WebSocket**, authenticated by a **cookie-only HttpOnly JSON Web Token (JWT)** with no Bearer tokens. When a researcher starts a run, the control plane acts as a **stateless supervisor over a durable Postgres lease**: a **`FlServerLauncher`** abstraction launches a **long-running, multi-run FL server keyed on `run_id`** via one of three backends — **Kubernetes Jobs (primary, production)**, **Amazon Elastic-Container-Service (ECS) RunTask (secondary)**, or a **dev-only `LocalProcessLauncher`** — and a **reconciler loop** drives lifecycle from the lease (so a JVM (Java Virtual Machine) restart never loses a run, a straggler never hangs a round because every round loop has a **deadline + minimum-quorum**, and per-org concurrency quotas + scale-to-zero bound cost). The FL server is the **custom Python (PyTorch) framework — no Flower / `flwr` dependency** — implementing **FedAvg + DeComFL** with the three correctness fixes (the `1/P` factor, Central-Processing-Unit-canonical (CPU-canonical) RNG, symmetric serializer), parameter chunking for models >300 MB, and a dual gRPC stub model (training stub + parallel heartbeat stub); clients are the **Tauri v2 desktop app** and the **native C++ libtorch mobile core** (both speaking the `fedlearn.v2` gRPC contract over Transport-Layer-Security plus mutual-TLS (TLS + mTLS), uploading scalars/seeds only). Every run writes content-addressed (Secure-Hash-Algorithm-256-bit / sha256) checkpoints to **S3 (Simple Storage Service) or MinIO**, a determinism manifest + lineage to self-hosted **MLflow**, per-round telemetry incrementally to the dashboard's **recharts communication-cost panel**, and a single **W3C `traceparent`** stitched end to end (JVM → spawned Python → client → mobile) into **OpenTelemetry (OTel) Collector → Grafana/Loki/Tempo/Prometheus**. The full topology, the three data-flows, and the eight architecture decisions are in `01-ARCHITECTURE-HLD.md`.

---

## 3. The complete document map (read top-to-bottom only via §4's order)

Every document in `docs/v2/build/` and the two on-disk DeComFL companion docs. **The "Read when" column tells you the moment to open each.** The "Owns" column is the single authority for that topic — if two documents seem to disagree, the one that *owns* the topic wins.

### 3.1 Foundation documents (read all four fully, once, before any code)

| Doc | Title | What it is | Owns (the authority for) | Read when |
|---|---|---|---|---|
| **00** | `00-OVERVIEW.md` (this file) | The single entry point: vision, architecture-in-a-paragraph, the document map, the reading order, the master glossary. | Navigation, orientation, the master glossary. Owns NO contract. | Second (right after 91). Return to it to look things up. |
| **01** | `01-ARCHITECTURE-HLD.md` | The High-Level Design (HLD): the five-unit map, the three end-to-end data-flows, the deployment topology (dev + production), the eight architecture decisions with reasoning. | The system *shape*: what the units are, how they talk, where boundaries fall, and *why*. It does NOT define signatures, columns, or bodies. | Third. Before any code, after 02 is not required — but 01 gives the map 02–04 fill in. |
| **02** | `02-TECH-STACK.md` | The LOCKED, pinned technology list with exact versions: §24 consolidated version-pin table, §25 the eleven hard invariants, plus per-layer choices (runtimes, Spring, gRPC/buf, PyTorch, Postgres, S3, MLflow, STOMP relay, frontend, Tauri, mobile, k8s substrate, DP, observability, supply chain, CI). | Every pinned version and every "you may use exactly this technology and no other" decision. GOLDEN RULE 1 forbids anything not in here. | Fourth. Before any code. The only list of technologies you may use. |
| **03** | `03-DATA-MODEL.md` | The full data model: the V1–V5 salvaged identity baseline, the three new Flyway migrations (V6 dataset registry + role-guard CHECK constraints, V7 `fl_runs` lease + `round_results` + `model_artifacts` + the `CLOB`→`JSONB` fix, V8 `determinism_manifests`), the Postgres portability fixes, the JPA mapping notes, the 12-item checklist. | Every table, column, type, constraint, index, and Flyway migration. Schema is owned by Flyway, JPA is validate-only. | Fifth. Before any code. |
| **04** | `04-API-CONTRACTS.md` | Every wire contract: the REST endpoints, the `fedlearn.v2` gRPC `.proto` (verbatim), the STOMP topics, the standard error envelope, the per-run scoped result token, and the W3C `traceparent` propagation contract. | Every endpoint path, HTTP method, Data-Transfer-Object (DTO) field name/type, gRPC message, STOMP topic, error code, and token format. GOLDEN RULE 2 forbids changing anything in here. | Sixth. Before any code. |

### 3.2 Low-Level Design documents (read the one for the unit you are about to build, just-in-time)

The LLDs give the concrete interface signatures, the tricky algorithms in real code/pseudocode, the file structure, the ordered build-task checklist, and the conformance checklist for one unit.

| Doc | Title | What it is | Owns (the authority for) | Read when |
|---|---|---|---|---|
| **10** | `10-LLD-backend-control-plane.md` | The control-plane LLD: the decomposed service layer (`AuthService`, `ProjectService`, `FlRunService`, `RunReconcilerService`, `AuthorizationService`, …), the JPA entity/repository mapping, the REST controllers, the run-token filter, and the STOMP wiring. | The Spring Boot control-plane internals: controller→service→repository layering, org-scoped repositories, the run lifecycle facade, the internal-callback ingest, the reconciler. | Immediately before building M4 (the control plane). |
| **11** | `11-LLD-fl-framework.md` | The Python FL-framework LLD: the `strategies/` pure-math layer (FedAvg, DeComFL), the `estimators/` CPU-canonical perturbation, the `transport/` codec + gRPC servicer, the dual-heartbeat client, the telemetry callbacks, and the determinism manifest builder. | The custom FL server/client: the DeComFL `1/P`/CPU-RNG/serializer fixes, parameter chunking, the dual heartbeat, the golden-vector fixtures, the `fedlearn.v2` servicer. | Immediately before building M3 (the framework), alongside the DeComFL spec/plan. |
| **12** | `12-LLD-orchestration-substrate.md` | The orchestration substrate LLD: §5 the interfaces (`FlServerLauncher`, `FlRunSpec`, `FlRunService`, `LeaseManager`, `OrgQuotaService`), §6 the reconciler loop + lease SQL (`FOR UPDATE SKIP LOCKED`) + admission control + readiness probe + round deadline/quorum, §13 the 19-task checklist, §14 the conformance checklist. | The `FlServerLauncher` abstraction, the durable lease, the reconciler, quotas, the round deadline/quorum, per-round checkpoints, the run-token-authenticated internal callbacks. | Immediately before building M5 (the substrate). |
| **13** | `13-LLD-frontend-dashboard.md` | The frontend dashboard LLD: §4 the exact module tree, §5 the interfaces (Axios instance, Zod schemas, V5 role types, query-key factory, STOMP hook), §6 the flows (the 401 silent-probe interceptor, the one shared STOMP connection, the run-observability surface), §13 the 25-task checklist. | The React 19 SPA: TanStack Query server-state, Zod wire-boundary validation, the V5 role types (which fix the dead-admin-UI bug), the one shared STOMP connection, the recharts communication-cost panel, the Vitest+Playwright+Mock-Service-Worker (MSW) test layer. | Immediately before building M8 (the frontend). |
| **14** | `14-LLD-desktop-tauri.md` | The desktop LLD: the Tauri v2 Rust command layer (`auth`/`keychain`/`hardware`/`launcher`/`native_runner`/`docker_runner`/`updater`), the fail-closed IPC bridge, the bollard Jetson device-mount rule, and the signed minisign updater flow. | The Tauri v2 desktop orchestrator: the Rust command surface, OS-keychain JWT storage, the PyInstaller subprocess model, the bollard Docker path, the code-signed auto-updater. | Immediately before building M9 (the desktop). |
| **15** | `15-LLD-mobile.md` | The mobile-FL LLD: the React Native TypeScript shell, the TurboModule (JSI) bridge, and the native C++ libtorch on-device DeComFL core gated by the golden-vector parity test. | The mobile FL client: the RN bridge, the CPU-canonical C++ ZO core, the golden-vector parity gate, NativeWind/react-native-reusables styling. | Immediately before building M10 (the mobile core). |
| **16** | `16-LLD-observability.md` | The observability LLD: the concrete Micrometer/Prometheus metric names, span names, structlog fields, the three Grafana dashboards, the OTel Collector pipeline, and the incremental per-round telemetry emitter. | The metrics/logs/traces stack and the FL-run telemetry pipeline; the W3C `traceparent` propagation wiring JVM→Python→client→mobile; the comm-cost panel data. | Immediately before building M6 (observability). |
| **17** | `17-LLD-data-and-artifacts.md` | The data-and-artifact LLD: the Python `DataSource`/`Partitioner` ABCs, the content-addressed npz partition format, the S3/MinIO `ArtifactStore` client, the MLflow Model Registry wiring, and the determinism-manifest builder. | The dataset/partition registry, content-addressed (sha256) artifact store, MLflow lineage, and the reproducibility/determinism manifest. | Immediately before building M7 (artifact + dataset + lineage). |
| **18** | `18-LLD-security-and-compliance.md` | The cross-cutting security & compliance LLD: §5 the platform-auth interfaces (role enums, `JwtTokenProvider`, `SecurityConfig`, `AuthorizationService`, run token, STOMP authz), §6 the hard algorithms in code (cookie-JWT flow, org-scope filter, run-token Hash-based-Message-Authentication-Code (HMAC), gRPC mTLS bind, DP, robust guard), §11 the verify-in-isolation checklist, §13 the 20-task checklist. | The three-layer role enum, org-scoped multi-tenant authorization (RLS-style), cookie-only JWT, the per-run scoped result token, gRPC mTLS, Differential Privacy (DP), the robust-mean/clipping guard, the SOC-2/HIPAA controls. | Immediately before M4 (it is the security foundation woven into the control plane), and again for M11 (DP/robust/compliance). |

> **LLD numbering note (authoritative — read once).** All nine LLDs (10- through 18-) are authored on disk: `10-LLD-backend-control-plane.md`, `11-LLD-fl-framework.md`, `12-LLD-orchestration-substrate.md`, `13-LLD-frontend-dashboard.md`, `14-LLD-desktop-tauri.md`, `15-LLD-mobile.md`, `16-LLD-observability.md`, `17-LLD-data-and-artifacts.md`, and `18-LLD-security-and-compliance.md`. The numbering used by these files is authoritative; where `01-ARCHITECTURE-HLD.md §3`'s original "LLD doc" column or `91-LOCAL-MODEL-GUIDE.md §1` differ, the on-disk file numbering above wins (see the supersession note in `01 §3`). Each unit in the `01 §3` map has exactly one dedicated LLD; security/compliance (18-) is the numbered cross-cutting LLD in addition to the ten unit LLDs.

### 3.3 Orchestration & meta documents (the build plan and your operating manual)

| Doc | Title | What it is | Owns (the authority for) | Read when |
|---|---|---|---|---|
| **90** | `90-BUILD-SEQUENCE.md` | The conductor's score: the milestone-ordered build plan (M0 monorepo/CI through M13 production deploy), the dependency graph, the ordering reasoning, the per-milestone done-conditions, the human-review gate after every milestone, and the milestone→audit-priority→risk traceability table. | The ORDER you build in and the acceptance gate per milestone. It sequences 01–04 and the LLDs; it never redefines a contract. | After the four foundation docs, to plan the build; then return before each milestone. |
| **91** | `91-LOCAL-MODEL-GUIDE.md` | Your operating manual: the reading order, how to parse every doc, the seven GOLDEN RULES, the TDD (Test-Driven-Development) cycle (RED→GREEN→COMMIT), the handoff/checkpoint protocol, and the ten-point self-check. | HOW you work: the rules, the stop-and-ask protocol, the commit discipline, the checkpoint package. | **FIRST, always.** This is the document you read before everything and return to between tasks. |

### 3.4 The two DeComFL correctness companion documents (read before touching the FL framework, M3)

These are on disk now and are the acceptance contract for the product core. They are NOT in `docs/v2/build/` — they live one level up.

| Doc | Path | What it is | Owns |
|---|---|---|---|
| DeComFL correctness **spec** | `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` | The design source of truth: the three bugs with `file:line` evidence, the blocking fixes and cleanups, the locked CPU-canonical-RNG decision, the determinism contract, the T1–T5 test plan that defines "done". | The three correctness fixes and the determinism contract. |
| DeComFL correctness **plan** | `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md` | The strict RED→GREEN→COMMIT TDD task sequence (8 tasks) with the exact `pytest` commands and the precise expected failure before each fix. | The exact M3 sub-order and the per-task commands. |

> The wider `docs/v2/` tree also holds `decisions/`, `cost/`, and `explainers/` for the DeComFL trifecta (indexed by `docs/v2/README.md`). Those are background/ADR reading, not build contracts; do not implement from them.

---

## 4. Where to start — the reading order for the local model (do not skip; do not reorder)

This is the canonical order. The first part is READ; the second part is BUILD.

### 4.1 Read order (foundation — once, before any code)

```
1. 91-LOCAL-MODEL-GUIDE.md   ── the operating rules (GOLDEN RULES, TDD, checkpoints). FIRST, ALWAYS.
2. 00-OVERVIEW.md            ── this file: the map, the vision, the glossary.
3. 01-ARCHITECTURE-HLD.md    ── the system shape: five units, three flows, topology, the 8 decisions.
4. 02-TECH-STACK.md          ── the LOCKED pinned versions + the §25 hard invariants.
5. 03-DATA-MODEL.md          ── every table/column/migration (Flyway owns the schema).
6. 04-API-CONTRACTS.md       ── every REST/gRPC/STOMP contract + error envelope + run token + traceparent.
```

Then, just-in-time, the LLD for the unit you are about to build (12, 13, or 18 — see §3.2), and the DeComFL spec/plan before the FL framework.

### 4.2 Build order (the milestones — full detail in `90-BUILD-SEQUENCE.md`)

Build in this exact milestone order. Each milestone ends with a hard human-review checkpoint; **do not start the next milestone until the previous milestone's review has passed.** The "Primary docs" column is what you read before that milestone.

| Milestone | Goal (one line) | Primary docs to read first |
|---|---|---|
| **M0** | Monorepo skeleton + PR-time CI + branch protection + version pins + buf scaffold. | `02 §22–§25` |
| **M1** | The complete Postgres schema via Flyway V1–V8, validated against real Postgres (Testcontainers). | `03` (full), `02 §5` |
| **M2** | The `fedlearn.v2` gRPC proto authored once via `buf` with a breaking-change gate. | `04 §10`, `02 §3` |
| **M3** | The custom Python FL framework with the DeComFL correctness trifecta fixed (the product core). | DeComFL spec + plan, `02 §4`, `04 §10` |
| **M4** | The Spring Boot control plane + the security foundation (role enum, cookie JWT, org-scoped authz, run token). | `18 §5–§13`, `04 §1–§9/§11–§13`, `03 §7` |
| **M5** | The orchestration substrate (`FlServerLauncher`, lease, reconciler, quotas, round deadline/quorum). | `12` (full), `04 §4/§5`, `03 §5.2` |
| **M6** | Platform observability (Micrometer/Prometheus, OTel Collector, Grafana/Loki/Tempo, end-to-end traceparent). | `02 §20`, `04 §14`, `04 §5/§11` |
| **M7** | Artifact + dataset + run-lineage stack (S3/MinIO content-addressed, MLflow, determinism manifest). | `02 §7–§9`, `04 §8.2/§9/§4.4`, `03 §5.3` |
| **M8** | The React 19 frontend dashboard (TanStack Query, Zod, one STOMP connection, the comm-cost panel). | `13` (full), `04` (wire shapes), `02 §12–§15` |
| **M9** | The Tauri v2 desktop app (reuse the M8 renderer, Rust command layer, signed minisign updater). | `02 §16` |
| **M10** | The native C++ libtorch mobile FL core (on-device DeComFL, golden-vector parity gate). | `02 §17`, M2 C++ stubs, M3 golden fixtures |
| **M11** | DP + robustness + compliance hardening (DP-SGD/scalar-DP, robust guard, SOC-2/HIPAA controls, SBOM). | `18 §6.6/§11/§13` tasks 16–20, `02 §19/§22` |
| **M12** | Full local end-to-end integration (browser → … → client, live comm-cost wedge, resume-from-checkpoint). | `01 §4` (the three flows) |
| **M13** | Production deployment (k8s Jobs primary, STOMP relay swap, quotas + scale-to-zero, default-secure gRPC). | `01 §5.1`, `02 §10/§11/§18` |

The dependency graph that produced this order (and the five ordering rules behind it) is in `90-BUILD-SEQUENCE.md §1`. The short version: **contracts before consumers** (M1 schema + M2 proto first), **the product core before the orchestrator** (M3 before M5), **security is a foundation not a coat of paint** (M4 builds it in), **producers before visualizers** (M6/M7 before M8), **clients last, hardening last** (M9/M10/M11).

---

## 5. Shared facts repeated here (so this file is self-contained)

These appear in multiple documents; they are repeated here verbatim so you never have to cross-reference to recall them.

### 5.1 The eleven hard invariants (from `02-TECH-STACK.md §25`; never violate any, on any milestone)

1. **No `flwr` / Flower dependency** anywhere; custom Protocol-Buffers only (`package fedlearn.v2`); remove `flwr-datasets`.
2. **Cookie-only HttpOnly JWT;** no `Authorization: Bearer` header in the frontend; no token in `localStorage`.
3. **Schema is owned by Flyway, not JPA;** a new field = a new `V{n}__*.sql`. The `test` profile keeps Flyway disabled (in-memory H2 `create-drop`) — never change that.
4. **gRPC defaults to TLS + mTLS;** never ship `insecure_channel` as the default (dev loopback plaintext is the only exception).
5. **DeComFL: the `1/P` averaging factor + CPU-canonical RNG + symmetric serializer;** a golden-vector Python↔C++ parity test gates determinism.
6. **The FL round loop has a deadline + minimum-quorum;** never hang on a straggler.
7. **Per-org concurrency quotas + scale-to-zero** before lifting the port cap.
8. **Delete the false "Byzantine-robust" claim;** market the DeComFL scalar-only DLG-resistance wedge truthfully.
9. **No Artificial-Intelligence (AI) attribution** in any commit, PR, comment, doc, or changelog — authorship is human-only.
10. **The dual gRPC stub model** (training stub + parallel heartbeat stub) is preserved; the heartbeat stub keeps the server from timing the client out during long rounds.
11. **Parameter chunking** for models >300 MB is preserved (required for LLM federations).

(Invariants 1–9 are listed verbatim in `90 §0.3`; 10–11 are the load-bearing FL-protocol invariants from `01 §1` / `04 §10.3`. The authoritative list is `02 §25`.)

### 5.2 The fixed ID types (from `04-API-CONTRACTS.md §1`; do not guess them)

| Entity | ID type | Wire form |
|---|---|---|
| `users.id` | `Long` / BIGINT | JSON number |
| `organizations.id`, `projects.id`, `fl_runs.id`, `datasets.id`, artifact ids | UUID (Universally Unique Identifier) | lowercase 8-4-4-4-12 string |

Mixed types appear in composite keys (e.g. `OrganizationMembershipId(UUID orgId, Long userId)`).

### 5.3 The three role layers (the V5 identity model; from `18` / `03 §3`)

- **Platform** — `PlatformRole { USER, PLATFORM_ADMIN }`. `PLATFORM_ADMIN` bypasses org-membership checks; `CustomUserDetailsService` emits authority `ROLE_PLATFORM_ADMIN`.
- **Organization** — `OrgRole { OWNER, ADMIN, MEMBER }`. Each tenant lives here.
- **Project** — `ProjectRole { MEMBER, CLIENT }` plus implicit owner via `projects.user_id`. Projects belong to exactly one org (`projects.org_id NOT NULL`).

### 5.4 The go/no-go gate (repeated from §1.4)

DeComFL IP title is owned by RIT under policy C03.0, not the founder (risk **R1**). The build proceeds in parallel, but **no public launch or moat claim is defensible until an RIT IPMO license/spin-out is executed.** Not a build milestone; never pretended away (`90 §2`, `01 §1.4`).

### 5.5 The single sentence to remember (from `91 §7`)

**Implement bodies behind frozen contracts using only the locked stack; write the test first where told; commit per task; and the moment a fact is missing, STOP and ask instead of guessing.**

---

## 6. Master glossary — EVERY acronym and term used across the v2 build set

Alphabetical. This is the union of the glossaries in every build document plus the terms introduced by name. If an acronym you see in any doc is not here, treat that as a documentation gap and flag it.

| Term | Full form / definition |
|---|---|
| ADR | Architecture Decision Record — a short document recording one decision, its alternatives, and the reasoning. |
| AI | Artificial Intelligence. |
| API | Application Programming Interface. |
| ARM64 | 64-bit Advanced RISC (Reduced Instruction Set Computer) Machine — the mobile/Jetson CPU architecture. |
| ASCII | American Standard Code for Information Interchange — the character set the normative diagrams use. |
| ARN | Amazon Resource Name — the identifier an ECS task is persisted under as `executor_ref`. |
| Aurora | AWS's managed MySQL/Postgres-compatible database; considered only at hyperscale, not used in v2 (Postgres/RDS is the choice). |
| AWS | Amazon Web Services. |
| Baggage | OpenTelemetry key-value context propagated with a trace; v2 rule: never put PII in baggage. |
| bollard | The Rust crate the Tauri desktop uses to talk to the Docker daemon (the Jetson client path). |
| buf | The single-source Protocol-Buffers toolchain (lint + breaking-change gate + codegen) for the `fedlearn.v2` proto. |
| Byzantine-robust | A (false, to-be-deleted) v1 README claim that aggregation tolerates adversarial clients; the v2 truthful story is the DLG-resistance wedge + opt-in DP + a robust-mean guard. |
| CC6/CC7/CC8 | SOC 2 Trust-Services-Criteria control categories (logical access, system operations, change management) mapped to codebase evidence in the controls doc. |
| CI | Continuous Integration. |
| Citus | A Postgres sharding extension; explicitly NOT used (control-plane tables are bounded). |
| CN | Common Name — the X.509 certificate field that binds gRPC client identity under mTLS. |
| Content-addressed | Storing/keying an artifact by the sha256 hash of its bytes, so identical bytes dedupe and any reference is integrity-checked. |
| CPU | Central Processing Unit. "CPU-canonical RNG" means perturbations are generated on the CPU with a local generator so Python and C++ produce bit-identical vectors. |
| CSP | Content-Security-Policy — an HTTP response header the frontend/control plane set. |
| DDL | Data Definition Language — the SQL that creates/alters tables (the Flyway migrations). |
| DeComFL | Dimension-Free Communication Federated Learning — the platform's zeroth-order FL algorithm that sends scalars instead of model weights. NOTE: the v1 wiki mis-expanded this as "Decomposed", which is wrong per the paper (`docs/audit/2026-05-29/B1-paper-alignment.md:33`). |
| Determinism manifest | The per-run record (seed, hyperparameters, library/dataset/model/golden-vector hashes, `rng_device='cpu'`) that makes a run reproducible; lives in `determinism_manifests` (V8). |
| Dirichlet split | The non-IID (non-Independent-and-Identically-Distributed) data partitioning the custom `Partitioner` owns (replacing the four v1 forks + `flwr-datasets`). |
| DLG | Deep Leakage from Gradients — the gradient-inversion attack family DeComFL's scalar-only uploads structurally eliminate. |
| DoD | Definition of Done. |
| DP | Differential Privacy. |
| DP-SGD | Differentially-Private Stochastic Gradient Descent (per-sample gradient clipping + Gaussian noise) — the FedAvg-path DP. |
| DTO | Data Transfer Object — a typed wire payload (REST request/response body). |
| E2E | End-to-End. |
| ECS | (AWS) Elastic Container Service — the secondary `FlServerLauncher` backend (RunTask). |
| EKS | (AWS) Elastic Kubernetes Service — where the primary Kubernetes Jobs launcher runs in production. |
| EOL | End Of Life (e.g. Spring Boot 3.4.5 was past OSS EOL in v1). |
| FedAvg | Federated Averaging — the classic FL strategy (clients send weight deltas; the server averages them). |
| FedRAMP | (US) Federal Risk and Authorization Management Program — explicitly deferred in v2. |
| FK | Foreign Key. |
| FL | Federated Learning. |
| `fl_runs` | The durable Postgres lease table that is the single source of truth for run state; the JVM is a stateless supervisor over it. |
| `FlServerLauncher` | The Java abstraction with three backends (Kubernetes Jobs / ECS RunTask / LocalProcess) that launches the long-running FL server. |
| Flyway | The database-migration tool that owns the schema (`V*__*.sql`); JPA runs validate-only. |
| Golden vector | A frozen `.npy` perturbation vector + manifest that is the language-neutral RNG contract the C++ mobile core must reproduce bit-for-bit. |
| GDPR | General Data Protection Regulation — informs the right-to-erasure FL note in the controls doc. |
| GPU | Graphics Processing Unit. |
| gRPC | Google Remote Procedure Call — the binary RPC protocol the FL server and clients speak. |
| HIPAA | Health Insurance Portability and Accountability Act — the healthcare/pneumonia demo makes HIPAA-readiness the architectural floor. |
| HLD | High-Level Design (document 01). |
| HMAC | Hash-based Message Authentication Code — used (SHA-256) to mint/verify the per-run scoped result token. |
| HSTS | HTTP Strict Transport Security — an HTTP response header. |
| HTTPS | HyperText Transfer Protocol Secure. |
| IID | Independent and Identically Distributed (the data assumption a Dirichlet split deliberately breaks for non-IID FL). |
| IP | Intellectual Property (the RIT C03.0 go/no-go gate). |
| IPC | Inter-Process Communication — the Tauri renderer↔Rust bridge (must fail closed in packaged builds). |
| IPMO | (RIT) Intellectual Property Management Office. |
| JDK | Java Development Kit. |
| jjwt | The Java JWT library pinned for token signing/verification. |
| JPA | Jakarta Persistence API — the Java ORM, run validate-only (Flyway owns the schema). |
| JSON | JavaScript Object Notation. |
| JVM | Java Virtual Machine. |
| JWT | JSON Web Token — the cookie-only HttpOnly auth token. |
| k8s | Kubernetes — the primary FL-server launcher backend (Jobs). |
| Lease | A row in `fl_runs` held with a deadline; the reconciler reaps expired leases (`FOR UPDATE SKIP LOCKED`). |
| libtorch | The C++ distribution of PyTorch the mobile FL core links (ARM64). |
| LLD | Low-Level Design (documents 12, 13, 18 on disk; others planned). |
| LLM | Large Language Model. |
| LM | Language Model (you, the ~30B implementer). |
| Loki | The log-aggregation backend in the observability stack. |
| LTS | Long-Term Support (Spring Boot 3.5 LTS line; Java 21 LTS). |
| MB | Megabyte. |
| Micrometer | The JVM metrics facade that exports to Prometheus. |
| MinIO | The self-hosted S3-compatible object store (dev/on-prem artifact store). |
| MLflow | The self-hosted (Apache-2.0) experiment/run-lineage and Model Registry. |
| MSW | Mock Service Worker — the frontend test request-mocking library. |
| mTLS | mutual Transport Layer Security — both gRPC peers present certificates; identity binds to the cert CN. |
| Multi-AZ | Multi Availability Zone (the RDS production deployment). |
| minisign | The signature scheme the mandatory Tauri auto-updater uses; unsigned updates are rejected by the framework. |
| Monorepo | One repository holding all five units + proto + deploy + docs, with affected-build CI. |
| NativeWind | The Tailwind-for-React-Native styling layer for the mobile app. |
| OKLCH | Oklab Lightness-Chroma-Hue — the perceptual color space the single design-token package uses. |
| OLTP | Online Transaction Processing — the control-plane database workload (PostgreSQL). |
| Opacus | The PyTorch DP-SGD library used on the FedAvg DP path where it fits. |
| ORM | Object-Relational Mapping (JPA/Hibernate). |
| OS | Operating System. |
| OTel | OpenTelemetry — the trace/metric/log instrumentation standard; the Collector fans out to Tempo/Loki/Prometheus. |
| P | The DeComFL perturbation/local-step count; the dropped `1/P` averaging factor is the R2 bug (default P=10 → 10× overshoot). |
| Partitioner | The custom interface that produces a content-addressed, reproducible non-IID data split (replaces `flwr-datasets`). |
| PHI | Protected Health Information (HIPAA scope). |
| PII | Personally Identifiable Information — never placed in trace baggage or histogram labels. |
| Playwright | The browser end-to-end test runner for the frontend. |
| PR | Pull Request. |
| Prometheus | The metrics time-series database scraping Micrometer on the internal management port. |
| Quorum | The minimum number of client reports a round must receive before it may close at its deadline (so a straggler never hangs the run). |
| RBAC | Role-Based Access Control. |
| RCE | Remote Code Execution — the v1 unsigned-auto-install vulnerability the Tauri signed updater structurally kills. |
| recharts | The React charting library rendering convergence + the communication-cost panel. |
| Reconciler | The boot-time + periodic loop that drives run lifecycle from the `fl_runs` lease (the JVM is a stateless supervisor). |
| RDS | (AWS) Relational Database Service — the managed PostgreSQL host. |
| Redis | One option for the multi-replica STOMP relay broker. |
| Renovate | The dependency-update bot configured in CI. |
| REST | Representational State Transfer — the control-plane HTTP API style. |
| RIT | Rochester Institute of Technology (DeComFL's origin; the IP gate). |
| RLS | Row-Level Security — the org-scoped query-filter pattern (`org_id IN (:scope)`). |
| RNG | Random Number Generator — CPU-canonical for DeComFL determinism. |
| RoundResult | The per-round telemetry payload (loss/accuracy + uplink/downlink bytes + scalars-transmitted) POSTed incrementally to the control plane. |
| RPC | Remote Procedure Call. |
| RPO / RTO | Recovery Point Objective / Recovery Time Objective (reliability targets referenced in the HLD/SRE). |
| Rust | The Tauri desktop command-layer language. |
| `run_id` | The UUID the long-running multi-run FL server is keyed on (the unit of orchestration). |
| S3 | (AWS) Simple Storage Service — the content-addressed artifact/checkpoint store. |
| Scale-to-zero | No FL-server compute is consumed when an org has no active runs. |
| SBOM | Software Bill of Materials — emitted (CycloneDX) and gated in CI. |
| sha256 | Secure Hash Algorithm 256-bit — the content-addressing hash for artifacts and golden vectors. |
| shadcn/ui | The component library (web + desktop) seeded from the OKLCH token package. |
| SOC 2 | System and Organization Controls 2 (Type 2) — the compliance program v2 targets. |
| SPA | Single-Page Application (the React frontend). |
| SQL | Structured Query Language. |
| SoT | Source of Truth. |
| SRE | Site Reliability Engineering. |
| SSR | Server-Side Rendering — explicitly NOT needed (hence React + Vite, no Next.js). |
| STOMP | Simple Text Oriented Messaging Protocol — carried over WebSocket for live logs/telemetry. |
| structlog | The Python structured-logging library carrying `project_id`/`round_idx`/`trace_id`. |
| Tailwind | The CSS framework (v4) underlying the design system. |
| TanStack Query | The frontend server-state/cache library (replacing v1's duplicate fetch triads). |
| Tauri | The v2 desktop shell (Rust + reused React renderer) replacing Electron. |
| TCP | Transmission Control Protocol (the v1 11-port cap was on TCP ports). |
| TDD | Test-Driven Development (RED → GREEN → COMMIT). |
| Tempo | The distributed-tracing backend in the observability stack. |
| Testcontainers | The library that spins a real PostgreSQL `17.10` in CI so migrations validate against Postgres, not H2. |
| TLS | Transport Layer Security. |
| `traceparent` | The W3C trace-context header propagated JVM → process-env → gRPC-metadata across every hop. |
| Trivy | A container/dependency vulnerability scanner run in CI. |
| TS | TypeScript. |
| URL | Uniform Resource Locator. |
| UUID | Universally Unique Identifier (lowercase 8-4-4-4-12). |
| v1 / v2 | Version 1 (the audited proof-of-concept) / Version 2 (this greenfield rebuild). |
| Valkey | A Redis-compatible option for the multi-replica STOMP relay. |
| Vite | The frontend build tool/dev server (v6). |
| Vitest | The frontend unit/component test runner. |
| WebSocket / WS | The transport carrying STOMP for live logs. |
| wry / tao | The Tauri webview / windowing crates (pinned in the desktop stack). |
| Zod | The TypeScript runtime-validation library used at the frontend wire boundary. |
| ZO | Zeroth-Order optimization — the DeComFL technique (estimate the gradient from loss values + a perturbation, no backprop). |
| `z` | The DeComFL perturbation vector; it is NEVER transmitted — clients send only the seed that regenerates it plus the scalar `g`. |

---

## 7. Source ledger

- **The map this document indexes:** every file in `docs/v2/build/` (00, 01, 02, 03, 04, 12, 13, 18, 90, 91) plus the two DeComFL companion docs in `docs/v2/specs/` and `docs/v2/plans/`.
- **The architecture and reasoning behind every choice:** `01-ARCHITECTURE-HLD.md` (the five units, three flows, topology, eight decisions).
- **The reference architecture, the per-unit decision table, and the risk register R1–R17:** `docs/audit/2026-05-29/README.md`.
- **The build order and the milestone gates:** `90-BUILD-SEQUENCE.md`.
- **Your operating rules:** `91-LOCAL-MODEL-GUIDE.md`.

*End of 00-OVERVIEW.md. This document owns no contract — it is the map. When a summary here and a named owning document differ, the owning document wins.*
