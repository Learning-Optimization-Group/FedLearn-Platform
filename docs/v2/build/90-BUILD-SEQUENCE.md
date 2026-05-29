# 90 — BUILD SEQUENCE (FedLearn Platform v2)

**Document type:** The master, milestone-ordered build plan for the v2 (version 2) greenfield rebuild.
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters) building the WHOLE platform from an empty repository to a working system, one milestone at a time.
**Status:** Build-authoritative orchestration plan. This document does **not** redefine any contract; it **sequences** the foundation docs (01–04) and the Low-Level Design documents (10–18) into the order a single agent should execute them, with a hard human-review gate after every milestone.
**Date authored:** 2026-05-29.
**Source of truth:** the v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md` (§5 "Prioritized next-brainstorm queue" gives the priority order this plan implements), plus every build doc under `/home/anurag/codebase/FedLearn-Platform/docs/v2/build/`.

---

## 0. How to read and use this document

This is the **conductor's score**. Each numbered milestone (M0–M13) tells you:

1. **Goal** — the one outcome the milestone delivers.
2. **Builds (units / LLDs)** — which build documents you implement, in which sub-order.
3. **Why here (ordering reasoning)** — why this milestone comes after its predecessors and before its successors, tied to a dependency or an audit finding.
4. **Steps** — the exact, ordered work, each pointing at the authoritative LLD section that already specifies the contract (you implement the body; the LLD gives the signature).
5. **Done-condition / acceptance check** — the exact command(s) or observable state that must hold before the milestone is complete. If a command is given, it must exit 0 / produce the stated output.
6. **CHECKPOINT: hand to external review** — a hard gate. **Do not start the next milestone until the human has run the external review over this milestone's diff and the review has passed.** (This is a build-process control, not a content dependency; the docs never reference the reviewer.)

### 0.1 Abbreviations (expanded on first use, per repo policy)

The first time any acronym appears below it is written in full in parentheses. The complete glossary is in §16. The ones you need immediately: **FL (Federated Learning)**; **DeComFL (Dimension-Free Communication Federated Learning)** — the platform's zeroth-order FL algorithm that sends scalars instead of weights (the v1 wiki's "Decomposed" expansion is wrong per the paper, `docs/audit/2026-05-29/B1-paper-alignment.md:33`); **CI (Continuous Integration)**; **LLD (Low-Level Design)**; **HLD (High-Level Design)**; **API (Application Programming Interface)**; **gRPC (Google Remote Procedure Call)**; **JWT (JSON (JavaScript Object Notation) Web Token)**; **STOMP (Simple Text Oriented Messaging Protocol)**; **DP (Differential Privacy)**; **mTLS (mutual Transport Layer Security)**.

### 0.2 The locked document set you sequence (do not invent others)

| Doc | Title | Owns |
|---|---|---|
| `01-ARCHITECTURE-HLD.md` | Architecture HLD | unit map, three data-flows, topology, the 8 architecture decisions |
| `02-TECH-STACK.md` | Tech Stack | every pinned version, the §24 pin table, the §25 hard invariants |
| `03-DATA-MODEL.md` | Data Model | the Flyway (schema-migration tool) migrations V6/V7/V8, the `fl_runs` lease, content-addressing |
| `04-API-CONTRACTS.md` | API Contracts | REST (Representational State Transfer), the `fedlearn.v2` gRPC proto, STOMP topics, error envelope, per-run scoped token, W3C `traceparent` |
| `10-` | Control-plane LLD (control plane + authorization) | *(in progress — its build slice is covered by 18- security + 03- data + 04- contracts until authored)* |
| `12-LLD-orchestration-substrate.md` | Orchestration substrate | `FlServerLauncher`, lease, reconciler, quotas, round deadline/quorum |
| `13-LLD-frontend-dashboard.md` | Frontend dashboard | React (UI library) 19 SPA (Single-Page Application), TanStack Query, Zod, STOMP client, recharts |
| `18-LLD-security-and-compliance.md` | Security & compliance (cross-cutting) | role enum, org-scoped authorization, cookie JWT, run token, gRPC mTLS, DP, robust guard, SOC 2 (System and Organization Controls 2)/HIPAA (Health Insurance Portability and Accountability Act) controls |

> **Numbering note (from `01 §3`, authoritative):** LLDs 11- (FL-framework), 14- (Desktop), 15- (Mobile-FL), 16- (Observability), 17- (Data-and-artifact) are referenced by the HLD but only 12-, 13-, and 18- are authored on disk at the time of writing. Where a milestone below builds a unit whose dedicated LLD is not yet authored, this plan points you at the authoritative contract that already exists (the proto in `04 §10`, the data model in `03`, the security/observability slices in `18`/`16` references), and flags the dependency. **Do not fabricate the missing LLD's contracts — implement only what 01–04, 12, 13, 18, and the DeComFL spec/plan pin.**

### 0.3 The single hardest constraint: never violate the §25 invariants

`02-TECH-STACK.md §25` lists 11 hard invariants. They apply to **every** milestone. Re-read them before each milestone. The load-bearing ones for sequencing:

1. **No `flwr` / Flower dependency** anywhere; custom protobuf only (`package fedlearn.v2`); remove `flwr-datasets`.
2. **Cookie-only HttpOnly JWT;** no `Authorization: Bearer` header in the frontend; no `localStorage` token.
3. **Schema is owned by Flyway, not JPA (Jakarta Persistence API);** new fields = a new `V{n}__*.sql`. The `test` profile keeps Flyway disabled (in-memory H2 (an embedded Java database) `create-drop`) — never change that.
4. **gRPC defaults to TLS (Transport Layer Security) + mTLS;** never ship `insecure_channel` as the default.
5. **DeComFL: `1/P` averaging factor + Central-Processing-Unit-canonical (CPU-canonical) Random-Number-Generator (RNG) + symmetric serializer;** a golden-vector Python↔C++ parity test gates determinism.
6. **FL round loop has a deadline + minimum-quorum;** never hang on a straggler.
7. **Per-org concurrency quotas + scale-to-zero** before lifting the port cap.
8. **Delete the false "Byzantine-robust" claim;** market the DeComFL scalar-only Deep-Leakage-from-Gradients (DLG)-resistance wedge truthfully.
9. **No Acquired-Intelligence (AI) attribution** in any commit, Pull Request (PR), comment, doc, or changelog — authorship is human-only.

---

## 1. The dependency graph (ASCII) — why the order is what it is

The milestones form a directed acyclic graph. An arrow `A ──▶ B` means "B consumes a contract or artifact that A produces; build A first." This is the spine of the whole plan.

```
            ┌──────────────────────────────────────────────────────────────┐
            │  M0  MONOREPO + CI SKELETON                                    │
            │  (Makefile, paths-filter, ci.yml, branch protection,          │
            │   Renovate, gitleaks, version pins from 02 §24, buf scaffold) │
            └───────────────┬──────────────────────────────────────────────┘
                            │  every later unit's CI job hangs off this
            ┌───────────────┴───────────────┬───────────────────────────────┐
            ▼                                ▼                               ▼
 ┌────────────────────┐         ┌────────────────────────┐      ┌────────────────────┐
 │ M1 DATA MODEL +    │         │ M2 PROTO / buf          │      │ (shared pins from  │
 │    MIGRATIONS      │         │   fedlearn.v2 (04 §10)  │      │  02 feed all)      │
 │  Postgres, Flyway  │         │  buf lint+breaking gate │      └────────────────────┘
 │  V6/V7/V8 (03)     │         └───────────┬─────────────┘
 └─────────┬──────────┘                     │  generates Py/Java/TS/C++ stubs
           │  fl_runs, round_results,       │
           │  determinism_manifests         │
           │                                │
           │            ┌───────────────────┴───────────────────┐
           │            ▼                                        ▼
           │  ┌────────────────────────┐         ┌──────────────────────────────┐
           │  │ M3 FL FRAMEWORK         │         │ (Java stubs consumed by M5,  │
           │  │   (Python, DeComFL fix) │         │  TS stubs by M8, C++ by M10) │
           │  │  1/P, CPU-RNG, codec,   │         └──────────────────────────────┘
           │  │  golden vectors         │
           │  │  (DeComFL spec+plan)    │
           │  └───────────┬─────────────┘
           │              │  the FL server the substrate launches
           ▼              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M4  CONTROL PLANE CORE + SECURITY FOUNDATION              │
 │   role enum, cookie JWT, org-scoped authz, REST surface   │
 │   (18- security + 04 §1-§9 + 03 entities) — needs M1      │
 └───────────────────────────┬──────────────────────────────┘
                             │  FlRunService seam, internal-callback auth
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M5  ORCHESTRATION SUBSTRATE                               │
 │   FlServerLauncher (k8s/ECS/local), fl_runs lease,        │
 │   reconciler, quotas, round deadline/quorum, run token    │
 │   (12-) — needs M1 (lease table), M3 (the server it       │
 │   launches), M4 (the controllers + token mint)            │
 └───────────────────────────┬──────────────────────────────┘
                             │  per-round RoundResult POST + STOMP fan-out
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M6  OBSERVABILITY                                         │
 │   Micrometer→Prometheus, OTel Collector, Grafana/Loki/    │
 │   Tempo, W3C traceparent JVM→Python→client→mobile (16-)   │
 │   — needs M3+M4+M5 to have something to instrument        │
 └───────────────────────────┬──────────────────────────────┘
                             │  telemetry the dashboard renders
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M7  ARTIFACT + DATASET + RUN-LINEAGE STACK                │
 │   S3/MinIO content-addressed, dataset registry, MLflow,   │
 │   determinism manifest wiring (17- + 03 registry)         │
 │   — needs M1 (registry tables), M3 (what it stores),      │
 │     M5 (checkpoints wired to fl_runs)                     │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M8  FRONTEND DASHBOARD                                    │
 │   React 19 + Vite 6 + TanStack Query + Zod + STOMP +      │
 │   recharts comm-cost panel (13-) — needs M2 (TS stubs is  │
 │   optional), M4 (REST), M5 (runs), M6 (STOMP topics)      │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M9  DESKTOP (Tauri v2)                                    │
 │   reuse M8 renderer, Rust command layer, bollard Docker,  │
 │   minisign signed updater, OS keychain (14-)              │
 │   — needs M8 (the renderer it reuses), M4 (auth)          │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M10 MOBILE FL CORE (native C++ libtorch + gRPC)           │
 │   on-device DeComFL, CPU-canonical RNG, golden-vector     │
 │   parity gate vs M3 (15-) — needs M2 (C++ stubs), M3      │
 │   (the golden vectors it must reproduce)                  │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M11 DP + ROBUSTNESS + COMPLIANCE HARDENING                │
 │   DP-SGD/scalar-DP, robust-mean guard, delete false claim,│
 │   SOC 2/HIPAA controls doc, supply-chain SBOM (18- §6.6+) │
 │   — layered LAST over a working pipeline (M3+M5+M11)      │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────┐
 │ M12 END-TO-END INTEGRATION + M13 PRODUCTION DEPLOY        │
 │   full local E2E (browser→…→client), then k8s/ECS deploy  │
 └──────────────────────────────────────────────────────────┘
```

**The five ordering rules that produced this graph (internalize these):**

1. **Contracts before consumers.** The schema (M1) and the proto (M2) are pure contracts that everything downstream binds to. They must be frozen first or every consumer re-churns. This is exactly the audit's "buf is the single source of truth + a breaking-change gate" decision (`02 §3.3`) and "Flyway owns the schema" (`03 R-A`).
2. **The product core before the orchestrator.** The FL framework (M3) — with the three DeComFL correctness fixes — is the FL server the substrate (M5) launches. You cannot test the launcher against a server that diverges P× per round (risk **R2**) or `KeyError`s on every transformer upload (risk **R4**). Fix the algorithm first, then build the thing that runs it.
3. **Security is a foundation, not a coat of paint.** The role enum, cookie JWT, and org-scoped authorization (M4) gate every REST and STOMP surface. The audit's headline backend bug is the bootstrap admin being 403'd everywhere (risk **R7**), and the multi-tenant leak (risk **R8**); both are structural, so they are built into M4, not bolted on later. (DP and robust aggregation — M11 — *are* a later layer, because they sit over a working aggregation step that must exist first.)
4. **Producers before visualizers.** Observability (M6) and the artifact/lineage stack (M7) need a running pipeline to instrument and store. The frontend (M8) renders telemetry that M5/M6 produce. Build bottom-up.
5. **Clients last, hardening last.** Desktop (M9) reuses the frontend renderer (M8). Mobile (M10) must reproduce the golden vectors frozen by M3. DP/compliance hardening (M11) is applied over the whole working system so its tests assert against real flows. This matches the audit P0→P1→P2→P3 priority queue (`README.md §5`).

> **Critical correctness ordering inside M3 (do not reorder):** the DeComFL fixes have an internal dependency chain proven by their own TDD (Test-Driven Development) plan (`docs/v2/plans/2026-05-29-decomfl-correctness-plan.md`): serializer symmetry (Bug 3) → shared `canonical_perturbation` helper + golden vectors → wire server+client to it (Bug 2) → `1/P` fix (Bug 1) → hoist + bounded history → fix the stale test. M3's steps reproduce that exact order because each later fix's test depends on the earlier fix being in place.

---

## M0 — Monorepo skeleton + CI + branch protection

**Goal:** an empty-but-governed monorepo: the directory layout, the pinned toolchain, the `buf` scaffold, a PR-time CI pipeline with `dorny/paths-filter` affected-builds, branch protection, Renovate, gitleaks, and a root task runner — so that *every later milestone's code is gated from its first commit*.

**Builds (units / LLDs):** the CI/monorepo decisions in `02-TECH-STACK.md §22` (supply chain), `§23` (Makefile + paths-filter + GitHub Actions), `§24` (the version-pin table), `§25` (the hard invariants). No application LLD yet.

**Why here (ordering reasoning):** the audit names PR-time CI "the highest-leverage, lowest-cost fix" and a P0 item (`README.md:210`, `02 §23`). v1 had **no PR CI**, so broken/vulnerable code could merge (risk **R13**). Building CI first means M1–M13 are gated from their first line. It has no code dependency on anything else, so it is the unique root of the graph.

**Steps:**
1. Create the repo root with the five-unit layout matching v1's salvageable shape: `backend/fl-platform-api/`, `framework/`, `frontend/`, `fedlearn-desktop/` (will become Tauri), `client-docker/`, plus `proto/`, `mobile_client/`, `deploy/`, `docs/`.
2. Pin the toolchain at repo root from `02 §24.1`: `.tool-versions` / `.nvmrc` (`node 24.4.0`), `rust-toolchain.toml` (`1.87.0`), the Gradle wrapper (`9.5.1`), Python pin (`3.12.9`), Java (Temurin 21 LTS `21.0.7+6`). Each is `verify-before-use` — run the resolution command in `02 §24` and pin the resolved value.
3. Create `proto/buf.yaml` + `proto/buf.gen.yaml` (managed mode) + an empty `proto/fedlearn/v2/fedlearn.proto` placeholder per `02 §3.3` canonical layout. Pin `buf` CLI `1.70.0`.
4. Write the root `Makefile` (or `Taskfile.yml`) with targets `lint`, `test`, `proto`, `build`, each delegating to the per-unit native tool (`02 §23`).
5. Write `.github/workflows/ci.yml` (orchestrator → `dorny/paths-filter`), plus per-unit job stubs `backend.yml`, `framework.yml`, `frontend.yml`, `desktop.yml`, `mobile.yml`, `proto.yml` (buf lint + breaking + freshness), `security.yml` (gitleaks + Trivy + pip-audit), `release.yml`. **Kill any duplicate `desktop-release.yml`; keep one** (`02 §23`, `README.md:142`). Keep macOS/Windows multi-arch builds **tag-gated**, not on the PR path.
6. Add the Renovate config (`02 §22`): grouped (Spring stack one PR, ML stack monthly). Add a one-time gitleaks history baseline before locking trunk (`02 §22`).
7. Configure branch protection on `main`: required status checks = the CI jobs; no direct pushes.

**Done-condition / acceptance check:**
- A trivial no-op PR triggers `ci.yml`; `paths-filter` runs **only** the affected jobs; all jobs pass green.
- Branch protection blocks a merge when any required check is red (verify by pushing a deliberately failing lint, confirm the merge button is blocked, then revert).
- `buf lint` runs in `proto.yml` and passes on the placeholder proto.
- `gitleaks` runs in `security.yml` with the baseline and reports zero new findings.
- No `flwr`/`flwr-datasets` appears in any manifest (grep the repo: `grep -rn "flwr" --include=*.toml --include=*.txt --include=*.gradle .` returns nothing).

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the M0 diff (CI config correctness, version pins matching `02 §24`, branch protection, no `flwr`, no AI attribution in any file). Do not start M1 until the review passes.

---

## M1 — Data model + Flyway migrations (Postgres)

**Goal:** the complete control-plane schema on managed-style PostgreSQL, owned by Flyway, validated against **real Postgres** (not H2) in CI: the salvaged V1–V5 identity baseline plus the three new v2 migrations V6 (dataset registry + role guards), V7 (`fl_runs` lease + `round_results` + `model_artifacts` + the `CLOB`→`JSONB` fix), V8 (`determinism_manifests`).

**Builds (units / LLDs):** `03-DATA-MODEL.md` in full — §1 (the four hard rules), §3 (V1–V5 baseline), §5 (the V6/V7/V8 DDL verbatim), §6 (Postgres portability fixes), §7 (JPA mapping notes), §9 (the 12-item checklist). The datastore decisions come from `02 §5` (Postgres 17.10, Flyway, Testcontainers) and `01 D6`.

**Why here (ordering reasoning):** the schema is a pure contract that the control plane (M4) maps entities onto and the substrate (M5) writes its lease into. The `fl_runs` table, its partial unique index, and the `round_results` comm-cost columns must exist before any Java entity, repository, or reconciler can compile. Building the schema first prevents the v1 anti-pattern of JPA `ddl-auto` inventing the schema (`03 R-A`). It depends only on M0 (the CI that will validate it).

**Steps (follow `03 §5` order exactly):**
1. **Prerequisite (`03 §1` note):** copy `V4__project_membership_and_model_hub.sql` and `V5__identity_foundations.sql` from `backend/.../build/resources/main/db/migration/` into `backend/.../src/main/resources/db/migration/` so the chain V1→V8 is complete in one directory. Use the verbatim DDL in `03 §3.3`.
2. Author `V6__dataset_registry.sql` per `03 §5.1`: the `chk_users_platform_role` and `chk_project_membership_role` CHECK constraints (the risk **R7** database guard), then `datasets`, `dataset_versions`, `partition_recipes`, then the `projects.dataset_version_id` / `partition_recipe_id` nullable FKs (Foreign Keys).
3. Author `V7__fl_runs_and_artifacts.sql` per `03 §5.2`: the `audit_events.metadata` `CLOB`→`JSONB` fix and the `TIMESTAMP`→`TIMESTAMP WITH TIME ZONE` block (§6), then `model_artifacts`, then `fl_runs` (with `idx_fl_runs_lease_active` and the partial unique index `uq_fl_runs_one_active_per_project`), wire `model_artifacts.fl_run_id` FK, then `round_results` (with `uplink_bytes`/`downlink_bytes`/`scalars_transmitted`), then the `round_result`→`round_results` backfill, then `DROP TABLE round_result`. **Before running it,** verify the legacy `optimizer` value domain with `SELECT DISTINCT optimizer FROM projects;` and adjust the `COALESCE`→`strategy` coercion if needed (`03 §5.2` note).
4. Author `V8__determinism_manifest.sql` per `03 §5.3`: `determinism_manifests` 1:1 with `fl_runs`, enforcing `rng_device = 'cpu'`.
5. Add a **Testcontainers-PostgreSQL** CI profile that runs V1→V8 against real PostgreSQL `17.10` (`03 §6` hard requirement). **Do not** change the `test` profile's Flyway-disabled / H2 `create-drop` behavior — add the Testcontainers profile *alongside* it.

**Done-condition / acceptance check (`03 §9` checklist):**
- `./gradlew flywayValidate` (or boot the `dev` profile against local Docker Postgres) applies V1→V8 cleanly.
- The partial unique index `uq_fl_runs_one_active_per_project` and `idx_fl_runs_lease_active` are present (`\d fl_runs` in psql shows both).
- `grep -rn "CLOB" backend/.../db/migration/V6 V7 V8` returns nothing; `audit_events.metadata` is `JSONB` after V7.
- Every tenant-owned new table (`datasets`, `model_artifacts`, `fl_runs`) has `org_id UUID NOT NULL`.
- The Testcontainers-Postgres CI job runs the full chain green; the `test` profile is unchanged.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the migrations (DDL correctness vs `03 §5`, the backfill safety, the `CLOB`/`TIMESTAMP` fixes, the partial unique index, no `users.id` migration per `03 R-D`). Do not start M2 until the review passes.

---

## M2 — Proto / buf single-source contract (`fedlearn.v2`)

**Goal:** the canonical `fedlearn.v2` gRPC contract authored once in `proto/fedlearn/v2/fedlearn.proto`, generating Python, Java, TypeScript, and C++ stubs via `buf`, with a breaking-change gate wired into CI.

**Builds (units / LLDs):** `04-API-CONTRACTS.md §10` (the full `.proto` is reproduced verbatim there — generate from it) and the buf decisions in `02 §3` (single source, managed mode, breaking gate, the C++-runtime caveat).

**Why here (ordering reasoning):** the proto is the second pure contract. The FL framework (M3) is the Python server/client that implements these services; the substrate (M5) consumes the Java stubs; the frontend (M8) may use the TS types; the mobile core (M10) links the C++ stubs. Freezing it now — with `buf breaking` guarding it — kills the v1 drift class where the mobile copy had `SubmitModelUpdateReque` (`02 §3`, `04 §10.1`). It depends only on M0 (the buf scaffold).

**Steps:**
1. Replace the M0 placeholder `proto/fedlearn/v2/fedlearn.proto` with the full authoritative proto from `04 §10.2`: `package fedlearn.v2`, `option java_package = "com.fedlearn.v2"`, the `FederatedLearningService` with all RPCs (lifecycle, FedAvg model transfer, DeComFL seeds/scalars, telemetry), and every message (`Tensor`, `ModelParameters`, registration, heartbeat, `ModelChunk`/`ModelUpdateChunk` with the v2 framing fields `codec`/`compressed`/`total_bytes`/`sha256`, the DeComFL `PerturbationSeeds`/`GradientScalars`/`RebuildHistory`, client metrics).
2. Configure `buf.gen.yaml` (managed mode, `java_package` out of the `.proto`) to emit Python (`grpcio-tools`-matched), Java, TypeScript, and C++ stubs.
3. Run `buf generate`; pin the resolved `protobuf` runtime and the matched `grpcio`/`grpcio-tools` pair (`02 §3.1/§3.2`, `verify-before-use`). The C++ gRPC runtime is **not** produced by buf — keep the cross-compile script, pinned `verify-before-use` (`02 §3.2`).
4. Wire `proto.yml` to run `buf lint` + `buf breaking` (against trunk) + a `buf generate` freshness check that fails if the working tree differs (`02 §3.3`).
5. Encode the non-proto framing rules from `04 §10.3` as a checklist the framework (M3) and clients (M10) must enforce: channel security default mTLS, cert-CN (Common-Name) identity, `codec` whitelist `{safetensors, lz4+safetensors}`, chunk symmetry + sha256, `max_payload_bytes` cap, **never transmit the perturbation vector `z`** (seeds only), round deadline + quorum, gRPC status mapping.

**Done-condition / acceptance check:**
- `buf lint` and `buf breaking` pass; `buf generate` is idempotent (freshness check green).
- Generated stubs exist for Python, Java, TS; the C++ stub generation step runs (runtime cross-compile is a documented separate step).
- The proto declares `package fedlearn.v2` and `java_package = "com.fedlearn.v2"`; no `fedlearn.v1` remains.
- A deliberate breaking edit (rename a field) makes `buf breaking` fail; revert it.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the proto (matches `04 §10.2` exactly, the framing fields present, the DeComFL messages carry seeds not weights, the breaking gate works). Do not start M3 until the review passes.

---

## M3 — FL framework with the DeComFL correctness trifecta

**Goal:** the custom Python FL framework (no `flwr`) with the three DeComFL correctness bugs fixed and pinned by a TDD suite that becomes the acceptance contract: serializer save/load symmetry (Bug 3), CPU-canonical perturbation RNG + frozen golden vectors (Bug 2), the `1/P` averaging fix (Bug 1), plus the bounded-history and hoist cleanups. This is the product.

**Builds (units / LLDs):** the FL-framework unit (HLD unit 4; LLD 11- referenced). The authoritative spec/plan are `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` and `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md` (an 8-task TDD plan). Versions from `02 §4` (PyTorch `2.12.0`, safetensors, the custom `DataSource`/`Partitioner` replacing `flwr-datasets`). The gRPC servicer implements the M2 proto.

**Why here (ordering reasoning):** the three bugs are all on the live DeComFL path and are the audit's P0 "DeComFL correctness trifecta" (`README.md:209`, risks **R2/R3/R4**). The substrate (M5) launches this server; observability (M6) instruments it; the comm-cost panel (M8) renders its scalar counts. None of those can be validated against a server that steps P× too far, corrupts aggregation on a GPU, or `KeyError`s on every transformer. So the algorithm is fixed before anything runs it. It depends on M2 (the proto the servicer implements) and is contract-gated to M10 (the golden vectors mobile must reproduce).

**Steps (follow the DeComFL plan task order exactly — each later fix's test depends on the earlier fix):**
1. **Task 0 — baseline.** `pip install -e framework && pytest`; record the red tests (expect ≥4 failures: `KeyError: 'parameters'` ×2+, the `seed_history.append` `AttributeError`).
2. **Task 1 — Bug 3 (serializer symmetry).** In `framework/src/fedlearn/communication/serializer.py`, change `torch.save(params, buffer)` to `torch.save({'parameters': params, 'num_examples': num_examples}, buffer)` so the save matches the load at `chunks_to_parameters`. Add the multi-chunk + transformer-shaped roundtrip tests. **This unblocks the chunked/LLM upload path.**
3. **Task 2 — shared `canonical_perturbation` + golden vectors.** Create `framework/src/fedlearn/estimators/perturbation.py` with `canonical_perturbation(seed, num_params, dtype=torch.float32)` generating `N(0, I_d)` on CPU with a local `torch.Generator`. Create `framework/tests/fixtures/decomfl_golden/generate.py` and freeze 3 golden `.npy` vectors + a `manifest.json` recording the torch/numpy versions — **this is the language-neutral RNG contract the M10 C++ port must pass.** Add `test_perturbation.py` (golden bit-exact + cross-device parity, skip-guarded).
4. **Task 3 — Bug 2 wiring.** Route both `ZerothOrderEstimator.generate_perturbation` and `DeComFL._generate_perturbation` through `canonical_perturbation(...).to(device)`, deleting the duplicated device-bound generators. Add `TestServerClientPerturbationAgree`.
5. **Task 4 — Bug 1 (`1/P` fix).** In `decomfl_strategy.py`, delete the spurious `* self.P` on the update line (`x_current = x_current - self.eta * delta`); `delta` is already `1/(N*P)`-averaged. Add `TestRebuildTrajectoryEquivalence` (the canary). Correct the two misleading "P cancels in derivation" wiki notes.
6. **Task 5 — B-2 (local RNG).** Replace `np.random.seed`/`torch.manual_seed` in `__init__` with a local `np.random.default_rng(seed)`; draw seeds from it. Add `TestNoGlobalRNGMutation`.
7. **Task 6 — C-1 (hoist z to O(K·P)).** Reassociate the aggregation sum so `z` is generated once per `(k,p)` and weighted by the summed gradient across clients. Add `TestOptimizedEqualsNaiveAggregate`.
8. **Task 7 — C-2 (bounded history).** Add `max_retained_rounds` + `evict_old_history()`. Add `TestBoundedHistory`.
9. **Task 8 — fix the stale `seed_history` test** to the round-keyed API; confirm the full suite is green.
10. **Dataset/partitioner:** implement the custom `DataSource` + `Partitioner` interface (`02 §4.3`) owning the Dirichlet non-IID split, collapsing the four `dirichlet_split` forks and **removing `flwr-datasets`** from the manifest. Split the two seed namespaces (`partition_recipes.data_seed` vs the optimizer seed) per `03 §4.3`.
11. **gRPC servicer:** implement the M2 `fedlearn.v2` service in the Python server (`grpc_servicer.py`), preserving **parameter chunking** (>300 MB) and the **dual heartbeat** (training stub + parallel heartbeat stub), and wire the `should_stop` flag the v1 hard-coded `False` (`04 §10.1` item 6).

**Done-condition / acceptance check (the DeComFL plan's self-review checklist):**
- `cd framework && pytest` — full suite green; CUDA/MPS parity tests `SKIPPED`, not failed; no `KeyError`, no `AttributeError`.
- T1 (`TestRebuildTrajectoryEquivalence`), T2 (`TestGoldenVectors` + `TestServerClientPerturbationAgree`), T3 (`TestChunkedRoundtrip` multi-chunk + transformer-shaped) all pass.
- `grep -rn "torch.Generator(device=" framework/src` returns only `perturbation.py` (CPU).
- `grep -n "np.random.seed\|torch.manual_seed" framework/src/fedlearn/server/decomfl_strategy.py` returns nothing.
- `grep -n "cancels in derivation" docs/wikis/framework/06_decomfl.md` returns nothing.
- The golden `manifest.json` exists with `torch_version` (`2.12.0`) + `numpy_version` recorded and 3 cases each with a sha256.
- No `flwr` / `flwr-datasets` in `framework/requirements.txt` or `pyproject.toml`.
- `ruff check src tests` and mypy `strict` on `perturbation.py` pass.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the framework diff (the three fixes are surgical and match the spec; the client `(eta/P)*delta` step is untouched; the golden vectors are frozen with versions; `flwr-datasets` is gone; the servicer implements the M2 proto with chunking + dual heartbeat). Do not start M4 until the review passes.

---

## M4 — Control plane core + security foundation

**Goal:** the Spring Boot 3.5 control plane on Java 21 with the security foundation built in from the first line: the three-layer role **enum** (killing the v1 admin-string drift), cookie-only HttpOnly JWT, org-scoped multi-tenant authorization (RLS-style query filters), the per-run scoped result-token machinery, STOMP topic-level authorization, the standard error envelope, and the full REST surface (auth, projects, runs-facade, users, admin, orgs, datasets, artifacts) mapped onto the M1 entities.

**Builds (units / LLDs):** `18-LLD-security-and-compliance.md` §5–§13 (role enums, `JwtTokenProvider`, `SecurityConfig`, `AuthorizationService` + org scope, run token, STOMP authz, CSP/HSTS, rate limit, audit, exception handler) and `04-API-CONTRACTS.md §1–§9, §11–§13` (every REST/STOMP/token contract). Entity mapping from `03 §7`. Stack from `02 §2` (Spring Boot `3.5.14`, jjwt `0.12.5`, Spring gRPC for the internal channel).

**Why here (ordering reasoning):** the control plane owns identity, projects, and the `fl_runs` row the substrate (M5) leases — so it must exist before M5. Security is a *foundation* here because the audit's two structural backend blockers are the bootstrap-admin lockout (risk **R7**) and the cross-org leak (risk **R8**); both are baked into the enum and the org-scoped chokepoint, not patched later. (DP and robust aggregation are deferred to M11 because they sit over a working aggregation step.) M4 depends on M1 (schema) and produces the `FlRunService` seam and the token mint that M5 consumes.

**Steps (interleave `18 §13` and the REST contracts):**
1. **Role enums** (`18 §13.1`): `PlatformRole {USER, PLATFORM_ADMIN}`, `OrgRole {OWNER, ADMIN, MEMBER}`, `ProjectRole {MEMBER, CLIENT}`. `CustomUserDetailsService` emits authority `ROLE_PLATFORM_ADMIN`.
2. **JWT hardening** (`18 §13.3`): add `iss`/`aud`/`jti`/`tokenVersion`/clock-skew + `isRevoked` to `JwtTokenProvider`.
3. **`SecurityConfig`** (`18 §13.4`): CSP, HSTS, nosniff, Referrer-Policy, Permissions-Policy, `frameOptions.deny`, the filter order. Cookie attributes `HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600` (`04 §1`).
4. **Org-scoped authorization** (`18 §13.5/§13.6`): `UserPrincipal`, `OrgScope`/`OrgScopeFilter` (populate `visibleOrgIds`, `ALL_ORGS` for platform admin), `AuthorizationService` (`requireOrgMember/requireOrgAdmin/requireParticipant/visibleOrgIds`) and the `org_id IN (:scope)` `TenantPredicate`.
5. **Align `@PreAuthorize`** (`18 §13.7`): replace every `hasRole('ADMIN')` with `hasRole('PLATFORM_ADMIN')`; **delete the v1 test that seeds the literal `"ADMIN"`** and add the `platform_admin_reaches_admin_routes` test.
6. **JPA entities + repositories** (`03 §7`): map `FlRun`, `RoundResult` (FK to `fl_run_id`), `ModelArtifact`, `Dataset`, `DatasetVersion`, `PartitionRecipe`, `DeterminismManifest`, `AuditEvent` (JSONB via `@JdbcTypeCode(SqlTypes.JSON)`; enums via `@Enumerated(EnumType.STRING)` matching the CHECK sets). JPA is **validate-only**.
7. **REST controllers + DTOs** (`04 §2–§9`): auth (register/login/me/logout/verify/forgot/reset, rate-limited), projects (`orgId` required on create, `serverPort` removed), the runs facade (start returns `202`; `FlRunService` is a seam M5 fills), users (`/me` only — delete list-all), admin (`PLATFORM_ADMIN`-gated), orgs, datasets, artifacts (pre-signed URLs). Every list is org-scoped.
8. **Run token + filter** (`18 §13.8`, `04 §13`): `RunTokenService.mint/verify` (HMAC-SHA256, constant-time), `RunTokenFilter` gating `/api/internal/**`; **delete `InternalApiKeyFilter` and `APP_INTERNAL_API_KEY`.**
9. **STOMP `JwtChannelInterceptor`** (`18 §13.9`, `04 §11`): parse the SUBSCRIBE destination, call `requireParticipant(projectId)` — close the v1 WS cross-tenant leak.
10. **Rate limit + audit + exception handler** (`18 §13.10–§13.12`): Bucket4j on the three unauth endpoints (`429 RATE_LIMITED`); `@Auditable` coverage; one `GlobalExceptionHandler` emitting the `04 §12.1` code registry.
11. **Boot guards** (`18 §13.18`): the base profile refuses to boot without the four required secrets; **remove the false "Byzantine-robust" README claim.**

**Done-condition / acceptance check:**
- `SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun` boots against local Docker Postgres (JPA validates against V1→V8).
- `AdminControllerIntegrationTest.platform_admin_reaches_admin_routes` passes; the old `"ADMIN"`-seeding test is gone.
- `TenantIsolationTest.orgA_cannot_read_orgB_project` passes.
- `WsSubscribeAuthzTest.non_participant_subscribe_rejected` passes.
- `RunTokenServiceTest` (mismatch=403, bad HMAC=401, terminal=409, constant-time) passes; `InternalApiKeyFilter` and `APP_INTERNAL_API_KEY` are deleted.
- `RateLimitTest.auth_endpoint_429_after_capacity` passes.
- `grep -rn "Bearer" frontend/` (n/a yet) — but `grep -rn "hasRole('ADMIN')" backend/` returns nothing.
- `grep -rni "byzantine" README.md` returns nothing.
- `./gradlew test` is green against the Testcontainers-Postgres profile.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the control-plane diff (role enum kills the drift; org-scoping is total; the run token replaces the global key; STOMP authz; cookie-only JWT with no Bearer; the error envelope; the false Byzantine claim is deleted). Do not start M5 until the review passes.

---

## M5 — Orchestration substrate (the rebuild)

**Goal:** the long-running, run-keyed FL orchestration substrate replacing v1's `ProcessBuilder`-per-project model: the `FlServerLauncher` abstraction with three backends (Kubernetes Jobs primary, AWS Elastic-Container-Service (ECS) RunTask secondary, `LocalProcessLauncher` dev-only), the durable `fl_runs` lease, the reconciler loop (stateless JVM supervisor), per-org concurrency quotas + admission control, the round deadline + minimum-quorum, per-round durable checkpoints, and the per-run scoped-token-authenticated internal callbacks.

**Builds (units / LLDs):** `12-LLD-orchestration-substrate.md` in full — §4 (module tree under `orchestration/`), §5 (interfaces: `FlServerLauncher`, `FlRunSpec`, `FlRunService`, `LeaseManager`, `OrgQuotaService`), §6 (the reconciler loop, lease SQL, admission control, readiness probe, round deadline/quorum), §13 (the 19-task checklist). Substrate decisions from `01 D1/D2/D4/D5/D7` and `02 §18`. Run API from `04 §4`; internal callbacks from `04 §5`.

**Why here (ordering reasoning):** the substrate launches the M3 FL server, writes the M1 `fl_runs` lease, and is called by the M4 controllers using the M4 run-token mint. So it strictly follows all three. It is the audit's single biggest **REBUILD** and a P1 foundation (`README.md:215`, risk **R9/R10**). Getting it right requires the algorithm (M3) to already be correct so the launcher's E2E tests assert against a converging server.

**Steps (follow `12 §13` exactly):**
1. Confirm `V7` (the lease table) from M1 is applied (`./gradlew flywayValidate`).
2. Write `FlRun`/`RoundResult`/`ModelArtifact` entities + repositories with the native lease/quota queries (most are reused from M4; add the native `FOR UPDATE SKIP LOCKED` lease query and the `FOR UPDATE` quota count).
3. Write the enums + records: `LauncherBackend {K8S_JOB, ECS_RUN_TASK, LOCAL_PROCESS}` (must equal the `fl_runs.launcher` CHECK), `FlRunSpec`, `LaunchResult`, `ExecutorState`, `LauncherException`, `OrgQuota`.
4. `FlRunStateMachine` — the `04 §4.3` transition table + `assertTransition`.
5. `LeaseManager` — atomic acquire/renew/release/findExpiredLeases/findPendingRuns (`12 §6.3` SQL).
6. `OrgQuotaService` — `quotaFor` (default + override) + `tryReserveSlot` (`FOR UPDATE` count); admission **before** launch.
7. `RunTokenService` integration (the mint already exists in M4); the substrate injects `FEDLEARN_RUN_TOKEN` into the executor env.
8. `FlServerLauncher` interface + `OrchestrationProperties` + the ArchUnit rule scaffold.
9. `LocalProcessLauncher` (dev-only): `supportsProfile` true only for `dev`; raised port range 50000–50100; reader thread keeps logs **off** the FL critical path (`C1-F7`).
10. `EcsRunTaskLauncher`: `runTask` (singleton `EcsClient`), persist the task ARN as `executor_ref`, `stop`=`StopTask`, `describe`=`DescribeTasks`.
11. `KubernetesJobLauncher` (primary): fill `k8s/fl-server-job.yaml.mustache` (image, resources, env incl. `FEDLEARN_*`+`TRACEPARENT`, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`, per-Job ServiceAccount); deterministic Job name `fl-run-{runId}`; `describe` maps Pod phases.
12. `FlRunService.startRun` (`12 §6.4`): org-member check, launcher-profile guard (`LOCAL_PROCESS` outside dev → `422 UNSUPPORTED_LAUNCHER`), dataset resolution, hyperparameter validation per `04 §4.2`, quota reserve, `PENDING` insert+flush, token mint, env build, **launch-after-commit**.
13. `FlRunService.stopRun` + `getStatus`.
14. Internal callbacks (`recordRoundResult`/`markFinished`/`recordCheckpoint`/`recordStatus`), each gated by a validated `RunContext` (the per-run token), idempotent on `(fl_run_id, round_idx)`, terminal-run guarded — and the per-round POST is **incremental** (`04 §5`).
15. `FlRunReconciler` (`12 §6.2`): `@Scheduled` + `@EventListener(ApplicationReadyEvent)`; reaps orphans on boot; re-adopts or marks INTERRUPTED; readiness probe (`12 §6.5`, replaces the v1 3-second sleep); the round watchdog backstop.
16. STOMP wiring: on every transition publish `ProjectStatusUpdatePayload` to `/topic/status/{projectId}` and `RunEventPayload` to `/topic/runs/{projectId}` (`04 §11`).
17. Micrometer meters `fedlearn_orchestration_runs_launched_total{launcher}` etc. (keep `client_id` off labels — the `02 §20` cardinality budget).
18. ArchUnit gate `Arch_noProcessBuilderOutsideLauncherPackage`.
19. **Delete v1:** remove `flower/FlowerServerManager.java` and its `ProjectService` lifecycle calls; route controllers to `FlRunService`.
20. **Wire the round deadline + minimum-quorum into the M3 FL server's round loop** (`12 §6.6`): a round completes when all expected clients reported OR (deadline elapsed AND received ≥ `min_quorum`); per-round checkpoint to S3 **before** advancing the round counter.

**Done-condition / acceptance check (`12 §14` conformance):**
- The JVM holds **no** in-heap run/process map; reality is always `launcher.describe(executor_ref)`.
- `FlRunService_secondConcurrentStart_returns409` passes (the partial unique index, not app code).
- `OrgQuotaService_atCap_returns409` and `_concurrentStarts_neverExceedCap` pass.
- `LocalProcessLauncher_rejectedOutsideDev` passes; a dev smoke run spawns a process on a 50000–50100 port.
- `Reconciler_*` tests pass (boot reclaim, re-adopt, readiness timeout, watchdog). **Kill the local FL process by PID, wait one reconcile interval (~15s), confirm the run is marked INTERRUPTED/FAILED, not phantom-RUNNING.**
- The round loop never hangs: a straggler test (one client never reports) closes the round at the deadline with ≥ quorum.
- `Arch_noProcessBuilderOutsideLauncherPackage` passes; the project compiles with no reference to `FlowerServerManager`.
- A full dev smoke (`04 §4` create project → start `LOCAL_PROCESS` run → `202` → poll `RUNNING`/`SUCCEEDED`) works; `/actuator/prometheus` shows the orchestration meters.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the substrate (stateless supervisor — no in-heap map; lease + reconciler correctness; quotas before launch; round deadline/quorum; `LOCAL_PROCESS` dev-only; per-run token callbacks; `FlowerServerManager` deleted; no `ProcessBuilder` outside `orchestration.launcher`). Do not start M6 until the review passes.

---

## M6 — Platform observability

**Goal:** the full observability stack — Micrometer→Prometheus on an internal management port, OpenTelemetry (OTel) Collector → Grafana/Loki/Tempo, structlog with `project_id`/`round_idx`/`trace_id`, and a single W3C `traceparent` stitched end to end JVM→spawned Python→client→mobile — plus the FL-run telemetry pipeline made incremental with the communication-cost panel data flowing.

**Builds (units / LLDs):** the observability unit (HLD unit 8; LLD 16- referenced) per `02 §20` (the pinned stack: Prometheus `3.12.0`, Grafana `13.0.1`, Loki `3.7.2`, Tempo `3.0`, OTel Collector `0.153.0`) and the `04 §14` `traceparent` propagation contract (REST → process-env `TRACEPARENT` → gRPC metadata). The telemetry pipeline shape is `04 §5` + `04 §11`; the comm-cost columns are the `03 §5.2` `round_results` fields.

**Why here (ordering reasoning):** observability needs a running pipeline to instrument — the M3 server, the M4 control plane, the M5 substrate. Building it now (before the frontend) means the dashboard (M8) renders telemetry that already flows. The audit makes this a P1 foundation ("deps pinned, imported nowhere" in v1 — risk **R14** reproducibility join + B3 rebuild, `README.md:217`).

**Steps:**
1. Add `micrometer-registry-prometheus` + `micrometer-tracing-bridge-otel` to the backend (`02 §2.1`); expose metrics on the internal management port.
2. Implement the `04 §14` `traceparent` carriers at each hop: browser→JVM HTTP header; JVM→executor as the env var `TRACEPARENT` (set by the M5 launcher alongside the run token); Python server `extract()` from `os.environ["TRACEPARENT"]`; FL server→client gRPC metadata key `traceparent`; client→mobile gRPC metadata. **Never put PII in baggage** (`04 §14` caveat).
3. structlog in the framework with `project_id`/`round_idx`/`trace_id`; the STOMP `LogLinePayload`/`RunEventPayload` carry `traceId` (`04 §11`).
4. Stand up the OTel Collector pipeline + Grafana/Loki/Tempo (local via docker-compose for dev, per `01 §5.2`).
5. Confirm the FL-run telemetry pipeline is **incremental** (the M5 internal callback already POSTs per-round): the `RoundResult` → `/api/internal/runs/{runId}/results` → STOMP → recharts path fires per round, carrying `uplinkBytes`/`downlinkBytes`/`scalarsTransmitted`/`modelParamCount` (`04 §5.1`) — the data the M8 comm-cost panel renders.
6. Enforce the cardinality budget: `client_id` **off** histogram labels; per-client detail goes to MLflow (M7), not Prometheus (`02 §20`).

**Done-condition / acceptance check:**
- `/actuator/prometheus` on the management port exposes JVM + orchestration meters.
- A run produces one Tempo trace whose spans stitch JVM → Python server → client (verify the `trace_id` is identical across a Grafana log query in Loki and the Tempo trace).
- The per-round `RoundResult` POST fires **during** the round loop (not batched at the end); the row carries the comm-cost columns populated for a DeComFL run.
- No histogram label carries `client_id` (grep the meter registrations).

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over observability (traceparent end-to-end across all hops; structlog fields; incremental per-round POST; comm-cost columns populated; cardinality budget honored; no PII in baggage). Do not start M7 until the review passes.

---

## M7 — Artifact + dataset + run-lineage stack

**Goal:** the content-addressed artifact store (S3 or MinIO, keyed by sha256), the dataset/partition registry wired end to end, self-hosted MLflow as the Model Registry, and the determinism manifest written into `fl_runs` at run creation and per-round checkpoints wired to runs — making any completed run reproducible and resumable.

**Builds (units / LLDs):** the data-and-artifact unit (HLD unit 9; LLD 17- referenced) per `02 §7` (S3/MinIO content-addressed), `§8` (MLflow `3.12.0`), `§9` (the V6 registry from M1). The REST surface is `04 §8.2` (datasets/partitions) + `04 §9` (artifacts, pre-signed URLs). The manifest contract is `04 §4.4` (`DeterminismManifestDto`) + `03 §5.3` (`determinism_manifests`).

**Why here (ordering reasoning):** the artifact store holds what M3 trains and what M5 checkpoints; the registry tables exist from M1; the manifest references the M3 golden-vector hash and the M1 dataset/partition hashes. Building it after the substrate means the per-round checkpoint (M5 step 20) has a real S3 target. The audit makes this a P1 foundation (artifact store "does not exist; only S3 TODOs" — risks **R14/R16**, `README.md:216`).

**Steps:**
1. Stand up MinIO locally (dev) and the S3 path (prod) with the same API (`01 §5.2`); pre-signed PUT/GET brokering so blob bytes never transit the JVM (`04 §9`).
2. Implement the artifacts API (`04 §9`): `upload-url`, `register`, `get`, `download-url`; `model_artifacts` rows keyed on `(org_id, sha256)`.
3. Implement the datasets/partitions API (`04 §8.2`): register a dataset version by content hash; create a partition recipe whose `contentHash` is a deterministic function of its params (closes the v1 stale-split pickle trap).
4. Wire the M3 `DataSource`/`Partitioner` to read the registered `dataset_version` + `partition_recipe` at run time (content-addressed, reproducible split).
5. Self-host MLflow; each run writes a tracking entry; `fl_runs.mlflow_run_id` links out.
6. Write the determinism manifest into `fl_runs`/`determinism_manifests` at run creation (`03 §5.3`): seed, hyperparameters, torch/numpy/framework-git-sha/proto versions, `initial_model_sha256`, `dataset_split_sha256`, `golden_vector_sha256`, `rng_device='cpu'`. Expose it at `GET /api/runs/{runId}/manifest` (`04 §4.4`).
7. Wire per-round checkpoints (the M5 callback `recordCheckpoint`) to content-addressed `model_artifacts` rows; expose `GET /api/runs/{runId}/checkpoints`.

**Done-condition / acceptance check:**
- Uploading a model via the pre-signed URL and registering it produces a `model_artifacts` row keyed on its sha256; the same bytes registered twice dedupe (`409 SHA_EXISTS`).
- A run pins a `dataset_version` + `partition_recipe`; re-running with the same recipe reproduces the identical split (assert the partition is byte-stable).
- `GET /api/runs/{runId}/manifest` returns the full `DeterminismManifestDto` with `rng_device='cpu'` and the golden-vector hash.
- A per-round checkpoint appears in `GET /api/runs/{runId}/checkpoints`; killing and resuming a run reads the last checkpoint (no restart from round 1).
- MLflow shows the run with its parameters/metrics; `fl_runs.mlflow_run_id` is set.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the lineage stack (content-addressing correctness; pre-signed URLs keep blobs off the JVM; the manifest captures the full reproducibility contract; checkpoint/resume works; MLflow wired). Do not start M8 until the review passes.

---

## M8 — Frontend dashboard

**Goal:** the React 19 + Vite 6 + TypeScript dashboard SPA with TanStack Query server-state, Zod validation at the wire boundary, one shared STOMP connection, the V5 role types (fixing the dead-admin-UI live bug), CSP/HSTS, the recharts telemetry surface including the **communication-cost panel** (the DeComFL bandwidth wedge made visible), and the Vitest + Playwright + Mock-Service-Worker (MSW) test layer.

**Builds (units / LLDs):** `13-LLD-frontend-dashboard.md` in full — §4 (the exact module tree), §5 (Axios instance, Zod schemas, V5 role types, query-key factory, STOMP hook), §6 (provider stack, the 401 silent-probe interceptor, the one shared STOMP connection, code-splitting, the run-observability surface), §13 (the 25-task checklist). Versions from `02 §12–§15`. Wire shapes from `04` (REST/STOMP/error envelope).

**Why here (ordering reasoning):** the frontend consumes the M4 REST surface, the M5 run lifecycle, and the M6 STOMP telemetry topics — all of which now exist and produce data. The recharts comm-cost panel renders the `round_results` columns that M3/M5/M6 populate. Building it after the backend pipeline means MSW handlers mirror real, frozen contracts. It is a P3 hardening item in the audit (`README.md:229`) but is sequenced here so the system is demonstrable before the clients and the final hardening.

**Steps (follow `13 §13` exactly — abridged):**
1. Scaffold + pin deps (`13 §3`): remove `react-icons`; add TanStack Query `5.100.14`, Zod `4.4.3`, Vitest, MSW `2.14.6`, Playwright. `.nvmrc`=`24`.
2. `lib/env.ts` + `.env.{development,ec2demo,production}` (fail-fast on missing prod URLs).
3. `vite.config.ts` (`strictPort:5173`, ec2demo proxy, `manualChunks`) + `vitest.config.ts` + `tsconfig.json` (strict) + `eslint.config.ts` (`tseslint strictTypeChecked`, `no-explicit-any:error`, jsx-a11y).
4. `@fedlearn/tokens` OKLCH (Oklab Lightness-Chroma-Hue) token package + `theme.css` (`@theme`) + shadcn init.
5. `api/schemas.ts` + `api/types.ts`: all Zod schemas mirroring `04`; export the V5 role types. **The `MeResponse` schema must reject the legacy `role:'ADMIN'` string** (regression-locks the dead-admin bug).
6. `api/parse.ts` + `api/axiosClient.ts` (single instance, `withCredentials:true`, the 401 silent-probe interceptor swallowing 401 only on `/me`).
7. `api/endpoints.ts` + `api/queryKeys.ts` (typed fn-per-endpoint + the key factory).
8. `query/*` hooks + `auth/*` (`useIdentity`, `RequireAuth`, `RequirePlatformAdmin`). The admin route guard regression-locks risk **R7** on the frontend.
9. `lib/wsUrl.ts` + `realtime/StompProvider.tsx` (one shared ref-counted connection, `wss://` derivation) + `useStompTopic` (validated subscribe) + `logStore`/`liveResultsStore`.
10. Pages + features: auth pages, `DashboardPage` + `ProjectGrid` (replacing the v1 548-line god component), `features/runs/StartRunModal` (per-strategy hyperparameter form, FedAvg vs DeComFL fields per `04 §4.2`, `LOCAL_PROCESS` hidden outside dev), `LogViewer` (history+live merge), and the observability surface: `ConvergenceChart`, **`CommunicationCostPanel`** (the wedge), `PerClientPanel`, `FederationOrrery` (real data, declarative animation).
11. `lib/errorCode.ts` mapping the full `04 §12.1` registry; CSP/HSTS meta fallback in `index.html`.
12. Test layer: `src/test/setup.ts`, MSW handlers/server, fixtures, the golden-path Playwright e2e; wire `tsc --noEmit`, `eslint .`, `vitest run --coverage`, `vite build` (bundle-size budget) into the PR CI.

**Done-condition / acceptance check (`13 §13`):**
- `npm ci` resolves; `react-icons` is absent; no `Authorization: Bearer` and no `localStorage` token anywhere (`grep -rn "Bearer\|localStorage" src/` returns nothing token-related).
- `schemas.meResponse.test.ts` passes (valid parses; legacy `role:'ADMIN'` throws).
- `RequirePlatformAdmin.test.tsx` passes (regression-locks the dead-admin bug).
- `stompProvider.refcount.test.tsx` passes (one shared connection, not v1's three).
- `communicationCost.test.tsx` + `convergenceChart.scope.test.tsx` pass; charts render from `/topic/results`.
- `vite build` initial chunk < 150 KB gzipped (code-split).
- `vitest run` green; `playwright test` green against MSW; coverage ≥ measured baseline.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the frontend (cookie-only, no Bearer/localStorage; Zod at the boundary; V5 role types fix the admin UI; one STOMP connection; the comm-cost panel renders the wedge; tests green). Do not start M9 until the review passes.

---

## M9 — Desktop (Tauri v2)

**Goal:** the end-user FL-client orchestrator as a Tauri v2 app reusing the M8 React renderer, with a small Rust command layer (spawn/kill the PyInstaller-bundled Python client subprocess, one `bollard` Docker call for the Jetson path, one OS-keychain call for the JWT), and a **mandatory code-signed minisign auto-updater** that structurally kills the v1 unsigned-auto-install Remote-Code-Execution (RCE) class.

**Builds (units / LLDs):** the desktop unit (HLD unit 7; LLD 14- referenced) per `02 §16` (Tauri `2.11.2`, bollard, the signing budget, the WebKitGTK/sidecar open risks). The auth model reuses the M4 cookie JWT; the keychain holds the token so the renderer never sees it.

**Why here (ordering reasoning):** the desktop **reuses the M8 renderer** — so M8 must be built first. It authenticates against the M4 control plane and connects clients to M5-launched runs. It is a P2 product item (`README.md:221`). Building it after the web SPA means the renderer is already a tested artifact; the desktop only adds the Rust command layer and the updater.

**Steps:**
1. Scaffold the Tauri v2 shell reusing the M8 React renderer (shared shadcn components); pin `tauri 2.11.2`, `wry 0.55.1`, `tao 0.35.3`.
2. Rust command layer: spawn/kill the FL-client subprocess (the PyInstaller-bundled Python client — keep training as a **subprocess/sidecar**, not in-process C++, so DeComFL RNG parity stays free and a libtorch crash cannot take down the UI, `02 §16.2`); stream stdout; one `bollard` `HostConfig` Docker call for the **Jetson** path preserving the invariant — **no `--runtime nvidia`, explicit `/dev/nvhost-*` device mounts** (`02 §16.2`, `02 §25.7`).
3. OS-keychain JWT via Tauri's keychain command; the renderer **never** sees the token.
4. The **mandatory signed minisign auto-updater**: unsigned updates are rejected by the framework itself (this is the structural RCE fix). Make the IPC bridge **fail-closed** in packaged builds.
5. Code-sign per OS (`02 §16.2` budget: Apple Developer $99/yr; Windows Authenticode; Linux unsigned is normal). Smoke-test framer-motion/recharts on WebKitGTK (Linux) and re-check the Tauri sidecar signing issues against `2.11.2` (`02 §16.1` open risks).

**Done-condition / acceptance check:**
- The desktop app launches, reuses the M8 renderer, and authenticates against the dev backend (cookie JWT in the keychain, not the renderer).
- Starting a client subprocess from the desktop connects it to a running M5 run (dev `LOCAL_PROCESS` launcher); training proceeds.
- The Jetson `bollard` path uses no `--runtime nvidia` and the `/dev/nvhost-*` mounts (verify the generated `HostConfig`).
- The updater **rejects an unsigned update artifact** (verify by feeding a tampered/unsigned update — it is refused).
- The IPC bridge fails closed in a packaged build.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the desktop (renderer reuse; subprocess training model; keychain token isolation; **signed updater rejects unsigned**; fail-closed IPC; Jetson invariant preserved). Do not start M10 until the review passes.

---

## M10 — Mobile FL core (native C++ libtorch + gRPC)

**Goal:** the on-device DeComFL core in native C++ (libtorch ARM64 (64-bit Advanced RISC Machine) + gRPC) using CPU-canonical RNG, hardened (correct dtype, a `requires_grad` filter), and **gated by the golden-vector parity test** so the C++ perturbation reproduces the exact vectors M3 froze in Python — the Python↔C++ determinism guarantee that makes mobile aggregation correct.

**Builds (units / LLDs):** the mobile-FL unit (HLD unit 5; LLD 15- referenced) per `02 §17` (React Native `0.8x`, NativeWind, react-native-reusables, libtorch matched to torch `2.12.0`, gRPC C++ cross-compiled). The proto is the M2 C++ stubs; the parity contract is the M3 golden fixtures.

**Why here (ordering reasoning):** the C++ core must reproduce the golden vectors frozen by M3 — so M3 must exist first. It links the M2 C++ stubs. It is a P3 item (`README.md:228`). Building it last among the clients means the determinism contract (the golden vectors) is already frozen and CI-gated, so the parity test is a real merge gate, not a moving target.

**Steps:**
1. Lift the `mobile_client/` subtree onto the trunk; reconcile its proto copy to the single M2 `fedlearn.v2` source (no vendored drift).
2. Pin libtorch to match the server torch `2.12.0` as closely as the ARM64 build allows (RNG parity across versions is the risk, `02 §17.3`); confirm an ARM64 libtorch build exists and the size budget.
3. Harden the ZO (Zeroth-Order) C++ core: correct dtype (float32 canonical, not silently following the model dtype), add a `requires_grad` filter, implement CPU-canonical RNG matching `canonical_perturbation`.
4. Implement the golden-vector parity test: the C++ core regenerates the M3 golden `.npy` vectors from the same seeds and must match bit-for-bit; **wire it as a CI merge gate** (`01 §6` — "the test is a merge gate, not a nicety").
5. Wire the dual gRPC channels + parameter chunking (already present in the C++ per `02 §17.3`); the heartbeat stub carries `run_id` and consumes `should_stop`.
6. Restyle to the OKLCH token package via NativeWind + react-native-reusables; **kill** the dead TF.js JS, the MNIST blobs, the disabled DeComFL UI, and the fabricated `exp(-loss)` confidence chart (`README.md:135-136`).

**Done-condition / acceptance check:**
- The C++ golden-vector parity test passes: C++ perturbation for each frozen `(seed, num_params)` case matches the Python golden `.npy` bit-for-bit; the test is a required CI check.
- The mobile app registers with a run via the M2 `fedlearn.v2` gRPC, runs on-device DeComFL, and uploads only seeds/scalars (never `z`).
- One canonical proto source; no vendored copy with `SubmitModelUpdateReque`-style drift.
- The fabricated confidence chart, TF.js JS, and MNIST blobs are removed.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the mobile core (golden-vector parity passes as a CI gate; CPU-canonical RNG; `requires_grad` filter; correct dtype; single proto source; dead/fabricated UI removed). Do not start M11 until the review passes.

---

## M11 — DP + robustness + compliance hardening

**Goal:** the privacy/robustness layer and the compliance program, layered over the now-working pipeline: DP-SGD (Differentially-Private Stochastic Gradient Descent) on the FedAvg path, calibrated scalar-DP on the DeComFL path, a robust-mean/clipping aggregation guard, the SOC 2 Type 2 + HIPAA-readiness controls checklist, and the supply-chain SBOM (Software Bill of Materials) gates as required CI checks.

**Builds (units / LLDs):** `18-LLD-security-and-compliance.md §6.6` (the FL threat model + `dp.py`/`robust.py`), `§11.x/§11.y` (HIPAA-readiness + the SOC 2 controls checklist → `docs/v2/build/controls/soc2-hipaa-controls-checklist.md`), `§13` tasks 16–20. Aggregation decisions from `02 §19`; supply chain from `02 §22`.

**Why here (ordering reasoning):** DP and the robust guard sit **over** the aggregation step that M3 built and M5 runs — they need a correct, working aggregation to wrap. The audit places them in P2 (`README.md:223-224`, risk **R12**) and explicitly says to **delete the false "Byzantine-robust" claim** (done in M4) and instead ship *real, opt-in* DP + a guard. Layering them last means their tests assert against real DeComFL/FedAvg flows. The compliance controls doc enumerates evidence that only exists once M0–M10 are built.

**Steps:**
1. `dp.py` (`18 §13.16`): `dpsgd_clip_and_noise` for FedAvg (per-sample gradient clipping + Gaussian noise; Opacus where it fits, `02 §19`); `scalar_dp` for DeComFL (clip the magnitude of `g`, add scalar Gaussian noise **before** the client uploads, using the CPU-canonical generator). The `04 §4.2` `dpEnabled`/`dpNoiseMultiplier`/`dpClipNorm` fields already exist in the contract — wire them.
2. `robust.py` (`18 §13.17`): coordinate-wise trimmed-mean + clip + NaN/Inf reject (the `04 §4.2` `robustClipTau`).
3. Confirm the false "Byzantine-robust" claim is gone (done in M4; re-verify).
4. CI security gates (`18 §13.19`): `security.yml` with gitleaks, pip-audit, OWASP/Gradle dependency-check, CycloneDX SBOM — as **required** checks blocking merge on high/critical findings.
5. Controls doc (`18 §13.20`): write `docs/v2/build/controls/soc2-hipaa-controls-checklist.md` from `18 §11.x/§11.y` — every CC6/CC7/CC8/Confidentiality row with its codebase evidence (mTLS in transit, scoped tokens, audit-event capture, in-region data residency, the GDPR right-to-erasure FL note from `18 §6.7`).

**Done-condition / acceptance check:**
- `DpTest.scalar_dp_clips_and_noises_deterministically` passes; a DeComFL run with `dpEnabled:true` clips + noises `g` deterministically under the CPU-canonical generator.
- `RobustTest.nan_inf_update_dropped` passes; a NaN/Inf client update is dropped.
- `security.yml` runs gitleaks + pip-audit + dependency-check + SBOM as required checks; a seeded high-severity dependency blocks the merge.
- `docs/v2/build/controls/soc2-hipaa-controls-checklist.md` exists with every CC6/CC7/CC8/Confidentiality row and its evidence.
- `grep -rni "byzantine" .` returns nothing in the public docs/README.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the hardening (DP math correct and opt-in; robust guard drops NaN/Inf; the false claim is gone; SBOM/scan gates block; the controls doc maps every control to real evidence). Do not start M12 until the review passes.

---

## M12 — End-to-end integration

**Goal:** a single, full, local end-to-end run proving the whole platform works together: browser login → create org/project/dataset → register a partition → start a DeComFL run → the substrate launches the FL server → clients (desktop + a Python/Docker client) connect over gRPC → per-round telemetry streams live to the dashboard's comm-cost panel → a checkpoint lands in MinIO → the run completes and the manifest is reproducible.

**Builds (units / LLDs):** none new — this milestone wires M1–M11 into the three core flows from `01 §4` (Flow A login/create, Flow B launch/train, Flow C telemetry/checkpoint/logs).

**Why here (ordering reasoning):** integration can only happen once every unit exists. It is the proof that the contracts (M1/M2) actually decoupled the units correctly. It precedes deploy (M13) because a green local E2E is the precondition for any production rollout.

**Steps:**
1. Bring up the local dev stack per `01 §5.2`: Vite `:5173`, Spring `dev` `:8081`, local Docker Postgres (not H2), MinIO, the OTel/Grafana compose, the `LocalProcessLauncher`.
2. Drive Flow A: register/login (cookie JWT) → create an org → create a project (`orgId` required) → register a dataset version + partition recipe.
3. Drive Flow B: `POST .../runs` (DeComFL) → `202` → the substrate writes the `PENDING` lease, mints the run token, launches the dev FL server → reconciler drives `STARTING`→`RUNNING` → desktop + Docker clients connect over gRPC (dev plaintext on loopback is allowed; everywhere else mTLS) → rounds proceed with the deadline + quorum.
4. Drive Flow C: per-round `RoundResult` POSTs (incremental) → STOMP `/topic/results/{projectId}` → recharts comm-cost panel updates live; a checkpoint lands content-addressed in MinIO; logs stream on `/topic/logs/{projectId}`; one Tempo trace stitches all hops.
5. Verify reproducibility: re-run with the same manifest seed + dataset/partition recipe and confirm the trajectory matches (the M3 rebuild-equivalence guarantee at the system level).

**Done-condition / acceptance check:**
- The full E2E completes a multi-round DeComFL run; the dashboard shows live loss/accuracy **and** the comm-cost panel showing scalars-transmitted ≪ equivalent FedAvg full-model bytes.
- Killing the FL process mid-run, then resuming, restarts from the last checkpoint (not round 1).
- A straggler (one client stops reporting) does not hang the run — the round closes at the deadline with ≥ quorum.
- `GET /api/runs/{runId}/manifest` is reproducible: a second run with the same seed/recipe lands on the same parameters.
- One Tempo trace spans browser→JVM→Python server→client.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the full E2E (all three flows; live comm-cost wedge; resume-from-checkpoint; straggler does not hang; reproducible manifest; one stitched trace). Do not start M13 until the review passes.

---

## M13 — Production deployment

**Goal:** the platform deployed on the production topology (`01 §5.1`): N control-plane replicas behind a load balancer with the STOMP relay swapped to Redis/RabbitMQ, managed PostgreSQL (RDS, Multi-AZ), S3 artifacts, the Kubernetes Jobs launcher primary (ECS RunTask secondary), default-secure gRPC (TLS+mTLS), per-org quotas + scale-to-zero live, and the observability stack collecting in-cluster.

**Builds (units / LLDs):** the production topology in `01 §5.1` and the relay/scale decisions in `02 §10/§11` (STOMP relay swap) + `02 §18` (k8s Jobs, quotas, scale-to-zero). Deploy scripting lives under `deploy/`.

**Why here (ordering reasoning):** deployment is last because it requires a green local E2E (M12) and every unit hardened (M11). The multi-replica topology is only safe once the STOMP relay is in place (`01 §5.1` — the in-memory broker cannot route `/topic/*` across replicas).

**Steps:**
1. Provision managed PostgreSQL `17.10` (RDS, Multi-AZ) and run V1→V8 against it (the Testcontainers profile already proved the chain on real Postgres).
2. Swap the STOMP simple broker to the relay: `enableStompBrokerRelay` against RabbitMQ (preferred — its STOMP plugin and MPL-2.0 license, `02 §11`) or Redis/Valkey. This is the one-line change that makes N replicas safe.
3. Deploy N control-plane replicas behind the load balancer (TLS termination + HSTS at the ingress).
4. Configure the `KubernetesJobLauncher` as primary (EKS, GPU nodes via nodeSelector); ECS RunTask as secondary. Default-secure gRPC (TLS+mTLS, cert-CN identity + enrollment token) — **never** `insecure_channel` outside dev.
5. Enforce per-org concurrency quotas + admission control + scale-to-zero **before** any production traffic (risk **R10**).
6. Point S3 (artifacts), MLflow (lineage), and the OTel Collector→Grafana/Loki/Tempo at the cluster; confirm the cardinality budget holds at scale.
7. Verify the HIPAA-readiness floor: data residency in-region, mTLS in transit, audit-event capture, scoped tokens (M11 controls doc).

**Done-condition / acceptance check:**
- A run launched in production lands as a k8s Job (`fl-run-{runId}`); `kubectl get jobs` shows it; the reconciler drives its lifecycle.
- STOMP telemetry fans out across all replicas (subscribe on replica A, publish via replica B, receive on A).
- gRPC refuses plaintext outside dev (`GrpcServerStartTest.refuses_plaintext_outside_dev` analog holds in the deployed config).
- A run beyond an org's quota is rejected with `409 ORG_QUOTA_EXCEEDED`; idle orgs scale to zero (no FL-server task-hours when no runs are active).
- The deployed observability stack shows metrics/logs/traces; no `client_id` on histograms.

**CHECKPOINT: hand to external review.** Stop. The human runs the external review over the deploy (relay swap correct; quotas + scale-to-zero live; default-secure gRPC; k8s Jobs primary; HIPAA-readiness floor met). This is the final gate before declaring v2 launch-ready (subject to the upstream IP go/no-go gate, below).

---

## 2. The non-technical go/no-go gate (state it; do not hide it)

DeComFL is research from RIT (Rochester Institute of Technology). Under RIT Intellectual-Property (IP) policy C03.0, **RIT — not the founder — likely owns it** (`README.md:27`, risk **R1**, the only Critical/go-no-go item). **The build (M0–M13) can proceed in parallel, but no public launch or moat claim is defensible until an RIT Intellectual Property Management Office (IPMO) license or spin-out is executed.** This is named here per `01 §1.4` because it is upstream of the entire product; it is not a build milestone (no code), but the plan must not pretend it away.

---

## 3. Milestone → audit-priority → risk-mitigation traceability

Every milestone maps to the audit's prioritized queue (`README.md §5`) and closes specific risks (`README.md §4`). This is the proof that the ordering is the audit's ordering.

| Milestone | Audit priority | Risks closed |
|---|---|---|
| M0 Monorepo+CI | P0 (PR-time CI) | R13 (no PR CI; EOL Spring Boot) |
| M1 Data model | P1 (Postgres cutover) | R14 (no lineage), R8 (org_id NOT NULL half) |
| M2 Proto/buf | P0 (proto drift gate) | (drift class; supports R4 fix) |
| M3 FL framework | **P0 (DeComFL trifecta)** | **R2 (1/P), R3 (CPU-RNG), R4 (chunked upload)** |
| M4 Control plane + security | P0 (identity/RBAC; stop-the-bleeding security) | R7 (admin lockout), R8 (org isolation), R6 (token reuse half), R12 (delete false claim) |
| M5 Orchestration substrate | P1 (substrate rebuild) | R9 (no run entity/checkpoint/straggler hang), R10 (quotas/scale-to-zero) |
| M6 Observability | P1 (observability rebuild) | R14 (reproducibility join), B3 |
| M7 Artifact/dataset/lineage | P1 (artifact+lineage stack) | R14, R16 (destructive save), R15 (stale-split) |
| M8 Frontend | P3 (frontend hardening) | R7 (admin UI half), R17 (one brand/no fabricated chart) |
| M9 Desktop | P2 (Tauri migration) | R5 (unsigned auto-install RCE) |
| M10 Mobile | P3 (mobile lift + harden) | R3 (Python↔C++ parity), R17 (fabricated chart) |
| M11 DP/robust/compliance | P2 (FL robustness/privacy; compliance) | R12 (no DP/robust), R11 (SOC2/HIPAA) |
| M12 E2E + M13 Deploy | (integration of all) | validates R6/R9/R10 end-to-end |
| (IP gate) | P0 (IP resolution) | R1 (go/no-go) |

---

## 4. Global done-condition (the whole platform is "built")

The platform is built when, in order, every milestone's CHECKPOINT has passed external review **and**:

1. `make lint && make test` is green across all units (the M0 root task runner).
2. The M12 full E2E completes a reproducible, resumable, straggler-tolerant DeComFL run with the live comm-cost wedge visible.
3. The M13 production topology runs a k8s-Job-launched run with the relay, quotas, scale-to-zero, and default-secure gRPC.
4. Every §0.3 / `02 §25` hard invariant holds (verified by the conformance greps embedded in each milestone's done-condition).
5. The SBOM + scan gates (M11) are required CI checks and green.
6. No AI attribution exists anywhere in the repo (commits, PRs, docs, comments).

---

## 5. Glossary (acronyms, alphabetical)

| Acronym | Full form |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| ARM64 | 64-bit Advanced RISC (Reduced Instruction Set Computer) Machine |
| ASCII | American Standard Code for Information Interchange |
| CI | Continuous Integration |
| CN | Common Name (of an X.509 certificate) |
| CPU | Central Processing Unit |
| CSP | Content-Security-Policy |
| DDL | Data Definition Language |
| DeComFL | Dimension-Free Communication Federated Learning |
| DLG | Deep Leakage from Gradients |
| DP | Differential Privacy |
| DP-SGD | Differentially-Private Stochastic Gradient Descent |
| DTO | Data Transfer Object |
| E2E | End-to-End |
| ECS | (AWS) Elastic Container Service |
| EKS | (AWS) Elastic Kubernetes Service |
| EOL | End Of Life |
| FedAvg | Federated Averaging |
| FK | Foreign Key |
| FL | Federated Learning |
| gRPC | Google Remote Procedure Call |
| HIPAA | Health Insurance Portability and Accountability Act |
| HLD | High-Level Design |
| HMAC | Hash-based Message Authentication Code |
| HSTS | HTTP Strict Transport Security |
| IP | Intellectual Property |
| IPC | Inter-Process Communication |
| IPMO | (RIT) Intellectual Property Management Office |
| JPA | Jakarta Persistence API |
| JSON | JavaScript Object Notation |
| JVM | Java Virtual Machine |
| JWT | JSON Web Token |
| k8s | Kubernetes |
| LLD | Low-Level Design |
| LLM | Large Language Model |
| LTS | Long-Term Support |
| MB | Megabyte |
| MSW | Mock Service Worker |
| mTLS | mutual Transport Layer Security |
| Multi-AZ | Multi Availability Zone |
| OKLCH | Oklab Lightness-Chroma-Hue (color space) |
| OLTP | Online Transaction Processing |
| OS | Operating System |
| OTel | OpenTelemetry |
| PHI | Protected Health Information |
| PII | Personally Identifiable Information |
| PR | Pull Request |
| RBAC | Role-Based Access Control |
| RCE | Remote Code Execution |
| RDS | (AWS) Relational Database Service |
| REST | Representational State Transfer |
| RIT | Rochester Institute of Technology |
| RLS | Row-Level Security |
| RNG | Random Number Generator |
| RPC | Remote Procedure Call |
| S3 | (AWS) Simple Storage Service |
| SBOM | Software Bill of Materials |
| SOC 2 | System and Organization Controls 2 |
| SPA | Single-Page Application |
| SQL | Structured Query Language |
| STOMP | Simple Text Oriented Messaging Protocol |
| TCP | Transmission Control Protocol |
| TDD | Test-Driven Development |
| TLS | Transport Layer Security |
| TS | TypeScript |
| UUID | Universally Unique Identifier |
| WS | WebSocket |
| ZO | Zeroth-Order (optimization) |

---

## 6. Source ledger

**Foundation + LLD docs sequenced (all under `docs/v2/build/`):**
- `01-ARCHITECTURE-HLD.md` — unit map (§3), three flows (§4), topology (§5), the 8 decisions (§7).
- `02-TECH-STACK.md` — every pin (§24), the §25 hard invariants, CI/monorepo (§22/§23), substrate backends (§18), observability (§20), DP (§19).
- `03-DATA-MODEL.md` — the V6/V7/V8 migrations (§5), Postgres fixes (§6), the §9 checklist.
- `04-API-CONTRACTS.md` — REST (§2–§9), the `fedlearn.v2` proto (§10), STOMP (§11), error envelope (§12), per-run token (§13), traceparent (§14).
- `12-LLD-orchestration-substrate.md` — `FlServerLauncher`/lease/reconciler/quotas (§5/§6), the 19-task checklist (§13).
- `13-LLD-frontend-dashboard.md` — module tree (§4), interfaces (§5), flows (§6), the 25-task checklist (§13).
- `18-LLD-security-and-compliance.md` — role enum/authz/run token/mTLS/DP/robust (§5/§6), the 20-task checklist (§13).

**DeComFL correctness (the M3 spine):**
- `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` — the three-bug design.
- `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md` — the 8-task TDD plan (the exact M3 sub-order).

**Audit synthesis (the priority order + risk register):**
- `docs/audit/2026-05-29/README.md` — §5 prioritized queue (the milestone order), §4 risk register (R1–R17), §2 decision table, §3 conflict resolutions.

*End of 90-BUILD-SEQUENCE.md. Every milestone's contracts trace to a build doc (01–04, 12, 13, 18) or the DeComFL spec/plan; every ordering decision traces to a dependency or an audit finding cited inline. Where an LLD is not yet authored (11-, 14-, 15-, 16-, 17-), the milestone points at the authoritative contract that already exists and flags the dependency — no missing-LLD contract is fabricated.*
