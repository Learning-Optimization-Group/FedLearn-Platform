# FedLearn Platform — v2 Audit & Research Phase Design

**Date:** 2026-05-29
**Branch:** `main-clean`
**Author:** Anurag (asnddev@gmail.com)
**Status:** Design — awaiting user review before agent dispatch

---

## 1. Goal

Treat the current FedLearn Platform as a **proof-of-concept** and produce the
decision-grade research needed to design a **production-grade v2** for a
**startup** context. Every recommendation is calibrated to: resource
optimization, practical cost analysis, scaling, security, efficiency,
maintainability, observability, and **performance observability of the FL
projects/runs users create**.

This document scopes the **audit + research phase only**. The v2 rebuild itself
decomposes into follow-on brainstorms — one per area we commit to. Keeping the
phases separate lets the audit honestly recommend "kill/replace X" rather than
pre-committing to porting it.

## 2. Locked decisions (from brainstorm)

| Decision | Value |
|---|---|
| End state | **Greenfield v2 design.** v1 is the reference implementation. Agents design toward an ideal end-state, not just bug lists. |
| Target | **Production-grade startup.** Think every micro-detail: cost, scale, security, efficiency, maintainability, observability, perf observability. |
| Compliance baseline | **Decide after agent research.** A dedicated agent proposes the floor (SOC2/GDPR/HIPAA/FedRAMP). |
| Scale baseline | **Agents propose tiers** (seed → Series-A → hyperscale) with concrete cost models. |
| Mobile | **Real and substantial.** Lives on `origin/fed-mobile:mobile_client/` — React Native + native C++ FL core (libtorch ARM64 + gRPC). Audited **in-place** (read-only); the subtree lift onto `main-clean` is deferred to a v2 implementation step. |
| Verification | **Adversarial.** Major findings are independently challenged by skeptic agents before they enter the synthesis. |
| Output | Decision-grade markdown reports + a master synthesis, in `docs/audit/2026-05-29/`. Builds on (does not overwrite) the 2026-05-27 audit. |

## 3. Mobile (`fed-mobile`) findings that shape the plan

- `mobile_client/` is **fully self-contained** — 162 files, nothing referenced
  outside the directory. A clean subtree lift is possible.
- The branch is **157 commits stale** relative to `main` (branched at `dfde813`;
  `fed-mobile` is only 2 commits ahead of the merge-base). A **whole-branch
  merge is wrong** — only `mobile_client/` should be lifted.
- It is a **native C++ FL core** (`mobile_client/shared/src/`): `DeComFLClient.cpp`,
  `ZerothOrderEstimator.cpp`, `FederatedLoop.cpp`, `FedLearnClient.cpp`,
  `ModelManager.cpp`, `DataLoader.cpp` — i.e. the Python framework's DeComFL
  ported to libtorch. Android (Kotlin/JNI/CMake) + iOS (ObjC++) bridges.
- It carries **two** `fedlearn.proto` copies (`shared/proto/` and
  `src/federated/protos/`). The shared copy already shows drift from the
  canonical `framework/.../protos/fedlearn.proto` (`package fedlearn.v1`) —
  e.g. a malformed `SubmitModelUpdate(SubmitModelUpdateReque…)`. Proto
  reconciliation against the canonical `fedlearn.v1` is a confirmed work item.

## 4. Agent roster (17 experts + 1 synthesis)

Every agent must: (a) cite `file:line` evidence for codebase claims and URLs for
market claims; (b) rate each module **salvage / refactor / rebuild / kill** with
a one-line rationale; (c) flag uncertainty explicitly rather than fabricate; (d)
calibrate to the production-grade-startup target; (e) keep the **DeComFL paper**
guarantees and the platform's stated invariants (no `flwr` dep, cookie-only
auth, Flyway-owned schema, dual heartbeat/training stubs, parameter chunking) in
view when proposing changes.

### Track A — Per-unit deep audits

| ID | Agent | Target | Key questions |
|---|---|---|---|
| **A1** | Backend auditor | `backend/fl-platform-api` (103 Java sources, V1–V5 Flyway) | Spring Boot architecture, auth/RBAC (the `ADMIN` vs `PLATFORM_ADMIN` break), FL-server lifecycle, **the `ProcessBuilder`-spawns-Python model as a scaling cliff**, DB design & multi-tenancy, DTO/PII boundaries, the legacy `flower` package. |
| **A2** | Frontend auditor | `frontend` (React 19, Vite 6, TS) | Component architecture, STOMP-over-WS, state management, **missing CSP**, bundle/code-split/perf, the zero-unit-test gap, `any`-typed sites, V5 identity type contract. |
| **A3** | Framework auditor | `framework` (Python, PyTorch, custom gRPC) | Serializer correctness (chunked-upload asymmetry bug), FedAvg + DeComFL strategies, gRPC contract, parameter chunking, parallel heartbeat lifecycle, memory bounds, `MAX_SAMPLES` cap. |
| **A4** | Client-docker auditor | `client-docker` | Multi-arch base images, Jetson L4T path, **the `flwr-datasets` runtime-dep leak** (contradicts "no Flower" rule), thin-wrapper discipline, supply-chain surface. |
| **A5** | Desktop auditor | `fedlearn-desktop` (Electron, dockerode) | Electron security model, **unsigned auto-install on quit** (supply-chain RCE), `safeStorage` keychain, IPC bridge fail-open fallback, dockerode Jetson device-mount path. |
| **A6** | Mobile auditor | `origin/fed-mobile:mobile_client/` (read-only) | RN + native C++ FL core, **proto drift vs canonical `fedlearn.v1`**, libtorch ARM64 build, **on-device DeComFL fidelity vs the Python impl**, FedAvg-with-Protobuf-replacing-pickle claim, battery/thermal/memory bounds, 1M/10M/100M TorchScript model handling, security of on-device model+data. |

### Track B — Cross-cutting research

| ID | Agent | Question it answers |
|---|---|---|
| **B1** | Paper-alignment | Does the implementation match the **DeComFL paper** (Algorithms 2/3/4, ZO estimator Eq.1, seed history `S^t`, gradient history `G^t`) — in **both** the Python framework **and** the native C++ mobile port? Where does it diverge, and does any divergence break the paper's convergence/communication guarantees? |
| **B2** | Tech-stack & architecture | Best production stack for an FL platform. Build-vs-adopt analysis vs **Flower / NVIDIA FLARE / OpenMined PySyft / FedML**. What to keep custom (chunking, parallel heartbeat, DeComFL) vs adopt off-the-shelf. Recommend the v2 component stack with rationale. |
| **B3** | Observability (platform + FL-run) | The emphasized ask: **observability of the FL projects users create** — per-round / per-client training telemetry, ML experiment tracking (MLflow / W&B), convergence curves, drift, client contribution. **Plus** platform observability: OTel, Prometheus/Grafana/Loki/Tempo, **correlation IDs across JVM → Python → client → mobile**. Wire up the currently-dead `opentelemetry`/`prometheus_client` deps and the empty `RoundResult` telemetry pipeline. |
| **B4** | Security / threat-model / compliance | **FL-specific threats**: gradient/data leakage, model & data poisoning, Byzantine & sybil clients, free-rider attacks; gRPC-plaintext-over-WAN; multi-tenant isolation; secrets. Proposes the **compliance floor** (SOC2 Type 2 / GDPR / HIPAA / FedRAMP) with rationale tied to likely FL verticals (incl. the pneumonia/healthcare demo). |
| **B5** | Desktop strategy: native vs Electron | The explicit question. **Electron vs Tauri vs per-OS native** (Swift/WinUI/GTK) vs **"desktop = thin shell over the same native C++ FL core the mobile app uses."** Trade-offs: bundle size, native FL perf, code signing, CPU/CUDA/Jetson matrix, maintenance cost across 3 OSes. Concrete recommendation. |
| **B6** | Scale / cost / infra economics | Proposes **tiered sizing** (seed → Series-A → hyperscale) with **concrete cloud cost models** at each tier; DB choice (Postgres vs distributed), k8s vs managed, model/artifact storage (S3), and the unit economics of **FL-server-per-project** at each tier. |
| **B7** | Coding standards / maintainability / DX | Monorepo tooling (**Nx / Turborepo / Bazel**), CI/CD (PR-time gates — the current load-bearing gap), **proto codegen across 4 languages** (Java/Python/TS/C++), test standards & coverage gates, linting/formatting, release engineering, dependency hygiene (Renovate). |

### Track C — Additions (selected in brainstorm)

| ID | Agent | Question it answers |
|---|---|---|
| **C1** | Reliability / fault-tolerance / SRE | FL runs last hours; clients churn; **the FL server is a spawned process with zero HA**. Failure modes, checkpointing, **round-recovery via `rebuild_model`**, the `startServerForProject` race, disaster recovery, graceful degradation, SLO/SLI definition for FL runs. |
| **C2** | Data engineering / partitioning | The data plane the platform exists to serve: **non-IID handling, Dirichlet splits, the `flwr-datasets` dependency**, dataset versioning/lineage, where client training data physically lives, privacy of data-at-rest, partition reproducibility. |
| **C3** | ML reproducibility / experiment lineage | **Can you reproduce a published result?** Deterministic FL runs (seed control across Python + C++), model registry/versioning, run lineage, artifact storage, config capture. Distinct from observability; strengthens the paper-alignment story. |
| **C4** | Business / GTM / pricing / IP | Startup viability: pricing model, competitive wedge vs Flower/FLARE, and critically — **RIT's IP ownership of DeComFL** and the commercialization/licensing implications of spinning research into a startup (Apache license, `flwr-datasets` license, model-weight IP). |

### Synthesis

| ID | Agent | Output |
|---|---|---|
| **S1** | Synthesizer | Merges all 17 reports into: **(1)** a v2 reference architecture (diagram + component table with chosen stacks), **(2)** a per-unit **salvage/refactor/rebuild/kill** decision table with rationale, **(3)** a cross-cutting risk register (security, compliance, reliability, cost), **(4)** the prioritized **next-brainstorm queue** for the v2 build. |

## 5. Run mechanics

Executed via the **Workflow** tool (ultracode is on) as a verification pipeline:

1. **Audit/Research phase** — all 17 agents run concurrently (capped by the
   workflow concurrency limit), each producing a structured report.
2. **Adversarial verification phase** — each report's *major* findings (the
   salvage/rebuild/kill calls and any "critical" severity claims) are handed to
   independent skeptic agents prompted to **refute**. A finding survives only if
   it is not refuted by a majority. Refuted findings are demoted with a note.
3. **Synthesis phase** — S1 consumes the verified findings and emits the master
   brief.

Each phase's structured output is schema-validated at the agent boundary.

## 6. Output structure

```
docs/audit/2026-05-29/
  README.md                 # S1 master synthesis (v2 architecture + decision table + queue)
  A1-backend.md
  A2-frontend.md
  A3-framework.md
  A4-client-docker.md
  A5-desktop.md
  A6-mobile.md
  B1-paper-alignment.md
  B2-tech-stack.md
  B3-observability.md
  B4-security-compliance.md
  B5-desktop-strategy.md
  B6-scale-cost.md
  B7-standards-dx.md
  C1-reliability-sre.md
  C2-data-engineering.md
  C3-reproducibility.md
  C4-business-gtm-ip.md
  _verification.md          # adversarial-verification ledger (what survived / was refuted)
```

## 7. Out of scope (this phase)

- Any v2 code, scaffolding, or migrations.
- The actual `mobile_client/` subtree lift (deferred to v2 implementation).
- The 2026-05-27 audit's Phase 0 fixes (those proceed independently if desired).

## 8. Next steps after this phase

The S1 synthesis produces a **next-brainstorm queue**. Each committed v2 area
then runs its own `brainstorming → writing-plans → implementation` cycle. This
audit does not itself write implementation plans.
