# FedLearn Platform — v2 Master Synthesis

**Date:** 2026-05-29
**Branch:** `main-clean` (mobile audited read-only on `origin/fed-mobile:mobile_client/`)
**Inputs:** 18 expert reports (A1–A6, B1–B7, C1–C5) in this directory, each major finding adversarially verified by 3 skeptics. See [`_verification.md`](_verification.md) for the survived/refuted ledger.
**Builds on:** the [2026-05-27 audit](../2026-05-27/README.md) — that audit's Phase 0–3 bug-list still stands; this synthesis is the **greenfield v2 design** that those fixes feed into. Where the two overlap, this document supersedes only the *forward architecture*, not the immediate Phase-0 fixes.

---

## 0. Executive summary

FedLearn v1 is a competent proof-of-concept with **one genuine, paper-backed differentiator and four classes of production blocker**.

**The differentiator (keep and build the company on it):** DeComFL — dimension-free, O(1)-per-round communication via zeroth-order optimization (ICLR 2025, RIT / Prof. Haibo Yang). It transmits ~1 MB total to fine-tune a billion-parameter model versus ~tens of TB for an equivalent FedAvg run, and its scalar-only uploads **structurally eliminate the deep-leakage-from-gradients (DLG) reconstruction attack family** (B1, B4-1, B6). No incumbent (Flower, NVIDIA FLARE, FedML, PySyft) ships it. This is the bandwidth/edge-LLM wedge and the privacy story.

**Blocker class 1 — DeComFL itself is currently broken in three independent ways** (all 0-refute, two reproduced):
- Server aggregation drops the `1/P` averaging factor → global model steps **P× (10× at default) too far** and diverges from every reconnecting client's rebuild trajectory (B1-C1).
- Server regenerates perturbations on CUDA while CPU/MPS clients use their own device; PyTorch does not guarantee cross-device RNG parity → **silent aggregation corruption** on any GPU server or heterogeneous fleet (B1-C2 / C3-1).
- Chunked/streaming upload (the path every transformer >100 MB takes) `KeyError`s on `'parameters'` — **LLM federations cannot complete a round** (A3-C1, carried unfixed from 05-27).

**Blocker class 2 — identity/RBAC is dead end-to-end.** The bootstrap admin gets `PLATFORM_ADMIN` but every `/api/admin/**` route and the entire frontend admin UI gate on the legacy `ADMIN` string, so the canonical admin is 403'd / sees nothing — and an integration test seeds the literal `"ADMIN"` production never produces, masking it (A1-F1, A2-F1).

**Blocker class 3 — the FL-server orchestration model is a scaling cliff and a control plane half-built.** One Python process per project via `ProcessBuilder`, capped at 11 ports, in-memory `Process` map lost on JVM restart, no run entity, no checkpoint/resume, a round loop with no timeout/quorum (one straggler hangs forever), and an ECS path that logs the task ARN and forgets it (A1-F2, B2, B6-1, C1).

**Blocker class 4 — supply-chain & security posture.** Unsigned Electron auto-install RCE (A5-C1, unfixed since 05-27); gRPC plaintext-by-default though full TLS+mTLS already exists in code (B4-2); no PR-time CI (B7-01); no org-level multi-tenant isolation in `AuthorizationService` (B4); Spring Boot 3.4.5 past OSS EOL (B7).

**The non-technical gate that dominates everything:** DeComFL is RIT research. Under RIT IP policy C03.0, **RIT — not the founder — almost certainly owns it**. No defensible moat claim, diligent raise, or self-branded open-source is possible until an RIT IPMO license/spin-out is executed (C4). This is a go/no-go gate the whole product is downstream of.

**Verdict:** Salvage the Spring Boot control plane and the React frontend; **rebuild the FL orchestration substrate** (long-running, run-keyed, durable); **keep DeComFL custom and fix it**; migrate desktop Electron→Tauri; stand up CI, observability, a dataset/run registry, and an artifact store. Compete vertical-first on the bandwidth wedge; **kill** horizontal "general FL platform" positioning that loses to Flower.

---

## 1. v2 Reference architecture

Reconciles B2 (tech-stack), B3 (observability), B5 (desktop), B6 (scale/cost), B7 (DX), B4 (security), C1–C3 (reliability/data/repro).

### 1.1 Component table (chosen stack per layer)

| Layer | v2 choice | Verdict vs v1 | Sourced from |
|---|---|---|---|
| **Control plane (API)** | Spring Boot 3.5+ LTS, Java 21, Gradle. Cookie-only JWT + STOMP. | **Salvage** (bump off EOL 3.4.5) | B2, B7, A1 |
| **AuthZ** | Add `org_id`-scoped checks + RLS-style query filters; collapse roles to an enum; per-run scoped result tokens. | **Refactor** | B4, A1 |
| **FL orchestration substrate** | **Rebuild:** long-running multi-run server keyed on `run_id` (SuperLink-style). `FlServerLauncher` abstraction: **k8s Jobs primary**, ECS RunTask salvaged, `LocalProcessLauncher` dev-only. Durable `fl_runs` lease table + reconciler. | **Rebuild** (kill ProcessBuilder-per-project for non-dev) | B2, A1, B6, C1 |
| **FL framework** | Custom Python (no `flwr`). FedAvg + **DeComFL** (fixed: 1/P factor, CPU-canonical RNG, typed/safetensors codec). Keep chunking + dual heartbeat. | **Salvage core, rebuild serializer** | B1, A3, B2 |
| **Mobile FL core** | Native C++ libtorch + gRPC (DeComFL). Stays mobile-only; CPU-canonical RNG + golden-vector parity test gate. | **Salvage** (lift `mobile_client/` in a v2 step) | A6, B5, C3 |
| **OLTP datastore** | **Managed Postgres (RDS)** + 1-yr RI through Series-A; Aurora only at hyperscale. **No Citus** (control-plane tables are bounded). Flyway-owned schema (already Postgres-dialect; fix `CLOB`→`TEXT/JSONB`). | **Rebuild** (kill H2 outside dev/test) | B6, B2 |
| **Artifact / model store** | **S3/MinIO, content-addressed (sha256)** checkpoints wired to `fl_runs`. Prerequisite for reproducibility + round recovery. | **Rebuild** (does not exist; only TODOs) | B2, C1, C3 |
| **Dataset / partition registry** | Flyway **V6**: `datasets` / `dataset_versions` / `partition_recipes` keyed on content hashes; one `DataSource`/`Partitioner` interface (kills the 4 forks, removes flwr-datasets). | **Rebuild** (does not exist) | C2 |
| **Experiment / run lineage** | `FlRun` aggregate + determinism manifest (seed, hyperparams, lib/dataset/model hashes); **MLflow** self-hosted (Apache-2.0, $0, data-resident) Model Registry. | **Rebuild** (no run entity today) | C3, B3 |
| **Real-time channel** | STOMP-over-WS; back the in-memory simple broker with a **Redis/RabbitMQ (Amazon MQ) relay** when multi-replica. | **Refactor** (one-line relay swap) | B6, A1, B2 |
| **Platform observability** | Micrometer→Prometheus on an internal mgmt port; Grafana + Loki + Tempo + OTel Collector; **W3C traceparent** through `ProcessBuilder` env / gRPC metadata (JVM→Python→client→mobile); structlog with `project_id`/`round_idx`/`trace_id`. | **Rebuild** (deps pinned, imported nowhere) | B3, B7 |
| **FL-run telemetry** | Salvage the existing `RoundResult`→`/api/internal/results`→STOMP→recharts pipeline; make per-round POST **incremental** (today it batches after the run); add a **communication-cost panel** (DeComFL's wedge) + per-client small-multiples; V6 column for bytes/scalars-transmitted. | **Salvage + extend** | B3, C5 |
| **Frontend** | React 19 + Vite 6 (no Next.js). TanStack Query for server-state. Add CSP+HSTS, Vitest+Playwright+MSW, code-split, V5 role types + Zod at the wire boundary. | **Salvage core, refactor** | A2, C5 |
| **Desktop** | **Tauri v2** (React renderer reused; ~5 main services → small Rust command layer w/ bollard + keychain). Mandatory signed minisign updater structurally kills the C5 RCE class. | **Rebuild shell, salvage subprocess model** | B5, A5 |
| **Design system** | One **OKLCH token package** (seeded from web `theme.css`) → **shadcn/ui** (web+desktop) + **react-native-reusables/NativeWind** (mobile). One brand. | **Rebuild** (3 disjoint palettes today) | C5 |
| **Monorepo / CI** | Makefile/Taskfile + `dorny/paths-filter` affected-builds (reject Bazel; defer Nx to the JS/TS triangle). **`buf`** single-source proto + breaking-change gate. PR-time `ci.yml` + branch protection. Renovate + per-stack vuln scans + SBOM. | **Rebuild** (no PR CI today) | B7 |
| **Aggregation robustness** | Add DP (DP-SGD on FedAvg, calibrated scalar-DP on DeComFL) + a robust-mean/clipping guard; **delete the false "Byzantine-robust" README claim**. | **Rebuild** (none today) | B4, B1 |

### 1.2 Data-flow (v2)

```
Browser ── HTTPS ──▶ nginx (TLS term) ──▶ Spring Boot control plane (:8081, HA, behind LB)
                                              │   owns users/orgs/projects/fl_runs (Postgres + RLS)
                                              │   STOMP relay (Redis/RabbitMQ) for /topic/* across replicas
                                              │
                            POST /start ──────┤  FlServerLauncher.launch(run_id, config)
                                              ▼
                              k8s Job (primary) / ECS RunTask / dev LocalProcess
                                  = one long-running FL server keyed on run_id
                                  │  reads model + dataset_version from S3 (content-addressed)
                                  │  per-round: writes checkpoint to S3, POSTs RoundResult+comm-bytes
                                  │            (incremental, per-round) to /api/internal/results/{run}
                                  │            with a per-run scoped token (not the global key)
                                  ▼
                       gRPC (TLS + mTLS, cert-CN-bound identity)
                                  │  scalars + seeds only (DeComFL) — no raw gradients
                                  ▼
        FL clients: desktop (Tauri→PyInstaller subprocess / Docker on Jetson)
                    + mobile (native C++ libtorch, CPU-canonical RNG)
                    raw data never leaves the device; CPU-canonical perturbation RNG everywhere

Telemetry: every hop carries a W3C traceparent → OTel Collector → Tempo/Loki/Prometheus → Grafana.
Reconciler: boot-time + periodic; the JVM is a stateless supervisor over the fl_runs DB lease.
Lineage: each run writes a determinism manifest + content-addressed artifacts to MLflow + S3.
```

---

## 2. Per-unit decision table (reconciled across all agents)

Verdicts reconciled where agents conflicted; conflicts resolved explicitly in §3. `(D)` = a related framing was demoted in verification (see `_verification.md`).

| Unit / module | Verdict | Rationale (agents) |
|---|---|---|
| Spring Boot control plane | **SALVAGE** | Mature; bump off EOL 3.4.5; decompose the 438-line ProjectService god-object (A1, B2). |
| Auth/RBAC role model | **REFACTOR** | Collapse `ADMIN`/`PLATFORM_ADMIN` drift to an enum; fix masked test; topic-level WS authz (A1-F1, A2-F1). |
| Multi-tenant org isolation | **REBUILD** | `AuthorizationService` never checks `org_id`; discover leaks cross-org PUBLIC metadata (B4). |
| Internal result callbacks | **REFACTOR** | Per-run scoped tokens replacing the single global key (A1-F6). |
| FL orchestration — local ProcessBuilder | **KILL** (non-dev) / keep as dev `LocalProcessLauncher` | 11-port cap, no isolation, state lost on restart (A1, B2, B6, C1). |
| FL orchestration — v2 substrate | **REBUILD** | `fl_runs` + `FlServerLauncher` + reconciler; k8s Jobs primary (A1, B2, C1). |
| ECS Fargate path | **SALVAGE→complete** | Persist task ARN, stop/poll, reconcile, per-org quota (A1, C1, B6). |
| Round loop | **REBUILD** | No timeout/quorum — one straggler hangs forever (C1). |
| Per-round checkpoint/resume | **REBUILD** | Does not exist; DeComFL state is tiny — build durable ledger + S3 artifact (C1). |
| `serializer.py` (chunked codec) | **REBUILD** | C1 upload break; typed/safetensors codec (A3-C1). |
| DeComFL algorithm core | **SALVAGE, fix** | Math correct; fix 1/P factor + CPU-canonical RNG (B1-C1, B1-C2). |
| README "Byzantine-robust" claim | **KILL** | False; paper makes no such claim; aggregation is unguarded mean (B1-H3, B4). |
| FedAvg/DeComFL aggregation robustness | **REBUILD** | Add DP + robust guard (B4). |
| Datastore (H2-file) | **REBUILD→Postgres** | Production-tier; ec2demo migration is config-level `(D)` (B2, B6). |
| Artifact/model store | **REBUILD** | Does not exist; only S3 TODOs (B2, C1, C3). |
| Dataset registry / partitioner | **REBUILD** + KILL flwr_datasets + collapse 4 `dirichlet_split` forks | No registry/lineage; pickle cache → content-addressed npz (C2). |
| Central ECG CSV in JAR | **REFACTOR** (was KILL `(D)`) | Hand server a pre-split test set; drop 5.7 MB from JAR — hygiene, not FL-premise breach (C2). |
| Run/experiment lineage + model registry | **REBUILD** | No run entity; MLflow + manifests (C3, B3). |
| Determinism / golden-vector tests | **REBUILD** | Zero exist; CPU-canonical RNG + Python↔C++ parity in CI (C3). |
| Platform observability | **REBUILD** | Micrometer/OTel wiring; correlation IDs (B3-02, B7). |
| FL-run telemetry pipeline | **SALVAGE + extend** | Producer exists `(D)`; make per-round POST incremental, add comm-cost panel (B3, C5). |
| `async_coordinator.py` | **KILL** | Dead RabbitMQ code (B3). |
| In-memory STOMP broker | **REFACTOR** (was REBUILD `(D)`) | One-line relay swap when multi-replica (B6, A1, B2). |
| Frontend React/Vite base | **SALVAGE** | Right tool; no SSR need (A2). |
| Cookie-auth + 401 interceptor | **SALVAGE** | Textbook-correct posture (A2). |
| V5 identity type contract | **REBUILD** | Live bug — admin UI dead (A2-F1). |
| Security headers (CSP/HSTS) | **REFACTOR** (was REBUILD `(D)`) | ~10 lines in `SecurityConfig.headers()` + frontend CSP (A2). |
| Frontend test layer | **REBUILD/stand-up** | Zero tests on auth/STOMP/role-gates (A2). |
| Server-state mgmt | **REFACTOR→TanStack Query** | Kills duplicate fetch triads (A2). |
| `react-icons` / `frontend/dist/` | **KILL** dep (low) / **no-op** `(D)` | Unused 2nd icon lib; dist never tracked (A2). |
| Desktop shell | **REBUILD→Tauri v2** | Mandatory signed updater kills RCE class (B5, A5). |
| Desktop auto-updater + signing | **REBUILD** | Unsigned auto-install RCE, unfixed (A5-C1). |
| Desktop fail-open IPC bridge | **REFACTOR** (hard release gate) | Fail closed in packaged builds (A5-C3). |
| Desktop auth token model | **REFACTOR** | Delete dead `accessToken` branch; source expiry from backend (A5-C2). |
| Desktop Jetson device-mount path | **SALVAGE** | Correct; no `--runtime nvidia` (A5, A4). |
| Per-OS native desktop apps | **KILL** | 3× UI, no payoff (B5). |
| Thin-shell-over-mobile-C++ (desktop) | **KILL** | No bundle win, fragile RNG parity (B5). |
| Mobile RN bridge / screens / nav | **SALVAGE** (RN), **REBUILD** styling | Sound TurboModule wiring; unthemed Bootstrap hex (A6, C5). |
| Mobile proto copies | **REBUILD** | Drift; one missing DeComFL `(D severity)` (A6, B7). |
| Mobile ZO C++ core | **REBUILD/harden** | float32, no requires_grad filter, untested RNG parity (A6, C3). |
| Mobile DeComFL UI path / TF.js JS / MNIST blobs | **KILL** | Disabled/dead/duplicated (A6). |
| Mobile inference "confidence" chart | **KILL** | Fabricated `exp(-loss)` proxy (C5). |
| Client-docker thin-wrapper + Jetson | **SALVAGE** | Single source of truth via pip (A4). |
| Client-docker Dockerfile build | **REFACTOR→multi-stage** (was REBUILD) | `--no-deps` + base bump (A4). |
| Dependency manifests / supply chain | **REBUILD** | Pin to digest, pip-audit/Trivy/SBOM (A4, B7). |
| Client-docker pickle cache | **REFACTOR→npz+sha256** | Versioning/integrity, not RCE `(D)` (A4, C2). |
| PR-time CI + branch protection | **REBUILD** | The highest-leverage, lowest-cost fix (B7-01). |
| `desktop-release.yml` | **KILL** | Duplicate release workflow on same tag (B7). |
| Backend static analysis | **REBUILD** | Spotless+Checkstyle+SpotBugs+JaCoCo+ArchUnit (B7). |
| Proto codegen | **REFACTOR→buf** (was REBUILD `(D)`) | Single source + breaking gate; DX hygiene (B7, B2). |
| Cross-surface design tokens / component lib | **REBUILD** | 3 palettes; one OKLCH token pkg → shadcn + RN-reusables (C5). |
| Brand identity | **REBUILD** | One FedLearn brand; retire FedMob/Desktop sub-brands (C5). |
| DeComFL as the wedge | **SALVAGE** | Only paper-backed differentiator (C4, B2). |
| DeComFL IP title | **REBUILD** | RIT C03.0 owns it; IPMO license = go/no-go gate (C4). |
| Horizontal general-FL positioning | **KILL** | Loses to Flower; vertical-first instead (C4). |
| Pricing / GTM / ToS-DPA | **REBUILD** | Open-core + usage-based; no legal terms exist (C4). |
| Compliance program | **REBUILD** | SOC 2 Type 2 + HIPAA-readiness; defer FedRAMP (B4). |

---

## 3. Explicitly resolved conflicts between agents

1. **FL substrate: KILL vs REBUILD vs REFACTOR (A1 / B2 / B6 vs C1).** Resolved: **KILL the local ProcessBuilder-per-project model for non-dev; REBUILD the substrate concept** as a long-running run-keyed launcher; keep ProcessBuilder as a dev-only `LocalProcessLauncher` behind the abstraction. The ECS path is **salvage-and-complete**, not kill. This satisfies all four agents — they were describing the same target at different altitudes.

2. **STOMP broker: REBUILD (B6) vs REFACTOR (A1, B2).** Resolved in favor of **REFACTOR** — it is a one-line `enableStompBrokerRelay` swap once the backend is multi-replica. Two sibling audits already rated it refactor; B6's "rebuild/high" was demoted in verification.

3. **flwr-datasets framing: "no-Flower invariant breach / critical" (A4, B2) vs "Apache-2.0, not legal risk / hygiene" (C4).** Resolved: **KILL the dependency** (all agree on the action) but **C4's framing wins on severity** — it is bundle-bloat + the platform's own self-advertised "no Flower" hygiene, **not** a legal/IP risk and not a hard invariant breach (flwr-datasets does not depend on the `flwr` FL framework). Severity: low/medium.

4. **Desktop auto-update RCE severity (A5 critical) vs verification (1 refuter: latent because no release feed metadata ships today).** Resolved: keep as a **hard release gate / Phase-0 fix** — the dangerous booleans and public feed are real; the fix (two booleans + signing) is trivial and the Tauri migration eliminates the class structurally. Treat as "fix before any public distribution."

5. **Central ECG CSV: KILL (C2) vs verification (2/3 refuted — server only builds a *test* loader, public benchmark, not PII).** Resolved: **REFACTOR** — hand the server a pre-split held-out test set and drop the 5.7 MB CSV from the JAR/git. Build hygiene, not an FL-premise contradiction.

6. **B3-01 "empty dashboard / no producer" (3/3 refuted — producer exists in `fl_server.py:561-587`).** Resolved: the producer exists but **POSTs in a batch after the run completes**. Re-scoped to **"make the per-round POST incremental"** so the chart populates live during training, not "build the producer."

7. **H2 datastore: KILL ec2demo now (B6) vs production-tier rebuild (B2).** Resolved: **REBUILD to managed Postgres for the production/multi-replica tier** (B2's framing); ec2demo's H2-on-EBS is acceptable until that cutover (B6's KILL was demoted). Must fix `audit_events.metadata CLOB`→`TEXT/JSONB` before the cutover.

8. **Security headers REBUILD (A2) — demoted to REFACTOR.** Spring Security already emits HSTS + nosniff by default; the real gap is CSP + Referrer-Policy, ~10 lines.

9. **Pickle cache: RCE (A4, demoted) vs versioning hygiene (C2, survived).** Resolved: **REFACTOR to content-addressed npz+sha256** for versioning/integrity and to fix the stale-split trap. The RCE framing does not hold (cache lives in the container's own image layer, not a bind-mount); the versioning rationale does.

---

## 4. Cross-cutting risk register

| # | Domain | Risk | Severity | Source | v2 mitigation |
|---|---|---|---|---|---|
| R1 | **IP** | RIT (not founder) owns DeComFL under C03.0; unfundable / can't claim moat until licensed | **Critical (go/no-go)** | C4 | Execute RIT IPMO license/spin-out **before** any raise or moat claim |
| R2 | Correctness | DeComFL server step P× too large; diverges from rebuild trajectory | Critical | B1-C1 | Remove the spurious `*P` in `aggregate_fit`; property test |
| R3 | Correctness | DeComFL perturbation RNG diverges CPU/CUDA/MPS → silent corruption | Critical | B1-C2/C3-1 | CPU-canonical RNG everywhere + golden-vector parity test in CI |
| R4 | Correctness | Chunked/LLM upload `KeyError` — transformer federations can't complete a round | Critical | A3-C1 | Wrap upload payload symmetrically; streaming roundtrip test |
| R5 | Security | Unsigned Electron auto-install on quit → supply-chain RCE | Critical | A5-C1 | Disable auto-install now; sign; migrate to Tauri minisign updater |
| R6 | Security | gRPC plaintext-by-default; self-asserted `client_id` (Sybil) though TLS+mTLS exists | Critical | B4-2 | Default-secure; bind identity to cert CN + backend enrollment token |
| R7 | AuthZ | Bootstrap admin 403'd from all admin routes; test masks it | Critical | A1-F1/A2-F1 | Collapse role to enum; fix the masked test; align FE/BE constants |
| R8 | Multi-tenant | No `org_id` isolation; cross-org PUBLIC metadata leak | High | B4 | Org-scoped query filters / RLS; per-run scoped result tokens |
| R9 | Reliability | No run entity, no checkpoint/resume; round loop hangs on one straggler; state lost on restart | High | C1, A1-F2 | `fl_runs` lease + reconciler; per-round S3 checkpoint; round deadline + min-quorum |
| R10 | Cost | FL-server unbounded once 11-port cap lifted; no quotas/scale-to-zero | High | B6-1 | Per-org concurrency quotas + admission control **before** lifting the cap; scale-to-zero orchestration |
| R11 | Compliance | No SOC2/HIPAA/GDPR program; healthcare demo makes HIPAA the floor; no ToS/DPA | High | B4, C4 | SOC 2 Type 2 + HIPAA-readiness architecture from day one; draft ToS/DPA before first regulated deal; defer FedRAMP |
| R12 | Security | No DP, no robust aggregation; false "Byzantine-robust" public claim | High | B4, B1 | Add DP-SGD/scalar-DP + robust guard; delete the false claim |
| R13 | DX | No PR-time CI; broken/vulnerable code can merge; EOL Spring Boot 3.4.5 | High | B7-01 | `ci.yml` + branch protection + Renovate + per-stack scans; bump to 3.5 LTS |
| R14 | Reproducibility | No seed/hyperparam/version/artifact-hash capture; can't reproduce own runs | High | C3, C2 | Determinism manifest + content-addressed artifacts + dataset registry (V6) |
| R15 | Data | Pickle split cache (no integrity/versioning, stale-split trap); flwr_datasets contamination | Medium | C2, A4 | Content-addressed npz+sha256; own Partitioner; remove flwr-datasets |
| R16 | Reliability | Destructive in-place model save; no off-host copy | Medium | C1 | Versioned content-addressed artifacts in S3 |
| R17 | UX/Brand | Three product names/palettes; fabricated mobile confidence chart erodes trust | Medium | C5 | One brand + OKLCH token package; remove fabricated chart |

**Mitigated/structural strengths to preserve:** DeComFL scalar-only uploads structurally kill the DLG reconstruction attack family (B4-1) — the platform's privacy wedge; the cookie-only HttpOnly JWT posture (A2); the Jetson `/dev/nvhost-*` device-mount path with no `--runtime nvidia` (A4, A5); Flyway-owned schema discipline (A1, B2).

---

## 5. Prioritized next-brainstorm queue (v2 build)

Each item below becomes its own `brainstorming → writing-plans → implementation` cycle. P0 = gates the company / blocks any v2 launch.

### P0 — gates everything
- **IP resolution (DeComFL / RIT C03.0).** Execute the IPMO license/spin-out. *Nothing downstream is defensible until this lands.* (C4)
- **DeComFL correctness trifecta.** Fix the `1/P` server-step factor, make perturbation RNG CPU-canonical with a golden-vector parity test, and fix the chunked-upload `KeyError`. This is the product. (B1, A3, C3)
- **PR-time CI + branch protection.** `ci.yml` (gradle/pytest/vitest/eslint + paths-filter), Renovate, gitleaks, per-stack vuln scans; kill the duplicate release workflow. (B7)
- **Identity/RBAC end-to-end fix.** Role enum, fix the masked test, align FE/BE constants, bump off EOL Spring Boot. (A1-F1, A2-F1, B7)
- **Stop-the-bleeding security.** gRPC default-secure (TLS+mTLS already coded); disable Electron auto-install + sign; org-scoped authz. (B4, A5)

### P1 — v2 architecture foundations
- **FL orchestration substrate rebuild:** `fl_runs` lease + `FlServerLauncher` (k8s Jobs primary) + reconciler + round deadline/quorum + per-org quotas + scale-to-zero. (A1, B2, B6, C1)
- **Artifact + dataset + run-lineage stack:** S3/MinIO content-addressed checkpoints, Flyway V6 dataset/partition registry, `FlRun` + determinism manifest + MLflow registry. (C1, C2, C3, B2)
- **Observability rebuild:** Micrometer/Prometheus + Grafana/Loki/Tempo + OTel Collector + W3C traceparent JVM→Python→client→mobile; make FL-run per-round POST incremental + add the comm-cost panel. (B3, B7, C5)
- **Managed Postgres cutover** (fix `CLOB`→`TEXT/JSONB`; Testcontainers in CI). (B2, B6)

### P2 — product & compliance
- **Desktop Tauri v2 migration** (signed minisign updater, Rust command layer, fail-closed bridge). (B5, A5)
- **Cross-surface design system:** OKLCH token package → shadcn + react-native-reusables; one brand; remove fabricated charts. (C5)
- **Compliance program:** SOC 2 Type 2 + HIPAA-readiness architecture; ToS/DPA; pricing (open-core + usage-based metering on rounds×clients). (B4, C4)
- **FL robustness/privacy:** DP layer + robust aggregation guard; delete the false Byzantine claim; market the scalar-only DLG-resistance wedge. (B4, B1)
- **Supply-chain hardening:** digest-pinned multi-stage Docker, SBOM, base bump. (A4, B7)

### P3 — mobile & polish
- **Mobile subtree lift + harden:** `mobile_client/` onto main; reconcile proto via buf; CPU-canonical ZO RNG + `requires_grad` filter + golden-vector test; kill TF.js-era JS, MNIST blobs, disabled DeComFL UI. (A6, C3, B7)
- **Frontend hardening:** TanStack Query, code-split, Zod wire boundary, Vitest+Playwright+MSW, CSP/HSTS. (A2)
- **Backend static analysis** (Spotless/Checkstyle/SpotBugs/JaCoCo/ArchUnit) + decompose the ProjectService god-object. (B7, A1)

---

*All file:line evidence and per-finding skeptic reasoning are in the sibling reports and [`_verification.md`](_verification.md). Demoted findings (9 of 96 verified items) are de-prioritized per that ledger.*
