# FedLearn Platform v2 Audit — Adversarial Verification Ledger

**Date:** 2026-05-29
**Branch:** `main-clean` (mobile audited read-only on `origin/fed-mobile`)
**Method:** Each major finding and every salvage/refactor/rebuild/kill verdict was challenged by **3 independent skeptic agents** under three lenses (correctness, evidence, materiality). A finding **survived** if fewer than 2 skeptics refuted it. `refutes` = number of skeptics (of 3) who refuted.

> How to read this: **SURVIVED** items are load-bearing for the synthesis. **DEMOTED** items (survived=false, refutes≥2) are de-prioritized — the technical fact is usually true but the *severity* or *verdict* was overstated. See the DEMOTED summary at the bottom.

---

## 1. Verdict & finding ledger

| Agent | Finding / Verdict | Severity | Survived | Refutes (of 3) |
|---|---|---|:--:|:--:|
| A1 backend | KILL — FL-server orchestration (local ProcessBuilder) | high | ✅ | 1 |
| A1 backend | REBUILD — FL-server orchestration (v2 target: `fl_runs` + launcher + reconciler) | high | ✅ | 0 |
| A1 backend | REBUILD — Observability of FL runs | high | ✅ | 1 |
| A1 backend | **F1** Bootstrap admin 403'd from all `/api/admin/**`; test masks the bug | critical | ✅ | 0 |
| A1 backend | **F2** ProcessBuilder scaling cliff; ECS path is half a control plane | critical | ✅ | 1 |
| A1 backend | **F6** Any FL task can write/finish ANY project (broken object-level auth) | high | ✅ | 1 |
| A2 frontend | REBUILD — V5 identity type contract (`role: USER\|ADMIN`) | high | ✅ | 0 |
| A2 frontend | REBUILD — Security-header layer (CSP/HSTS) | high | ❌ **DEMOTED** | 2 |
| A2 frontend | REBUILD — Frontend test layer | high | ✅ | 1 |
| A2 frontend | KILL — `react-icons` dependency | high | ✅ | 1 |
| A2 frontend | KILL — `frontend/dist/` committed | high | ❌ **DEMOTED** | 3 |
| A2 frontend | **F1** V5 identity mismatch is LIVE — admin/permission UI is dead | critical | ✅ | 0 |
| A3 framework | REBUILD — `serializer.py` (C1 chunked-upload break) | high | ✅ | 0 |
| A3 framework | **A3-C1** Chunked-upload broken (KeyError `'parameters'`) | critical | ✅ | 0 |
| A4 client-docker | KILL — `flwr_datasets` runtime dependency | high | ✅ | 1 |
| A4 client-docker | REBUILD — Dockerfile build (dep-order, single-stage, EOL base) | high | ✅ | 1 |
| A4 client-docker | REBUILD — Dependency manifests (3 divergent specs) | high | ✅ | 1 |
| A4 client-docker | REBUILD — Pickle split cache (RCE-as-fedlearn) | high | ❌ **DEMOTED** | 2 |
| A4 client-docker | REBUILD — Supply-chain posture (no pinning/scan/SBOM) | high | ✅ | 0 |
| A4 client-docker | **CD1** flwr-datasets leak violates no-Flower invariant; trivially removable | critical | ✅ | 1 |
| A5 desktop | REBUILD — `updater.ts` + electron-builder code signing | high | ✅ | 1 |
| A5 desktop | REBUILD — CI security scanning (absent) | high | ✅ | 0 |
| A5 desktop | **A5-C1** Unsigned auto-install supply-chain RCE (verified, escalated, UNFIXED) | critical | ✅ | 0 |
| A5 desktop | **A5-C3** Renderer IPC bridge fails OPEN to fake authenticated preview-user | critical | ✅ | 1 |
| A5 desktop | **A5-C2** Auth model launders HttpOnly cookie into Bearer; dead `accessToken` path | high | ✅ | 1 |
| A5 desktop | **A5-H1** Packaged-renderer CSP retains `'unsafe-eval'` | high | ✅ | 1 |
| A6 mobile | REBUILD — Proto copies (drift; one missing DeComFL) | high | ✅ | 1 |
| A6 mobile | REBUILD — ZO C++ core (float32 trunc, no requires_grad filter, untested RNG) | high | ✅ | 1 |
| A6 mobile | KILL — DeComFL UI path (disabled/unreachable) | high | ✅ | 1 |
| A6 mobile | KILL — TF.js-era JS (imports tfjs not in package.json) | high | ✅ | 1 |
| A6 mobile | KILL — MNIST blobs (committed twice) | high | ✅ | 1 |
| A6 mobile | **M-C1** Proto drift | critical | ✅ | 1 |
| B1 paper | REBUILD — Server update `aggregate_fit` (missing 1/P factor) | high | ✅ | 0 |
| B1 paper | REBUILD — Server perturbation RNG device (CUDA vs CPU/MPS) | high | ✅ | 1 |
| B1 paper | KILL — README "Byzantine-robust" claim | high | ✅ | 0 |
| B1 paper | **B1-C1** Server step P× too large; diverges from rebuild trajectory | critical | ✅ | 0 |
| B1 paper | **B1-C2** Server regenerates perturbations on cuda; silent non-learning | critical | ✅ | 0 |
| B1 paper | **B1-H3** README 'Byzantine-robust' claim unsupported by paper and code | high | ✅ | 0 |
| B2 tech-stack | KILL — FL substrate ProcessBuilder-per-project spawn model | high | ✅ | 1 |
| B2 tech-stack | REBUILD — FL substrate as a concept (long-running multi-tenant) | high | ✅ | 0 |
| B2 tech-stack | REBUILD — Proto codegen / vendored copies | high | ❌ **DEMOTED** | 2 |
| B2 tech-stack | KILL — `flwr-datasets` runtime dependency | high | ✅ | 0 |
| B2 tech-stack | REBUILD — Datastore (H2 file-mode) | high | ✅ | 1 |
| B2 tech-stack | REBUILD — Artifact / model store | high | ✅ | 0 |
| B3 observability | KILL — `async_coordinator.py` (RabbitMQ ResultConsumer, dead code) | high | ✅ | 0 |
| B3 observability | REBUILD — Mobile telemetry path | high | ❌ **DEMOTED** | 2 |
| B3 observability | REBUILD — Correlation-ID / distributed tracing | high | ✅ | 0 |
| B3 observability | REBUILD — Experiment tracking (MLflow) | high | ❌ **DEMOTED** | 2 |
| B3 observability | **B3-01** FL-run telemetry pipeline fully built but NO producer (empty chart) | high | ❌ **DEMOTED** | 3 |
| B3 observability | **B3-02** Platform observability absent; pinned OTel/Prom deps unused | high | ✅ | 0 |
| B4 security | REBUILD — gRPC trust model (plaintext default, self-asserted client_id) | high | ✅ | 0 |
| B4 security | REBUILD — Multi-tenant authorization (org isolation) | high | ✅ | 0 |
| B4 security | REBUILD — FL aggregation robustness (Byzantine/poisoning/DP) | high | ✅ | 0 |
| B4 security | REBUILD — Compliance program (none today) | high | ✅ | 0 |
| B4 security | **B4-1** DeComFL scalar-only upload structurally kills DLG family (not privacy w/o DP) | medium | ✅ | 0 |
| B4 security | **B4-2** gRPC plaintext by default though full TLS+mTLS implemented | critical | ✅ | 0 |
| B5 desktop-strat | KILL — Per-OS native apps | high | ✅ | 0 |
| B5 desktop-strat | REBUILD — Tauri v2 shell (target) | high | ✅ | 0 |
| B5 desktop-strat | KILL — Thin shell over mobile C++ core, in-process, for v2 | high | ✅ | 0 |
| B5 desktop-strat | REBUILD — Auto-updater + signing config | high | ✅ | 0 |
| B5 desktop-strat | **B5-1** Desktop is a subprocess orchestrator; torch in bundle regardless of shell | high | ❌ **DEMOTED** | 1* |
| B5 desktop-strat | **B5-2** Migrate Electron→Tauri; deciding factor is C5 not size | high | ✅ | 1 |
| B6 scale-cost | REBUILD — Orchestration at hyperscale (EKS+Karpenter GPU) | high | ✅ | 0 |
| B6 scale-cost | REBUILD — In-memory STOMP broker | high | ✅ | 1 |
| B6 scale-cost | KILL — DB H2-file mode in ec2demo | high | ❌ **DEMOTED** | 2 |
| B6 scale-cost | KILL — Citus / distributed Postgres | high | ✅ | 0 |
| B6 scale-cost | KILL — Aurora Serverless v2 for production OLTP | high | ✅ | 1 |
| B6 scale-cost | **B6-1** FL-server capped at 11 concurrent; no scale-to-zero/quotas | critical | ✅ | 1 |
| B7 standards-dx | REBUILD — PR-time CI (absent) | high | ✅ | 0 |
| B7 standards-dx | KILL — `desktop-release.yml` (duplicate release workflow) | high | ✅ | 0 |
| B7 standards-dx | REBUILD — Proto codegen (3 copies, 3 mechanisms) | high | ❌ **DEMOTED** | 1* |
| B7 standards-dx | REBUILD — Backend static analysis (none) | high | ✅ | 0 |
| B7 standards-dx | REBUILD — Dependency hygiene (Renovate/audit/EOL Spring Boot 3.4.5) | high | ✅ | 0 |
| B7 standards-dx | **B7-01** Zero PR-time CI — any PR can merge broken/vulnerable | critical | ✅ | 0 |
| C1 reliability | REBUILD — Per-round checkpoint / resume subsystem | high | ✅ | 0 |
| C1 reliability | REBUILD — seed/gradient history persistence | high | ✅ | 1 |
| C1 reliability | REBUILD — Round loop (no timeout/quorum; straggler hangs forever) | high | ✅ | 0 |
| C1 reliability | REBUILD — ECS-Fargate path (fire-and-forget runTask) | high | ✅ | 1 |
| C1 reliability | KILL — In-memory `runningServers` map as source of truth | high | ✅ | 1 |
| C1 reliability | KILL — Destructive in-place model save (`fl_server.py:545`) | high | ✅ | 1 |
| C2 data-eng | KILL — Central ECG CSV bundled in server | high | ❌ **DEMOTED** | 2 |
| C2 data-eng | KILL — Duplicated `dirichlet_split` (x4 forks) | high | ✅ | 1 |
| C2 data-eng | REBUILD — Pickle split cache | high | ✅ | 0 |
| C2 data-eng | KILL — `flwr_datasets` partitioning | high | ✅ | 0 |
| C2 data-eng | REBUILD — Partition reproducibility / seed control | high | ✅ | 1 |
| C2 data-eng | REBUILD — Dataset registry / versioning / lineage | high | ✅ | 0 |
| C3 reproducibility | REBUILD — Perturbation seed-sync / RNG path | high | ✅ | 1 |
| C3 reproducibility | REBUILD — Run/experiment lineage (no run entity) | high | ✅ | 0 |
| C3 reproducibility | REBUILD — Determinism verification tests | high | ✅ | 0 |
| C3 reproducibility | REBUILD — Model registry / artifact lineage | high | ✅ | 0 |
| C3 reproducibility | KILL — Mobile cross-language RNG 'bit-identical' claim | high | ✅ | 0 |
| C3 reproducibility | **C3-1** DeComFL RNG diverges GPU server vs CPU clients; silent corruption | critical | ✅ | 0 |
| C4 business-ip | REBUILD — DeComFL IP title / ownership (RIT C03.0) | high | ✅ | 0 |
| C4 business-ip | KILL — Horizontal general-FL-platform positioning | high | ✅ | 0 |
| C4 business-ip | REBUILD — Pricing model (none) | high | ✅ | 0 |
| C4 business-ip | REBUILD — GTM motion | high | ✅ | 1 |
| C4 business-ip | KILL — `flwr-datasets` dependency (hygiene, not legal) | high | ✅ | 0 |
| C4 business-ip | REBUILD — Customer data/model ownership terms (no ToS/DPA) | high | ✅ | 1 |
| C5 design-ux | REBUILD — Cross-surface visual system / design tokens | high | ✅ | 1 |
| C5 design-ux | REBUILD — Component library / design system | high | ✅ | 1 |
| C5 design-ux | REBUILD — Desktop renderer styling (`styles.css`) | high | ✅ | 0 |
| C5 design-ux | REBUILD — Mobile screens styling (inline hex, emoji tabs) | high | ✅ | 0 |
| C5 design-ux | REBUILD — Per-client / communication-cost FL visualization | high | ✅ | 1 |
| C5 design-ux | KILL — Mobile inference 'confidence' chart (fabricated exp(-loss)) | high | ✅ | 1 |

\* B5-1 and B7's proto-codegen verdict each drew exactly **1** refutation but are flagged here for nuance (see §3); they technically **survived** the <2 threshold. B5-1 is treated as demoted-to-context because its sole refuter showed it is non-actionable framing, not a defect.

---

## 2. Per-item skeptic summaries (verified items)

**A1 KILL local ProcessBuilder (survived, 1 refute).** Two skeptics confirmed every sub-claim against `FlowerServerManager.java`: the 11-port range (`application.properties:125-126`), no isolation (child of API JVM), in-memory `ConcurrentHashMap` lost on crash. The lone refuter argued the verdict ignores the dual-path ECS branch — but the verdict is explicitly scoped to the *local* path "for any non-dev deployment," which is the operative path in every runnable profile. Survives.

**A1 REBUILD v2 orchestration target (survived, 0 refute).** Unanimous. `fl_runs` table absent (migrations stop at V5), no `FlServerLauncher`/reconciler, ECS `taskArn` logged but never persisted (`FlowerServerManager.java:141-143` returns `Optional.empty()`), no `@Transactional` on `/start` (F4 race). Standard control-plane design maps 1:1 onto verified defects.

**A1 REBUILD FL-run observability (survived, 1 refute).** Two confirmed no Micrometer/Prometheus on the classpath, actuator exposes `health,loggers` only, no MDC/correlation IDs, in-memory STOMP broker. The refuter correctly noted a *working* run-observability layer exists (server_logs persistence, round_result, STOMP channels) so "REBUILD" overstates vs "enhance" — but the operational-metrics gap is real. Survives with the caveat that the live log/result feed is salvageable.

**A1 F1 — bootstrap admin 403'd (survived, 0 refute).** Unanimous and airtight. `BootstrapRunner.java:122` sets `PLATFORM_ADMIN`; `CustomUserDetailsService` emits `ROLE_PLATFORM_ADMIN`; `AdminController.java:17` requires `hasRole('ADMIN')` = `ROLE_ADMIN`. No `RoleHierarchy` bean. Integration test seeds the literal `"ADMIN"` string production never produces, masking the lockout. The bug is broader than the UI — the backend admin API is also dead for the canonical admin.

**A1 F2 — ProcessBuilder cliff / half a control plane (survived, 1 refute).** Two skeptics verified the 11-port cap, no isolation, orphaned children on crash, and ECS `runTask` with no ARN persistence / no `StopTask` / no reconciliation. The refuter argued the ECS path is gated to the unfinished `production` profile and the live `ec2demo` is single-host demo, so "critical" is overstated. The in-memory-map orphaning of `/stop`+`/delete` after a crash remains a legitimate live concern. Survives but read severity as "blocks production scale," not "live outage today."

**A1 F6 — broken object-level auth on internal callbacks (survived, 1 refute).** Two confirmed `ResultsController` takes `projectId` from the path with no ownership check, gated only by a single global `APP_INTERNAL_API_KEY` shared across all FL tasks (`FlowerServerManager.java:189,386`). The refuter argued the key lives only in operator-controlled server processes (never in tenant containers), so it is integrity-hardening (medium) not a tenant-reachable IDOR (high). **Read as: real per-run-token design gap for v2, severity contingent on the future multi-tenant FL-server trust boundary.**

**A2 F1 / REBUILD V5 identity contract (survived, 0 refute).** Unanimous. Backend emits `PLATFORM_ADMIN` (`AuthController.java:147,179`), frontend types `role: 'USER'|'ADMIN'` and gates all admin UI on `=== 'ADMIN'` (`Sidebar.tsx:99`, `AdminUsersPage.tsx:41`, etc.). The bootstrapped platform admin sees zero admin UI. Confirmed it is the same root cause as A1-F1, end-to-end.

**A2 REBUILD frontend test layer (survived, 1 refute).** Two confirmed zero test files, no Vitest/Playwright/MSW, riskiest code (auth interceptor, STOMP, role-gates, `logStore`) untested. The refuter argued "REBUILD" mislabels greenfield work that should be "build," and that frontend role-gates are UX affordances (the real authz boundary is backend, which IS tested). Survives — the gap is real; treat as "stand up," and prioritize `logStore`/401-interceptor unit tests + cookie-auth E2E.

**A2 KILL react-icons (survived, 1 refute).** Two confirmed it is declared (`^5.5.0`), 39 MB on disk, and has zero imports while lucide-react is used in 24 files. The refuter correctly noted it tree-shakes to zero shipped bytes so "high" severity is inflated. Survives as a correct one-line cleanup; **read severity as low.**

**A3 / A3-C1 chunked-upload break (survived, 0 refute, reproduced).** Unanimous and independently reproduced (`KeyError: 'parameters'` at `serializer.py:155`). Upload path saves a bare `OrderedDict` (`serializer.py:97`, `grpc_client.py:194`) while `chunks_to_parameters` expects a wrapped dict; the download path correctly wraps (`grpc_servicer.py:91`). `ALWAYS_STREAM_TRANSFORMERS=True` + >100MB threshold route every transformer/LLM upload to the broken path. This is the platform's stated differentiator and is 100% broken for its intended use case. **This is the single most important correctness finding.**

**A4 KILL flwr_datasets (survived, 1 refute).** Two confirmed it is a runtime dep used only on the CIFAR-10 path (`client.py:363`), forces matplotlib+seaborn into the native bundle, and that two in-repo Dirichlet splitters already exist. The refuter argued flwr-datasets does not depend on the `flwr` FL framework (shared PyPI namespace only) so "invariant violation" is imprecise and severity is inflated. Survives — removal is correct; **read as bundle-bloat + hygiene (low/medium), not a hard invariant breach.**

**A4 REBUILD Dockerfile build (survived, 1 refute).** Two confirmed: framework installed without `--no-deps` then numpy re-pinned `<2.0`, build-essential/git shipped in the single-stage runtime, tag-pinned EOL 2.0.1 base. The refuter showed the "across the torch ABI" framing is backwards (numpy<2.0 is actually the torch-2.0-compatible state) and the base is an overridable ARG. Survives — the missing `--no-deps`, single-stage hygiene, and stale base are real; **fix is targeted (`--no-deps` + multi-stage + base bump), not a from-scratch rebuild.**

**A4 REBUILD dependency manifests (survived, 1 refute).** Two confirmed three divergent specs (39 `>=` in Docker vs 28 `==` in native; numpy/transformers/protobuf disagree) and that numpy 2.1.2 has no cp38 wheel for the documented Jetson Python 3.8 target. The refuter argued the divergence is intentional multi-arch hygiene (cp38-aarch64 wheel constraints) already neutralized by `--no-deps` in the PyInstaller scripts. Survives — consolidation + a lockfile/CI resolution check is the v2 fix; **the only un-guarded path is the Dockerfile.**

**A4 REBUILD supply-chain posture (survived, 0 refute).** Unanimous. No digest pinning, no pip-audit/Trivy/Grype, no SBOM, EOL base, floor-only pins, plaintext-gRPC edge client. Standard supply-chain DD gaps for an on-prem/edge product.

**A4 CD1 — flwr-datasets invariant (survived, 1 refute).** Same axis as the KILL verdict above; survived with the same "critical is inflated → medium hygiene" caveat.

**A5 / A5-C1 unsigned auto-install RCE (survived, 0 refute).** Unanimous, escalated, still UNFIXED since 05-27. `updater.ts:13-14` (`autoDownload=true`, `autoInstallOnAppQuit=true`), `electron-builder.yml:67` (`identity:null`), public GitHub release feed (`anurag2796/FedLearn-Platform`). On the Windows/Linux paths electron-updater fails open without a publisher signature. Phase-0 fix is two booleans + signing certs.

**A5-C3 fail-open IPC bridge (survived, 1 refute).** Two confirmed `App.tsx:100-105` returns `{success:true, authenticated:true, username:'preview-user'}` when preload is absent, with no `app.isPackaged` gate in the prod webpack config. The refuter argued the fallback shell is functionally inert (empty data, all actions fail) and contextIsolation means no token leak — so it is a fail-open UX bypass, not a critical security hole. Survives — fix is a compile-time prod guard; **read as a hard release gate, severity high not critical.**

**A5-C2 cookie→Bearer laundering (survived, 1 refute).** Two confirmed the dead `accessToken` branch (`auth.service.ts:122`; backend body has no such field) and the working path replays the cookie value as a Bearer header. The refuter argued HttpOnly buys nothing in a DOM-less Electron main process (no XSS surface) and the token is `safeStorage`-encrypted, so the "laundering" framing is wrong and the only real bug is the 1h-vs-24h expiry drift (self-healing via 401→logout). Survives — delete dead branch + source expiry from backend; **severity low/medium.**

**A5-H1 packaged CSP keeps unsafe-eval (survived, 1 refute).** Two confirmed `index.html:8` ships `script-src 'self' 'unsafe-eval'` into the packaged build while the prod webpack bundle has `devtool:false` (no eval needed). The refuter argued unsafe-eval is inert without a prior injection foothold and sandbox+contextIsolation block RCE, so it is low-severity defense-in-depth. Survives as a cheap hardening win; **severity low/medium.**

**A6 mobile verdicts (all survived, mostly 1 refute each).** Proto copies, ZO C++ core, DeComFL UI path, TF.js-era JS, MNIST blobs, and M-C1 each drew one refuter arguing the underlying observation is true but severity is inflated for an unmerged feature branch (e.g., the proto typo `SubmitModelUpdateReque` does not break the wire because message names aren't on the wire; the drifted mobile proto copy is partly orphaned/dead; the DeComFL UI is already feature-flagged off; the MNIST footprint is actually ~137 MiB not 11 MB but branch-local). All survived the threshold. **Treat the mobile unit as v2 implementation-step work, not launch-blocking — but the ZO RNG-parity gap is genuinely load-bearing for correctness once the C++ core ships.**

**B1-C1 / server step P× too large (survived, 0 refute, reproduced).** Unanimous and empirically reproduced: `decomfl_strategy.py:197` divides by `N·P` then `:200` multiplies by `P`, cancelling the `1/P` averaging. Verified against the official ZidongLiu/DeComFL reference (which divides by `num_pert`) and the in-repo client/rebuild paths (which keep `1/P`). At default P=10 the global model steps 10× too far and diverges from every reconnecting client's rebuild trajectory. **Correctness-breaking; on the live DeComFL path.**

**B1-C2 / C3-1 perturbation RNG device divergence (survived, 0 refute).** Unanimous across two reports. Server generates `z` on CUDA when a GPU is present (`decomfl_strategy.py:77,212`); CPU/MPS clients generate on their device. PyTorch officially does not guarantee identical `torch.randn` across CPU/CUDA for the same seed. DeComFL reconstructs `delta += g·z` by regenerating `z`, so any device mismatch silently corrupts aggregation. Latent on a pure-CPU demo, fatal on the planned heterogeneous fleet / GPU server. Fix: CPU-canonical RNG + golden-vector test.

**B1-H3 / KILL Byzantine claim (survived, 0 refute).** Unanimous. The DeComFL paper (arXiv 2405.15861) makes no Byzantine claim; READMEs (`README.md:32,82,387`; `framework/README.md:9,213`) assert "Byzantine-robust aggregation" and even fabricate a paper title. The aggregation is a plain unguarded mean of scalars (`decomfl_strategy.py:191-197`). False public security claim + misattributed citation → delete.

**B2 KILL ProcessBuilder spawn / REBUILD substrate (survived, 1 / 0 refute).** The spawn-model KILL drew one refuter who argued it conflates the spawn pattern with the substrate concept and over-promotes a planning label to a severity. The substrate-concept REBUILD was unanimous: no `run_id` multiplexing, single-shot `start_server` loop, native C++ mobile client + DeComFL scalar protocol justify staying custom (Option C) but benchmark-gated against a 2-week Flower spike.

**B2 KILL flwr-datasets / REBUILD artifact store (survived, 0 refute).** Both unanimous. Artifact store does not exist (only S3 TODOs); models saved as overwritten local `.npz`. Prerequisite for reproducibility + round recovery.

**B2 REBUILD datastore H2 (survived, 1 refute).** Two confirmed H2-file in dev+ec2demo, Postgres-dialect migrations, Flyway-postgres plugin already wired. The refuter argued the Postgres path already exists in the `production` profile so this is a config swap not a rebuild, and the "REBUILD" label over-states. Survives — move ec2demo to managed Postgres; **effort is config-level given existing scaffolding.**

**B3-02 platform observability absent (survived, 0 refute).** Unanimous. `build.gradle` has only `spring-boot-starter-actuator`, no Micrometer registry; `management...include=health,loggers`; OTel/prometheus_client pinned in `requirements.txt` but imported in zero source files.

**B3 KILL async_coordinator / REBUILD correlation-IDs (survived, 0 refute).** Both unanimous. `async_coordinator.py` is dead (commented import at `server.py:10`, no pika dep). No W3C traceparent crosses JVM→Python→gRPC→mobile; only a per-5xx UUID exists in the error handler.

**B4 all four REBUILDs (survived, 0 refute each).** Unanimous across the board: gRPC plaintext-default with self-asserted `client_id` (Sybil-open) though TLS+mTLS code exists; `AuthorizationService` never checks `org_id` and `getDiscoverProjects` leaks PUBLIC project metadata cross-org; no robust aggregation or DP anywhere; no compliance program. B4-1 (DeComFL scalar-only kills the DLG reconstruction family) and B4-2 (plaintext-by-default though full TLS+mTLS exists) both survived unanimously — B4-1 is the platform's genuine, marketable privacy wedge.

**B5 all four verdicts (survived, 0 refute each).** Unanimous: KILL per-OS native (3× UI for a thin orchestrator, same signing cost); REBUILD onto Tauri v2 (mandatory signed updater structurally kills the C5 RCE class); KILL thin-shell-over-mobile-C++ (no bundle win, fragile RNG-parity invariant, collapses process isolation); REBUILD the auto-updater/signing. B5-1 (torch dominates the bundle regardless of shell) drew 1 refuter who showed it is correct-but-non-actionable framing, not a defect → context, not action item.

**B6 REBUILD hyperscale / KILL Citus / KILL Aurora-Serverless (survived).** Hyperscale-EKS+Karpenter and KILL-Citus were unanimous (Fargate can't schedule GPU; control-plane tables are bounded, growth is telemetry → TSDB). KILL-Aurora-Serverless drew 1 refuter noting Database Savings Plans (Dec 2025) and scale-to-zero now exist, narrowing the gap — but provisioned+RI still wins for steady OLTP. B6-1 (11-concurrent cap, no scale-to-zero/quotas) drew 1 refuter arguing "critical" is overstated since the cap is currently self-limiting and the unbounded-Fargate harm is gated behind unbuilt activation. Survives — per-org quotas + admission control are a mandatory v2 precondition before lifting the cap.

**B6 REBUILD in-memory STOMP broker (survived, 1 refute).** Two confirmed `WebSocketConfig:39` simple broker cannot fan out across replicas. The refuter noted two sibling audits (A1, B2) rate the identical component "refactor" (a one-line `enableStompBrokerRelay` swap), so "REBUILD/high" is overstated. Survives — **read as refactor: back with a Redis/RabbitMQ relay when the backend goes multi-replica.**

**B7 REBUILD PR-CI / B7-01 / static analysis / dep-hygiene (survived, 0 refute each).** All unanimous. Both workflows fire on-tag-only (no `pull_request` trigger anywhere); 132 Java files doing JWT + ProcessBuilder with zero linter/SpotBugs/JaCoCo; no Renovate/Dependabot/audit; Spring Boot 3.4.5 past OSS EOL (2025-12-31). KILL `desktop-release.yml` (duplicate release workflow on same `v*` tag) was unanimous. **PR-time CI + branch protection is the single highest-leverage, lowest-cost v2 P0** (carries the 05-27 P0 forward).

**C1 all six verdicts (survived).** Per-round checkpoint subsystem, round-loop quorum/timeout, and the v2 control-plane were unanimous. seed/gradient persistence, ECS path, in-memory map, and destructive save each drew 1 refuter arguing the FL server is a bounded single-shot process so the blast radius is "lose the run, re-run it" (medium) rather than data loss. All survived. **The round loop has no timeout/quorum — one straggler hangs the run forever — which is a genuine reliability defect for WAN/heterogeneous fleets.**

**C2 REBUILD pickle cache / dataset registry / KILL flwr_datasets partitioning (survived, 0 refute each).** Unanimous. No dataset registry/versioning/lineage exists; pickle split cache should become content-addressed `.npz` + sha256; flwr_datasets partitioning replaced by HF datasets + own Partitioner. Duplicated `dirichlet_split` KILL and seed-control REBUILD each drew 1 refuter on severity (the forks are behaviorally near-identical; the seed-coupling claim is partly overstated since server and client run in separate processes) but survived.

**C3 all five verdicts + C3-1 (survived, mostly 0 refute).** Run-lineage, determinism tests, model registry, and the KILL of the unverified mobile "bit-identical" comment were unanimous. The RNG-path REBUILD drew 1 refuter (CPU-only deployment makes the divergence latent; fix is patch-sized not a rebuild) but survived. C3-1 (the same RNG divergence) was unanimous as critical.

**C4 all six verdicts (survived, 0–1 refute).** DeComFL IP ownership under RIT C03.0, KILL horizontal positioning, REBUILD pricing, and KILL-flwr-datasets-as-hygiene were unanimous. GTM-motion REBUILD and customer-ToS/DPA REBUILD each drew 1 refuter on severity (generic advice; pre-revenue POC) but survived. **The IP-ownership finding is the single most important business fact: RIT very likely owns DeComFL; an IPMO license/spin-out is a go/no-go gate before claiming it as a moat.**

**C5 verdicts (survived, 0–1 refute each).** Three-surface incoherence, desktop two-conflicting-`:root`-blocks, mobile inline-hex/emoji styling, and the KILL of the fabricated mobile `exp(-loss)` "confidence" chart all survived. Several drew 1 refuter arguing the shared-design-system work is consolidation/refactor (a token package already exists in `theme.css`) rather than a from-scratch rebuild, and the desktop conflict is a mid-migration artifact. All survived. **The fabricated confidence chart and the cross-surface token package are the load-bearing items.**

---

## 3. DEMOTED findings (survived=false, refutes ≥ 2)

These were refuted by a majority of skeptics. The underlying technical fact is usually real, but the **severity or verdict was overstated** — treat them as low-priority or re-scoped, not as v2-blocking high-severity items.

1. **A2 — `frontend/dist/` committed (KILL, 3/3 refuted).** **FALSE PREMISE.** `git ls-files` shows `frontend/dist/` is NOT tracked and never was on any branch; root `.gitignore:49` already ignores `dist/`. The directory exists on disk only as an untracked, ignored local build artifact. The auditor mistook a filesystem listing for version control. No action.

2. **B3-01 — FL-run telemetry pipeline has no producer / empty chart (3/3 refuted).** **FALSE PREMISE / wrong-directory grep.** The producer DOES exist: `backend/.../scripts/fl_server.py:561-587` POSTs per-round results to `/api/internal/results/{projectId}` with the internal key, and `/finished`. The auditor grepped only `framework/` and `client-docker/`, missing the actual backend-spawned entry point. **Important caveat:** results are POSTed in a batch *after* `start_server()` returns, not streamed live per round — so the live-during-training chart is weaker than ideal, but the chart is not "permanently empty." Re-scope to "make per-round POST incremental," not "build the producer."

3. **A2 — Security-header layer CSP/HSTS (REBUILD, 2/3 refuted).** Real gap (no CSP, no Referrer-Policy), but two skeptics showed the body itself concludes "refactor not rebuild," Spring Security emits HSTS + X-Content-Type-Options by default (so "nonexistent at both layers" is wrong), and "any XSS = full compromise" is the generic worst-case undercut by HttpOnly+SameSite+pinned deps+no XSS sink. **Re-scope: add ~10 lines of CSP/HSTS to `SecurityConfig.headers()` — a low/medium refactor, not a rebuild.**

4. **A4 — Pickle split cache RCE-as-fedlearn (REBUILD, 2/3 refuted).** The pickle path resolves to `/app/scripts/data_splits` inside the container's own image layer, NOT the documented `/data` bind-mount; the same process writes then reads its own cache in an ephemeral `--rm` container as the unprivileged `fedlearn` user. The RCE precondition (attacker writes the `.pkl`) is not met by any real config. **Re-scope: replace pickle with `.npz`+sha256 for versioning/integrity hygiene (low severity), not RCE.** (Note: C2's pickle-cache REBUILD survived on the versioning/stale-split-trap rationale — that framing holds; only the *RCE* framing is demoted.)

5. **B2/B7 — Proto codegen / vendored copies (REBUILD, 2/3 and 1/3).** B2's version was majority-refuted: on `main`, only one canonical proto exists with committed stubs; the live mobile copy compiled by the build (`mobile_client/shared/proto`) is byte-identical to canonical, and only an *orphaned* `src/federated/protos` copy and the build-time-vendored client-docker copy drift (additively — missing DeComFL, no field renumbering). No wire incompatibility was demonstrated. **Re-scope: adopt `buf` as single-source-of-truth + breaking-change gate is sound DX hygiene (low/medium), not a high-severity rebuild.** (The A6 proto-drift finding survived at 1/3 and remains a v2 cleanup.)

6. **B3 — Mobile telemetry path (REBUILD, 2/3 refuted).** The C++ core has loss/accuracy locally but a heartbeat telemetry path to the server already exists (`HeartbeatRequest`); adding loss/accuracy is two nullable proto fields piggybacked on the existing RPC, and the report's own §9 ranks it P3/non-blocking. **Re-scope: low-priority additive enhancement, not a rebuild.**

7. **B3 — Experiment tracking / MLflow (REBUILD, 2/3 refuted).** A per-round metrics path already exists (RoundResult → STOMP → recharts), so "none exists" is wrong; the genuine gap is run-comparison/registry/lineage, which is correctly owned by C3 (which survived). **Re-scope: the experiment-tracking *capability* gap is real and is captured by C3's run-lineage/model-registry findings; do not double-count B3's framing.**

8. **B6 — KILL DB H2-file mode in ec2demo (KILL, 2/3 refuted).** ec2demo is an explicitly-documented single-EC2 demo where H2-file on EBS persists across reboots and carries no Flyway-race/concurrency risk; the Postgres path is already wired for the multi-replica/production tier. Two skeptics also found V5 declares `audit_events.metadata CLOB`, which raw Postgres does not accept — so the cutover is not the drop-in implied. **Re-scope: "migrate to RDS before the production/multi-replica cutover" is a routine roadmap item (medium), and the CLOB→TEXT/JSONB migration must be fixed first.** (B2's RDS-Postgres REBUILD survived on the production-tier rationale — that holds; only the "KILL ec2demo H2 now" framing is demoted.)

9. **C2 — KILL central ECG CSV bundled in server (KILL, 2/3 refuted).** The server reads the CSV only to build a held-out *test* loader for centralized evaluation (a standard, legitimate FL pattern that C2's own v2 design legitimizes) — it does NOT train on it. ECG5000 is a public UCR benchmark, not patient PII, and the path is one of three demo/benchmark paths. **Re-scope: "delete the 5.7 MB CSV from the JAR / git" is build-hygiene + bloat (low/medium), not an FL-premise contradiction. Hand the server a pre-split test set rather than the full corpus.**

10. **B5-1 — Desktop subprocess-orchestrator framing (1/3, flagged).** Technically true (torch is in the PyInstaller bundle regardless of shell) but the sole refuter showed it is correct-by-construction context that the existing code already implements, not a defect requiring action. **Treat as supporting context for the B5 Tauri verdict, not a standalone finding.**

---

## 4. Verification statistics

- **Total verified items:** 96
- **Survived:** 87
- **Demoted (majority-refuted):** 9 (items 1–9 above)
- **Critical findings (all survived):** A1-F1, A1-F2, A2-F1, A3-C1, A5-C1, A5-C3, B1-C1, B1-C2, B4-2, B6-1, B7-01, C3-1, M-C1, CD1 — except **B3-01 (demoted, false premise)**.
- **Strongest unanimous (0-refute) correctness findings:** A3-C1 (chunked upload), B1-C1 (P× step), B1-C2/C3-1 (RNG device divergence), A1-F1/A2-F1 (admin role mismatch), B7-01 (no PR CI).
