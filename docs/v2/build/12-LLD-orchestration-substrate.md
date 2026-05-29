# 12 — Low-Level Design (LLD): FL Orchestration Substrate

**Document type:** Production build specification — Low-Level Design (LLD) for one unit.
**Unit:** the FL (Federated Learning) **orchestration substrate** — the rebuild of v1's scaling-cliff `ProcessBuilder` model.
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters, e.g. Qwen/Llama 32B on an Apple M4 Max). Every interface, signature, table, environment variable, command, and failure path below is **pre-decided**. Implement the bodies; do not redesign the contracts. Where a body is genuinely tricky (the reconciler loop, lease acquisition, quota admission, round deadline/quorum) real code or precise pseudocode is given — copy its structure.
**Status:** build-authoritative for v2 (version 2). Conforms to and never contradicts the four foundation docs: `01-ARCHITECTURE-HLD.md`, `02-TECH-STACK.md`, `03-DATA-MODEL.md`, `04-API-CONTRACTS.md`.
**Date authored:** 2026-05-29.
**Source of truth:** the v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md`, plus depth reports `A1-backend.md`, `B2-tech-stack.md` (captured in `02-TECH-STACK.md §18`), `B6-scale-cost.md`, `C1-reliability-sre.md`. Every existing-code claim cites `path:line`; every external/market claim cites a source Uniform Resource Locator (URL).

---

## 0. How to read this document (and the abbreviation key)

The first time an acronym appears it is written in full followed by the short form in parentheses; thereafter the short form is used. The complete glossary is in §15.

**Acronyms you need immediately:** FL (Federated Learning), LLD (Low-Level Design), API (Application Programming Interface), JVM (Java Virtual Machine), gRPC (Google Remote Procedure Call), k8s (Kubernetes), ECS (Elastic Container Service), ARN (Amazon Resource Name), UUID (Universally Unique Identifier), DB (Database), JPA (Jakarta Persistence API), STOMP (Simple Text Oriented Messaging Protocol), S3 (Simple Storage Service), DeComFL (the platform's zeroth-order FL strategy — note: `04-API-CONTRACTS.md §0` flags that v1 mis-expanded it as "Decomposed"; the paper-correct expansion is "Dimension-Free Communication Federated Learning"), FedAvg (Federated Averaging), RNG (Random Number Generator), HMAC (Hash-based Message Authentication Code).

**Normative keywords:** "MUST" = a build constraint no code may violate; "SHOULD" = a strong default; "MUST NOT" = forbidden. These mirror the foundation docs.

**This unit's relationship to the foundation docs:** the HLD (`01`) §4.2 Flow B and §7 D1/D2/D4/D5/D7 describe this substrate at high level; the data model (`03`) §5.2 defines the `fl_runs`/`round_results`/`model_artifacts` tables; the API contracts (`04`) §4, §5, §10, §11, §13, §14 define the REST/gRPC/STOMP/token/trace surfaces this unit consumes and exposes. **This LLD is the only document that specifies the Java `FlServerLauncher` interface, its three implementations, the reconciler loop, the round deadline/quorum logic, the per-org quota/admission code, scale-to-zero, and checkpoint/resume wiring.**

---

## 1. Purpose & single responsibility

The FL orchestration substrate is the control-plane subsystem that **launches, supervises, and tears down one long-running FL server per run, keyed on `run_id`, across three pluggable executor backends (Kubernetes Jobs, AWS ECS RunTask, and a dev-only local process), using a durable Postgres `fl_runs` lease as the single source of truth so the JVM is a stateless supervisor.** It owns the run lifecycle state machine, the reconciler loop (boot-time + periodic) that reconciles real executor state to the DB, the round-deadline + minimum-quorum enforcement contract, per-org concurrency quotas + admission control, scale-to-zero, and the wiring of per-round checkpoint/resume through the artifact store. It does **not** implement the FL algorithm itself (that is the Python FL framework, FL-framework LLD `11-` — here referenced as the "FL server"), nor the identity/authorization layer (control-plane LLD `10-`), nor the artifact-store internals (data-and-artifact LLD `17-`).

---

## 2. Position in the system — dependencies & interfaces

### 2.1 What this unit depends on (CONSUMES)

| Dependency | What this unit needs from it | Contract reference |
|---|---|---|
| Postgres `fl_runs` table + `round_results` + `model_artifacts` | The durable lease, run state, executor binding, lineage pins, per-round metrics, content-addressed checkpoints | `03-DATA-MODEL.md §5.2` (`V7__fl_runs_and_artifacts.sql`) |
| Authorization layer (control-plane LLD 10) | Resolve caller `org_id`/role; `requireParticipant(...)`; org-scoped query filters | `04-API-CONTRACTS.md §1.1` role enums, §11 topic authz |
| Per-run scoped result token minting/validation | Mint `flrun_<...>` at launch; validate on `/api/internal/runs/{runId}/**` | `04-API-CONTRACTS.md §13` |
| Artifact store (data-and-artifact LLD 17) | Pre-signed S3/MinIO URLs; content-addressed (sha256) checkpoint objects | `04-API-CONTRACTS.md §9`; `03-DATA-MODEL.md §5.2 model_artifacts` |
| The FL server (Python framework) | Reads `FEDLEARN_*` env, exposes gRPC `GetServerStatus`, enforces round deadline/quorum, POSTs incremental `RoundResultDto`, writes checkpoints | `04-API-CONTRACTS.md §10` (gRPC `fedlearn.v2`), §5 (internal callbacks) |
| STOMP relay / WebSocketService | Broadcast run lifecycle + log events to `/topic/status/{projectId}`, `/topic/runs/{projectId}`, `/topic/logs/{projectId}` | `04-API-CONTRACTS.md §11` |
| OpenTelemetry (OTel) propagator | Serialize the launch span's W3C `traceparent` into the executor env | `04-API-CONTRACTS.md §14` |
| Kubernetes Java client / AWS ECS SDK | Submit/stop/describe k8s Jobs and ECS tasks | `02-TECH-STACK.md §18.1` |

### 2.2 What depends on this unit (EXPOSES)

| Consumer | What it calls / receives | Contract reference |
|---|---|---|
| Runs REST controller (control-plane LLD 10) | `FlRunService.startRun(...)`, `.stopRun(...)`, `.getStatus(...)` behind `POST /api/projects/{projectId}/runs`, `POST /api/runs/{runId}/stop`, `GET /api/runs/{runId}/status` | `04-API-CONTRACTS.md §4` |
| Internal callbacks controller (control-plane LLD 10) | `FlRunService.recordRoundResult(...)`, `.markFinished(...)`, `.recordCheckpoint(...)`, `.recordStatus(...)` behind `/api/internal/runs/{runId}/**` | `04-API-CONTRACTS.md §5` |
| The FL server (launched executor) | Receives `FEDLEARN_RUN_ID`, `FEDLEARN_RUN_TOKEN`, `FEDLEARN_BACKEND_URL`, `FEDLEARN_PROJECT_ID`, `TRACEPARENT` env vars + run config flags | `04-API-CONTRACTS.md §13` env table, §14 |
| Frontend dashboard (frontend LLD 13) | Live run state via STOMP `/topic/status/{projectId}` + `/topic/runs/{projectId}` | `04-API-CONTRACTS.md §11` |

### 2.3 Position diagram (where this unit sits)

```
   Runs REST controller (LLD 10)                 Internal callbacks controller (LLD 10)
        │  startRun/stopRun/getStatus                   │  recordRoundResult/markFinished/...
        ▼                                               ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                THIS UNIT — FL ORCHESTRATION SUBSTRATE                        │
 │                                                                             │
 │  FlRunService ── admission/quota ──▶ writes fl_runs lease (PENDING)         │
 │       │                                                                     │
 │       └──▶ FlServerLauncher.launch(spec) ─┐                                 │
 │                                           │  KubernetesJobLauncher (primary)│
 │  Reconciler (@Scheduled + boot) ──────────┤  EcsRunTaskLauncher  (secondary)│
 │       │ poll()/describe()/stop()          │  LocalProcessLauncher (dev only)│
 │       └──▶ reconcile executor → fl_runs → STOMP                             │
 └───────────────────────────────────────────────────────────────────────────┘
        │  fl_runs lease (source of truth)        │  launch executor (env + flags)
        ▼                                         ▼
   Postgres (LLD 17)                       k8s Job / ECS task / local process
                                                  = one long-running FL server (run_id)
                                                  │ reads/writes S3 (LLD 17)
                                                  │ gRPC GetServerStatus (readiness)
                                                  │ POST /api/internal/runs/{runId}/** (incremental)
                                                  ▼
                                            FL clients (desktop/jetson/mobile)
```

---

## 3. Tech stack for this unit (pinned, from `02-TECH-STACK.md`)

| Technology | Pinned version | One-line reasoning |
|---|---|---|
| Java | Temurin JDK 21 `21.0.7+6` (verify-before-use) | LTS (Long-Term Support); virtual threads (Project Loom) for the reconciler/heartbeat fan-in; record patterns for the enums (`02 §1.1`). |
| Spring Boot | `3.5.14` | Salvaged control plane off EOL (End-Of-Life) 3.4.5; `@Scheduled`, `@Transactional`, `ApplicationReadyEvent` are all first-party (`02 §2.1`). |
| Gradle (committed wrapper) | `9.5.1` | Project invariant — never switch to Maven (`02 §2.3`). |
| Kubernetes (EKS) | `1.36.x` | **Primary, production** executor backend; one FL server = one k8s `Job` with cgroup isolation, `activeDeadlineSeconds`, `ttlSecondsAfterFinished` (`02 §18.1`). |
| `io.kubernetes:client-java` | verify-before-use (latest) | The mature JVM Kubernetes client used by `KubernetesJobLauncher` (`02 §18.1`). |
| AWS ECS | managed; SDK `software.amazon.awssdk:ecs` `2.25.11` (verify-before-use) | **Secondary** backend; salvaged-and-completed from v1's fire-and-forget `startEcsFargateServer` (`02 §18.1`, `A1-F2`). |
| PostgreSQL (RDS) | `17.10` | The `fl_runs` lease lives here; partial-unique-index closes the one-active-run race (`03 §4.1`, `02 §5.1`). |
| Flyway | Boot 3.5 BOM-managed | Owns the schema; the substrate's tables are `V7` (`03 §5.2`). |
| Micrometer (+ otel bridge) | Boot 3.5 BOM | Emits substrate metrics (runs launched/failed, reconcile latency) on the internal management port (`02 §20`). |
| OTel propagator | Boot 3.5 BOM (`micrometer-tracing-bridge-otel`) | Serializes `traceparent` into the executor env at launch (`04 §14`). |

**MUST NOT** introduce any other orchestration technology. Bazel is rejected; Nx is deferred (`02 §23`). No Flower/`flwr` anywhere (`02 §25 rule 1`).

---

## 4. Module / file structure

All paths are under `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/`. The v1 package `flower/` (the legacy "Flower" name — there is no Flower dependency) is **replaced** by a new `orchestration/` package. The v1 `FlowerServerManager.java` is **deleted**; its lifecycle responsibilities move into the files below.

```
orchestration/
  FlRunService.java                 # Public service facade: startRun/stopRun/getStatus/recordRoundResult/markFinished/recordCheckpoint/recordStatus. The only seam the controllers call.
  FlRunSpec.java                    # Immutable launch spec (Java record): runId, projectId, orgId, strategy, config, env, image, resources, traceparent, runToken.
  FlRunStateMachine.java            # Validates legal status transitions (PENDING→STARTING→RUNNING→…); single source of the transition table.
  quota/
    OrgQuotaService.java            # Per-org concurrency quota lookup + atomic admission check (counts active runs under a row lock).
    OrgQuota.java                   # Record: orgId, maxConcurrentRuns, maxRoundsPerRun (defaults + per-org overrides).
  launcher/
    FlServerLauncher.java           # THE interface (§5.1). launch/stop/describe + supports(profile).
    LaunchResult.java               # Record returned by launch(): executorRef, grpcEndpoint(nullable), launcher enum.
    ExecutorState.java              # Record returned by describe(): phase enum (PENDING/RUNNING/SUCCEEDED/FAILED/MISSING), exitReason(nullable).
    LauncherBackend.java            # Enum: K8S_JOB, ECS_RUN_TASK, LOCAL_PROCESS (mirrors fl_runs.launcher CHECK).
    KubernetesJobLauncher.java      # PRIMARY. Submits a k8s Job per run; stop=delete Job; describe=read Job/Pod status.
    EcsRunTaskLauncher.java         # SECONDARY. EcsClient.runTask; stop=StopTask; describe=DescribeTasks. Singleton EcsClient bean.
    LocalProcessLauncher.java       # DEV ONLY. ProcessBuilder spawn behind the interface; gated to the `dev` profile, rejected elsewhere.
  reconcile/
    FlRunReconciler.java            # @Scheduled periodic + @EventListener(ApplicationReadyEvent) boot reconciliation; the supervisor loop (§6.2).
    LeaseManager.java               # Atomic lease acquire/renew/release via single-row UPDATEs (§6.3); supervisor instance id.
  token/
    RunTokenService.java            # Mint + validate the per-run scoped token (HMAC-SHA256), the §13 contract.
  config/
    OrchestrationProperties.java    # @ConfigurationProperties("fedlearn.orchestration"): lease TTL, reconcile interval, readiness timeout, quotas, k8s/ecs config.
  repository/
    FlRunRepository.java            # Spring Data JPA repo for fl_runs + the native lease/quota UPDATE queries.
    RoundResultRepository.java      # Spring Data JPA repo for round_results (idempotent per-round upsert).
    ModelArtifactRepository.java    # Spring Data JPA repo for model_artifacts.
  entity/
    FlRun.java                      # @Entity mapping fl_runs (the aggregate root). UUID id == run_id.
    RoundResult.java                # @Entity mapping round_results.
    ModelArtifact.java              # @Entity mapping model_artifacts.
```

External-but-owned support files:
```
src/main/resources/
  k8s/fl-server-job.yaml.mustache   # Job template KubernetesJobLauncher fills per run (image, resources, env, deadline, ttl).
  scripts/run_fl_server.sh          # The wrapper LocalProcessLauncher exec's in dev (carried from v1, hardened).
```

---

## 5. Key interfaces & type signatures (FULL — implement the bodies)

### 5.1 `FlServerLauncher` — the abstraction (THE interface, three backends)

```java
package com.federated.fl_platform_api.orchestration.launcher;

import java.util.Optional;

/**
 * One FL server per run, launched on a pluggable executor backend.
 * The control plane "submits a run to the substrate" — it does NOT fork a Python
 * process and grab a port (B2-tech-stack: the v1 anti-pattern). All three backends
 * sit behind this interface so FlRunService and FlRunReconciler are backend-agnostic.
 */
public interface FlServerLauncher {

    /** Which backend this implementation is. Matches fl_runs.launcher CHECK values. */
    LauncherBackend backend();

    /**
     * Launch one long-running FL server for spec.runId().
     * MUST be idempotent on executorRef: if called twice for the same runId+backend,
     * it MUST NOT create a second executor (guard on a deterministic executor name,
     * e.g. "fl-run-{runId}"). Returns the durable handle to persist in fl_runs.executor_ref.
     * @throws LauncherException on a terminal admission/submission error (caller marks run FAILED).
     */
    LaunchResult launch(FlRunSpec spec) throws LauncherException;

    /**
     * Request termination of the executor identified by executorRef.
     * MUST be idempotent: stopping an already-gone executor returns normally (no throw).
     * Issues a graceful stop (SIGTERM / k8s delete with grace) before force.
     */
    void stop(String executorRef);

    /**
     * Observe the real executor state. The reconciler's only window into reality.
     * MUST return ExecutorState.MISSING (not throw) when the executor is not found,
     * so the reconciler can decide orphan vs terminal based on the lease.
     */
    ExecutorState describe(String executorRef);

    /**
     * Whether this backend may be used under the given Spring profile.
     * LocalProcessLauncher.supportsProfile returns true ONLY for "dev".
     */
    boolean supportsProfile(String activeProfile);
}
```

```java
// LaunchResult.java
public record LaunchResult(
    String executorRef,            // k8s Job name | ECS task ARN | "pid:<n>" for local
    Optional<String> grpcEndpoint, // host:port if known at launch (k8s Service DNS); empty until reconciler learns it
    LauncherBackend backend
) {}

// ExecutorState.java
public record ExecutorState(Phase phase, Optional<String> exitReason, Optional<String> grpcEndpoint) {
    public enum Phase { PENDING, RUNNING, SUCCEEDED, FAILED, MISSING }
}

// LauncherBackend.java  — MUST equal the fl_runs.launcher CHECK domain
public enum LauncherBackend { K8S_JOB, ECS_RUN_TASK, LOCAL_PROCESS }

// LauncherException.java
public class LauncherException extends RuntimeException {
    public LauncherException(String message, Throwable cause) { super(message, cause); }
}
```

### 5.2 `FlRunSpec` — the immutable launch spec

```java
package com.federated.fl_platform_api.orchestration;

import java.util.Map;
import java.util.UUID;

/**
 * Everything an executor needs to run, computed by FlRunService at admission time.
 * config is the strategy-specific JSON persisted in fl_runs.config (03 §5.2).
 * env carries the locked FEDLEARN_* + TRACEPARENT vars (04 §13 env table, §14).
 */
public record FlRunSpec(
    UUID runId,
    UUID projectId,
    UUID orgId,
    Strategy strategy,            // FedAvg | DeComFL
    int numRounds,
    int minClients,               // the minimum quorum floor (04 §4.1)
    int roundDeadlineSeconds,     // per-round wall-clock deadline (no infinite hang)
    long seed,
    String containerImage,        // resolved per strategy/model band
    ResourceRequest resources,    // cpu/mem/gpu requests for the executor
    Map<String,String> config,    // -> fl_runs.config JSONB (K,P,eta,mu,dp,...)
    Map<String,String> env        // FEDLEARN_RUN_ID, FEDLEARN_RUN_TOKEN, FEDLEARN_BACKEND_URL,
                                  // FEDLEARN_PROJECT_ID, TRACEPARENT (04 §13/§14)
) {
    public enum Strategy { FedAvg, DeComFL }
    public record ResourceRequest(String cpu, String memory, int gpuCount) {}
}
```

### 5.3 `FlRunService` — the public facade (what the controllers call)

```java
package com.federated.fl_platform_api.orchestration;

import java.util.UUID;

public interface FlRunService {

    /** POST /api/projects/{projectId}/runs (04 §4). Admission + quota + lease(PENDING) + launch handoff.
     *  Returns 202-shaped RunDto. Throws domain exceptions mapping to the §12 error codes. */
    RunDto startRun(UUID projectId, StartRunRequest req, AuthContext caller);

    /** POST /api/runs/{runId}/stop (04 §4). Transition to STOPPING + launcher.stop(). */
    RunDto stopRun(UUID runId, AuthContext caller);

    /** GET /api/runs/{runId}/status (04 §4). Lightweight poll-friendly view. */
    RunStatusDto getStatus(UUID runId, AuthContext caller);

    // ---- internal callbacks (04 §5); caller is a validated RunContext from the per-run token (§13) ----

    /** POST /api/internal/runs/{runId}/results — incremental per-round (idempotent on (runId,round)). */
    void recordRoundResult(UUID runId, RoundResultDto dto, RunContext token);

    /** POST /api/internal/runs/{runId}/finished — terminal: SUCCEEDED|FAILED + final artifact. */
    void markFinished(UUID runId, RunFinishedDto dto, RunContext token);

    /** POST /api/internal/runs/{runId}/checkpoint — record a per-round content-addressed checkpoint pointer. */
    void recordCheckpoint(UUID runId, CheckpointReportDto dto, RunContext token);

    /** POST /api/internal/runs/{runId}/status — executor self-report between rounds; first report carries grpcEndpoint. */
    void recordStatus(UUID runId, RunStatusReportDto dto, RunContext token);
}
```

> `RunDto`, `RunStatusDto`, `StartRunRequest`, `RoundResultDto`, `RunFinishedDto`, `CheckpointReportDto`, `RunStatusReportDto`, `AuthContext`, `RunContext` are exactly the shapes in `04-API-CONTRACTS.md §4.1/§4.4/§5.1/§13`. Do not invent new field names — copy them.

### 5.4 `LeaseManager` — atomic lease primitives

```java
package com.federated.fl_platform_api.orchestration.reconcile;

import java.time.Duration;
import java.util.List;
import java.util.UUID;

public interface LeaseManager {

    /** A stable id for THIS supervisor instance (pod name / host id). Read once at startup. */
    String supervisorId();

    /** Atomically claim the run's lease (optimistic, single-row UPDATE). Returns true iff this instance won. */
    boolean acquire(UUID runId);

    /** Extend lease_expires_at = now + ttl, ONLY if this instance still owns it. Returns false if lost. */
    boolean renew(UUID runId, Duration ttl);

    /** Release ownership (lease_owner=NULL, lease_expires_at=NULL) — used on graceful terminal transitions. */
    void release(UUID runId);

    /** Non-terminal runs whose lease has expired (lease_expires_at < now) — the reconciler's reclaim set. */
    List<FlRunLeaseRow> findExpiredLeases(int limit);

    /** PENDING runs eligible for launch when quota allows — the reconciler's admit set. */
    List<FlRunLeaseRow> findPendingRuns(int limit);

    record FlRunLeaseRow(UUID runId, UUID projectId, UUID orgId, String status,
                         String launcher, String executorRef) {}
}
```

### 5.5 `OrgQuotaService` — admission control

```java
package com.federated.fl_platform_api.orchestration.quota;

import java.util.UUID;

public interface OrgQuotaService {
    /** Resolve effective quota (per-org override, else platform default). */
    OrgQuota quotaFor(UUID orgId);

    /**
     * Atomically check the org has headroom and reserve a slot by inserting the PENDING fl_runs row
     * in the SAME transaction. MUST run inside FlRunService.startRun's @Transactional boundary and
     * MUST take a row lock on the count so two concurrent starts cannot both pass (see §6.4).
     * @return true if admitted; false if at capacity (caller -> 409 ORG_QUOTA_EXCEEDED).
     */
    boolean tryReserveSlot(UUID orgId);
}
```

```java
// OrgQuota.java
public record OrgQuota(UUID orgId, int maxConcurrentRuns, int maxRoundsPerRun) {}
```

### 5.6 DB row types (mapping `03-DATA-MODEL.md §5.2`)

`FlRun` entity — the aggregate root; UUID `id` **is** the `run_id`. Columns map 1:1 to `fl_runs` (status, lease_owner, lease_expires_at, launcher, executor_ref, grpc_endpoint, round_idx, strategy, config JSONB, dataset_version_id, partition_recipe_id, initial_model_artifact_id, final_model_artifact_id, mlflow_run_id, requested_by, created_at, updated_at, started_at, ended_at). `status` ∈ `{PENDING,STARTING,RUNNING,SUCCEEDED,FAILED,STOPPED}` (note: the run state machine in `04 §4.3` exposes `STOPPING` as a transient API-facing state mapped onto DB `RUNNING`+stop-requested; see §6.1). `launcher` ∈ `{K8S_JOB,ECS_RUN_TASK,LOCAL_PROCESS}`.

`RoundResult` entity — maps `round_results` (fl_run_id, round_idx, loss, accuracy, val_loss, val_accuracy, num_clients_reported, uplink_bytes, downlink_bytes, scalars_transmitted, gpu_utilization, round_started_at, round_ended_at). **Unique `(fl_run_id, round_idx)`** makes the incremental per-round POST idempotent.

`ModelArtifact` entity — maps `model_artifacts` (org_id, sha256, storage_uri, size_bytes, kind ∈ `{INITIAL,CHECKPOINT,FINAL}`, fl_run_id, round_idx). **Unique `(org_id, sha256)`** dedupes identical bytes per tenant.

---

## 6. Core algorithms & flows (real code / precise pseudocode)

### 6.1 Run state machine (the transition table — implement exactly)

DB `fl_runs.status` is the durable truth. The API-facing `STOPPING` (`04 §4.3`) is a derived view: a run is reported `STOPPING` when DB `status='RUNNING'` and a stop has been requested but not yet confirmed by the reconciler; the DB itself moves `RUNNING → STOPPED` when the reconciler confirms termination.

```
 PENDING ──(launcher.launch ok)──▶ STARTING ──(reconciler: gRPC ready & minClients seen)──▶ RUNNING
    │                                  │                                                       │
    │(admission/launch error)          │(executor never becomes ready before timeout)          │(all rounds done)
    ▼                                  ▼                                                       ▼
  FAILED                             FAILED                                                SUCCEEDED
                                                                                              │
 RUNNING ──(/stop -> launcher.stop, reconciler confirms gone)──▶ STOPPED                      │
 RUNNING/STARTING ──(executor exits non-zero | deadline-kill | lease expired & MISSING)──▶ FAILED
```

`FlRunStateMachine.assertTransition(from, to)` MUST reject any edge not in the table above and throw `IllegalRunTransitionException` (maps to `409 RUN_NOT_STOPPABLE` / `409 RUN_TERMINAL` per `04 §12`). Terminal states `{SUCCEEDED, FAILED, STOPPED}` accept no outgoing edge.

### 6.2 The reconciler loop (REAL pseudocode — the heart of "stateless supervisor")

The JVM holds **no** in-memory `Process`/`Job` map (this is the v1 fatal `ConcurrentHashMap<UUID,Process>` at `flower/FlowerServerManager.java:78`, lost on restart — `A1-F4`, `C1-F4`, risk **R9**). The reconciler is the only writer of `RUNNING/SUCCEEDED/STOPPED/FAILED` from executor polling.

```text
# Runs on BOTH:
#   @EventListener(ApplicationReadyEvent)   -> boot reconciliation (reaps orphans from a prior JVM)
#   @Scheduled(fixedDelayString = "${fedlearn.orchestration.reconcile-interval-ms:15000}")
#
# All reconciler work is per-run and idempotent. Concurrency between replicas is mediated by the lease.

function reconcileOnce():
    # ---- PHASE A: reclaim expired leases (a previous owner pod died) ----
    for row in leaseManager.findExpiredLeases(limit = BATCH):          # status in {STARTING,RUNNING}
        if not leaseManager.acquire(row.runId):                        # atomic UPDATE ... WHERE lease expired
            continue                                                   # another replica grabbed it; skip
        launcher = launcherFor(row.launcher)                           # K8S_JOB | ECS_RUN_TASK | LOCAL_PROCESS
        state    = launcher.describe(row.executorRef)                  # the ONLY window into reality
        switch state.phase:
            case RUNNING:
                leaseManager.renew(row.runId, LEASE_TTL)               # re-adopt: the executor is alive, keep supervising
                if state.grpcEndpoint.present and row.status == STARTING and quorumLikelyReady(row):
                    transition(row.runId, RUNNING, grpcEndpoint=state.grpcEndpoint)
            case SUCCEEDED:
                transition(row.runId, SUCCEEDED)                       # finalize lineage; release lease
                leaseManager.release(row.runId)
            case FAILED:
                transition(row.runId, FAILED, reason = state.exitReason)
                leaseManager.release(row.runId)
            case MISSING:
                # executor is gone AND the lease had expired -> orphan/crash
                if row.status == STARTING:
                    transition(row.runId, FAILED, reason = "executor vanished before ready")
                else:  # was RUNNING
                    markInterruptedAndMaybeResume(row)                 # see §6.6 (checkpoint/resume)
                leaseManager.release(row.runId)
            case PENDING:
                leaseManager.renew(row.runId, LEASE_TTL)               # still scheduling; keep waiting

    # ---- PHASE B: readiness promotion for runs THIS replica owns and that are STARTING ----
    for runId in ownedRuns(status = STARTING):
        endpoint = gRpcReadinessProbe(runId)                          # GetServerStatus until READY or timeout (§6.5)
        if endpoint != null:
            transition(runId, RUNNING, grpcEndpoint = endpoint)
            leaseManager.renew(runId, LEASE_TTL)
        elif startingFor(runId) > READINESS_TIMEOUT:
            launcher.stop(executorRefOf(runId))
            transition(runId, FAILED, reason = "readiness timeout")
            leaseManager.release(runId)

    # ---- PHASE C: round-progress deadline watchdog (defense in depth; the FL server enforces too) ----
    for runId in ownedRuns(status = RUNNING):
        if now() - lastRoundAdvance(runId) > (roundDeadlineSeconds(runId) + GRACE):
            # the FL server SHOULD have advanced or self-killed; if it has not, the server is wedged
            launcher.stop(executorRefOf(runId))
            transition(runId, FAILED, reason = "round deadline exceeded with no progress")
            leaseManager.release(runId)
        else:
            leaseManager.renew(runId, LEASE_TTL)                      # supervisor heartbeat keeps the lease alive

    # ---- PHASE D: admit PENDING runs if org quota allows (also drives reconciler-initiated launch) ----
    for row in leaseManager.findPendingRuns(limit = BATCH):
        if leaseManager.acquire(row.runId) and orgQuotaService.hasHeadroom(row.orgId):
            launchAndBindExecutor(row)                                 # same path as FlRunService.startRun handoff
```

**Why both boot + periodic:** boot reconciliation reaps orphans a crashed prior JVM left behind (the v1 split-brain — `C1-F4`); periodic reconciliation re-adopts runs whose owning pod died mid-flight and advances state. Together they make a redeploy, OOM (Out-Of-Memory) kill, or `SIGKILL` non-fatal to a run (`01 §4.2` reconciler block, risk **R9**).

### 6.3 Lease acquisition / renewal (REAL SQL — atomic, single-row)

The lease is two columns on `fl_runs`: `lease_owner VARCHAR(128)` and `lease_expires_at TIMESTAMPTZ` (`03 §5.2`). All lease mutations are single-row conditional UPDATEs — no application lock, no race.

```sql
-- acquire(runId): win the lease iff nobody owns it OR the owner's lease has expired.
UPDATE fl_runs
   SET lease_owner = :supervisorId,
       lease_expires_at = now() + (:ttlSeconds * interval '1 second'),
       updated_at = now()
 WHERE id = :runId
   AND status IN ('PENDING','STARTING','RUNNING')
   AND (lease_owner IS NULL OR lease_expires_at < now());
-- acquire() returns true iff rowCount == 1.

-- renew(runId): extend ONLY if this instance still owns it (lost-lease detection).
UPDATE fl_runs
   SET lease_expires_at = now() + (:ttlSeconds * interval '1 second'),
       updated_at = now()
 WHERE id = :runId
   AND lease_owner = :supervisorId;
-- renew() returns false (rowCount==0) if the lease was stolen; caller MUST stop touching the run.

-- release(runId)
UPDATE fl_runs SET lease_owner = NULL, lease_expires_at = NULL, updated_at = now()
 WHERE id = :runId AND lease_owner = :supervisorId;
```

The reclaim scan uses the partial index `idx_fl_runs_lease_active` (`03 §5.2`):
```sql
SELECT id, project_id, org_id, status, launcher, executor_ref
  FROM fl_runs
 WHERE status IN ('STARTING','RUNNING') AND lease_expires_at < now()
 ORDER BY lease_expires_at ASC LIMIT :batch
   FOR UPDATE SKIP LOCKED;   -- two replicas reconciling in parallel never fight over the same rows
```

**Lease TTL vs reconcile interval (locked defaults, §8):** `reconcile-interval = 15s`, `lease-ttl = 60s`. Rule: `lease-ttl MUST be >= 3 × reconcile-interval` so a single missed reconcile (GC pause, brief network blip) does not falsely expire a healthy owner's lease.

### 6.4 Admission control + quota + one-active-run (REAL code, the start path)

`FlRunService.startRun` is `@Transactional`. Two protections fire in one transaction: the **partial unique index** (one active run per project) and the **org quota** (concurrency cap). The first closes v1's check-then-act duplicate-spawn race declaratively (`A1-F4`, `C1-F3`); the second closes the unbounded-cost hole (`B6 §6.1`, risk **R10**).

```java
@Transactional
public RunDto startRun(UUID projectId, StartRunRequest req, AuthContext caller) {
    Project project = projects.requireOrgMember(projectId, caller);          // 403/404 from LLD 10
    if (req.launcher() == LOCAL_PROCESS && !activeProfileIsDev())
        throw new UnsupportedLauncherException();                            // 422 UNSUPPORTED_LAUNCHER
    UUID datasetVersionId = resolveDatasetVersion(req, project);             // 422 NO_DATASET_VERSION if absent
    validateHyperparameters(req);                                            // 400 VALIDATION_FAILED (04 §4.2)

    // (1) per-org concurrency admission — count active runs UNDER A LOCK, then reserve.
    OrgQuota quota = orgQuotaService.quotaFor(project.orgId());
    int active = flRunRepository.countActiveByOrgForUpdate(project.orgId()); // SELECT ... FOR UPDATE (see SQL below)
    if (active >= quota.maxConcurrentRuns())
        throw new OrgQuotaExceededException();                               // 409 ORG_QUOTA_EXCEEDED
    if (req.numRounds() > quota.maxRoundsPerRun())
        throw new ValidationException("numRounds", "exceeds org max");       // 400 VALIDATION_FAILED

    // (2) insert the PENDING lease row. The partial unique index enforces one-active-run-per-project:
    //     a concurrent start for the same project fails here -> 409 RUN_ALREADY_ACTIVE.
    FlRun run = FlRun.pending(projectId, project.orgId(), caller.userId(),
                              req.strategy(), req.numRounds(), req.minClients(),
                              req.roundDeadlineSeconds(), req.seed(), toConfigJson(req),
                              datasetVersionId, resolvePartitionRecipe(req, project),
                              launcherEnum(req.launcher()));
    try {
        run = flRunRepository.saveAndFlush(run);                             // flush NOW to surface the unique violation
    } catch (DataIntegrityViolationException e) {
        if (isPartialUniqueViolation(e)) throw new RunAlreadyActiveException(); // 409 RUN_ALREADY_ACTIVE
        throw e;
    }

    // (3) mint the per-run scoped token + build env (04 §13/§14) and the spec.
    String runToken = runTokenService.mint(run.id(), projectId, project.orgId());
    Map<String,String> env = buildExecutorEnv(run, runToken, currentTraceparent());
    FlRunSpec spec = buildSpec(run, env, datasetVersionId);

    // (4) handoff to the launcher. Done AFTER commit (see §9 "launch-after-commit") so a launch
    //     failure cannot leave a committed RUNNING row with no executor, and vice-versa.
    registerAfterCommit(() -> launchAndBind(run.id(), spec));

    return RunDto.from(run); // 202 Accepted, status STARTING is set by launchAndBind/reconciler
}
```

```sql
-- countActiveByOrgForUpdate(orgId): lock the org's active rows so two concurrent starts serialize.
SELECT count(*) FROM fl_runs
 WHERE org_id = :orgId AND status IN ('PENDING','STARTING','RUNNING')
 FOR UPDATE;
```

`launchAndBind` (called after commit) acquires the lease, sets `status=STARTING`, calls `launcher.launch(spec)`, persists `executor_ref`/`grpc_endpoint`, and renews the lease. If `launch` throws `LauncherException`, it transitions the run to `FAILED` with the message and releases the lease. **No 3-second sleep** — readiness is the reconciler's Phase B gRPC probe (§6.5), fixing v1's `process.waitFor(3, SECONDS)` at `flower/FlowerServerManager.java:214` that proved only "didn't crash in 3s," not "healthy" (`A1`, `C1-F3`).

### 6.5 Readiness probe (replaces the 3s sleep)

```text
function gRpcReadinessProbe(runId) -> grpcEndpoint | null:
    endpoint = flRunRepository.grpcEndpoint(runId)   # learned from launcher.describe() or /api/internal/.../status
    if endpoint == null: return null                 # executor hasn't reported its endpoint yet
    try:
        resp = FederatedLearningServiceStub(endpoint).GetServerStatus({run_id: runId})  # 04 §10.2
        if resp.server_state in {WAITING_FOR_CLIENTS, TRAINING}:
            return endpoint                          # booted AND serving
        return null
    except gRPC.UNAVAILABLE: return null             # not up yet; reconciler retries next tick
```

The FL server's first `POST /api/internal/runs/{runId}/status` (`04 §5.1 RunStatusReportDto`) carries `grpcEndpoint`, so `recordStatus` persists it onto `fl_runs.grpc_endpoint`; the probe then has a target. Under k8s the endpoint is the Job's `Service` DNS name and is known at launch.

### 6.6 Round deadline + minimum-quorum (the contract this unit ENFORCES via the FL server)

The round loop's deadline + quorum is enforced **inside the FL server** (the Python framework), but its parameters are owned here in `fl_runs.config` and surfaced over gRPC `GetServerStatusResponse.round_deadline_unix_ms` / `required_clients_for_round` (`04 §10.2`). This unit's responsibilities:

1. **Pass the parameters down.** `numRounds`, `minClients`, `roundDeadlineSeconds` go into `fl_runs.config` and the executor env/flags. They came from `StartRunRequest` (`04 §4.1`).
2. **Watchdog (reconciler Phase C).** If `now - last_round_advance > roundDeadlineSeconds + GRACE` and the server has not advanced or self-killed, the reconciler force-stops it and marks `FAILED` (a wedged server, not a slow straggler).
3. **The semantics the FL server MUST implement** (restated from `02 §18.2` / `04 §10.3` so this doc is self-contained — the framework LLD owns the body):

```text
# Inside the FL server's round loop. THIS is why "one straggler never hangs the run" (C1-F5, R9).
def run_round(run, clients, round_idx):
    deadline   = now() + run.round_deadline_seconds          # per-round wall-clock
    min_quorum = run.min_clients                             # the floor from StartRunRequest
    received   = []
    while now() < deadline and len(received) < len(clients):
        r = wait_for_next_result(timeout = deadline - now())
        if r: received.append(r)
    if len(received) < min_quorum:
        report_status(run_id, FAILED, reason="quorum_not_met")  # POST .../finished FAILED — do NOT hang
        return
    # DeComFL averages over clients actually received (1/N of received) — partial quorum is mathematically fine;
    # FedAvg weights renormalize naturally. (C1 §3.2.)
    aggregate(received)                                      # DeComFL: 1/P factor (R2); CPU-canonical RNG (R3)
    write_checkpoint_to_s3(run, round_idx)                  # content-addressed, BEFORE advancing round counter
    POST /api/internal/runs/{run_id}/checkpoint  {round, artifactId, sha256, sizeBytes}
    POST /api/internal/runs/{run_id}/results     {serverRound, loss, ..., uplinkBytes, scalarsTransmitted}  # incremental
    run.round_idx = round_idx                                # advance AFTER durable persist (C1 §3.1 ordering)
```

The control plane (this unit) also wires `HeartbeatResponse.should_stop=true` on `/stop` so clients drain gracefully (`04 §10.1 item 6`, `C1 §3.2 item 3`).

### 6.7 End-to-end launch sequence (ASCII)

```
Runs controller     FlRunService        OrgQuota/LeaseMgr     FlServerLauncher        Reconciler        FL server
     │ POST .../runs     │                    │                     │                     │                 │
     │──────────────────▶│ @Transactional     │                     │                     │                 │
     │                   │ requireOrgMember    │                     │                     │                 │
     │                   │ tryReserveSlot ────▶│ count active FOR UPDATE                   │                 │
     │                   │ INSERT fl_runs PENDING (partial-unique => 409 if dup)           │                 │
     │                   │ mint runToken; build env (FEDLEARN_* + TRACEPARENT)             │                 │
     │◀── 202 RunDto ────│ registerAfterCommit(launchAndBind)        │                     │                 │
     │                   │ -- COMMIT --        │                     │                     │                 │
     │                   │ launchAndBind: acquire lease; status=STARTING                   │                 │
     │                   │ launch(spec) ───────────────────────────▶│ submit k8s Job/ECS  │                 │
     │                   │ persist executor_ref/grpc_endpoint; renew lease                 │                 │
     │                   │                     │                     │                     │ Phase B probe   │
     │                   │                     │                     │                     │ GetServerStatus─▶ boots,
     │                   │                     │                     │                     │◀── WAITING ─────│ reads S3
     │                   │                     │                     │                     │ status=RUNNING  │
     │                   │                     │                     │                     │ STOMP /topic/status
     │                   │                     │                     │  ROUND LOOP: incremental results POST ─▶ (clients)
     │                   │                     │                     │                     │ Phase C watchdog│
     │                   │                     │                     │                     │ on finish: SUCCEEDED + release
```

---

## 7. Data it owns

This unit **owns** three tables (defined in `03-DATA-MODEL.md §5.2`, migration `V7__fl_runs_and_artifacts.sql`). It reads but does not own `projects`, `organizations`, `dataset_versions`, `partition_recipes`.

| Table | Columns this unit reads/writes | Key invariants enforced |
|---|---|---|
| `fl_runs` | id(=run_id), project_id, org_id, status, **lease_owner, lease_expires_at**, launcher, executor_ref, grpc_endpoint, round_idx, strategy, config(JSONB), dataset_version_id, partition_recipe_id, initial/final_model_artifact_id, mlflow_run_id, requested_by, created_at, updated_at, started_at, ended_at | `UNIQUE (project_id) WHERE status IN ('PENDING','STARTING','RUNNING')` (one active run/project, `A1-F4`); `idx_fl_runs_lease_active` partial index for the reclaim scan; status `CHECK`; launcher `CHECK`. |
| `round_results` | id, fl_run_id, round_idx, loss, accuracy, val_loss, val_accuracy, num_clients_reported, **uplink_bytes, downlink_bytes, scalars_transmitted**, gpu_utilization, round_started_at, round_ended_at | `UNIQUE (fl_run_id, round_idx)` makes the incremental per-round POST idempotent (`04 §5`). |
| `model_artifacts` | id, org_id, sha256, storage_uri, size_bytes, kind(INITIAL/CHECKPOINT/FINAL), fl_run_id, round_idx | `UNIQUE (org_id, sha256)` dedupes identical bytes per tenant; content-addressed (`03 §4.2`). |

**In-memory structures (deliberately minimal — the JVM holds no run-state map):**

| Structure | Purpose | Why it is safe to lose on restart |
|---|---|---|
| `String supervisorId` | This pod's lease-owner id | Re-derived at boot from `HOSTNAME`/pod name. |
| `Map<LauncherBackend, FlServerLauncher>` | Backend dispatch table (Spring beans) | Rebuilt by Spring on boot; stateless. |
| Per-tick reconcile working set (local vars) | The batch of rows reconciled this tick | Re-fetched from the DB every tick; nothing persisted in heap between ticks. |

There is **no** persistent in-heap `Process`/`Job` handle map. Reconstructing reality is always `launcher.describe(executor_ref)` against the durable `fl_runs.executor_ref` — this is the entire point of the rebuild (`C1 §3.3`, risk **R9**).

---

## 8. Configuration & environment variables

### 8.1 Spring config (`OrchestrationProperties`, prefix `fedlearn.orchestration`)

| Key | Type | Default | Profile(s) | Meaning |
|---|---|---|---|---|
| `fedlearn.orchestration.reconcile-interval-ms` | long | `15000` | all | Reconciler `@Scheduled` fixedDelay. |
| `fedlearn.orchestration.lease-ttl-seconds` | int | `60` | all | Lease lifetime; MUST be `>= 3 ×` reconcile interval. |
| `fedlearn.orchestration.readiness-timeout-seconds` | int | `120` | all | Max time in STARTING before FAILED (replaces v1's 3s sleep). |
| `fedlearn.orchestration.reconcile-batch-size` | int | `50` | all | Rows reclaimed/admitted per tick. |
| `fedlearn.orchestration.round-deadline-grace-seconds` | int | `60` | all | Watchdog grace beyond `roundDeadlineSeconds` before force-kill. |
| `fedlearn.orchestration.default-quota.max-concurrent-runs` | int | `5` | all | Per-org default concurrency cap (`B6 §6.1`, R10). |
| `fedlearn.orchestration.default-quota.max-rounds-per-run` | int | `1000` | all | Per-org default round ceiling (matches `04 §4.1` numRounds 1..1000). |
| `fedlearn.orchestration.default-backend` | enum | `K8S_JOB` | ec2demo/production | Backend used when the request omits one (LOCAL_PROCESS in dev). |
| `fedlearn.orchestration.k8s.namespace` | string | `fl-runs` | production | Namespace for FL-server Jobs. |
| `fedlearn.orchestration.k8s.service-account` | string | `fl-run-sa` | production | Per-Job ServiceAccount (least privilege). |
| `fedlearn.orchestration.k8s.job-ttl-seconds` | int | `3600` | production | `ttlSecondsAfterFinished` (auto-GC finished Jobs). |
| `fedlearn.orchestration.k8s.active-deadline-seconds` | int | `86400` | production | Job `activeDeadlineSeconds` (hard wall on a stuck run). |
| `fedlearn.orchestration.ecs.cluster-name` | string | (empty) | ec2demo | ECS cluster; presence selects the ECS backend in v1 — v2 selects by `launcher` field, not by this presence (`A1`). |
| `fedlearn.orchestration.ecs.task-definition` | string | (empty) | ec2demo | ECS task definition family:revision. |
| `fedlearn.orchestration.ecs.subnets` / `.security-groups` | csv | (empty) | ec2demo | `awsvpc` network config for `runTask`. |
| `fedlearn.orchestration.local.port-range-start` | int | `50000` | dev only | Dev `LocalProcessLauncher` port floor (carried from `application.properties:120`). |
| `fedlearn.orchestration.local.port-range-end` | int | `50100` | dev only | **Raised from v1's 50010** (`B6 §1.3` "raise to 50000–50100"); dev-only so the cap is irrelevant in prod. |

### 8.2 Executor environment variables (locked names — set by the launcher into the spawned env)

These are exactly the `04-API-CONTRACTS.md §13` env table plus `§14` trace; the launcher MUST set all of them into the k8s Job env / ECS task override / dev process env:

| Env var | Value |
|---|---|
| `FEDLEARN_RUN_ID` | the run UUID |
| `FEDLEARN_RUN_TOKEN` | `flrun_<base64url(payload)>.<base64url(hmac)>` (the §13 per-run scoped token) |
| `FEDLEARN_BACKEND_URL` | base URL for `/api/internal/...` (VPC-internal/HTTPS outside dev) |
| `FEDLEARN_PROJECT_ID` | project UUID (display/log convenience) |
| `TRACEPARENT` | the W3C trace context of the launch span (`04 §14`) |

Plus run-config flags passed as additional env or CLI args (mirroring v1's `run_fl_server.sh` contract): strategy, num-rounds, min-clients, round-deadline-seconds, seed, dataset-version uri, initial-model uri. **MUST NOT** put any secret other than `FEDLEARN_RUN_TOKEN` in the env, and that token is short-lived + run-scoped (`04 §13`).

### 8.3 Secrets

| Secret | Source | Used by |
|---|---|---|
| `app.internal.run-token-secret` | secrets manager (AWS Secrets Manager / Vault) | `RunTokenService` HMAC-SHA256 signing/verification (`04 §13`). |
| Kubernetes credentials | in-cluster ServiceAccount (when running in EKS) or kubeconfig | `KubernetesJobLauncher`. |
| AWS credentials | instance role / IRSA (IAM Roles for Service Accounts) | `EcsRunTaskLauncher` (singleton `EcsClient`). |

---

## 9. Error handling & edge cases (enumerate the real failure modes)

| # | Failure mode | Exact handling | Audit tie |
|---|---|---|---|
| E1 | Two concurrent `POST .../runs` for one project | Second insert violates the partial unique index → `DataIntegrityViolationException` → `409 RUN_ALREADY_ACTIVE`. No app lock. | `A1-F4`, `C1-F3` |
| E2 | Org at concurrency cap | `countActiveByOrgForUpdate >= maxConcurrentRuns` → `409 ORG_QUOTA_EXCEEDED`, no row inserted. | `B6 §6.1`, R10 |
| E3 | `launcher.launch` throws after commit | `launchAndBind` catches `LauncherException`, transitions PENDING→FAILED with the message, releases lease. The 202 already returned; the frontend sees FAILED via STOMP/poll. | A1 |
| E4 | Executor never becomes ready (boots then crashes at 4s) | Reconciler Phase B: after `readiness-timeout-seconds` in STARTING, `launcher.stop` + FAILED. (Fixes v1's "didn't crash in 3s ≠ healthy", `flower/FlowerServerManager.java:214`.) | `C1-F3` |
| E5 | Owning pod dies mid-run (OOM/SIGKILL/redeploy) | Lease expires; another replica's reconciler Phase A reclaims via `acquire`, `describe`s the executor: RUNNING→re-adopt; MISSING→INTERRUPTED→resume from last checkpoint (§6.6/E13) or FAILED. No orphan, no phantom-RUNNING. | `A1-F4`, `C1-F4`, R9 |
| E6 | One straggler client never reports | The FL server's round loop hits `round_deadline` and proceeds with `>= min_clients` or FAILs with `quorum_not_met`; never hangs. Reconciler Phase C watchdog is the backstop. | `C1-F5`, R9 |
| E7 | FL server wedged (no round progress, not crashed) | Reconciler Phase C: `now - last_round_advance > roundDeadlineSeconds + grace` → `launcher.stop` + FAILED. | `C1-F5/F7` |
| E8 | `LOCAL_PROCESS` requested outside `dev` | `FlRunService.startRun` → `422 UNSUPPORTED_LAUNCHER`; `LocalProcessLauncher.supportsProfile` returns false. | `A1`, `02 §18.1` |
| E9 | Internal callback with a bad/foreign per-run token | `RunTokenService.validate` → `401 RUN_TOKEN_INVALID` (HMAC mismatch/expired) or `403 RUN_TOKEN_MISMATCH` (token runId ≠ path runId). | `A1-F6`, `04 §13` |
| E10 | Callback for a terminal run | `recordRoundResult`/`markFinished` check `status ∈ TERMINAL` → `409 RUN_TERMINAL`; no write. | `04 §5` |
| E11 | Duplicate per-round POST (retry) | `UNIQUE (fl_run_id, round_idx)` → upsert/ignore; idempotent, never a duplicate row. | `04 §5` |
| E12 | Lease stolen between renew calls (clock skew / long GC) | `renew` returns false (rowCount 0) → this instance MUST stop acting on the run this tick; the new owner takes over. No double-stop because `stop` is idempotent. | `C1 §3.3` |
| E13 | Resume from checkpoint after INTERRUPTED | `markInterruptedAndMaybeResume`: if a `model_artifacts` CHECKPOINT row exists for the run, re-launch a new executor with `--resume-round = max(round_idx)+1` and the checkpoint sha256 in env; the FL server hydrates ledger + model and continues. DeComFL per-round state is `O(K·P)` scalars — cheap to resume (`C1 §0/§3.1`). | `C1-F1/F2`, R9/R16 |
| E14 | STOMP subscriber slow / log pipe backpressure | Logs flow on a separate channel and MUST NOT be in the FL-progress critical path; the executor writes stdout to the platform log agent, not through the JVM pipe (kills v1's 64KB stdout wedge). The substrate publishes *progress events* via the internal API, not raw stdout. | `C1-F7`, `A1` |
| E15 | `stop` on an already-terminal run | `409 RUN_NOT_STOPPABLE`; `FlRunStateMachine.assertTransition` rejects the edge. | `04 §4.3/§12` |
| E16 | ECS task ARN lost / not persisted (v1's bug) | v2 persists `executor_ref` in the same transaction as STARTING before returning; `stop`/`describe` always have the ARN. | `A1-F2` (v1 `startEcsFargateServer` returned `Optional.empty()` and never persisted the ARN) |
| E17 | k8s API transient `UNAVAILABLE` during `describe` | Treat as "unknown, not MISSING": return `ExecutorState.PENDING` (keep the lease, retry next tick). Only a definitive 404/NotFound maps to `MISSING`. | `C1 §3.3` |

---

## 10. Testing strategy

Frameworks: **JUnit 5** + **Spring Boot Test** + **Testcontainers-Postgres** (the substrate's DB invariants MUST be tested against real Postgres, not H2 — `A1-F10`, `B2`) + **Mockito** for launcher fakes + **ArchUnit** (forbid `ProcessBuilder` outside `orchestration.launcher`, `02 §2.3`).

| Test (name) | Type | Asserts |
|---|---|---|
| `FlRunService_startRun_insertsPendingLeaseRow` | integration (Testcontainers) | A new run is `PENDING`, `lease_owner` NULL, config persisted; returns 202 RunDto. |
| `FlRunService_startRun_secondConcurrentStart_returns409` | integration | Two parallel `startRun` for one project → exactly one row, the other gets `RUN_ALREADY_ACTIVE` (partial-unique index). |
| `OrgQuotaService_atCap_returns409` | integration | With `maxConcurrentRuns=2` and 2 active runs, the 3rd `startRun` → `ORG_QUOTA_EXCEEDED`; no row inserted. |
| `OrgQuotaService_concurrentStarts_neverExceedCap` | integration | N parallel starts with cap K admit exactly K (the `FOR UPDATE` count holds). |
| `LeaseManager_acquire_winsOnlyWhenExpiredOrUnowned` | integration | `acquire` succeeds on NULL/expired lease, fails when a fresh owner holds it. |
| `LeaseManager_renew_failsAfterLeaseStolen` | integration | After another owner `acquire`s, the original `renew` returns false. |
| `Reconciler_bootReclaimsOrphan_marksInterruptedThenResumes` | integration | A RUNNING row with expired lease + MISSING executor + existing CHECKPOINT → re-launch with resume-round. |
| `Reconciler_describeRunning_readoptsAndRenews` | integration | Expired-lease RUNNING row whose executor is alive → lease renewed, status unchanged. |
| `Reconciler_readinessTimeout_failsRun` | integration (clock-advanced) | STARTING past `readiness-timeout` with no endpoint → `launcher.stop` called + status FAILED. |
| `Reconciler_roundWatchdog_failsWedgedServer` | unit (fake clock) | No round advance beyond `deadline+grace` → force-stop + FAILED reason set. |
| `StateMachine_rejectsIllegalTransition` | unit | SUCCEEDED→RUNNING and STOPPED→anything throw `IllegalRunTransitionException`. |
| `KubernetesJobLauncher_launch_isIdempotentOnRunId` | unit (mock k8s client) | Two `launch` for the same runId create one Job (deterministic name guard). |
| `KubernetesJobLauncher_describe_mapsPodPhases` | unit | Job/Pod statuses map to the correct `ExecutorState.Phase`; NotFound→MISSING; ApiException(503)→PENDING (E17). |
| `EcsRunTaskLauncher_launch_persistsArn_stopCallsStopTask` | unit (mock EcsClient) | `runTask` ARN captured; `stop` issues `StopTask` with it (fixes v1 fire-and-forget). |
| `LocalProcessLauncher_rejectedOutsideDev` | unit | `supportsProfile("production")` is false; `startRun` with LOCAL_PROCESS outside dev → `UNSUPPORTED_LAUNCHER`. |
| `RunTokenService_mintThenValidate_roundTrips` | unit | A minted token validates; a tampered payload/sig → `RUN_TOKEN_INVALID`; foreign runId → `RUN_TOKEN_MISMATCH`. |
| `RecordRoundResult_duplicateRound_isIdempotent` | integration | POST round 7 twice → one `round_results` row (unique constraint). |
| `RecordRoundResult_terminalRun_returns409` | integration | Callback to a SUCCEEDED run → `RUN_TERMINAL`, no write. |
| `Arch_noProcessBuilderOutsideLauncherPackage` | ArchUnit | No class outside `orchestration.launcher` references `java.lang.ProcessBuilder`. |

**Coverage gate:** JaCoCo line coverage ≥ 0.70 on the `orchestration` package once the launcher fakes are wired (`02 §2.3`).

---

## 11. Build & run (verify this unit in isolation)

All commands run from `backend/fl-platform-api/`. Java 21 + the committed Gradle wrapper (`02 §2.3`).

```bash
# 1. Compile + run the substrate's unit tests only.
./gradlew test --tests "com.federated.fl_platform_api.orchestration.*"

# 2. Run the Testcontainers-Postgres integration tests (real Postgres, NOT H2 — A1-F10).
#    Requires a local Docker daemon; Testcontainers pulls postgres:17.10.
./gradlew test --tests "com.federated.fl_platform_api.orchestration.*IntegrationTest"

# 3. Boot the control plane locally with the dev profile (LocalProcessLauncher is the only legal backend).
#    Dev uses local Docker Postgres + MinIO per 01 §5.2 (NOT H2).
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun     # :8081

# 4. Smoke a run end-to-end against the local backend (dev):
#    - create a project (04 §3), then start a run on the LOCAL_PROCESS backend:
curl -i --cookie "jwtToken=$DEV_JWT" -H 'Content-Type: application/json' \
  -d '{"strategy":"DeComFL","numRounds":3,"minClients":1,"roundDeadlineSeconds":60,
       "launcher":"LOCAL_PROCESS","seed":42,
       "hyperparameters":{"learningRate":0.001,"mu":0.001,"numPerturbations":10,"numLocalSteps":5}}' \
  http://localhost:8081/api/projects/$PROJECT_ID/runs
#    Expect: 202 with RunDto.status=STARTING and a run UUID.

# 5. Poll status until RUNNING/SUCCEEDED (the reconciler drives the transitions):
curl --cookie "jwtToken=$DEV_JWT" http://localhost:8081/api/runs/$RUN_ID/status

# 6. Verify the lease + reconciler are alive: kill the local FL process by PID, wait one reconcile
#    interval (15s), and confirm the run is marked INTERRUPTED/FAILED (NOT phantom-RUNNING):
kill -9 $FL_PID; sleep 20; curl --cookie "jwtToken=$DEV_JWT" http://localhost:8081/api/runs/$RUN_ID/status

# 7. Substrate metrics on the internal management port (Micrometer/Prometheus):
curl http://localhost:8081/actuator/prometheus | grep fedlearn_orchestration
```

**Verify-in-isolation checklist:** (a) `startRun` writes a PENDING lease row; (b) a second concurrent start 409s; (c) quota cap rejects with 409; (d) killing the executor leads to reconciler-driven FAILED/INTERRUPTED within one reconcile interval (no phantom-RUNNING); (e) `LOCAL_PROCESS` is refused outside `dev`.

---

## 12. Reasoning & alternatives (why this design; what was rejected)

### D-1 Long-running run-keyed server behind a launcher abstraction, not `ProcessBuilder`-per-project
- **Chosen:** one FL server per `run_id`, launched via `FlServerLauncher`, with a durable `fl_runs` lease + reconciler making the JVM stateless.
- **Rejected — keep v1's `ProcessBuilder` + 11-port map:** capped at 11 concurrent runs (`application.properties:120-121` `50000–50010`), no isolation (FL child shares the API JVM's CPU/RAM/namespace), and the `ConcurrentHashMap<UUID,Process>` at `flower/FlowerServerManager.java:78` is lost on JVM restart → orphans + phantom-RUNNING (`A1-F2/F4`, `C1-F4`, risk **R9**, **R10**).
- **Rejected — adopt Flower/FLARE wholesale:** they multiplex runs cleanly, but neither ships a first-class native C++ on-device client and DeComFL's scalar protocol fits Flower's `Parameters` model awkwardly (`02 §18.1`, `01 §7 D1`). Custom substrate, rebuilt.
- **Audit tie:** synthesis "Rebuild" (`README.md:43,100`); `01 §7 D1`.

### D-2 Three backends behind one interface (k8s primary, ECS secondary, local dev-only)
- **Chosen:** `KubernetesJobLauncher` (production — cgroup isolation, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`), `EcsRunTaskLauncher` (secondary — salvaged + completed: persist the ARN, `StopTask`/`DescribeTasks`, singleton `EcsClient`), `LocalProcessLauncher` (dev-only).
- **Rejected — pick one backend:** the platform must run on a laptop with zero Kubernetes (dev), on the existing AWS investment (ECS, half-coded in v1), and on production k8s. Hardcoding any one blocks the others (`01 §7 D2`).
- **Rejected — EKS at seed tier:** EKS adds a flat ~$73/mo control-plane fee; Fargate RunTask wins below ~30–50 steady concurrent tasks (`B6 §1.3`). So ECS is the right Series-A primitive and k8s is the hyperscale/primary-production substrate; both stay behind the interface.
- **Audit tie:** conflict resolution #1 (`README.md:157`); `B6 §1.3`, `02 §18.1`.

### D-3 Durable Postgres lease + reconciler (stateless JVM supervisor)
- **Chosen:** lease columns on `fl_runs`; atomic single-row UPDATEs; boot + periodic reconciler; `FOR UPDATE SKIP LOCKED` so replicas never fight.
- **Rejected — FLARE-style hot/standby FL server:** overkill for a startup — a single FL server per run made cheaply resumable via checkpoints recovers in seconds and costs nothing idle; the HA that matters is control-plane HA + durable checkpoints, not warm-standby FL servers (`C1 §3.3`). DeComFL's tiny state makes cold-resume fast enough.
- **Rejected — in-memory map + `@PreDestroy` cleanup (v1):** `@PreDestroy` only fires on graceful shutdown; it does nothing for SIGKILL/OOM/ECS task replacement (`C1-F4`).
- **Audit tie:** `C1 §3.3`, `A1-F4`, risk **R9**; `01 §7 D1`.

### D-4 Round deadline + minimum-quorum (no infinite hang)
- **Chosen:** the FL server completes a round when all expected clients reported OR (deadline elapsed AND received ≥ `min_clients`); the reconciler watchdog backstops a wedged server.
- **Rejected — v1's "wait for exactly `clients_per_round`":** one client crashing mid-round hangs the entire run forever, pinning a process+port until a human hits `/stop` (`C1-F5`). v1 even had the liveness machinery (`is_client_alive`, `should_stop`) but never consulted it.
- **Why safe:** DeComFL averages over clients actually received; FedAvg renormalizes — partial quorum is mathematically correct (`C1 §3.2`).
- **Audit tie:** risk **R9**; `01 §7 D4`; `04 §10.3`.

### D-5 Per-org concurrency quotas + admission control before lifting the port cap
- **Chosen:** `OrgQuotaService.tryReserveSlot` inside the start transaction; default `max-concurrent-runs=5`.
- **Rejected — lift the cap with no quota:** "the 11-port cap is accidentally protecting you today; once it's lifted, nothing stops one tenant from spawning unbounded FL servers… the Fargate/EKS bill is unbounded" (`B6 §6.1`, risk **R10**).
- **Audit tie:** `B6 §6 item 1`, `04 §4` (`409 ORG_QUOTA_EXCEEDED`).

### D-6 Per-round checkpoint/resume via content-addressed S3, never destructive in-place
- **Chosen:** the FL server writes a per-round checkpoint to S3 (content-addressed sha256) and POSTs a `CheckpointReportDto` *before* advancing the round counter; resume reads the last checkpoint.
- **Rejected — v1's terminal-only destructive save (`fl_server.py:545`):** a 6-hour run dying at hour 5 restarts at round 1; no off-host copy, no versioning (`C1-F1/F6`, risk **R16**).
- **Audit tie:** `C1 §3.1/§3.4`, `01 §7 D5`, `03 §4.2`.

### D-7 Per-run scoped token + launch-after-commit + no stdout-pipe in the critical path
- **Chosen:** mint an HMAC-signed run-scoped token, inject via env; launch only after the start transaction commits; logs flow on a separate channel.
- **Rejected — v1's single global `APP_INTERNAL_API_KEY`:** any task could POST results for any project (`A1-F6`). Rejected v1's in-transaction launch coupling: a launch failure inside the tx would roll back the row but the executor might already exist; launch-after-commit makes the DB row the durable anchor and the launcher idempotent on `executor_ref`.
- **Audit tie:** `A1-F6`, `04 §13`; `C1-F7`.

---

## 13. Build task checklist for the local model (ordered, dependency-first)

Execute in order. Each task is ~one file/feature with an explicit done-condition. Do not start a task until its predecessors' done-conditions hold.

1. **Migration prerequisite.** Confirm `V7__fl_runs_and_artifacts.sql` exists per `03 §5.2` (copy V4/V5 into `src/main/resources/db/migration/` first if absent, per `03 §1` note). **Done when:** `./gradlew flywayValidate` (or boot in `dev`) applies V6→V8 against local Postgres with the partial unique index `uq_fl_runs_one_active_per_project` and `idx_fl_runs_lease_active` present.
2. **Entities + repositories.** Write `FlRun`, `RoundResult`, `ModelArtifact` `@Entity` classes (JPA validate-only) and `FlRunRepository`, `RoundResultRepository`, `ModelArtifactRepository` with the native lease/quota queries from §6.3/§6.4. **Done when:** a Testcontainers test loads/saves an `FlRun` and the unique-constraint test compiles.
3. **Enums + records.** `LauncherBackend`, `FlRunSpec`, `LaunchResult`, `ExecutorState`, `LauncherException`, `OrgQuota`. **Done when:** they compile and `LauncherBackend` equals the `fl_runs.launcher` CHECK domain.
4. **`FlRunStateMachine`.** Implement the §6.1 transition table + `assertTransition`. **Done when:** `StateMachine_rejectsIllegalTransition` passes.
5. **`LeaseManager`.** Implement `acquire/renew/release/findExpiredLeases/findPendingRuns` with the §6.3 SQL (`FOR UPDATE SKIP LOCKED`). **Done when:** `LeaseManager_acquire_*` and `LeaseManager_renew_failsAfterLeaseStolen` pass.
6. **`OrgQuotaService` + `OrgQuota`.** Implement `quotaFor` (default + override) and `tryReserveSlot` with the `FOR UPDATE` count. **Done when:** `OrgQuotaService_atCap_returns409` and `_concurrentStarts_neverExceedCap` pass.
7. **`RunTokenService`.** Implement HMAC-SHA256 mint/validate exactly per `04 §13` (constant-time compare). **Done when:** `RunTokenService_mintThenValidate_roundTrips` passes.
8. **`FlServerLauncher` interface** + `OrchestrationProperties`. **Done when:** the interface + config bind and `ArchUnit` rule scaffold compiles.
9. **`LocalProcessLauncher` (dev-only).** ProcessBuilder spawn behind the interface; `supportsProfile` true only for `dev`; raised port range; reader-thread hazard fixed (logs not in critical path). **Done when:** `LocalProcessLauncher_rejectedOutsideDev` passes and a dev smoke run spawns a process bound to a 50000–50100 port.
10. **`EcsRunTaskLauncher`.** `runTask` (singleton `EcsClient` bean), persist ARN as `executor_ref`, `stop`=`StopTask`, `describe`=`DescribeTasks`→`ExecutorState`. **Done when:** `EcsRunTaskLauncher_launch_persistsArn_stopCallsStopTask` passes (mock SDK).
11. **`KubernetesJobLauncher` (primary).** Fill `k8s/fl-server-job.yaml.mustache` (image, resources, env incl. `FEDLEARN_*`+`TRACEPARENT`, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`, per-Job ServiceAccount); deterministic Job name `fl-run-{runId}`; `describe` maps Pod phases (NotFound→MISSING, 503→PENDING). **Done when:** `KubernetesJobLauncher_launch_isIdempotentOnRunId` and `_describe_mapsPodPhases` pass (mock client).
12. **`FlRunService.startRun`.** Implement §6.4 (org-member check, launcher-profile guard, dataset resolution, hyperparameter validation per `04 §4.2`, quota reserve, PENDING insert+flush, token mint, env build, launch-after-commit). **Done when:** `FlRunService_startRun_insertsPendingLeaseRow` and `_secondConcurrentStart_returns409` pass.
13. **`FlRunService.stopRun` + `getStatus`.** `stop`→STOPPING (API view) + `launcher.stop`; `getStatus`→`RunStatusDto`. **Done when:** `StateMachine`/`stop` integration tests pass and `getStatus` returns the `04 §4.1` shape.
14. **Internal callbacks** (`recordRoundResult/markFinished/recordCheckpoint/recordStatus`), each gated by a validated `RunContext` (per-run token), idempotent on `(fl_run_id, round_idx)`, terminal-run guarded. **Done when:** `RecordRoundResult_duplicateRound_isIdempotent` and `_terminalRun_returns409` pass.
15. **`FlRunReconciler`.** Implement §6.2 Phases A–D on `@Scheduled` + `@EventListener(ApplicationReadyEvent)`; readiness probe (§6.5); round watchdog (Phase C); resume wiring (E13). **Done when:** `Reconciler_*` tests pass (boot reclaim, re-adopt, readiness timeout, watchdog).
16. **STOMP wiring.** On every state transition, publish `ProjectStatusUpdatePayload` to `/topic/status/{projectId}` and granular `RunEventPayload` to `/topic/runs/{projectId}` (`04 §11`). **Done when:** a transition emits the exact `04 §11.1` payloads (verified in an integration test with a STOMP test client).
17. **Metrics.** Register Micrometer meters `fedlearn_orchestration_runs_launched_total{launcher}`, `_runs_failed_total{reason}`, `_reconcile_duration_seconds`, `_active_runs{org}` (keep `client_id` off labels — `02 §20` cardinality budget). **Done when:** `/actuator/prometheus` shows the meters during a smoke run.
18. **ArchUnit gate.** `Arch_noProcessBuilderOutsideLauncherPackage`. **Done when:** the rule passes and fails if `ProcessBuilder` is referenced outside `orchestration.launcher`.
19. **Delete v1.** Remove `flower/FlowerServerManager.java` and its `ProjectService` lifecycle calls; route the controllers to `FlRunService`. **Done when:** the project compiles with no reference to `FlowerServerManager` and `./gradlew test` is green.

---

## 14. Conformance checklist (must hold before this unit is "done")

- [ ] JVM holds **no** persistent in-heap run/process map; reality is always `launcher.describe(executor_ref)` against `fl_runs` (`C1 §3.3`, R9).
- [ ] One active run per project enforced by the **partial unique index**, not app code (`A1-F4`).
- [ ] Per-org concurrency quota + admission enforced **before** launch (`B6 §6.1`, R10).
- [ ] Round loop has a **deadline + minimum-quorum**; reconciler watchdog backstops a wedged server (`C1-F5`, R9).
- [ ] Per-round checkpoint written to S3 **before** advancing the round counter; resume reads it (`C1 §3.1`, R16).
- [ ] `LOCAL_PROCESS` is **dev-only**; k8s Jobs primary, ECS secondary (`02 §18.1`).
- [ ] Internal callbacks authenticated by a **per-run scoped token**, never a global key (`A1-F6`, `04 §13`).
- [ ] `TRACEPARENT` propagated into the executor env at launch (`04 §14`).
- [ ] Tested against **real Postgres** via Testcontainers, not H2 (`A1-F10`).
- [ ] No `ProcessBuilder` outside `orchestration.launcher`; no `flwr` dependency anywhere (`02 §25`).

---

## 15. Glossary (acronyms, alphabetical)

| Acronym | Full form |
|---|---|
| API | Application Programming Interface |
| ARN | (AWS) Amazon Resource Name |
| DB | Database |
| DeComFL | Dimension-Free Communication Federated Learning (the platform's zeroth-order FL strategy) |
| ECS | (AWS) Elastic Container Service |
| EKS | (AWS) Elastic Kubernetes Service |
| EOL | End Of Life |
| FedAvg | Federated Averaging |
| FL | Federated Learning |
| gRPC | Google Remote Procedure Call |
| HA | High Availability |
| HMAC | Hash-based Message Authentication Code |
| IAM | (AWS) Identity and Access Management |
| IRSA | IAM Roles for Service Accounts |
| JPA | Jakarta Persistence API |
| JVM | Java Virtual Machine |
| k8s | Kubernetes |
| LLD | Low-Level Design |
| LLM | Large Language Model |
| LTS | Long-Term Support |
| OOM | Out Of Memory |
| OTel | OpenTelemetry |
| RDS | (AWS) Relational Database Service |
| RNG | Random Number Generator |
| S3 | (AWS) Simple Storage Service |
| SDK | Software Development Kit |
| STOMP | Simple Text Oriented Messaging Protocol |
| TTL | Time-To-Live |
| URL | Uniform Resource Locator |
| UUID | Universally Unique Identifier |

---

## 16. Source ledger

**Foundation docs (conformed to):**
- `docs/v2/build/01-ARCHITECTURE-HLD.md` — §4.2 Flow B, §7 D1/D2/D4/D5/D7, §5 topology.
- `docs/v2/build/02-TECH-STACK.md` — §18 (FlServerLauncher backends, lease+reconciler, round-deadline pseudocode), §2 (Spring Boot/Gradle), §5 (Postgres), §20 (observability), §25 (invariants).
- `docs/v2/build/03-DATA-MODEL.md` — §4.1 (lease table reasoning), §4.2 (content-addressing), §5.2 (`V7` DDL: `fl_runs`/`round_results`/`model_artifacts`, indexes).
- `docs/v2/build/04-API-CONTRACTS.md` — §4 (Runs API + state machine), §5 (internal callbacks), §10 (`fedlearn.v2` gRPC), §11 (STOMP topics), §12 (error envelope/codes), §13 (per-run token), §14 (traceparent).

**Audit reports (read for depth):**
- `docs/audit/2026-05-29/README.md` — decision table (`:99-103`), conflict resolution #1 (`:157`), risk register R9/R10/R16.
- `docs/audit/2026-05-29/A1-backend.md` — F2 (scaling cliff), F4 (start race + lease), F6 (global key), F10 (H2), v2 orchestration target (`:108-121`).
- `docs/audit/2026-05-29/B6-scale-cost.md` — §1 (FL-server-per-project), §1.3 (tiered orchestration), §6 (cost-control: quotas + scale-to-zero).
- `docs/audit/2026-05-29/C1-reliability-sre.md` — F1/F2/F4/F5/F7 (failure modes), §3.1 (checkpoint/resume), §3.2 (round deadline/quorum), §3.3 (HA/orchestration), decision table.

**Existing v1 code cited (verified during authoring against `main-clean`):**
- 11-port range `50000–50010` — `backend/fl-platform-api/src/main/resources/application.properties:120-121` (verified: `fl.server.port-range.start/end`).
- In-memory process map + lifecycle — `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/flower/FlowerServerManager.java:78` (`Map<UUID,Process> runningServers = new ConcurrentHashMap<>()`), `:83` (`startEcsFargateServer`), `:185` (`runningServers.put`), `:214` (`process.waitFor(3, TimeUnit.SECONDS)`), `:233/248` (`destroyForcibly`). (The audit cites slightly different line numbers, e.g. `:85` for the map; on `main-clean` the map is at `:78` — the code paths are the same.)

**External / market sources (URLs):**
- DeComFL paper (ICLR 2025) — https://arxiv.org/abs/2405.15861
- Kubernetes JobSet — https://jobset.sigs.k8s.io/docs/overview/
- Kubernetes Pod Failure Policy (GA 1.31) — https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/
- NVIDIA FLARE High Availability — https://nvflare.readthedocs.io/en/2.6/programming_guide/high_availability.html
- AWS Fargate pricing — https://aws.amazon.com/fargate/pricing/
- AWS EKS pricing — https://aws.amazon.com/eks/pricing/

**Uncertainty flagged:** GPU need for server-side zeroth-order reconstruction at 7B params is inferred, not benchmarked (`B6 §8`); the resume-compute ceiling for large `missed_rounds × K × P × model_dim` on mobile clients is unbenchmarked (`C1 §6`) — the resume path (E13) is correct in mechanism but may need a periodic full-model anchor checkpoint to bound replay depth, which the FL-framework LLD owns. v1 line numbers differ by a few lines from the audit's citations on this branch (noted above); the code paths are unchanged.

*End of 12-LLD-orchestration-substrate.md.*
