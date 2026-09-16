# 01 - Architecture & Core Concepts

This document outlines the foundational architecture of the FedLearn Spring Boot 3 API.

## 1. High-Level Architecture

The FedLearn backend is designed as an **Orchestration API**. It does not perform the heavy mathematical computations required for Machine Learning itself; instead, it manages users, projects, and security, and then dynamically provisions Python-based Federated Learning (FL) aggregation servers to handle the ML workloads.

The backend acts as the central control plane connecting the React web interface to the distributed Python FL ecosystem.

> ✅ **Branch reality.** The backend runs on **PostgreSQL** for every profile (H2 has been retired): `dev`/`ec2demo` against a local Postgres (`backend/fl-platform-api/docker-compose.yml` → `docker compose up -d`), `test` against Testcontainers Postgres (`jdbc:tc:postgresql:16.6-alpine`), and deployed envs override `SPRING_DATASOURCE_*`. The highest committed Flyway migration is **`V23`** (`V23__training_arm_ova_lp.sql`); `V20`–`V23` add project derivation, the `BASE_REF` unique index, and the two training-arm migrations — see the table in [the section README](README.md). The full **identity / multi-tenancy / audit subsystem IS present**: the `audit/`, `bootstrap/`, and `email/` packages, the `Organization` / `OrganizationMembership` / `ProjectMembership` / `ProjectAccessRequest` / `AuditEvent` entities, `PlatformRole` / `OrgScopeFilter` / `AuthorizationService`, and the `V4`–`V7` identity migrations (the original `users.role IN (USER, ADMIN)` from `V2` has been superseded by the layered `PlatformRole`). See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md). The orchestration package (`orchestration/`, renamed from the legacy `flower/` — DA-12), project, results, logging, security-filter, config, controller, and DTO machinery described below is current.

### Tech Stack
* **Language:** Java 21 (`JavaLanguageVersion.of(21)` in `build.gradle`)
* **Framework:** Spring Boot 3.4.5
* **Database:** PostgreSQL 16 for every profile (H2 retired; accessed via Spring Data JPA / Hibernate). Local dev via Docker Compose; tests via Testcontainers; deployed envs override `SPRING_DATASOURCE_*`.
* **Authentication:** Stateless JWT (JSON Web Tokens)
* **Real-time Comms:** Spring WebSocket (STOMP protocol)
* **FL server orchestration:** local Python processes, launched from the `fl-runtime/` scripts (no cloud SDK — see the orchestration note below)

> ✅ **Branch reality — orchestration is single-VM, not ECS/Fargate.** The only supported deployed
> architecture is the **hardened single VM**: FL servers run as **local Python processes** on the same
> host as the API. AWS Fargate orchestration *was* implemented once (`1239dda`), but the AWS SDK was
> later removed (`9124b62`), which deleted the implementation with it — there is **no `software.amazon.awssdk`
> dependency in `build.gradle` and no ECS code in the tree**. What survives is a single fail-closed knob:
> `ecs.cluster-name=${ECS_CLUSTER_NAME:}` (blank default) in `application.properties` and
> `application-production.properties`. `FlOrchestrationModeValidator` carries a `@PostConstruct` that throws
> `IllegalStateException` **at boot** if that property is set to a non-blank value, and it carries **no
> `@Profile`**, so the gate applies in every profile (`OP-14`, `8d5dfdc`). The blank default is the supported
> path, so the app always boots. `production` is the single-VM profile — it is *not* an "ECS Fargate profile".
> Managed-task orchestration is **deferred to `OP-12`**.

---

## 2. Directory Structure

The source code is located at `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/`.

| Directory / Package | Purpose |
|---|---|
| `audit/` | Declarative audit trail. The `@Auditable` annotation, the `AuditAspect` `@Around` advice that writes `audit_events` rows after successful mutations, and the `AuditContext` thread-local metadata sidecar. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md). |
| `bootstrap/` | Startup work. `BootstrapRunner` (`@Profile("!test")`) idempotently creates the first `PLATFORM_ADMIN` user and a default `Organization` from `app.bootstrap.*` env vars. `StartupReconciler` + `ProcessProbe` are the BA-3 crash-recovery pair: on boot they re-adopt FL-server children that outlived a backend restart (matching PID **and** OS start instant) and reap the runs whose processes are gone. |
| `config/` | Application configuration. `SecurityConfig` owns the Spring Security filter chain **and** the CORS `CorsConfigurationSource` (there is no separate `WebConfig` — CORS lives here, keyed on `app.cors.allowed-origins`); `WebSocketConfig` configures the STOMP broker and interceptor order; `AsyncConfig` the bounded worker pool; `InferenceRequestSizeFilter` caps inference payloads. It also holds the three fail-closed boot validators: `FlOrchestrationModeValidator` (OP-14, rejects a non-blank `ecs.cluster-name` — see the orchestration note above), `FlBoundaryAuthPolicyValidator` (SE-14) and `FlSecretDistinctnessValidator` (SE-20). |
| `controller/` | REST API endpoints — **24 `*Controller` classes**. Authentication (`AuthController`), projects (`ProjectController`), runs (`RunController`), results and benchmark ingest (`ResultsController`, `BenchmarkIngestController`), inference (`InferenceController`), the recipe catalog (`ModelRecipeController`), the artifact registry + marketplace (`ArtifactController`, `ArtifactLineageController`, `InternalArtifactController`, `MarketplaceController`), and the identity surface: memberships, access requests, owner-promotion requests, profile, admin console, user search, and the FL-client API. |
| `dto/` | Data Transfer Objects. POJOs used to decouple the external JSON payloads from the internal JPA entities. |
| `email/` | Pluggable email layer. The `EmailService` interface with a `LoggingEmailService` (dev) and `SmtpEmailService` (prod) adapter, selected by `EmailConfig` on `app.email.provider`. |
| `exception/` | Custom runtime exceptions and the `@ControllerAdvice` global exception handler that translates them into standardized HTTP responses. |
| `orchestration/` | The core orchestration layer (renamed from the legacy `flower/` — DA-12). `FlServerManager` spawns the Python ML servers as local processes through the `FlServerProcessRunner` seam (DA-8, default impl `LocalProcessFlServerRunner`, returning a `SpawnedFlProcess`) and tracks them in a `ConcurrentHashMap<UUID, ProcessHandle>`. |
| `model/` | JPA entities and the enums that bound their columns: `Project.java`, `User.java`, `Run.java`, `RunEnrollment.java`, `RoundResult.java`, `ServerLog.java`, the benchmark rows, the artifact-registry trio (`ArtifactBlob`, `ModelArtifact`, `ArtifactLineage`), the identity entities (`Organization`, `OrganizationMembership`, `ProjectMembership`, `ProjectAccessRequest`, `OwnerPromotionRequest`, `ProjectDeletionRequest`, `AuditEvent`), and the vocabulary enums (`PlatformRole`, `OrgRole`, `MembershipRole`, `ProjectVisibility`, `ProjectStatus`, `ProjectInitStatus`, `RunStatus`, `TrainingArm`, `ArtifactKind`, …). |
| `repository/` | Spring Data JPA interfaces extending `JpaRepository` for database access. |
| `security/` | JWT generation/validation (`JwtTokenProvider`), the authentication and internal-API-key filters, WebSocket handshake/channel/subscription interceptors, the request-scoped `OrgScope`/`OrgScopeFilter` multi-tenant gate, the FL connection-token and run-token machinery (`ConnectionTokenService`, `RunTokenRegistry`, `FlClientCertificateAuthority`), JWT revocation (`TokenRevocationService`), the login rate limiter, and the login auditing success/failure handlers. |
| `service/` | Business logic layer. Controllers delegate to services (like `ProjectService`) to handle complex operations and transactional boundaries. `AuthorizationService` centralises the role/org-scope checks; `ProjectStatusService` derives a project's status from its active run (BA-4); `ModelRecipeService` caches the `recipes.py` catalog. |
| `validation/` | One reusable constraint: `@ValueOfEnum` + `ValueOfEnumValidator`, for DTO fields whose value must name a constant of a given enum. |

---

## 3. Core Domain Models (JPA Entities)

The database is organized around the following core entities (the identity/multi-tenancy entities below are present on this branch — see the banner above):

### `User`
Represents an authenticated platform user.
- Contains credentials (hashed password) and a single `platformRole` (a `PlatformRole` enum — `USER`, `PROJECT_OWNER`, or `PLATFORM_ADMIN`). The old single `role` column was renamed to `platform_role` in the V5 migration; `PROJECT_OWNER` was added to the column's CHECK constraint in V7.
- **Lifecycle / profile columns:** `status` (`UserStatus` — `PENDING`/`ACTIVE`/`SUSPENDED`, defaults `ACTIVE`), `deletedAt` (soft-delete tombstone), `emailVerified`, `displayName`, `avatarUrl`, `lastLoginAt`.
- Has a one-to-many relationship with `Project`. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md) for the full role model.

### `Project`
The central entity for an FL experiment.
- **Key Fields:** `modelType` (the recipe key — `PNEUMONIA_CNN`, `CNN`, `CIFAR_RESNET18`, `MLP`, `TRANSFORMER`, `LLM_LORA`, `TINYNET_GOLDEN`), `modelName`, `optimizer`, `taskType` (`SEQ_CLASSIFICATION`/`CAUSAL_LM`, `LLM_LORA` only), `modelPath`, `serverPort`, and `activeRunId`.
- **Training arm:** `trainingArm` (a `TrainingArm` enum — `FULL` / `FROZEN_HEAD` / `OVA_LP`, defaulting to `FULL`) decides which parameters a run trains **and federates**, and which objective it trains under. Backed by `V22`/`V23`. See [03 - Project Management Lifecycle](03_project_management.md).
- **Status is derived, not stored-and-trusted.** The `status` column still exists as a mirror, but every DTO-building caller asks `ProjectStatusService.currentStatus(project)` instead (BA-4). That method consults the one-time init phase first — `initStatus` (`ProjectInitStatus` — `INITIALIZING`/`DONE`/`FAILED`, `V14`) — and only once init is `DONE` does it derive from the active `Run`: `PENDING`/`STARTING`/`RUNNING` → `RUNNING`, `COMPLETED` → `COMPLETED`, `STOPPED` → `STOPPED`, `FAILED` → `FAILED`, no active run → `CREATED`. The `ProjectStatus` enum therefore has six values (`INITIALIZING`, `CREATED`, `RUNNING`, `STOPPED`, `COMPLETED`, `FAILED`), not four. The derivation exists because the denormalized column drifted — a run that ended `FAILED` used to leave its project stuck reading `RUNNING`.
- **DP policy (SE-11, `V17`):** `regulated` and `dpEnabled` (both NOT NULL, default `FALSE`) plus the nullable `dpTargetEpsilon` / `dpDelta` / `dpClipNorm`. Completeness is validated in Java at creation and re-checked at the spawn seam, not by a CHECK constraint.
- **Derivation (DA-14, `V20`):** `initFromPretrained` (NOT NULL, default `FALSE`), `baseRefSha256`, `derivationSpec`. All opt-in; a NULL derivation is an ordinary from-scratch recipe project.
- **Multi-tenancy:** `orgId` (a `UUID`, **NOT NULL**) pins every project to an `Organization`; `visibility` (`ProjectVisibility` — three tiers: `PUBLIC` / `RESTRICTED` / `PRIVATE`, defaulting to `PRIVATE`) controls discoverability. Plus Model-Hub publish columns (`modelPublished`, `modelDescription`, `modelTags`, `modelPublishedAt`) and the optional `requirementsOverride` JSON (`V9`).
- **Relationships:** Owned by a `User`. Contains many `RoundResult`, `ServerLog`, and `ProjectMembership` entries, and its `Run` sub-tree cascades on delete (`V19`).

### `Run`
One training execution of a project (`V8`), and the source of truth for live FL-server state.
- **Key Fields:** `strategy`, `numRounds`, `minClients`, `clientsPerRound`, `partitioningMode`, `status` (`RunStatus`), `serverHost`/`serverPort`, `seed`, `torchVersion`, `recipeKey`, `createdBy`, and the timestamps (all `timestamptz` since `V13`).
- **Process identity (BA-3):** `serverPid` + `processStartedAt` (`V15`) let `StartupReconciler` tell a surviving child from a dead one — or from an unrelated process that recycled the PID — and `internalTokenHash` (`V16`) lets it rehydrate the in-memory `RunTokenRegistry` for exactly the runs it re-adopts.
- **`RunEnrollment`** carries per-run client enrollment: `(runId, userId)` PK, a run-scoped `partitionId` under `UNIQUE(run_id, partition_id)`, and `clientKind`.

### Identity & Multi-Tenancy Entities

Added by the identity subsystem (full detail in [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)):

- **`Organization`** — tenant boundary. `UUID` PK, unique `slug`, soft-delete `deletedAt`.
- **`OrganizationMembership`** — per-user org role (`OrgRole`: `OWNER`/`ADMIN`/`MEMBER`), composite PK `(org_id, user_id)`.
- **`ProjectMembership`** — per-user project role (`MembershipRole`: `OWNER`/`MEMBER`/`CLIENT`), composite PK `(project_id, user_id)`; carries `partitionId` and `joinedVia` provenance.
- **`ProjectAccessRequest`** — the join-request workflow for **RESTRICTED** projects (`AccessRequestStatus`: `PENDING`/`APPROVED`/`DENIED`). PUBLIC projects join outright and PRIVATE ones are invite-only and 404 to outsiders, so neither files a request.
- **`OwnerPromotionRequest`** / **`ProjectDeletionRequest`** (V7) — the two admin-approved workflows, sharing that same `AccessRequestStatus` vocabulary.
- **`AuditEvent`** — append-only audit log row: `action` (`AuditAction` enum), `actorUserId`, `orgId`, `targetType`/`targetId`, JSONB `metadata`, `requestIp`, `userAgent`.

### `RoundResult`
Stores the output of a single federated learning round (e.g., FedAvg).
- **Key Fields:** `serverRound`, `loss`, `accuracy`, `gpuUtilization`.
- These are written by the Python FL server calling the `/api/internal/projects/{id}/results` endpoint.

### `ServerLog`
Persistent storage for stdout logs generated by the Python FL Server.
- **Key Fields:** `level` (INFO, ERROR, etc.), `message`, `stackTrace`, `timestamp`.
- Used to export logs to text files or display historical data on the React dashboard after a project finishes.

---

## 4. The Data Flow Summary

1. **User Action:** A user creates a project on the React dashboard, then clicks "Start".
2. **REST Request:** The React app sends an HTTP POST to the Spring Boot `ProjectController`; the browser's `jwtToken` cookie rides along.
3. **Service Logic:** `createProject` checks that the caller may create projects at all (`PROJECT_OWNER` or `PLATFORM_ADMIN`), validates the DP config, persists the project shell as `INITIALIZING`, and returns **201 immediately**. Model initialization — a Python spawn that writes the project's initial `.npz` architecture — is dispatched to a bounded async worker *after* the transaction commits (BA-1); the worker flips `init_status` to `DONE` or `FAILED` and broadcasts the change.
4. **Orchestration:** On `/start`, `FlServerManager` provisions a Python FL Server as a **local process**, shelling out to the `fl-runtime/` launch scripts (`fl-runtime/run_fl_server.sh`, resolved from the `python.script.*` properties) through the `FlServerProcessRunner` seam, on a port from the `50000-50010` range.
5. **Real-time Observability:** The Python FL Server streams its logs back to Spring Boot. `WebSocketService` intercepts these logs, saves them to the `server_logs` table, and broadcasts them via STOMP to the React dashboard.
6. **Results Storage:** As the Python FL Server completes training rounds, it POSTs to Spring Boot's `/api/internal/**` endpoints to save `RoundResult` (and benchmark) data. Those calls carry **two** credentials: the shared `X-Internal-Key` and the per-run `X-Internal-Run-Token`, which binds the callback to exactly one project (SE-7 — see [02 - Security and Authentication](02_security_and_auth.md)).
7. **Artifact Registration:** On run completion the final model is additionally registered as a versioned, content-addressed artifact with an eval card attached. See [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md).

> This `.npz` file (step 3) is the project's *initial*, pre-training architecture — that mechanic is
> unchanged and current. What is **not** current is treating this file as the only place a *trained*
> model ever lives: on run completion the final model is additionally registered as a versioned,
> content-addressed artifact. See [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md).
