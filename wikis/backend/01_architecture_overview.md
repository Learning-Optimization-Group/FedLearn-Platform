# 01 - Architecture & Core Concepts

This document outlines the foundational architecture of the FedLearn Spring Boot 3 API.

## 1. High-Level Architecture

The FedLearn backend is designed as an **Orchestration API**. It does not perform the heavy mathematical computations required for Machine Learning itself; instead, it manages users, projects, and security, and then dynamically provisions Python-based Federated Learning (FL) aggregation servers to handle the ML workloads.

The backend acts as the central control plane connecting the React web interface to the distributed Python FL ecosystem.

> ✅ **Branch reality.** The backend runs on **PostgreSQL** for every profile (H2 has been retired): `dev`/`ec2demo` against a local Postgres (`backend/fl-platform-api/docker-compose.yml` → `docker compose up -d`), `test` against Testcontainers Postgres (`jdbc:tc:postgresql:16.6-alpine`), and deployed envs override `SPRING_DATASOURCE_*`. The highest committed Flyway migration is **`V19`**. The full **identity / multi-tenancy / audit subsystem IS present**: the `audit/`, `bootstrap/`, and `email/` packages, the `Organization` / `OrganizationMembership` / `ProjectMembership` / `ProjectAccessRequest` / `AuditEvent` entities, `PlatformRole` / `OrgScopeFilter` / `AuthorizationService`, and the `V4`–`V7` identity migrations (the original `users.role IN (USER, ADMIN)` from `V2` has been superseded by the layered `PlatformRole`). See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md). The orchestration package (`orchestration/`, renamed from the legacy `flower/` — DA-12), project, results, logging, security-filter, config, controller, and DTO machinery described below is current.

### Tech Stack
* **Language:** Java 21
* **Framework:** Spring Boot 3.4.1
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
| `bootstrap/` | First-run seeding. `BootstrapRunner` idempotently creates the first `PLATFORM_ADMIN` user and a default `Organization` from `app.bootstrap.*` env vars. |
| `config/` | Application configuration. Contains CORS settings (`WebConfig`), WebSocket broker configuration (`WebSocketConfig`), and Spring Security filter chains (`SecurityConfig`). Also holds `FlOrchestrationModeValidator`, the boot-time gate that rejects a non-blank `ecs.cluster-name` (see the orchestration note above). |
| `controller/` | REST API endpoints. Exposes routes for authentication (`AuthController`), projects (`ProjectController`), results (`ResultsController`), plus the identity surface: memberships, access requests, admin console, user search, and the FL-client API. |
| `dto/` | Data Transfer Objects. POJOs used to decouple the external JSON payloads from the internal JPA entities. |
| `email/` | Pluggable email layer. The `EmailService` interface with a `LoggingEmailService` (dev) and `SmtpEmailService` (prod) adapter, selected by `EmailConfig` on `app.email.provider`. |
| `exception/` | Custom runtime exceptions and the `@ControllerAdvice` global exception handler that translates them into standardized HTTP responses. |
| `orchestration/` | The core orchestration layer (renamed from the legacy `flower/` — DA-12). Contains `FlServerManager`, which spawns the Python ML servers as local processes through the `FlServerProcessRunner` seam (DA-8) and tracks them in a `ConcurrentHashMap<UUID, ProcessHandle>`. |
| `model/` | JPA Entities defining the database schema: `Project.java`, `User.java`, `RoundResult.java`, `ServerLog.java`, plus the present identity entities (`Organization.java`, `OrganizationMembership.java`, `ProjectMembership.java`, `ProjectAccessRequest.java`, `AuditEvent.java`). |
| `repository/` | Spring Data JPA interfaces extending `JpaRepository` for database access. |
| `security/` | JWT generation, API Key filters, WebSocket handshake interceptors, the request-scoped `OrgScope`/`OrgScopeFilter` multi-tenant gate, and the login auditing success/failure handlers. |
| `service/` | Business logic layer. Controllers delegate to services (like `ProjectService`) to handle complex operations and transactional boundaries. `AuthorizationService` centralises the role/org-scope checks. |

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
- **Key Fields:** `modelType` (e.g., CNN, LLM), `optimizer`, `status` (CREATED, RUNNING, STOPPED, COMPLETED), and `serverPort` (if running locally).
- **Multi-tenancy:** `orgId` (a `UUID`, **NOT NULL**) pins every project to an `Organization`; `visibility` (`ProjectVisibility` — three tiers: `PUBLIC` / `RESTRICTED` / `PRIVATE`, defaulting to `PRIVATE`) controls discoverability. Plus Model-Hub publish columns (`modelPublished`, `modelDescription`, `modelTags`, `modelPublishedAt`).
- **Relationships:** Owned by a `User`. Contains many `RoundResult`, `ServerLog`, and `ProjectMembership` entries.

### Identity & Multi-Tenancy Entities

Added by the identity subsystem (full detail in [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)):

- **`Organization`** — tenant boundary. `UUID` PK, unique `slug`, soft-delete `deletedAt`.
- **`OrganizationMembership`** — per-user org role (`OrgRole`: `OWNER`/`ADMIN`/`MEMBER`), composite PK `(org_id, user_id)`.
- **`ProjectMembership`** — per-user project role (`MembershipRole`: `OWNER`/`MEMBER`/`CLIENT`), composite PK `(project_id, user_id)`; carries `partitionId` and `joinedVia` provenance.
- **`ProjectAccessRequest`** — the join-request workflow for PRIVATE projects (`PENDING`/`APPROVED`/`DENIED`).
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

1. **User Action:** A user clicks "Start Project" on the React dashboard.
2. **REST Request:** The React app sends an HTTP POST to the Spring Boot `ProjectController` with a valid JWT.
3. **Service Logic:** `ProjectService` validates ownership and asks the `ModelInitializer` to build a local `.npz` weights file.
4. **Orchestration:** `FlServerManager` provisions a Python FL Server as a **local process**, shelling out to the `fl-runtime/` launch scripts (`fl-runtime/run_fl_server.sh`, resolved from the `python.script.*` properties) through the `FlServerProcessRunner` seam, on a port from the `50000-50010` range.
5. **Real-time Observability:** The Python FL Server streams its logs back to Spring Boot. `WebSocketService` intercepts these logs, saves them to the `server_logs` table, and broadcasts them via STOMP to the React dashboard.
6. **Results Storage:** As the Python FL Server completes training rounds, it sends POST requests to Spring Boot's internal endpoints (secured by API Key) to save `RoundResult` data.

> This `.npz` file (step 3) is the project's *initial*, pre-training architecture — that mechanic is
> unchanged and current. What is **not** current is treating this file as the only place a *trained*
> model ever lives: on run completion the final model is additionally registered as a versioned,
> content-addressed artifact. See [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md).
