# 01 - Architecture & Core Concepts

This document outlines the foundational architecture of the FedLearn Spring Boot 3 API.

## 1. High-Level Architecture

The FedLearn backend is designed as an **Orchestration API**. It does not perform the heavy mathematical computations required for Machine Learning itself; instead, it manages users, projects, and security, and then dynamically provisions Python-based Federated Learning (FL) aggregation servers to handle the ML workloads.

The backend acts as the central control plane connecting the React web interface to the distributed Python FL ecosystem.

> ⚠️ **Branch reality.** Two parts of this overview describe designed-but-not-yet-committed work:
> 1. **Database** — the backend runs on **PostgreSQL** for every profile (H2 has been retired): `dev`/`ec2demo` against a local Postgres (`backend/fl-platform-api/docker-compose.yml` → `docker compose up -d`), `test` against Testcontainers Postgres (`jdbc:tc:postgresql:16.6-alpine`), and deployed envs override `SPRING_DATASOURCE_*`. The highest committed Flyway migration is **`V19`**.
> 2. **Identity / multi-tenancy / audit** — the `audit/`, `bootstrap/`, and `email/` packages, the `Organization` / `OrganizationMembership` / `ProjectMembership` / `ProjectAccessRequest` / `AuditEvent` entities, `PlatformRole` / `OrgScope` / `AuthorizationService`, and the `V4`–`V6` migrations live on a **separate identity-foundations branch and are _not present_ on this branch.** This branch ships only `users.role IN (USER, ADMIN)` (migration `V2`). See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).
>
> The orchestration (`flower/`), project, results, logging, security-filter, config, controller, and DTO machinery described below is current.

### Tech Stack
* **Language:** Java 21
* **Framework:** Spring Boot 3.4.1
* **Database:** PostgreSQL 16 for every profile (H2 retired; accessed via Spring Data JPA / Hibernate). Local dev via Docker Compose; tests via Testcontainers; deployed envs override `SPRING_DATASOURCE_*`.
* **Authentication:** Stateless JWT (JSON Web Tokens)
* **Real-time Comms:** Spring WebSocket (STOMP protocol)
* **Cloud Integration:** AWS SDK v2 (ECS/Fargate)

---

## 2. Directory Structure

The source code is located at `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/`.

| Directory / Package | Purpose |
|---|---|
| `audit/` | Declarative audit trail. The `@Auditable` annotation, the `AuditAspect` `@Around` advice that writes `audit_events` rows after successful mutations, and the `AuditContext` thread-local metadata sidecar. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md). |
| `bootstrap/` | First-run seeding. `BootstrapRunner` idempotently creates the first `PLATFORM_ADMIN` user and a default `Organization` from `app.bootstrap.*` env vars. |
| `config/` | Application configuration. Contains CORS settings (`WebConfig`), WebSocket broker configuration (`WebSocketConfig`), and Spring Security filter chains (`SecurityConfig`). |
| `controller/` | REST API endpoints. Exposes routes for authentication (`AuthController`), projects (`ProjectController`), results (`ResultsController`), plus the identity surface: memberships, access requests, admin console, user search, and the FL-client API. |
| `dto/` | Data Transfer Objects. POJOs used to decouple the external JSON payloads from the internal JPA entities. |
| `email/` | Pluggable email layer. The `EmailService` interface with a `LoggingEmailService` (dev) and `SmtpEmailService` (prod) adapter, selected by `EmailConfig` on `app.email.provider`. |
| `exception/` | Custom runtime exceptions and the `@ControllerAdvice` global exception handler that translates them into standardized HTTP responses. |
| `flower/` | The core orchestration layer. Contains `FlowerServerManager` which interfaces with AWS or local processes to spawn the ML servers. |
| `model/` | JPA Entities defining the database schema. On this branch: `Project.java`, `User.java`, `RoundResult.java`, `ServerLog.java`. The identity entities (`Organization.java`, `AuditEvent.java`, …) belong to the designed identity-foundations branch — see the banner above. |
| `repository/` | Spring Data JPA interfaces extending `JpaRepository` for database access. |
| `security/` | JWT generation, API Key filters, WebSocket handshake interceptors, the request-scoped `OrgScope`/`OrgScopeFilter` multi-tenant gate, and the login auditing success/failure handlers. |
| `service/` | Business logic layer. Controllers delegate to services (like `ProjectService`) to handle complex operations and transactional boundaries. `AuthorizationService` centralises the role/org-scope checks. |

---

## 3. Core Domain Models (JPA Entities)

The database is organized around the following core entities (the identity/multi-tenancy entities below belong to the designed identity-foundations branch — see the banner above):

### `User`
Represents an authenticated platform user.
- Contains credentials (hashed password) and a single `platformRole` (a `PlatformRole` enum — `USER` or `PLATFORM_ADMIN`). The old single `role` column was renamed to `platform_role` in the V5 migration.
- **Lifecycle / profile columns:** `status` (`UserStatus` — `PENDING`/`ACTIVE`/`SUSPENDED`, defaults `ACTIVE`), `deletedAt` (soft-delete tombstone), `emailVerified`, `displayName`, `avatarUrl`, `lastLoginAt`.
- Has a one-to-many relationship with `Project`. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md) for the full role model.

### `Project`
The central entity for an FL experiment.
- **Key Fields:** `modelType` (e.g., CNN, LLM), `optimizer`, `status` (CREATED, RUNNING, STOPPED, COMPLETED), and `serverPort` (if running locally).
- **Multi-tenancy:** `orgId` (a `UUID`, **NOT NULL**) pins every project to an `Organization`; `visibility` (`ProjectVisibility` — `PRIVATE` default / `PUBLIC`) controls discoverability. Plus Model-Hub publish columns (`modelPublished`, `modelDescription`, `modelTags`, `modelPublishedAt`).
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
4. **Orchestration:** `FlowerServerManager` provisions a Python FL Server (either locally via `ProcessBuilder` or on AWS Fargate).
5. **Real-time Observability:** The Python FL Server streams its logs back to Spring Boot. `WebSocketService` intercepts these logs, saves them to the `server_logs` table, and broadcasts them via STOMP to the React dashboard.
6. **Results Storage:** As the Python FL Server completes training rounds, it sends POST requests to Spring Boot's internal endpoints (secured by API Key) to save `RoundResult` data.

> This `.npz` file (step 3) is the project's *initial*, pre-training architecture — that mechanic is
> unchanged and current. What is **not** current is treating this file as the only place a *trained*
> model ever lives: on run completion the final model is additionally registered as a versioned,
> content-addressed artifact. See [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md).
