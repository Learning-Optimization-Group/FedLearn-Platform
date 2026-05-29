# 10 — LOW-LEVEL DESIGN: Backend Control Plane (FedLearn Platform v2)

**Unit:** the BACKEND CONTROL PLANE — Spring Boot 3.5 (Long-Term-Support line), Java 21 (Long-Term-Support).
**Document type:** Production build specification — Low-Level Design (LLD).
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters) that will implement the
method bodies. Every interface, signature, package path, version, environment-variable name, and command
below is **pre-decided**. Do not choose alternatives, do not infer missing pieces, do not add unrequested
features. Where a body must be implemented, the signature/shape is given; you fill the body.

**Status:** authoritative for v2. Conforms to and does not contradict:
`docs/v2/build/02-TECH-STACK.md`, `docs/v2/build/03-DATA-MODEL.md`, `docs/v2/build/04-API-CONTRACTS.md`.
Where this document references those, the cited document is the source of truth.

**Date authored:** 2026-05-29.

---

## 0. Abbreviation key (first-use expansions; thereafter the short form is used)

| Short form | Full form |
|---|---|
| LLD | Low-Level Design |
| LLM | Large Language Model |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| RPC | Remote Procedure Call |
| gRPC | Google Remote Procedure Call |
| STOMP | Simple Text Oriented Messaging Protocol |
| WS | WebSocket |
| HTTP | HyperText Transfer Protocol |
| HTTPS | HTTP Secure |
| JWT | JSON Web Token |
| JSON | JavaScript Object Notation |
| JSONB | JSON Binary (PostgreSQL binary JSON column type) |
| UUID | Universally Unique Identifier |
| BIGINT | 64-bit signed integer SQL type |
| RBAC | Role-Based Access Control |
| RLS | Row-Level Security |
| FL | Federated Learning |
| DeComFL | Dimension-Free Communication Federated Learning (zeroth-order optimization) |
| FedAvg | Federated Averaging |
| DTO | Data Transfer Object |
| JPA | Jakarta Persistence API |
| ORM | Object-Relational Mapping |
| SQL | Structured Query Language |
| DDL | Data Definition Language |
| SpEL | Spring Expression Language |
| AOP | Aspect-Oriented Programming |
| HMAC | Hash-based Message Authentication Code |
| sha256 | Secure Hash Algorithm 256-bit |
| S3 | (AWS) Simple Storage Service |
| MinIO | self-hosted S3-compatible object store (not an acronym) |
| MLflow | Machine-Learning lifecycle tool (not an acronym) |
| ECS | (AWS) Elastic Container Service |
| EKS | (AWS) Elastic Kubernetes Service |
| ARN | (AWS) Amazon Resource Name |
| k8s | Kubernetes |
| RDS | (AWS) Relational Database Service |
| OTel | OpenTelemetry |
| W3C | World Wide Web Consortium |
| TLS | Transport Layer Security |
| mTLS | mutual TLS |
| CORS | Cross-Origin Resource Sharing |
| CSRF | Cross-Site Request Forgery |
| CSP | Content-Security-Policy |
| TTL | Time-To-Live |
| HA | High Availability |
| JVM | Java Virtual Machine |
| BOM | Bill Of Materials |
| EOL | End Of Life |
| PII | Personally Identifiable Information |
| PHI | Protected Health Information |
| SOC 2 | System and Organization Controls 2 |
| HIPAA | Health Insurance Portability and Accountability Act |
| SLO | Service-Level Objective |
| ArchUnit | the Java architecture-test library (not an acronym) |
| VBU | `verify-before-use` (confirm exact version exists before pinning) |

---

## 1. Purpose & single responsibility

The backend control plane is the **stateless supervisor and system-of-record** for the FedLearn platform.
Its single responsibility is: **own identity, tenancy, projects, and the durable run lease; expose the
REST API and STOMP channel; and COMMAND (never host) the FL orchestration substrate.** Concretely it:

1. Owns the OLTP schema (`03-DATA-MODEL.md`): users, organizations, memberships, projects, datasets,
   `fl_runs`, `round_results`, `model_artifacts`, `determinism_manifests`, `audit_events`.
2. Serves the REST contract (`04-API-CONTRACTS.md §2–§9`) over cookie-only JWT, and the STOMP topics
   (`§11`), and the internal callback API (`§5`) authenticated by per-run scoped tokens (`§13`).
3. **Commands** the substrate through the `FlServerLauncher` interface (`02-TECH-STACK.md §18`). In
   production the control plane does **not** spawn FL servers itself; it writes an `fl_runs` lease row,
   calls `FlServerLauncher.launch(...)`, and a reconciler reconciles executor state → database → STOMP.

**What it is NOT responsible for (hard boundary):**
- It does not run FL training, aggregation, or zeroth-order math (that is `framework/`).
- It does not host FL servers in production (only `LocalProcessLauncher`, dev-only, does).
- It does not store raw training data or model bytes (those are client-private / S3/MinIO; the control
  plane stores metadata, leases, and non-reversible fingerprints — `03-DATA-MODEL.md §1`).

**Audit driver:** A1 verdict is **"keep Spring Boot … salvage the auth/data layers with surgical fixes,
rebuild FL-server orchestration as a proper control plane"** (`A1-backend.md:14`). This LLD encodes that
verdict: salvage the layering, decompose the 438-line `ProjectService` god-object (`A1-backend.md:43`),
fix the dead admin RBAC path (`A1-F1`), enforce `org_id` scoping (`A1-F9`, `B4 §2.1`), and replace the
in-memory `ConcurrentHashMap<UUID,Process>` with the durable `fl_runs` lease (`A1-F2`, `C1-F4`).

---

## 2. Position in the system

### 2.1 Depends-on (this unit calls / requires these)

| Dependency | Contract | Reasoning |
|---|---|---|
| **PostgreSQL 17.10 (RDS)** | `03-DATA-MODEL.md` schema; JPA `validate`-only; Flyway owns DDL. | OLTP system-of-record. v1 H2-file-mode is KILLed outside dev/test (`A1-F10`, `B6 §4`). |
| **FL orchestration substrate** | `FlServerLauncher` interface (`02-TECH-STACK.md §18.1`): `KubernetesJobLauncher` (primary), `EcsRunTaskLauncher` (secondary), `LocalProcessLauncher` (dev only). | The control plane submits a run, it does not fork a process (`A1-F2`, `B2-tech-stack.md:148`). |
| **Object store (S3 / MinIO)** | Pre-signed URL broker only; **blob bytes never transit the JVM** (`04-API-CONTRACTS.md §9`). | Avoids the v1 in-JVM 2× memory blowups on large models (`B6 §1`, `A3 M4`). |
| **MLflow (self-hosted)** | Link-out: `fl_runs.mlflow_run_id`. | Experiment/model registry layered on the artifact store (`02-TECH-STACK.md §20.1`). |
| **STOMP relay (RabbitMQ / Redis)** | `enableStompBrokerRelay` swap, multi-replica only. | In-memory simple broker caps to one replica (`A1-backend.md:157`, `B6 §1.2`). Single-replica seed tier uses the in-memory simple broker. |
| **Email provider** | `EmailService` interface; `LoggingEmailService` (dev/test) / `SmtpEmailService` (deployed). | Email verify / password reset (`04-API-CONTRACTS.md §2`). |

### 2.2 Depended-by (these call this unit)

| Caller | Surface consumed |
|---|---|
| **React frontend** | REST `§2–§9`; STOMP `§11`; cookie-only JWT auth (`§1`). |
| **Desktop / mobile clients** | REST `§2–§9` (same cookie auth); they obtain `enrollment_token` and run config from REST, then connect to the FL server over gRPC (`§10`) — gRPC is NOT this unit. |
| **FL server (spawned executor)** | Internal callback API `§5` (`/api/internal/runs/{runId}/**`) authenticated by the per-run scoped token `§13`. |

### 2.3 Interfaces EXPOSED (by exact name from `04-API-CONTRACTS.md`)

- **REST:** `/api/auth/*` (§2), `/api/projects/*` (§3), `/api/projects/{projectId}/runs` + `/api/runs/*`
  (§4), `/api/internal/runs/{runId}/*` (§5), `/api/users/*` (§6), `/api/admin/*` (§7),
  `/api/orgs/*` + `/api/datasets/*` (§8), `/api/artifacts/*` (§9).
- **STOMP topics:** `/topic/logs/{projectId}`, `/topic/results/{projectId}`, `/topic/status/{projectId}`,
  `/topic/runs/{projectId}` (§11). WS endpoint `/ws-logs`.
- **Error envelope:** the single standard envelope (§12) with the stable `code` registry (§12.1).

### 2.4 Interfaces CONSUMED (defined elsewhere; this unit only calls them)

- `FlServerLauncher` (defined by the orchestration unit, `02-TECH-STACK.md §18.1`) — see §5.6 for the exact
  Java signature this unit programs against.
- gRPC `fedlearn.v2` (§10) is between FL clients and the FL server; the control plane does NOT speak it.

```
                         ┌───────────────────────────────────────────────┐
  React / Desktop  ──────▶  BACKEND CONTROL PLANE (this unit)             │
  (REST + STOMP)         │  Spring Boot 3.5.14 / Java 21                  │
                         │  controllers → services → repositories        │
                         │  security (JWT, per-run token) / audit / WS    │
                         └───────┬───────────────┬───────────────┬────────┘
                                 │ JPA (validate)│ FlServerLauncher│ pre-signed URL
                                 ▼               ▼  (.launch/.stop/.poll)  ▼
                          PostgreSQL 17.10   k8s Job / ECS task / dev proc   S3 / MinIO
                                 ▲               │ (FL server)                ▲
                                 │               │ /api/internal/runs/{id}/** │
                                 └───────────────┴──── per-run token ─────────┘
```

---

## 3. Tech stack for this unit (pinned; one-line reasoning each)

All versions are from `02-TECH-STACK.md`. "VBU" = `verify-before-use` (the listed doc pins it as such).

| Technology | Pinned version | One-line reasoning (source) |
|---|---|---|
| Java | `21` (Long-Term-Support) | Salvage the Java control plane; LTS to 2031; virtual threads for the reconciler/heartbeat fan-in (`02-TECH-STACK.md §1.1`). |
| Spring Boot | `3.5.14` | Bump off EOL 3.4.5; lowest-risk supported line keeping the cookie-JWT/STOMP/Flyway posture (`02-TECH-STACK.md §2.1`). |
| Gradle (wrapper committed) | `9.5.1` | Project invariant: Gradle, not Maven; CI entrypoint is `./gradlew` (`02-TECH-STACK.md §2.3`). |
| Spring Boot starters | Boot 3.5 BOM-managed | `web`, `security`, `data-jpa`, `validation`, `websocket`, `actuator`, `test` mirror v1's working set (`02-TECH-STACK.md §2.1`). |
| `io.jsonwebtoken:jjwt-api/-impl/-jackson` | `0.12.5` | Carried from v1; cookie JWT signing/validation (`02-TECH-STACK.md §2.1`). |
| PostgreSQL (RDS) | `17.10` | Managed Postgres replaces H2-file-mode; bounded control-plane tables, no sharding (`02-TECH-STACK.md §5.1`). |
| Flyway (+ `flyway-database-postgresql`) | Boot 3.5 BOM (Flyway 10+/11+), do not override | Schema owned by Flyway, `validate`-only JPA; new fields = new `V{n}__*.sql` (`02-TECH-STACK.md §5.2`). |
| Hibernate (JPA provider) | Boot 3.5 BOM (Hibernate 6) | Native JSONB mapping via `@JdbcTypeCode(SqlTypes.JSON)`; no third-party JSONB lib (`03-DATA-MODEL.md §7`). |
| `io.kubernetes:client-java` | latest, VBU | `KubernetesJobLauncher` (primary production backend) (`02-TECH-STACK.md §18.1`). |
| `software.amazon.awssdk:ecs` | `2.25.11` (VBU) | `EcsRunTaskLauncher` secondary backend, singleton client (`02-TECH-STACK.md §18.1`; `A1-F2`). |
| `software.amazon.awssdk:s3` | current 2.x, VBU | Pre-signed URL broker for the artifact store (`02-TECH-STACK.md §7`). |
| `io.micrometer:micrometer-registry-prometheus` | Boot 3.5 BOM, VBU | FL-run + platform metrics; v1 had none (`A1-backend.md:157`, `02-TECH-STACK.md §20`). |
| `io.micrometer:micrometer-tracing-bridge-otel` | Boot 3.5 BOM, VBU | W3C `traceparent` propagation JVM → process env → gRPC (`04-API-CONTRACTS.md §14`). |
| Testcontainers (PostgreSQL) | VBU | CI runs V1→V8 against real Postgres `17.10`; `test` profile stays H2 `create-drop`/Flyway-disabled (`03-DATA-MODEL.md §6`, `A1-F10`). |
| Rate limiter (Bucket4j) | VBU | Throttle the three unauthenticated auth mutations; `RATE_LIMITED` code (`04-API-CONTRACTS.md §2`, §12.1). Named in API contract; no version pinned in `02-TECH-STACK.md` — pin latest stable, VBU. |
| Password encoder (`spring-security-crypto`, BOM) | Boot 3.5 BOM | `BCryptPasswordEncoder` (carried from v1's Spring Security). No new dependency. |

> **Uncertainty flag:** `02-TECH-STACK.md` does **not** pin a Bucket4j version or a password-encoder algorithm.
> The API contract (§2) names Bucket4j for rate limiting and the v1 codebase uses Spring Security's
> `PasswordEncoder`. Use Bucket4j (latest stable, VBU) and `BCryptPasswordEncoder`; do not introduce a
> different rate-limit library or hashing algorithm without an explicit decision in `02-TECH-STACK.md`.

---

## 4. Module / file structure

Root package: `com.federated.fl_platform_api`. The legacy package name `flower` and the class
`FlowerServerManager` are historical (there is no Flower/`flwr` dependency); v2 **renames** the package to
`orchestration` and the manager to `FlRunService` + launcher classes, eliminating the misleading name.

```
backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/
├── FlPlatformApiApplication.java        # @SpringBootApplication entry point
│
├── config/
│   ├── SecurityConfig.java              # SecurityFilterChain, CORS, CSRF, cookie attrs, route authz
│   ├── WebSocketConfig.java             # /ws-logs, simple broker /topic, /app prefix, interceptors
│   ├── JacksonConfig.java               # WRITE_DATES_AS_TIMESTAMPS=false; Instant as ISO-8601 Z
│   ├── OpenApiConfig.java               # API surface documentation (optional, no behaviour)
│   ├── RateLimitConfig.java             # Bucket4j buckets for auth endpoints
│   └── AsyncConfig.java                 # virtual-thread executor + @EnableScheduling for reconciler
│
├── controller/
│   ├── AuthController.java              # /api/auth/*            (§2)
│   ├── ProjectController.java           # /api/projects/*        (§3)
│   ├── RunController.java               # /api/projects/{id}/runs + /api/runs/*  (§4)
│   ├── InternalRunController.java       # /api/internal/runs/{runId}/*  (§5)
│   ├── UserController.java              # /api/users/me*         (§6)  (list-all DELETED — A1-F3)
│   ├── AdminController.java             # /api/admin/*           (§7)
│   ├── OrgController.java               # /api/orgs/*            (§8.1)
│   ├── DatasetController.java           # /api/datasets/*        (§8.2)
│   └── ArtifactController.java          # /api/artifacts/*       (§9)
│
├── service/                            # decomposed from the v1 438-line ProjectService god-object
│   ├── AuthService.java                 # register/login/verify/forgot/reset; emailVerified gate (A1-F5)
│   ├── ProjectService.java             # project CRUD + visibility ONLY (was god-object — A1-backend.md:43)
│   ├── FlRunService.java               # run lifecycle: start/stop, lease write, launcher call, quota (A1-F2/F4, B6 §6)
│   ├── ProjectLogService.java          # log paging + text/plain export (split out — A1-backend.md:43)
│   ├── DiscoveryService.java           # PUBLIC-project discovery, org-scoped (split out; B4 §2.1 leak fix)
│   ├── RunReconcilerService.java       # @Scheduled lease reconciler (C1 §3.3, A1-backend.md:117)
│   ├── ResultIngestService.java        # internal-callback ingest: round_results/finished/checkpoint/status
│   ├── DatasetService.java             # datasets/versions/partition recipes (§8.2)
│   ├── ArtifactService.java            # pre-signed URL broker + artifact metadata (§9)
│   ├── OrgService.java                 # orgs + org memberships (§8.1)
│   ├── UserService.java                # /api/users/me profile + password change (§6)
│   ├── AdminService.java               # admin user/role/status/audit listing (§7; LAST_ADMIN — A1-F1)
│   └── AuthorizationService.java       # the single org-scoped authz chokepoint (A1-F9, B4 §2.1)
│
├── security/
│   ├── JwtTokenProvider.java           # mint/validate cookie JWT with iss/aud/jti/skew (A1-F7, B4 §2.3)
│   ├── JwtCookieFilter.java            # reads jwtToken cookie → SecurityContext
│   ├── CustomUserDetailsService.java   # loads user; emits ROLE_PLATFORM_ADMIN authority (A1-F1)
│   ├── RunTokenService.java            # mint/validate per-run scoped token (§13; A1-F6)
│   ├── RunTokenAuthFilter.java         # filter on /api/internal/** → RunContext (§13)
│   ├── JwtHandshakeInterceptor.java    # WS handshake cookie auth (salvaged from v1)
│   ├── JwtChannelInterceptor.java      # CONNECT + SUBSCRIBE-frame topic authz (A1 WS gap)
│   └── PlatformUserPrincipal.java      # authenticated principal (userId, platformRole, orgRoles)
│
├── orchestration/                      # renamed from `flower` (no Flower dep — the project conventions)
│   ├── FlServerLauncher.java           # INTERFACE consumed from substrate unit (§5.6)
│   ├── KubernetesJobLauncher.java      # primary production backend (impl body by orchestration unit)
│   ├── EcsRunTaskLauncher.java         # secondary backend (impl body by orchestration unit)
│   ├── LocalProcessLauncher.java       # dev-only ProcessBuilder backend (hard-gated to dev)
│   ├── FlRunSpec.java                  # launch input record (§5.6)
│   ├── LaunchResult.java               # launch output record (§5.6)
│   └── RunState.java                   # poll() result enum (§5.6)
│
├── audit/
│   ├── Auditable.java                  # @Auditable annotation (action, targetType, targetIdParam)
│   ├── AuditAspect.java                # proceeds-then-writes; Jackson-serialized metadata (A1 audit note)
│   └── AuditAction.java               # enum of audited actions
│
├── email/
│   ├── EmailService.java               # interface
│   ├── LoggingEmailService.java        # dev/test: writes .eml files
│   └── SmtpEmailService.java           # ec2demo/production
│
├── bootstrap/
│   └── BootstrapRunner.java            # first PLATFORM_ADMIN + Platform org on startup
│
├── domain/                             # JPA entities (one file each)
│   ├── User.java  Organization.java  OrganizationMembership.java
│   ├── Project.java  ProjectMembership.java  ProjectAccessRequest.java
│   ├── FlRun.java  RoundResult.java  ModelArtifact.java  DeterminismManifest.java
│   ├── Dataset.java  DatasetVersion.java  PartitionRecipe.java  AuditEvent.java
│   └── id/  OrganizationMembershipId.java  ProjectMembershipId.java   # composite (UUID, Long)
│
├── domain/enums/
│   ├── PlatformRole.java  OrgRole.java  ProjectRole.java
│   ├── RunStatus.java  RunStrategy.java  LauncherKind.java
│   ├── ProjectStatus.java  UserStatus.java  ArtifactKind.java
│   ├── Modality.java  PartitionerKind.java  AuditAction.java
│
├── repository/                         # Spring Data JPA; org-scoped query methods (A1-F9)
│   ├── UserRepository.java  OrganizationRepository.java  OrganizationMembershipRepository.java
│   ├── ProjectRepository.java  ProjectMembershipRepository.java  ProjectAccessRequestRepository.java
│   ├── FlRunRepository.java  RoundResultRepository.java  ModelArtifactRepository.java
│   ├── DeterminismManifestRepository.java  DatasetRepository.java  DatasetVersionRepository.java
│   ├── PartitionRecipeRepository.java  AuditEventRepository.java
│
├── dto/
│   ├── auth/        RegisterRequest, RegisterResponse, LoginRequest, MeResponse, ...
│   ├── project/     CreateProjectRequest, UpdateProjectRequest, ProjectResponseDto
│   ├── run/         StartRunRequest, RunDto, RunStatusDto, DeterminismManifestDto, CheckpointDto, HyperparametersDto
│   ├── internal/    RoundResultDto, RunFinishedDto, CheckpointReportDto, RunStatusReportDto
│   ├── admin/       AdminUserDto, UpdateUserRoleRequest, UpdateUserStatusRequest, AuditEventDto
│   ├── org/         CreateOrgRequest, OrgDto, OrgMemberDto, AddOrgMemberRequest, UpdateOrgMemberRoleRequest
│   ├── dataset/     CreateDatasetRequest, DatasetDto, CreateDatasetVersionRequest, DatasetVersionDto,
│   │                CreatePartitionRecipeRequest, PartitionRecipeDto
│   ├── artifact/    ArtifactUploadUrlRequest/Response, RegisterArtifactRequest, ArtifactDto
│   └── error/       ErrorResponse, FieldError
│
├── web/
│   └── GlobalExceptionHandler.java     # @RestControllerAdvice → the ONE error envelope (§12)
│
└── ws/
    └── WebSocketService.java           # broadcasts to /topic/{...}; structured JSON payloads (§11)

backend/fl-platform-api/src/main/resources/
├── db/migration/                       # V1..V5 (copy V4/V5 from build/resources — 03-DATA-MODEL §1)
│   ├── V6__dataset_registry.sql        V7__fl_runs_and_artifacts.sql  V8__determinism_manifest.sql
├── application.properties              # base (fails fast without required secrets)
├── application-dev.properties          application-ec2demo.properties  application-production.properties
└── application-test.properties         # in-memory H2, Flyway disabled, create-drop (DO NOT CHANGE)
```

**Architecture rule (enforced by ArchUnit, §10):** controllers call services only; **controllers never
touch repositories.** v1's PII leak (`A1-F3`) and unauthenticated result-write (`A1-F6`) both lived in
controllers because there was no service seam (`A1-backend.md:41`).

---

## 5. Key interfaces & type signatures (full)

### 5.1 Role enums (the A1-F1 fix — string drift becomes structurally impossible)

```java
package com.federated.fl_platform_api.domain.enums;

public enum PlatformRole { USER, PLATFORM_ADMIN }
public enum OrgRole      { OWNER, ADMIN, MEMBER }
public enum ProjectRole  { MEMBER, CLIENT }        // OWNER is implicit via projects.user_id
```

**Why enums, not strings (A1-F1):** v1 had `@PreAuthorize("hasRole('ADMIN')")` while
`CustomUserDetailsService` emitted `ROLE_PLATFORM_ADMIN` — the bootstrap admin was 403'd from every admin
route, and a test seeding the literal `"ADMIN"` masked it (`A1-backend.md:53-67`). With an enum:
- Jackson deserializes `UpdateUserRoleRequest.platformRole` to `PlatformRole`, so `valueOf` rejects any
  typo at the wire boundary → `400 VALIDATION_FAILED` (`04-API-CONTRACTS.md §7`).
- The database `CHECK (platform_role IN ('USER','PLATFORM_ADMIN'))` added in V6 (`03-DATA-MODEL.md §5.1`)
  is the database-level guard.
- `CustomUserDetailsService` emits authority `"ROLE_" + platformRole.name()` = `ROLE_PLATFORM_ADMIN`, and
  every `@PreAuthorize` uses `hasRole('PLATFORM_ADMIN')` — the two now provably agree.

### 5.2 `AuthorizationService` — the single org-scoped chokepoint (A1-F9, B4 §2.1)

Every project/run/org read and write funnels through this one bean. It is the database-level half of the
"RLS-style query filters" decision plus the application-level guard.

```java
package com.federated.fl_platform_api.service;

public interface AuthorizationService {

    /** Resolve the authenticated principal from the SecurityContext (never from a request body). */
    PlatformUserPrincipal currentPrincipal();

    /** PLATFORM_ADMIN bypasses org-membership checks (audited). */
    boolean isPlatformAdmin(PlatformUserPrincipal p);

    /** AUTH: throws 401 NOT_AUTHENTICATED if no valid principal. */
    PlatformUserPrincipal requireAuthenticated();

    /** ORG_MEMBER(orgId): caller is a member of orgId (any OrgRole) OR PLATFORM_ADMIN. else 403 FORBIDDEN. */
    void requireOrgMember(UUID orgId);

    /** ORG_ADMIN(orgId): caller OrgRole in {OWNER,ADMIN} of orgId OR PLATFORM_ADMIN. else 403. */
    void requireOrgAdmin(UUID orgId);

    /** PROJECT_PARTICIPANT(projectId): owner (projects.user_id) OR project_memberships row
     *  OR ORG_ADMIN(project.org) OR PLATFORM_ADMIN. else 403. Resolves org_id server-side. */
    void requireProjectParticipant(UUID projectId);

    /** Run-scoped variant: resolves run.projectId/run.orgId, then requireProjectParticipant. */
    void requireRunParticipant(UUID runId);

    /** PLATFORM_ADMIN only. else 403. */
    void requirePlatformAdmin();

    /** Returns the caller's org-ids; repository layer ANDs every tenant query with these. */
    Set<UUID> callerOrgIds();
}
```

**Reasoning (B4 §2.1, A1-F9):** v1's `AuthorizationService` "never references `org_id`" — a user in Org A
added as a CLIENT to an Org B project got full access regardless of org boundary, and `getDiscoverProjects`
leaked every PUBLIC project's name/owner across all tenants (`B4-security-compliance.md:75-76`). v2 makes
`org_id` mandatory in this chokepoint AND in every repository query method (§7.3). The PLATFORM_ADMIN
bypass is intentional (the project conventions identity layers) but **must be `@Auditable`** so the bypass is logged
(`B4-security-compliance.md:77`).

### 5.3 `FlRunService` — run lifecycle (decomposed from the god-object; A1-F2/F4)

```java
package com.federated.fl_platform_api.service;

public interface FlRunService {

    /** POST /api/projects/{projectId}/runs (§4). Writes PENDING lease, checks quota, calls launcher.
     *  @return RunDto with status STARTING (202).
     *  @throws ConflictException(RUN_ALREADY_ACTIVE) on the partial-unique-index violation (A1-F4)
     *  @throws ConflictException(ORG_QUOTA_EXCEEDED)  when org active-run count >= quota (B6 §6)
     *  @throws UnprocessableException(NO_DATASET_VERSION) when no dataset version pinned
     *  @throws UnprocessableException(UNSUPPORTED_LAUNCHER) when LOCAL_PROCESS requested outside dev */
    RunDto startRun(UUID projectId, StartRunRequest req);

    /** POST /api/runs/{runId}/stop (§4). Sets STOPPING, calls launcher.stop(executorRef).
     *  @throws ConflictException(RUN_NOT_STOPPABLE) when the run is already terminal. */
    RunDto stopRun(UUID runId);

    RunDto getRun(UUID runId);
    List<RunDto> listRunsForProject(UUID projectId, int page, int size, RunStatus statusFilter);
    RunStatusDto getRunStatus(UUID runId);
    DeterminismManifestDto getManifest(UUID runId);   // 409 MANIFEST_NOT_READY before run started
    List<CheckpointDto> listCheckpoints(UUID runId);
}
```

### 5.4 `RunReconcilerService` — the stateless-supervisor loop (A1-backend.md:117, C1 §3.3)

```java
package com.federated.fl_platform_api.service;

public interface RunReconcilerService {

    /** @Scheduled(fixedDelayString = "${app.reconciler.interval-ms:30000}")
     *  For each non-terminal fl_run: poll the launcher, reconcile RunState -> fl_runs.status -> STOMP,
     *  renew the lease for runs this instance owns, and claim runs whose lease has expired. */
    void reconcileTick();

    /** @EventListener(ApplicationReadyEvent.class)
     *  On boot, reap orphans: claim expired leases, poll, mark MISSING+lease-expired runs FAILED. */
    void reconcileOnBoot();
}
```

### 5.5 `RunTokenService` — per-run scoped token (A1-F6, `04-API-CONTRACTS.md §13`)

```java
package com.federated.fl_platform_api.security;

public record RunContext(UUID runId, UUID projectId, UUID orgId) {}

public interface RunTokenService {
    /** Mint at launch. Format: "flrun_<base64url(payload)>.<base64url(hmac)>" (§13).
     *  payload = {runId, projectId, orgId, issuedAt, expiresAt, nonce}; HMAC-SHA256 over payloadB64
     *  with secret app.internal.run-token-secret. NOT stored in a table (03-DATA-MODEL §8). */
    String mint(UUID runId, UUID projectId, UUID orgId, Duration ttl);

    /** Validate per §13 pseudocode. Throws the exact codes:
     *  401 RUN_TOKEN_INVALID (bad prefix/sig/expiry), 403 RUN_TOKEN_MISMATCH (token.runId != pathRunId),
     *  404 RUN_NOT_FOUND, 409 RUN_TERMINAL. */
    RunContext validate(String authorizationHeader, UUID pathRunId);
}
```

### 5.6 `FlServerLauncher` — the interface this unit CONSUMES (defined by the substrate unit)

The control plane programs against this; the bodies belong to `KubernetesJobLauncher` /
`EcsRunTaskLauncher` / `LocalProcessLauncher` (`02-TECH-STACK.md §18.1`).

```java
package com.federated.fl_platform_api.orchestration;

public record FlRunSpec(
    UUID    runId,
    UUID    projectId,
    UUID    orgId,
    RunStrategy strategy,            // DeComFL | FedAvg
    int     numRounds,
    int     minClients,
    int     roundDeadlineSeconds,
    UUID    datasetVersionId,
    UUID    partitionRecipeId,
    long    seed,
    Map<String,Object> config,       // serialized to fl_runs.config JSONB (03-DATA-MODEL §5.4)
    String  runToken,                // FEDLEARN_RUN_TOKEN injected into executor env (§13)
    String  backendUrl,              // FEDLEARN_BACKEND_URL
    String  traceparent              // W3C trace context for the launch span (§14)
) {}

public record LaunchResult(String executorRef, String grpcEndpoint) {}  // grpcEndpoint may be null until RUNNING

public enum RunState { PENDING, STARTING, RUNNING, SUCCEEDED, FAILED, STOPPED, MISSING }

public interface FlServerLauncher {
    LaunchResult launch(FlRunSpec spec);   // submit a run; do NOT block until reachable (202 semantics, §4)
    void         stop(String executorRef); // StopTask / delete Job / destroy process
    RunState     poll(String executorRef); // real executor state for the reconciler
    LauncherKind kind();                    // K8S_JOB | ECS_RUN_TASK | LOCAL_PROCESS
}
```

### 5.7 Selected request/response DTO shapes (exact — from `04-API-CONTRACTS.md`)

These are the load-bearing ones; the rest map 1:1 to `04-API-CONTRACTS.md` JSON shapes. Use Java records.

```java
// MeResponse (§2.1) — typed platformRole + orgs array (v2 fix vs v1's drifted string)
public record MeResponse(long userId, String username, String email,
                         PlatformRole platformRole, List<OrgMembershipView> orgs, boolean emailVerified) {}
public record OrgMembershipView(UUID orgId, String orgName, OrgRole orgRole) {}

// StartRunRequest (§4.1) — launcher enum on the wire is KUBERNETES|ECS|LOCAL_PROCESS
public record StartRunRequest(RunStrategy strategy, int numRounds, int minClients,
                              int roundDeadlineSeconds, String launcher, UUID datasetVersionId,
                              HyperparametersDto hyperparameters, long seed) {}

// RoundResultDto (§5.1) — the internal-callback per-round shape (comm-cost wedge)
public record RoundResultDto(int serverRound, Double loss, Double accuracy, Double gpuUtilization,
                             Long uplinkBytes, Long downlinkBytes, Long scalarsTransmitted,
                             Long modelParamCount, Double roundDurationSeconds,
                             Double aggregationSeconds, Integer activeClients) {}

// Error envelope (§12)
public record ErrorResponse(String timestamp, int status, String code, String message,
                            String path, String traceId, List<FieldError> fieldErrors) {}
public record FieldError(String field, String message) {}
```

### 5.8 Key JPA entity → table mappings (must match the DDL; JPA is `validate`-only — `03-DATA-MODEL.md §7`)

| Entity | Table | PK | Notable mappings |
|---|---|---|---|
| `FlRun` | `fl_runs` | `@Id UUID` | `status`/`strategy`/`launcher` as `@Enumerated(EnumType.STRING)` matching the `CHECK` sets exactly; `config` as `@JdbcTypeCode(SqlTypes.JSON)`; lineage FKs `@ManyToOne`. |
| `RoundResult` | `round_results` | `@Id UUID` | `@ManyToOne FlRun flRun` on `fl_run_id` (NOT `project_id` — repointed per C3 §5.3); `round_idx` is the wire `serverRound`. |
| `AuditEvent` | `audit_events` | `@Id UUID` | `metadata` as `@JdbcTypeCode(SqlTypes.JSON)` (was `String`/CLOB — §6 fix). |
| `OrganizationMembership` | `organization_memberships` | `@IdClass OrganizationMembershipId(UUID orgId, Long userId)` | mixed key (UUID, Long). |
| `ProjectMembership` | `project_memberships` | `@IdClass ProjectMembershipId(UUID projectId, Long userId)` | mixed key (UUID, Long). |
| `DeterminismManifest` | `determinism_manifests` | `@Id UUID` | `@OneToOne FlRun` on `fl_run_id UNIQUE`. |

---

## 6. Core algorithms & flows

### 6.1 Start-run flow (the rebuilt orchestration surface — A1-F2/F4, B6 §6)

`POST /api/projects/{projectId}/runs` → `RunController.startRun` → `FlRunService.startRun`. **`202`, not
`200`:** start is asynchronous against an external executor; the control plane writes the lease and hands
off (`04-API-CONTRACTS.md §4`).

```
Browser/Desktop        RunController        FlRunService          FlRunRepository       FlServerLauncher
     │  POST .../runs        │                    │                      │                      │
     │ (StartRunRequest)     │                    │                      │                      │
     ├──────────────────────▶│ requireOrgMember   │                      │                      │
     │                       ├───────────────────▶│ validate hyperparams (§4.2)                 │
     │                       │                    │ resolve project.org_id (server-side)        │
     │                       │                    │ check launcher allowed in profile           │
     │                       │                    │ check NO_DATASET_VERSION (422)              │
     │                       │                    │ check org active-run quota (409)            │
     │                       │                    ├─ INSERT fl_runs(status=PENDING,            │
     │                       │                    │     project_id,org_id,strategy,config,...) ─▶│
     │                       │                    │   ── partial-unique-index on (project_id)    │
     │                       │                    │      WHERE status IN active ── violation? ──▶ 409 RUN_ALREADY_ACTIVE
     │                       │                    │ mint RunTokenService.mint(run,proj,org)     │
     │                       │                    ├─ launcher.launch(FlRunSpec{token,traceparent,...}) ─▶
     │                       │                    │◀─ LaunchResult(executorRef, grpcEndpoint=null)
     │                       │                    ├─ UPDATE fl_runs SET status=STARTING,        │
     │                       │                    │     executor_ref=?, lease_owner=instanceId, │
     │                       │                    │     lease_expires_at=now()+TTL              │
     │                       │                    │ @Auditable(RUN_START) row written            │
     │◀──────────────────────┤◀───────────────────┤ 202 RunDto(status=STARTING)                 │
```

**The A1-F4 race is closed declaratively.** Two concurrent starts both attempt the `PENDING` INSERT; the
partial unique index `uq_fl_runs_one_active_per_project ON fl_runs(project_id) WHERE status IN
('PENDING','STARTING','RUNNING')` (`03-DATA-MODEL.md §5.2`) makes the second INSERT raise a constraint
violation, which `FlRunService` catches and maps to `409 RUN_ALREADY_ACTIVE`. **No application lock, no
`@Transactional` pessimistic row-lock needed** — the database is the arbiter. This is strictly better than
v1's check-then-act, which produced two FL servers on two ports (`A1-F4`, `C1-F3`).

```java
// FlRunService.startRun — the conflict-mapping core (pseudocode of the catch)
try {
    flRunRepository.saveAndFlush(pendingRun);        // forces the unique-index check now
} catch (DataIntegrityViolationException e) {
    if (isPartialUniqueViolation(e, "uq_fl_runs_one_active_per_project"))
        throw new ConflictException(ErrorCode.RUN_ALREADY_ACTIVE);
    throw e;
}
```

### 6.2 Reconciler loop (the JVM is a stateless supervisor over the DB lease — A1/C1)

```
@Scheduled(fixedDelay=30s)  RunReconcilerService.reconcileTick()
  for run in flRunRepository.findNonTerminal():            // status PENDING|STARTING|RUNNING
      state = launcher(run.launcher).poll(run.executorRef)  // K8S/ECS/local
      switch state:
        RUNNING   -> if run.grpcEndpoint==null && launcher reports endpoint: set it
                     renew lease: UPDATE fl_runs SET lease_expires_at=now()+TTL WHERE id=? AND lease_owner=me
                     if run.status != RUNNING: set RUNNING, broadcast /topic/status/{projectId}
        SUCCEEDED -> set SUCCEEDED, ended_at=now(), broadcast
        FAILED    -> set FAILED, errorMessage, broadcast
        STOPPED   -> set STOPPED, broadcast
        MISSING   -> if run.lease_expires_at < now(): set FAILED (orphan/crash), broadcast, emit metric
                     else: leave (another instance owns it / transient)
  // claim expired leases left by a dead instance (optimistic):
  UPDATE fl_runs SET lease_owner=me, lease_expires_at=now()+TTL
    WHERE status IN active AND lease_expires_at < now()
```

**Why a lease, not the v1 in-memory map (A1-F2.3, C1-F4):** v1 tracked processes in
`ConcurrentHashMap<UUID,Process>`; a JVM restart orphaned every Python child with no database record, and
`/stop` found nothing. The lease makes the database the source of truth: any supervisor instance can adopt
a run whose lease expired, poll the real executor, and reconcile state. Restarts and deploys become safe.
The lease index `idx_fl_runs_lease_active` (`03-DATA-MODEL.md §5.2`) makes the expired-lease scan cheap.

### 6.3 Internal result callback (per-round, incremental — A1-F6, B3)

`POST /api/internal/runs/{runId}/results` is called **by the FL server, per round**, authenticated by the
per-run token (not a cookie, not the v1 global key).

```
FL server               RunTokenAuthFilter         InternalRunController     ResultIngestService
   │ POST .../runs/{id}/results                         │                          │
   │ Authorization: Bearer flrun_<...>                  │                          │
   ├───────────────────────▶│ validate(header, pathId)  │                          │
   │                        │  HMAC-SHA256 const-time    │                          │
   │                        │  expiry / runId match      │                          │
   │                        │  run exists / not terminal │                          │
   │   401/403/404/409 ◀────┤ (any check fails)          │                          │
   │                        │ RunContext(run,proj,org) ─▶│ requireRunToken context  │
   │                        │                            ├─ validate RoundResultDto ─▶│
   │                        │                            │                          ├─ UPSERT round_results
   │                        │                            │                          │   ON CONFLICT (fl_run_id,round_idx)
   │                        │                            │                          │   DO NOTHING  (idempotent)
   │                        │                            │                          ├─ broadcast /topic/results/{projectId}
   │   202 (empty) ◀────────┴────────────────────────────┴──────────────────────────┘
```

**Why `runId`-keyed + token-bound (A1-F6):** v1 keyed callbacks on `projectId` and authorized only with
one global `APP_INTERNAL_API_KEY`, so any task could POST fabricated results for any project — a
multi-tenant integrity break (`A1-backend.md:143`). v2 binds the token to one `runId`; the control plane
resolves `runId → projectId/orgId` from the `fl_runs` row server-side; the caller never asserts the
project/org. **Idempotency:** the `UNIQUE (fl_run_id, round_idx)` constraint (`03-DATA-MODEL.md §5.2`) makes
a re-POSTed round a no-op, so the best-effort per-round callback can retry safely.

### 6.4 STOMP SUBSCRIBE-frame authorization (closes the A1 WS gap)

```java
// JwtChannelInterceptor.preSend — on a SUBSCRIBE frame only
if (StompCommand.SUBSCRIBE.equals(accessor.getCommand())) {
    String dest = accessor.getDestination();               // e.g. /topic/results/{projectId}
    UUID projectId = parseProjectId(dest);                 // reject malformed -> deny
    authorizationService.requireProjectParticipant(projectId);  // throws -> frame rejected
}
```

**Why (A1 WS gap, B4 §2.1):** v1 authenticated the handshake but never checked the SUBSCRIBE destination, so
any authenticated user could subscribe to any project's live telemetry — a cross-tenant leak
(`A1-backend.md:75`). Topics are keyed on `projectId` (not `runId`) because the dashboard subscribes when a
user opens a project, before a run exists (`04-API-CONTRACTS.md §11`). Also drop the v1 anonymous
`AuthenticationException` subclasses (`A1-backend.md:28`).

### 6.5 Cookie JWT mint/validate (hardened — A1-F7, B4 §2.3)

Mint with `sub` (userId), `iss` (`app.auth.jwt.issuer`), `aud` (`app.auth.jwt.audience`), `jti` (random
UUID), `iat`, `exp`. Validate with `requireIssuer` + `requireAudience` + `allowedClockSkewSeconds` +
optional `jti` deny-list. The cookie is `HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600`
(`04-API-CONTRACTS.md §1`). `Secure` is driven by `app.auth.cookie.secure` (true outside dev).

**Why (A1-F7, B4 §2.3):** v1 JWT had only `sub`+`exp` — no issuer/audience binding (a dev token was
indistinguishable from a prod token), no `jti` (no revocation; logout was cookie-clear only), no clock
skew. `iss`/`aud` stop cross-environment replay; `jti` enables a deny-list for revocation.

---

## 7. Data it owns

The control plane owns the entire `03-DATA-MODEL.md` schema. JPA runs `validate`-only; Flyway owns DDL.

### 7.1 Tables (exact names from `03-DATA-MODEL.md`)

| Table | Owner-relevant columns this unit reads/writes | Source |
|---|---|---|
| `users` | `id BIGINT`, `username`, `email`, `password`, `platform_role`, `status`, `email_verified`, `display_name`, `last_login_at` | §3.3, V1/V2/V5 |
| `organizations` | `id UUID`, `name`, `slug`, `created_at TIMESTAMPTZ` (V7 fix), `deleted_at` | §3.3 |
| `organization_memberships` | `org_id UUID`, `user_id BIGINT`, `org_role` (OWNER/ADMIN/MEMBER) | §3.3 |
| `projects` | `id UUID`, `user_id BIGINT`, `org_id UUID NOT NULL`, `name`, `model_type`, `model_name`, `optimizer`, `status`, `visibility`, `dataset_version_id`, `partition_recipe_id` | §3.3, V6 |
| `project_memberships` | `project_id UUID`, `user_id BIGINT`, `role` (MEMBER/CLIENT/OWNER), `partition_id` | §3.3 |
| `project_access_requests` | `project_id`, `user_id`, `status` (PENDING/APPROVED/DENIED) | §3.3 |
| `fl_runs` | `id UUID`, `project_id`, `org_id NOT NULL`, `status`, `lease_owner`, `lease_expires_at`, `launcher`, `executor_ref`, `grpc_endpoint`, `round_idx`, `strategy`, `config JSONB`, lineage FKs, `mlflow_run_id`, `requested_by`, timestamps | §5.2, V7 |
| `round_results` | `id UUID`, `fl_run_id NOT NULL`, `round_idx`, `loss`, `accuracy`, `val_loss`, `val_accuracy`, `num_clients_reported`, `uplink_bytes`, `downlink_bytes`, `scalars_transmitted`, `gpu_utilization`, timestamps | §5.2, V7 |
| `model_artifacts` | `id UUID`, `org_id NOT NULL`, `sha256 CHAR(64)`, `storage_uri`, `size_bytes`, `kind`, `fl_run_id`, `round_idx` | §5.2, V7 |
| `determinism_manifests` | `id UUID`, `fl_run_id UNIQUE NOT NULL`, version/RNG/hash fields, `manifest_json JSONB` | §5.3, V8 |
| `datasets` / `dataset_versions` / `partition_recipes` | the content-hashed registry (§5.1, V6) | §5.1 |
| `audit_events` | `id UUID`, `occurred_at TIMESTAMPTZ` (V7 fix), `actor_user_id`, `org_id`, `action`, `target_type`, `target_id`, `metadata JSONB` (V7 fix), `request_ip`, `user_agent` | §3.3, §6 fix |

### 7.2 In-memory structures (the only ones; everything durable is in PostgreSQL)

| Structure | Type | Purpose | Reasoning |
|---|---|---|---|
| Rate-limit buckets | `Map<String, Bucket>` (Bucket4j, Caffeine-backed, per-IP/endpoint) | Throttle `/register`, `/login`, `/password/forgot` | `RATE_LIMITED` (§12.1); seed-tier in-process is acceptable, move to a distributed bucket if multi-replica. |
| JWT `jti` deny-list (optional) | short-TTL cache | Revocation on logout / forced logout | A1-F7; bounded by token expiry. |
| Supervisor instance id | `String` (pod name / host id) | `lease_owner` value | C1 §3.3; identifies which instance owns a run lease. |

**There is NO in-memory `Process` map.** That v1 structure is deleted — it was the single point of amnesia
(`A1-F2.3`, `C1-F4` verdict **kill**). All run state lives in `fl_runs`.

### 7.3 Repository org-scoping rule (A1-F9)

Every tenant-owned repository query method takes `org_id` (or a `Set<UUID> orgIds` from
`AuthorizationService.callerOrgIds()`) and ANDs it into the `WHERE`. Example signatures:

```java
public interface ProjectRepository extends JpaRepository<Project, UUID> {
    List<Project> findByOrgIdIn(Collection<UUID> orgIds);                 // GET /api/projects (org-scoped)
    Optional<Project> findByIdAndOrgIdIn(UUID id, Collection<UUID> orgIds);
    List<Project> findByVisibilityAndOrgIdIn(Visibility v, Collection<UUID> orgIds); // discovery, org-scoped (B4 §2.1)
    boolean existsByNameIgnoreCaseAndOrgId(String name, UUID orgId);      // PROJECT_NAME_TAKEN within org
}

public interface FlRunRepository extends JpaRepository<FlRun, UUID> {
    @Query("select r from FlRun r where r.status in :active")
    List<FlRun> findNonTerminal(@Param("active") Collection<RunStatus> active);   // reconciler
    long countByOrgIdAndStatusIn(UUID orgId, Collection<RunStatus> active);       // quota check (B6 §6)
}
```

---

## 8. Configuration & environment variables

Properties resolve from environment variables; the base profile **fails fast** if a required secret is
absent (`A1-backend.md:153`). Profiles: `dev`, `ec2demo`, `production`, `test`.

| Property (env var) | Type | Default | Profile / notes |
|---|---|---|---|
| `APP_JWT_SECRET` (`app.auth.jwt.secret`) | string | none (boot fails) | all deployed; HMAC signing key. |
| `app.auth.jwt.issuer` | string | `fedlearn` | NEW (A1-F7): `iss` claim. |
| `app.auth.jwt.audience` | string | `fedlearn-api` | NEW: `aud` claim. |
| `app.auth.jwt.clock-skew-seconds` | int | `60` | NEW: validation skew tolerance. |
| `app.auth.cookie.secure` | bool | `true` (`false` in `dev`) | `Secure` cookie attribute; **must be true in `ec2demo`** behind TLS (B4 §2.3). |
| `app.auth.cookie.same-site` | enum | `Strict` | tightened from v1 `Lax` (`04-API-CONTRACTS.md §1`). |
| `app.internal.run-token-secret` | string | none (boot fails outside dev) | NEW (A1-F6): HMAC secret for per-run tokens (§13). |
| `CORS_ALLOWED_ORIGINS` (`app.cors.allowed-origins`) | csv | none (boot fails) | allowlist; `:5173` for dev frontend. |
| `app.reconciler.interval-ms` | int | `30000` | reconciler `@Scheduled` cadence (C1 §3.3). |
| `app.reconciler.lease-ttl-seconds` | int | `120` | lease renewal window. |
| `app.run.org-quota-max-active` | int | `8` (seed) | per-org concurrent-run cap (B6 §6, `ORG_QUOTA_EXCEEDED`). |
| `app.run.default-round-deadline-seconds` | int | `600` | round deadline floor (`04-API-CONTRACTS.md §4.1`). |
| `app.launcher.kind` | enum | `KUBERNETES` (`LOCAL_PROCESS` in dev) | active `FlServerLauncher` backend (§18.1). |
| `app.launcher.k8s.namespace` | string | `fl-runs` | `KubernetesJobLauncher` target namespace. |
| `ecs.cluster-name` (`app.launcher.ecs.cluster`) | string | empty | `EcsRunTaskLauncher` cluster; empty disables ECS. |
| `FEDLEARN_BACKEND_URL` (`app.internal.backend-url`) | url | none | base URL injected into executor env for callbacks (§13). |
| `app.artifact.bucket` | string | `fedlearn-artifacts` | S3/MinIO bucket for pre-signed URLs (§9). |
| `app.artifact.presign-ttl-seconds` | int | `900` | pre-signed URL TTL. |
| `SPRING_DATASOURCE_URL` / `_USERNAME` / `_PASSWORD` | string | none (deployed) | RDS Postgres 17.10 (`dev` H2-file; `test` in-memory H2). |
| `APP_BOOTSTRAP_ADMIN_EMAIL` / `_USERNAME` / `_PASSWORD` | string | optional | first PLATFORM_ADMIN + Platform org (the project conventions). |
| `app.email.provider` | enum | `logging` (`smtp` deployed) | selects `EmailService` impl. |
| `management.endpoints.web.exposure.include` | csv | `health,info,prometheus` | add `prometheus` (A1 observability gap). |

**Profile table (what activates where):**

| Profile | Datasource | Flyway | Launcher | cookie.secure | Notes |
|---|---|---|---|---|---|
| `dev` | H2 file-mode (Postgres mode) | enabled | `LOCAL_PROCESS` allowed | `false` | H2 console; public dev secrets. **Never deploy.** |
| `test` | in-memory H2 | **disabled** (`create-drop`) | n/a | n/a | **DO NOT CHANGE** Flyway-disabled behaviour (`03-DATA-MODEL.md §6`). |
| `ec2demo` | RDS Postgres 17.10 (was H2 — KILLed) | enabled (validate JPA) | `LOCAL_PROCESS` or `ECS` | `true` | behind TLS (`B4 §2.3`). |
| `production` | RDS Postgres 17.10 | enabled (validate JPA) | `KUBERNETES` | `true` | secrets via secrets manager. |

> **Uncertainty flag:** `02-TECH-STACK.md` and `B6` move `ec2demo` off H2 to RDS Postgres, contradicting the
> v1 project conventions which described `ec2demo` as H2 file-mode. This LLD follows the v2 docs (RDS Postgres for
> `ec2demo`/`production`; H2 only in `dev`/`test`).

---

## 9. Error handling & edge cases

All non-2xx go through one `@RestControllerAdvice GlobalExceptionHandler` → the single envelope (§12) with
the stable `code` registry (§12.1). v1 had two coexisting error contracts (`A1-backend.md:44`); v2 has one.

| # | Failure mode | Detection | Exact handling |
|---|---|---|---|
| 1 | Second concurrent `/start` for one project | `DataIntegrityViolationException` on `uq_fl_runs_one_active_per_project` | `409 RUN_ALREADY_ACTIVE` (§6.1). |
| 2 | Org over its active-run quota | `countByOrgIdAndStatusIn >= app.run.org-quota-max-active` | `409 ORG_QUOTA_EXCEEDED` (B6 §6). |
| 3 | Start with no dataset version (project has none, request omits) | null check in `FlRunService` | `422 NO_DATASET_VERSION`. |
| 4 | `LOCAL_PROCESS` launcher requested outside `dev` | profile check | `422 UNSUPPORTED_LAUNCHER`. |
| 5 | Bad/expired per-run token on internal callback | `RunTokenService.validate` (§13) | `401 RUN_TOKEN_INVALID`. |
| 6 | Token `runId` ≠ path `runId` | `RunTokenService.validate` | `403 RUN_TOKEN_MISMATCH`. |
| 7 | Callback for a run already terminal | `run.status in TERMINAL` | `409 RUN_TERMINAL`. |
| 8 | Duplicate per-round result POST | `UNIQUE (fl_run_id, round_idx)` | idempotent no-op (`ON CONFLICT DO NOTHING`); return `202` (§6.3). |
| 9 | FL server crash / orphan (lease expired, executor MISSING) | reconciler `poll()` returns `MISSING` + lease expired | set `FAILED`, broadcast `/topic/status`, emit `rounds_lost`/orphan metric (C1-F4). |
| 10 | JVM restart mid-run | `reconcileOnBoot()` on `ApplicationReadyEvent` | claim expired leases, poll, reconcile — no orphans (C1-F4). |
| 11 | Demote/remove the last PLATFORM_ADMIN | `countByPlatformRole(PLATFORM_ADMIN) <= 1` (counts the enum, not `"ADMIN"`) | `409 LAST_ADMIN` (A1-F1 co-symptom now fires). |
| 12 | Demote/remove the last org OWNER | `countByOrgIdAndOrgRole(orgId, OWNER) <= 1` | `409 LAST_OWNER`. |
| 13 | Login by a `PENDING` (unverified) account | `status != ACTIVE` at auth | `403 ACCOUNT_NOT_VERIFIED` (A1-F5). |
| 14 | Register/login/forgot flood | Bucket4j bucket empty | `429 RATE_LIMITED` (A1-F5). |
| 15 | DTO bean-validation failure | `MethodArgumentNotValidException` | `400 VALIDATION_FAILED` + `fieldErrors` (§12). |
| 16 | Typo'd role/status in admin request | Jackson enum `valueOf` fails | `400 VALIDATION_FAILED` (A1-F1 class can't recur). |
| 17 | Cross-tenant read attempt | `requireProjectParticipant`/`requireOrgMember` throws | `403 FORBIDDEN` (A1-F9, B4 §2.1). |
| 18 | SUBSCRIBE to a non-participant project topic | `JwtChannelInterceptor` SUBSCRIBE branch throws | frame rejected (A1 WS gap). |
| 19 | Manifest requested before run started | manifest row absent | `409 MANIFEST_NOT_READY`. |
| 20 | Unhandled / unexpected | catch-all in `GlobalExceptionHandler` | `500 INTERNAL_ERROR`, generic message, details only in logs (never a stack trace on the wire). |

**Launcher failure semantics:** `FlServerLauncher.launch` throwing (k8s API down, ECS error) → `FlRunService`
marks the just-written run `FAILED` with `errorMessage`, broadcasts, and returns the envelope; it does NOT
leave a `PENDING` lease dangling. `stop()` on an already-gone executor is treated as success (idempotent).

---

## 10. Testing strategy

Frameworks: JUnit 5 + Spring Boot Test + Mockito (`spring-boot-starter-test`, Boot BOM); Testcontainers
(PostgreSQL 17.10) for migration/integration tests; ArchUnit for the layering rule.

| Test (named) | Type | What it asserts |
|---|---|---|
| `PlatformRoleAuthorizationTest` | integration | A bootstrap `PLATFORM_ADMIN` (the **enum**, seeded as `PLATFORM_ADMIN`, not the literal `"ADMIN"`) can call every `/api/admin/**` route → `200`. **This is the test that v1 got wrong (A1-F1):** it must seed `PLATFORM_ADMIN`, never `"ADMIN"`. |
| `AdminRoleEnumValidationTest` | unit | `UpdateUserRoleRequest` with `"ADMINN"` → `400 VALIDATION_FAILED` (Jackson enum rejects the typo). |
| `LastAdminGuardTest` | integration | Demoting the only `PLATFORM_ADMIN` → `409 LAST_ADMIN` (the guard counts the enum). |
| `OrgScopeIsolationTest` | integration | Org A user reading an Org B project → `403 FORBIDDEN`; `GET /api/projects` returns only the caller's org's projects (A1-F9, B4 §2.1). |
| `DiscoveryOrgScopeTest` | integration | Discovery returns only PUBLIC projects in the caller's org(s); Org A cannot enumerate Org B project names (B4 §2.1). |
| `StartRunConcurrencyTest` | integration (Testcontainers) | Two concurrent `POST .../runs` for one project: exactly one `202`, the other `409 RUN_ALREADY_ACTIVE`; exactly one `fl_runs` active row (A1-F4). |
| `OrgQuotaTest` | integration | Starting beyond `app.run.org-quota-max-active` → `409 ORG_QUOTA_EXCEEDED` (B6 §6). |
| `RunTokenValidationTest` | unit | Bad signature → `401`; expired → `401`; `runId` mismatch → `403`; valid → `RunContext` with the token's triple (§13, A1-F6). |
| `InternalResultIdempotencyTest` | integration | Re-POSTing the same `(runId, round)` → `202` and exactly one `round_results` row (§6.3). |
| `ReconcilerOrphanReapTest` | integration | A `RUNNING` run with an expired lease and a `MISSING` executor → reconciler sets `FAILED` and broadcasts (C1-F4). |
| `ReconcilerBootReconcileTest` | integration | After a simulated restart, `reconcileOnBoot` adopts expired leases and reconciles state (no phantom RUNNING) (C1-F4). |
| `JwtClaimsHardeningTest` | unit | Minted token carries `iss`/`aud`/`jti`; validation rejects a token with the wrong `iss`/`aud` and honors clock skew (A1-F7). |
| `WsSubscribeAuthzTest` | integration | A non-participant SUBSCRIBE to `/topic/results/{projectId}` is rejected at the frame (A1 WS gap). |
| `UsersEndpointNoPiiTest` | integration | `/api/users` (list-all) does not exist (404/no mapping); only `/api/users/me` returns a DTO, never the JPA entity (A1-F3). |
| `MigrationV1toV8PostgresTest` | integration (Testcontainers PG 17.10) | V1→V8 applies cleanly on real Postgres; `audit_events.metadata` is JSONB; the partial unique index exists (A1-F10, `03-DATA-MODEL.md §6`). |
| `ErrorEnvelopeContractTest` | integration | Every error path returns the §12 envelope with a `code` from the §12.1 registry; never a raw stack trace. |
| `ControllerRepositoryArchTest` | ArchUnit | No controller references any `*Repository` type (the seam that would have caught F3/F6 — `A1-backend.md:41`). |

**What NOT to do:** do not disable Flyway anywhere except `test`; do not test against H2 for migration
correctness (that is exactly the v1 gap — H2 emulation hid the `CLOB` defect, `A1-F10`).

---

## 11. Build & run (this unit in isolation)

```bash
# from repo root; working directory for all backend commands:
#   backend/fl-platform-api

# 1. Run locally (dev profile, H2 file-mode, LocalProcessLauncher, H2 console on)
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun          # :8081

# 2. Unit + integration tests (in-memory H2, Flyway disabled, create-drop)
SPRING_PROFILES_ACTIVE=test ./gradlew test

# 3. A single test
./gradlew test --tests "com.federated.fl_platform_api.StartRunConcurrencyTest"

# 4. Migration validation against real Postgres 17.10 (Testcontainers profile — NEW, A1-F10)
./gradlew test --tests "com.federated.fl_platform_api.MigrationV1toV8PostgresTest"

# 5. Full build (compile + all tests)
./gradlew build

# 6. Fat JAR
./gradlew bootJar                                     # build/libs/*.jar

# 7. Verify health + metrics exposure
curl -s http://localhost:8081/actuator/health         # {"status":"UP"}
curl -s http://localhost:8081/actuator/prometheus | head   # micrometer metrics present
```

**Pre-run checklist (must be true before `production` boot):** `APP_JWT_SECRET`,
`app.internal.run-token-secret`, `CORS_ALLOWED_ORIGINS`, and `SPRING_DATASOURCE_*` are set; the base
profile refuses to boot otherwise (A1-backend.md:153). Copy `V4`/`V5` into
`src/main/resources/db/migration/` first (`03-DATA-MODEL.md §1`) or Flyway's chain is incomplete.

---

## 12. Reasoning & alternatives (what was rejected and why)

| Decision | Rejected alternative | Why (audit) |
|---|---|---|
| Keep Spring Boot 3.5; salvage the layering | Rewrite the backend in Go/FastAPI | "Don't rewrite the healthy organ"; the Java control plane is the most valuable, least-broken layer (`A1-backend.md:14`, `02-TECH-STACK.md §1.1`). |
| Decompose `ProjectService` into `ProjectService` / `FlRunService` / `ProjectLogService` / `DiscoveryService` | Keep the 438-line god-object | Field-injects 11 collaborators, mixes CRUD + lifecycle + paging + notifications; untestable without Spring (`A1-backend.md:43`). |
| Role **enums** + DB `CHECK` + enum-validated DTOs | Keep string roles | `ADMIN` vs `PLATFORM_ADMIN` drift 403'd the bootstrap admin and a test masked it (`A1-F1`). Enums + `CHECK` make the class structurally impossible. |
| Durable `fl_runs` lease + reconciler | In-memory `ConcurrentHashMap<UUID,Process>` | A JVM restart orphaned every Python child with no DB record; `/stop` found nothing (`A1-F2.3`, `C1-F4`, verdict **kill**). |
| Partial unique index closes the start race | `@Transactional` + pessimistic row lock | The index is declarative, needs no application lock, and survives multiple replicas; the lock only narrows the v1 race, the index eliminates it (`A1-F4`, `04-API-CONTRACTS.md §4.3`). |
| `202 Accepted` + reconciler-driven status | v1's blocking 3s startup probe + captured stdout | "Didn't crash in 3s" is not "healthy"; a slow boot that crashes at 4s reported RUNNING (`A1-F2`, `C1-F3`). |
| Per-run scoped token (HMAC, bound to runId/projectId/orgId) | One global `APP_INTERNAL_API_KEY` | Any task could write any project's results — a multi-tenant integrity break (`A1-F6`, `B4 §2.1`). Token is not a table (signed/validated; `03-DATA-MODEL.md §8`). |
| `org_id`-scoped `AuthorizationService` chokepoint + org-scoped repository queries | v1 project-only authz | `AuthorizationService` "never references `org_id`"; cross-org access and a discovery metadata leak (`A1-F9`, `B4 §2.1`). |
| `FlServerLauncher` interface; control plane commands, never hosts, FL servers | Continue spawning `python fl_server.py` from the JVM | 11-port cap, no isolation, no HA, state lost on restart (`A1-F2`, `B6 §1`). `LocalProcessLauncher` kept dev-only. |
| RDS Postgres 17.10 everywhere outside dev/test + Testcontainers CI | H2 file-mode in `ec2demo` | H2 emulation hid the `CLOB`/`TIMESTAMP` defects; H2 is single-writer (`A1-F10`, `B6 §4`). |
| Pre-signed URL artifact broker; bytes never transit the JVM | Stream model bytes through the API | v1's `getvalue()`/slice 2× memory blowup on large models (`B6 §1`, `A3 M4`). |
| Logs off the FL-progress critical path (progress via internal API → STOMP) | Stream stdout through the JVM daemon thread | A wedged reader fills the 64KB pipe and blocks the FL `write()` mid-round (`A1-F2.4`, `C1-F7`). |

---

## 13. Build task checklist for the ~30B local model (ordered, dependency-first)

Each task is one file/feature with a done-condition. Do them in order; later tasks depend on earlier ones.

1. **Migrations present.** Copy `V4`/`V5` from `build/resources` into `src/main/resources/db/migration/`;
   author `V6`/`V7`/`V8` exactly per `03-DATA-MODEL.md §5`. **Done:** `MigrationV1toV8PostgresTest`
   (Testcontainers PG 17.10) is green and `audit_events.metadata` is JSONB.
2. **Enums.** Create `PlatformRole`, `OrgRole`, `ProjectRole`, `RunStatus`, `RunStrategy`, `LauncherKind`,
   `ProjectStatus`, `UserStatus`, `ArtifactKind`, `Modality`, `PartitionerKind`, `AuditAction` in
   `domain/enums`. **Done:** names match the `CHECK` sets in §5.
3. **JPA entities + composite ids.** Implement `domain/*` per §5.8; JSONB via
   `@JdbcTypeCode(SqlTypes.JSON)`. **Done:** boot in `dev` with JPA `validate` passes (no schema mismatch).
4. **Repositories.** Implement `repository/*` with the org-scoped query methods of §7.3. **Done:**
   `findByOrgIdIn`, `findNonTerminal`, `countByOrgIdAndStatusIn`, `countByPlatformRole` exist.
5. **Error envelope.** `dto/error/*`, `web/GlobalExceptionHandler.java`, the `ErrorCode` enum mirroring
   §12.1. **Done:** `ErrorEnvelopeContractTest` green.
6. **Security: JWT.** `JwtTokenProvider` (iss/aud/jti/skew), `JwtCookieFilter`,
   `CustomUserDetailsService` (emits `ROLE_PLATFORM_ADMIN`), `PlatformUserPrincipal`. **Done:**
   `JwtClaimsHardeningTest` green.
7. **Security: config.** `SecurityConfig` (CORS allowlist, CSRF posture, cookie attrs, route authz with
   `hasRole('PLATFORM_ADMIN')`). **Done:** unauthenticated `/api/admin/**` → 401/403.
8. **AuthorizationService.** Implement §5.2 with `org_id` scoping; PLATFORM_ADMIN bypass `@Auditable`.
   **Done:** `OrgScopeIsolationTest` green.
9. **Audit.** `@Auditable`, `AuditAspect` (proceeds-then-writes, Jackson metadata), `AuditAction`.
   **Done:** an audited mutation writes one `audit_events` row; caller rollback rolls back the row.
10. **AuthService + AuthController.** register/login/verify/forgot/reset; `emailVerified` gate;
    `EmailService` + `LoggingEmailService`; Bucket4j `RateLimitConfig`. **Done:** PENDING login → `403
    ACCOUNT_NOT_VERIFIED`; flood → `429 RATE_LIMITED`.
11. **OrgService + OrgController.** `/api/orgs/*` (§8.1) incl. `LAST_OWNER` guard. **Done:** create-org
    makes the creator OWNER; last-owner removal → `409 LAST_OWNER`.
12. **ProjectService + ProjectController.** CRUD/visibility only; `orgId` required on create, validated
    against membership (§3). **Done:** create in a foreign org → `403`; duplicate name in org → `409
    PROJECT_NAME_TAKEN`.
13. **DiscoveryService.** Org-scoped PUBLIC-project discovery. **Done:** `DiscoveryOrgScopeTest` green.
14. **DatasetService + DatasetController.** registry CRUD by content hash (§8.2). **Done:** duplicate
    `content_hash` → `409 VERSION_SHA_EXISTS`.
15. **ArtifactService + ArtifactController.** pre-signed URL broker; bytes never transit JVM (§9). **Done:**
    `upload-url` returns a pre-signed PUT; register with an existing `sha256` → `409 SHA_EXISTS`.
16. **FlServerLauncher interface + records.** `orchestration/FlServerLauncher.java`, `FlRunSpec`,
    `LaunchResult`, `RunState` (§5.6). Stub `LocalProcessLauncher` for dev. **Done:** compiles; dev launch
    returns a `LaunchResult`.
17. **RunTokenService + RunTokenAuthFilter.** mint/validate per §13. **Done:** `RunTokenValidationTest`
    green.
18. **FlRunService + RunController.** start/stop/get/list/status/manifest/checkpoints (§4); quota check;
    partial-unique-index conflict → `409 RUN_ALREADY_ACTIVE`. **Done:** `StartRunConcurrencyTest` and
    `OrgQuotaTest` green.
19. **ResultIngestService + InternalRunController.** `/api/internal/runs/{runId}/*` (§5); idempotent
    per-round UPSERT. **Done:** `InternalResultIdempotencyTest` green.
20. **RunReconcilerService.** `@Scheduled reconcileTick` + `reconcileOnBoot`; lease renew/claim; broadcast.
    **Done:** `ReconcilerOrphanReapTest` and `ReconcilerBootReconcileTest` green.
21. **WebSocketConfig + JwtHandshakeInterceptor + JwtChannelInterceptor + WebSocketService.** `/ws-logs`,
    `/topic/*`, SUBSCRIBE-frame authz (§11). **Done:** `WsSubscribeAuthzTest` green.
22. **UserService + UserController.** `/api/users/me*` DTOs only; **no list-all** (A1-F3). **Done:**
    `UsersEndpointNoPiiTest` green.
23. **AdminService + AdminController.** `/api/admin/*` (§7); `LAST_ADMIN` counts the enum. **Done:**
    `PlatformRoleAuthorizationTest`, `AdminRoleEnumValidationTest`, `LastAdminGuardTest` green.
24. **BootstrapRunner.** first PLATFORM_ADMIN + Platform org; never logs the password in deployed profiles
    (B4 §2.2). **Done:** boot with `APP_BOOTSTRAP_ADMIN_EMAIL` creates exactly one PLATFORM_ADMIN.
25. **ArchUnit rule.** `ControllerRepositoryArchTest`: controllers never reference repositories. **Done:**
    rule green across the whole `controller` package.
26. **Observability wiring.** add `micrometer-registry-prometheus` + `micrometer-tracing-bridge-otel`;
    expose `prometheus`; FL-run counters (`runs_started`, `rounds_lost{reason}`, orphan-reap). **Done:**
    `/actuator/prometheus` shows the FL-run metrics.

---

*End of 10-LLD-backend-control-plane.md. All claims about existing code or audit findings cite the audit
reports under `docs/audit/2026-05-29/` (A1-backend.md, B4-security-compliance.md, C1-reliability-sre.md,
B6-scale-cost.md) or the foundation docs (02-TECH-STACK.md, 03-DATA-MODEL.md, 04-API-CONTRACTS.md).
Uncertainties (Bucket4j/password-encoder pins absent from 02-TECH-STACK.md; the ec2demo H2→Postgres
change) are flagged inline rather than papered over.*
