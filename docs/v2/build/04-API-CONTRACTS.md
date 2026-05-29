# 04 — API Contracts (FedLearn Platform v2)

**Document type:** Production build specification — interface surface.
**Audience:** an implementing language model (~30B parameters). Every contract here is pre-decided. Where a body must be implemented, the signature/shape is given; you fill the body, you do not redesign the contract.
**Status:** authoritative for v2. Supersedes all v1 endpoint shapes.
**Date authored:** 2026-05-29.

> **Abbreviation key (first-use expansions, repeated here for self-containment):**
> API (Application Programming Interface), REST (Representational State Transfer), gRPC (Google Remote Procedure Call), STOMP (Simple Text Oriented Messaging Protocol), WS (WebSocket), JWT (JSON Web Token), JSON (JavaScript Object Notation), HTTP (HyperText Transfer Protocol), HTTPS (HTTP Secure), URL (Uniform Resource Locator), UUID (Universally Unique Identifier), RBAC (Role-Based Access Control), RLS (Row-Level Security), FL (Federated Learning), DeComFL (Dimension-Free Communication Federated Learning — the platform's zeroth-order optimization strategy; note: the v1 wiki mis-expanded this as "Decomposed", which is wrong per the paper, see `docs/audit/2026-05-29/B1-paper-alignment.md:33`), ZO (Zeroth-Order), RNG (Random Number Generator), CRUD (Create-Read-Update-Delete), DTO (Data Transfer Object), S3 (Simple Storage Service), MinIO (the self-hosted S3-compatible object store), RDS (Relational Database Service), MLflow (the self-hosted experiment/model registry), OTel (OpenTelemetry), W3C (World Wide Web Consortium), CN (Common Name, of a TLS certificate), TLS (Transport Layer Security), mTLS (mutual TLS), DP (Differential Privacy), CSP (Content-Security-Policy), HSTS (HTTP Strict Transport Security), SBOM (Software Bill of Materials), DLG (Deep Leakage from Gradients), SLA (Service-Level Agreement), HIPAA (Health Insurance Portability and Accountability Act), SOC 2 (System and Organization Controls 2), k8s (Kubernetes), ECS (Elastic Container Service), ARN (Amazon Resource Name), HMAC (Hash-based Message Authentication Code), sha256 (Secure Hash Algorithm 256-bit), CSV (Comma-Separated Values), ECG (Electrocardiogram), gzip (GNU zip compression), lz4 (the LZ4 compression codec).

---

## 0. How to read this document

This file defines **the contracts that decouple the five deployable units** (control-plane API, Python FL framework, React frontend, Tauri desktop, mobile C++ core). It contains four contract families:

1. **REST API** (§2–§9) — browser/desktop/mobile ⇄ Spring Boot control plane. HTTP/JSON, cookie-only JWT auth.
2. **gRPC** (§10) — FL clients (Python desktop, Docker, native C++ mobile) ⇄ the long-running FL server. Package `fedlearn.v2`.
3. **STOMP-over-WebSocket** (§11) — Spring Boot ⇄ browser, for live logs/results/status.
4. **Cross-cutting** — the standard error envelope (§12), the per-run scoped result token (§13), and the W3C `traceparent` propagation contract (§14).

**Reasoning is inline.** Every nontrivial contract choice states *why this shape and not the alternative*, tied to the v2 audit findings (cited as `file:line` or by finding id, e.g. `A1-F6`).

**Versioning rule (locked):** REST is versioned by **route prefix discipline** (`/api/...`, no `/v2/` segment — the whole backend is v2; v1 is retired, not co-hosted). gRPC is versioned by **proto package** (`fedlearn.v2`); the v1 package `fedlearn.v1` is dead. `buf` is the single source of truth for the proto with a breaking-change gate (see `README.md:56`).

---

## 1. Global conventions (apply to every REST endpoint)

| Concern | Rule | Reasoning |
|---|---|---|
| Base path | All control-plane routes live under `/api`. | Single nginx/ALB routing prefix; everything else (`/ws-logs`, `/actuator/**`) is non-`/api`. |
| Content type | Request and response bodies are `application/json; charset=utf-8` unless a row says otherwise (log export is `text/plain`). | One serializer (Jackson) everywhere. |
| Auth transport | **HttpOnly cookie `jwtToken` only.** No `Authorization: Bearer`. Every browser/desktop call sets `withCredentials: true`. | v1 posture is textbook-correct and salvaged (`A2`, `README.md:119`). A JS-readable token is an XSS exfiltration target; the cookie is `HttpOnly` + `SameSite` + `Secure`. |
| Auth cookie attributes | `HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600`. `Secure` is controlled by `app.auth.cookie.secure` (true outside `dev`). | Matches existing `AuthController.java:101-107`; `SameSite=Strict` default tightened from v1's `Lax`. |
| Internal-callback auth | The FL server → backend callbacks (`/api/internal/**`) use a **per-run scoped token** in header `Authorization: Bearer flrun_<...>` (§13), **not** a cookie and **not** the v1 global `X-Internal-Api-Key`. | v1's single global `APP_INTERNAL_API_KEY` lets any task write results for any project — broken object-level auth, `A1-F6` / `ResultsController.java:38-65`. Per-run tokens close it. |
| Tenancy scoping | Every tenant-owned read/write is filtered by `org_id` derived from the caller's JWT, enforced at the repository/RLS layer — never trust an `org_id` from the request body. | v1 `AuthorizationService` never checks `org_id` → cross-org leak (`A1-F9`, `B4`, `README.md` R8). |
| IDs on the wire | `users.id` is a `Long` (BIGINT) serialized as a JSON **number**. `organizations.id`, `projects.id`, `fl_runs.id`, `round_results.id` (the V7 table that replaces v1 `round_result`), `datasets.id`, `artifact` ids are **UUID strings** (canonical lowercase 8-4-4-4-12). | Mixed key strategy is inherited from V5 (`A1-F8`, the project conventions identity section). v2 keeps it but documents it explicitly so the local model never guesses a type. New top-level entities (`fl_runs`, `datasets`) are **UUID**. |
| Timestamps | ISO-8601 UTC with `Z` suffix, e.g. `2026-05-29T16:32:04.123Z`. Java `Instant`, serialized by Jackson `WRITE_DATES_AS_TIMESTAMPS=false`. | One temporal format across REST/STOMP/MLflow. |
| Pagination | List endpoints that can be large take `?page=<int,0-based>&size=<int>`; `size` is server-clamped (hard max 500). Response is a bare JSON array (not a `Page` envelope) for parity with v1 `ProjectController.getProjectLogs`. | Matches `ProjectController.java:72-77`; keeps the frontend `TanStack Query` keys simple. |
| Errors | Every non-2xx returns the **standard error envelope** (§12). Never a raw stack trace, never a bare string. | v1 has two error contracts coexisting (`A1` "Two error-handling contracts coexist"); v2 collapses to one. |
| Validation | Bean Validation (`jakarta.validation`) on every request DTO; failures → `400` with `code=VALIDATION_FAILED` and a `fieldErrors` map. The frontend re-validates with Zod at the wire boundary. | Defense in depth; the local model implements both sides from the same field table. |
| Idempotency of mutations | `POST /api/projects/{id}/start` is **not** idempotent but is **conflict-guarded**: a second concurrent start returns `409 RUN_ALREADY_ACTIVE` via the `fl_runs` partial-unique index, not a duplicate server. | Closes the v1 check-then-act race (`A1-F4`) declaratively. |

### 1.1 Role model (locked enum — used by every auth requirement column)

v1's bug was string roles drifting (`ADMIN` vs `PLATFORM_ADMIN`, the bootstrap admin 403'd from every admin route, `A1-F1`). v2 collapses to three orthogonal **enums**, all per-user, exactly as the V5 identity model (the project conventions "Identity layers (V5)"):

| Layer | Enum | Values | Meaning |
|---|---|---|---|
| Platform | `PlatformRole` | `USER`, `PLATFORM_ADMIN` | `PLATFORM_ADMIN` bypasses org-membership checks. The bootstrap admin is `PLATFORM_ADMIN`. |
| Organisation | `OrgRole` | `OWNER`, `ADMIN`, `MEMBER` | Per-tenant. Carried in `organization_memberships(org_id UUID, user_id BIGINT, org_role)`. |
| Project | `ProjectRole` | `MEMBER`, `CLIENT` | Per-project, plus an implicit owner via `projects.user_id`. Carried in `project_memberships`. |

**Auth-requirement notation used in the tables below:**
- `PUBLIC` — no auth (permitAll).
- `AUTH` — any authenticated user (valid `jwtToken`).
- `ORG_MEMBER(p)` — authenticated AND a member of the org that owns resource `p` (any `OrgRole`).
- `ORG_ADMIN(p)` — authenticated AND `OrgRole ∈ {OWNER, ADMIN}` of the org that owns `p`, OR `PlatformRole=PLATFORM_ADMIN`.
- `PROJECT_PARTICIPANT(p)` — owner of project `p` (`projects.user_id`) OR a `project_memberships` row OR `ORG_ADMIN(p)` OR `PLATFORM_ADMIN`.
- `PLATFORM_ADMIN` — `PlatformRole=PLATFORM_ADMIN` only.
- `RUN_TOKEN(r)` — a valid per-run scoped token bound to run `r` (§13); used only on `/api/internal/**`.

`@PreAuthorize` SpEL must check the enum, never a literal string. Example for an admin route: `@PreAuthorize("hasRole('PLATFORM_ADMIN')")` where `CustomUserDetailsService` emits authority `ROLE_PLATFORM_ADMIN` — these must agree (the v1 break was exactly this disagreement).

---

## 2. Auth API — `/api/auth/*`

Cookie-only JWT. The login body returns identity only; the token is never in the body.

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/auth/register` | `PUBLIC` (rate-limited) | `RegisterRequest` | `201` `RegisterResponse` | `400 VALIDATION_FAILED`, `409 USER_ALREADY_EXISTS`, `429 RATE_LIMITED` |
| `POST` | `/api/auth/login` | `PUBLIC` (rate-limited) | `LoginRequest` | `200` `MeResponse` + `Set-Cookie: jwtToken` | `400 VALIDATION_FAILED`, `401 BAD_CREDENTIALS`, `403 ACCOUNT_NOT_VERIFIED`, `429 RATE_LIMITED` |
| `GET` | `/api/auth/me` | `AUTH` (silent 401 probe) | — | `200` `MeResponse` | `401 NOT_AUTHENTICATED` |
| `POST` | `/api/auth/logout` | `AUTH` | — | `204` + `Set-Cookie: jwtToken` (Max-Age=0) | — |
| `POST` | `/api/auth/verify-email` | `PUBLIC` | `VerifyEmailRequest` | `200` `{ "verified": true }` | `400 VALIDATION_FAILED`, `400 TOKEN_INVALID`, `410 TOKEN_EXPIRED` |
| `POST` | `/api/auth/password/forgot` | `PUBLIC` (rate-limited) | `ForgotPasswordRequest` | `202` `{ "status": "accepted" }` (always, to avoid account enumeration) | `400 VALIDATION_FAILED`, `429 RATE_LIMITED` |
| `POST` | `/api/auth/password/reset` | `PUBLIC` | `ResetPasswordRequest` | `200` `{ "reset": true }` | `400 VALIDATION_FAILED`, `400 TOKEN_INVALID`, `410 TOKEN_EXPIRED` |

**Reasoning on the new endpoints vs v1:** v1 had register/login/me/logout only and never enforced `emailVerified` at login (`A1-F5`). v2 adds the verify/forgot/reset trio so `status=PENDING` accounts cannot log in until verified, and adds rate limiting (Bucket4j) on the three unauthenticated mutating endpoints. The 401 on `/me` is deliberately distinct from 401 elsewhere — the Axios interceptor swallows 401 *only* on `/me* so the SPA can probe "am I logged in?" without a redirect loop (preserved from v1, `AuthController.java:131-141`).

### 2.1 Request/response JSON shapes (exact)

```jsonc
// RegisterRequest  (POST /api/auth/register)
{
  "username": "string, 3..50 chars, required",
  "email":    "string, valid email, <=100 chars, required",
  "password": "string, 8..100 chars, required"   // v2 raises v1 min from 6 to 8
}

// RegisterResponse  (201)
{
  "message": "User registered successfully. Check your email to verify.",
  "userId":  1234,            // JSON number (users.id is BIGINT/Long)
  "status":  "PENDING"        // PENDING until email verified; never ACTIVE on register
}

// LoginRequest  (POST /api/auth/login)
{
  "username": "string, required (accepts username OR email)",
  "password": "string, required"
}

// MeResponse  (returned by /login body and GET /api/auth/me)
{
  "userId":       1234,
  "username":     "alice",
  "email":        "alice@example.com",
  "platformRole": "USER",                 // PlatformRole enum
  "orgs": [                               // every org the user belongs to + their role
    { "orgId": "0f1c...uuid", "orgName": "Acme Health", "orgRole": "ADMIN" }
  ],
  "emailVerified": true
}

// VerifyEmailRequest
{ "token": "opaque-url-safe-token, required" }

// ForgotPasswordRequest
{ "email": "string, valid email, required" }

// ResetPasswordRequest
{ "token": "opaque-url-safe-token, required",
  "newPassword": "string, 8..100 chars, required" }
```

**Reasoning — `MeResponse` shape change vs v1:** v1 returned `{username, email, role}` where `role` was the drifted string (`AuthController.java:114-118`). v2 returns the **typed `platformRole` enum** plus the user's `orgs` array so the frontend can render org-scoped UI without a second round-trip. The frontend's Zod schema (`MeResponseSchema`) validates this exact shape at the wire boundary (`README.md:53`).

---

## 3. Projects API — `/api/projects`

A project is the long-lived configuration object (model type, owning org). A **run** (`fl_runs`) is one execution of that project. v1 conflated the two (the project carried `serverPort`/`status`); v2 separates them — start/stop now operate on runs (§4) but keep project-scoped convenience routes for the dashboard.

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/projects` | `ORG_MEMBER` (org from body) | `CreateProjectRequest` | `201` `ProjectResponseDto` | `400 VALIDATION_FAILED`, `403 FORBIDDEN`, `404 ORG_NOT_FOUND`, `409 PROJECT_NAME_TAKEN` |
| `GET` | `/api/projects` | `AUTH` | — | `200` `ProjectResponseDto[]` (org-scoped to caller) | — |
| `GET` | `/api/projects/{projectId}` | `PROJECT_PARTICIPANT` | — | `200` `ProjectResponseDto` | `403 FORBIDDEN`, `404 PROJECT_NOT_FOUND` |
| `PATCH` | `/api/projects/{projectId}` | `ORG_ADMIN` | `UpdateProjectRequest` | `200` `ProjectResponseDto` | `400`, `403`, `404`, `409 RUN_ACTIVE` (can't edit config mid-run) |
| `DELETE` | `/api/projects/{projectId}` | `ORG_ADMIN` | — | `200` `{ "projectId": "...", "message": "Project deleted successfully" }` | `403`, `404`, `409 RUN_ACTIVE` |
| `GET` | `/api/projects/{projectId}/results` | `PROJECT_PARTICIPANT` | — | `200` `RoundResultDto[]` (ordered by `serverRound`) | `403`, `404` |
| `GET` | `/api/projects/{projectId}/logs` | `PROJECT_PARTICIPANT` | `?page&size` | `200` `ServerLogDto[]` | `403`, `404` |
| `GET` | `/api/projects/{projectId}/logs/export` | `PROJECT_PARTICIPANT` | — | `200` `text/plain` attachment | `403`, `404` |

**Removed vs v1:** the legacy `POST /api/projects/{id}/delete` alias (`ProjectController.java:123-127`, `@Deprecated`) is **deleted** in v2 — there is exactly one delete verb (`DELETE`). Reasoning: a second mutating verb on the same resource is an audit-coverage and CSRF surface with no caller left after the desktop/web migration.

### 3.1 JSON shapes (exact)

```jsonc
// CreateProjectRequest  (POST /api/projects)
{
  "name":           "string, non-empty, required",
  "orgId":          "uuid, required",        // NEW in v2: project must belong to exactly one org
  "modelType":      "string, non-empty, required",  // e.g. "cnn", "transformer", "ecg_resnet"
  "modelName":      "string, optional",
  "optimizer":      "string, optional",
  "pretrainEpochs": 0,                        // integer >= 0, required
  "datasetVersionId": "uuid, optional"        // NEW in v2: pin to a dataset_versions row (§8)
}

// UpdateProjectRequest  (PATCH) — all fields optional; only provided fields change
{
  "name":             "string, optional",
  "modelName":        "string, optional",
  "optimizer":        "string, optional",
  "datasetVersionId": "uuid, optional"
}

// ProjectResponseDto  (response)
{
  "id":          "uuid",
  "name":        "Pneumonia Federation A",
  "orgId":       "uuid",
  "modelType":   "transformer",
  "modelName":   "distilbert-base",
  "optimizer":   "adam",
  "status":      "IDLE",          // project-level lifecycle: IDLE | HAS_ACTIVE_RUN | ARCHIVED
  "datasetVersionId": "uuid | null",
  "activeRunId": "uuid | null",   // NEW: the currently RUNNING/STARTING run, if any
  "createdAt":   "2026-05-29T16:00:00Z",
  "updatedAt":   "2026-05-29T16:05:00Z"
}
```

**Reasoning — `orgId` is required on create, never inferred from a default:** v1 set `Project.orgId` ad hoc (see recent commits `f7dbd37`/`dac7121` "set Project.orgId on creation to satisfy V5 NOT NULL"). v2 makes it an explicit required field validated against the caller's org membership, so a project can never be created orphaned or in an org the caller doesn't belong to (closes the F9 tenancy gap at create time). `serverPort` is **removed** from the project DTO — port is a run-level, executor-internal detail and is meaningless for the k8s/ECS launchers (v1 set it to `null` for ECS, `A1-F2`).

---

## 4. Runs API — `/api/projects/{projectId}/runs` and `/api/runs/*`

This is the **rebuilt** orchestration surface. v1 had `POST /{id}/start` and `POST /{id}/stop` that mutated the project and tracked a `Process` in an in-memory map lost on JVM restart (`A1-F2`). v2 makes a run a durable `fl_runs` row (the JVM is a stateless supervisor over the DB lease; a reconciler reconciles executor state → DB → STOMP).

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/projects/{projectId}/runs` | `ORG_MEMBER` | `StartRunRequest` | `202` `RunDto` (status `STARTING`) | `400 VALIDATION_FAILED`, `403`, `404 PROJECT_NOT_FOUND`, `409 RUN_ALREADY_ACTIVE`, `409 ORG_QUOTA_EXCEEDED`, `422 NO_DATASET_VERSION` |
| `GET` | `/api/projects/{projectId}/runs` | `PROJECT_PARTICIPANT` | `?page&size&status` | `200` `RunDto[]` (newest first) | `403`, `404` |
| `GET` | `/api/runs/{runId}` | `PROJECT_PARTICIPANT(run.project)` | — | `200` `RunDto` | `403`, `404 RUN_NOT_FOUND` |
| `POST` | `/api/runs/{runId}/stop` | `ORG_MEMBER(run.org)` | — | `202` `RunDto` (status `STOPPING`) | `403`, `404`, `409 RUN_NOT_STOPPABLE` (already terminal) |
| `GET` | `/api/runs/{runId}/status` | `PROJECT_PARTICIPANT` | — | `200` `RunStatusDto` (lightweight, poll-friendly) | `403`, `404` |
| `GET` | `/api/runs/{runId}/manifest` | `PROJECT_PARTICIPANT` | — | `200` `DeterminismManifestDto` | `403`, `404`, `409 MANIFEST_NOT_READY` |
| `GET` | `/api/runs/{runId}/checkpoints` | `PROJECT_PARTICIPANT` | — | `200` `CheckpointDto[]` | `403`, `404` |

**Reasoning — `202 Accepted`, not `200`:** start/stop are asynchronous against an external executor (k8s Job, ECS RunTask, or dev LocalProcess). The control plane writes the `fl_runs` lease row and hands off; it does **not** block until the FL server is reachable (v1 blocked 3 s and surfaced captured stdout — `A1-F2`/the project conventions FL lifecycle step 4). The client polls `GET /api/runs/{runId}/status` or subscribes to `/topic/status/{projectId}` (§11). `202` tells the caller "accepted, not yet running."

**Reasoning — quota enforcement before launch (`409 ORG_QUOTA_EXCEEDED`):** v1 had no per-org concurrency cap; lifting the 11-port limit without admission control is an unbounded cloud bill (`B6-1`, `README.md` R10). `FlRunService` checks the org's active-run count against its quota before writing the lease row.

### 4.1 JSON shapes (exact)

```jsonc
// StartRunRequest  (POST /api/projects/{projectId}/runs)
{
  "strategy":       "DeComFL",        // enum: "FedAvg" | "DeComFL"  (FedProx dropped in v2 — not in framework strategies)
  "numRounds":      20,               // int 1..1000  (v1 capped at 100; v2 raises for LLM federations)
  "minClients":     3,                // int 1..1000  — MINIMUM QUORUM to start/continue a round
  "roundDeadlineSeconds": 600,        // int >=1, default 600 — per-round wall-clock deadline (NO infinite hang)
  "launcher":       "KUBERNETES",     // enum: "KUBERNETES" | "ECS" | "LOCAL_PROCESS" (LOCAL_PROCESS rejected outside dev profile -> 422)
  "datasetVersionId": "uuid | null",  // overrides project default; required if project has none -> else 422 NO_DATASET_VERSION
  "hyperparameters": {                // strategy-specific; validated per strategy (see 4.2)
    "learningRate":  0.001,
    "mu":            0.001,           // DeComFL ZO smoothing radius
    "numPerturbations": 10,           // DeComFL P
    "numLocalSteps":    5,            // DeComFL K
    "gradEstimateMethod": "forward",  // "forward" | "central" (B1-H2)
    "dpEnabled":     false,           // Differential Privacy toggle
    "dpNoiseMultiplier": null,        // required (>0) iff dpEnabled
    "dpClipNorm":    null,            // required (>0) iff dpEnabled
    "robustClipTau": null             // optional scalar-magnitude clip for the robust-mean guard
  },
  "seed":           42                // int; recorded in the determinism manifest
}

// RunDto  (full)
{
  "id":            "uuid",
  "projectId":     "uuid",
  "orgId":         "uuid",
  "status":        "STARTING",  // see 4.3 state machine
  "strategy":      "DeComFL",
  "launcher":      "KUBERNETES",
  "executorRef":   "fl-run-7a3c... (k8s job name) | arn:aws:ecs:... | pid:dev-12345",
  "grpcEndpoint":  "fl-run-7a3c.fl.svc.cluster.local:50051 | null until RUNNING",
  "numRounds":     20,
  "minClients":    3,
  "roundDeadlineSeconds": 600,
  "currentRound":  0,
  "datasetVersionId": "uuid",
  "modelArtifactId":  "uuid | null",   // content-addressed final model (§9), null until SUCCEEDED
  "requestedByUserId": 1234,
  "seed":          42,
  "startedAt":     "2026-05-29T16:10:00Z | null",
  "finishedAt":    "null",
  "createdAt":     "2026-05-29T16:09:58Z",
  "errorMessage":  "null"               // populated only on FAILED
}

// RunStatusDto  (lightweight, for polling — no executor internals)
{
  "runId":        "uuid",
  "status":       "RUNNING",
  "currentRound": 7,
  "numRounds":    20,
  "activeClients": 3,
  "minClients":   3,
  "lastRoundAt":  "2026-05-29T16:18:00Z | null",
  "updatedAt":    "2026-05-29T16:18:02Z"
}
```

### 4.2 Hyperparameter validation (per strategy — implement exactly)

| Field | FedAvg | DeComFL | Rule |
|---|---|---|---|
| `learningRate` | required, `>0` | required, `>0` | — |
| `mu` | ignored (reject if present) | required, `>0` | ZO smoothing radius. |
| `numPerturbations` (P) | ignored | required, int `1..256` | DeComFL only. |
| `numLocalSteps` (K) | ignored | required, int `1..1000` | DeComFL only. |
| `gradEstimateMethod` | ignored | optional, `forward`\|`central`, default `forward` | `B1-H2`. |
| `dpEnabled` | optional bool | optional bool | DP-SGD on FedAvg; calibrated scalar-DP on DeComFL. |
| `dpNoiseMultiplier`/`dpClipNorm` | required iff `dpEnabled` | required iff `dpEnabled` | `B4`. |
| `robustClipTau` | optional | optional | Robust-mean/clipping guard on the scalar `g` (DeComFL) or per-coordinate (FedAvg). |

**Reasoning — DP and robust-clip fields exist from day one even if defaulted off:** the v1 README falsely claimed "Byzantine-robust" while the code did a plain mean (`B1-H3`, `README.md:106`). v2 deletes that false claim and instead ships real, opt-in DP + a robust guard. Putting the fields in the contract now means the framework and frontend agree on the shape before the feature is wired, and the local model never invents a different field name.

### 4.3 Run state machine (the `status` enum — locked)

```
PENDING ─► STARTING ─► RUNNING ─► SUCCEEDED
   │           │          │   │
   │           │          │   └─► STOPPING ─► STOPPED
   │           └──────────┴──────► FAILED
   └──────────────────────────────► FAILED   (admission/launch error)
```

| Status | Set by | Meaning |
|---|---|---|
| `PENDING` | `FlRunService` on `POST .../runs` | Lease row written; launcher not yet invoked. |
| `STARTING` | launcher | Executor (k8s Job / ECS task / dev process) submitted, gRPC not yet reachable. |
| `RUNNING` | reconciler when gRPC endpoint is healthy AND `minClients` registered | Rounds proceeding. |
| `STOPPING` | `FlRunService` on `/stop` | `stop(executorRef)` issued; awaiting confirmation. |
| `SUCCEEDED` | reconciler / internal callback `…/finished` | All rounds complete; final model artifact written. |
| `STOPPED` | reconciler | Executor confirmed terminated after a stop request. |
| `FAILED` | reconciler / launcher | Executor exited non-zero, deadline-killed, or admission error; `errorMessage` set. |

The `fl_runs` table has a **partial unique index** `(project_id) WHERE status IN ('PENDING','STARTING','RUNNING')` so a second concurrent `POST .../runs` for the same project fails with `409 RUN_ALREADY_ACTIVE` declaratively (closes `A1-F4`). The reconciler is the only writer of `RUNNING`/`SUCCEEDED`/`STOPPED`/`FAILED` from executor polling; the JVM holds no in-memory `Process` map (the v1 failure mode where a restart orphans children with no DB record — `A1-F2.3`).

### 4.4 Auxiliary run DTOs

```jsonc
// DeterminismManifestDto  (GET /api/runs/{runId}/manifest)
// The reproducibility contract: anyone can re-run this run byte-for-byte.
{
  "runId":            "uuid",
  "seed":             42,
  "strategy":         "DeComFL",
  "hyperparameters":  { /* exact StartRunRequest.hyperparameters echo */ },
  "torchVersion":     "2.12.0",        // pins RNG (CPU-canonical) parity, B1-C2 / spec §6
  "numpyVersion":     "1.26.4",
  "frameworkGitSha":  "abc1234...",
  "datasetVersionId": "uuid",
  "datasetSha256":    "hex64",
  "partitionRecipeId":"uuid",
  "modelInitSha256":  "hex64",          // hash of the initial global model
  "goldenVectorSha256":"hex64",         // RNG golden-vector fixture hash (spec §6)
  "createdAt":        "2026-05-29T16:10:00Z"
}

// CheckpointDto  (GET /api/runs/{runId}/checkpoints) — one per round, content-addressed in S3
{
  "runId":      "uuid",
  "round":      7,
  "artifactId": "uuid",
  "sha256":     "hex64",
  "sizeBytes":  104857600,
  "createdAt":  "2026-05-29T16:18:00Z"
}
```

**Reasoning — manifest + per-round checkpoint are first-class API objects:** v1 had no run entity, no checkpoint/resume, and a destructive in-place model save (`README.md` R14/R16, `C1`/`C3`). The determinism manifest (seed + library/dataset/model hashes) makes runs reproducible; the per-round content-addressed S3 checkpoint makes them resumable. Exposing both over REST lets the frontend show lineage and lets a reconnecting/rebuilding client (DeComFL) or a restarted server resume from the durable record.

---

## 5. Internal callbacks API — `/api/internal/*` (FL server → control plane)

These are called **by the FL server** (the spawned k8s/ECS/dev process), not by browsers. Auth is the **per-run scoped token** (§13) in `Authorization: Bearer flrun_<...>`, never a cookie, never the v1 global key. The token is bound to one `runId`; the path `runId` must match the token's `runId` or the call is `403`.

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/internal/runs/{runId}/results` | `RUN_TOKEN(runId)` | `RoundResultDto` | `202` (empty) | `401 RUN_TOKEN_INVALID`, `403 RUN_TOKEN_MISMATCH`, `404 RUN_NOT_FOUND`, `409 RUN_TERMINAL` |
| `POST` | `/api/internal/runs/{runId}/finished` | `RUN_TOKEN(runId)` | `RunFinishedDto` | `202` (empty) | `401`, `403`, `404`, `409 RUN_TERMINAL` |
| `POST` | `/api/internal/runs/{runId}/checkpoint` | `RUN_TOKEN(runId)` | `CheckpointReportDto` | `202` (empty) | `401`, `403`, `404` |
| `POST` | `/api/internal/runs/{runId}/status` | `RUN_TOKEN(runId)` | `RunStatusReportDto` | `202` (empty) | `401`, `403`, `404` |

**Reasoning — path moved from `/api/internal/results/{projectId}` to `/api/internal/runs/{runId}/results`:** v1 keyed callbacks on `projectId` and authorized only with the global key, so any task could POST results for any project (`A1-F6`, `ResultsController.java:38-65`). v2 keys on `runId` and binds the token to that run, so a task can write only its own run's telemetry. The control plane resolves `runId → projectId/orgId` server-side; the caller never asserts the project/org.

**Reasoning — per-round POST is incremental, not batched:** v1's producer existed but POSTed the whole history after the run finished, so the live chart was empty during training (`B3-01` resolution, `README.md` §3 item 6). v2 contract requires one `POST .../results` **per round, during the round loop**, immediately after aggregation/eval. The FL server must fire this best-effort (short timeout, try/except) so a telemetry failure never crashes the run (`B3` risk #9).

### 5.1 JSON shapes (exact)

```jsonc
// RoundResultDto  (POST /api/internal/runs/{runId}/results)  -- EXTENDED from v1
{
  "serverRound":          7,            // int, required
  "loss":                 0.2314,       // double | null
  "accuracy":             0.9012,       // double | null
  "gpuUtilization":       0.0,          // double | null (carried from v1)
  // --- DeComFL communication-cost wedge (NEW in v2, B3 §6.2) ---
  "uplinkBytes":          240,          // long | null  — bytes clients -> server this round
  "downlinkBytes":        240,          // long | null  — bytes server -> clients this round
  "scalarsTransmitted":   50,           // long | null  — K*P scalars this round (the O(K*P) proof)
  "modelParamCount":      66000000,     // long | null  — model dimension d (for dimension-free comparison)
  // --- per-round timing (NEW, feeds Grafana comm-vs-compute panel) ---
  "roundDurationSeconds":   4.2,        // double | null
  "aggregationSeconds":     0.1,        // double | null
  "activeClients":          3           // int | null
}

// RunFinishedDto  (POST .../finished)
{
  "finalStatus":      "SUCCEEDED",      // "SUCCEEDED" | "FAILED"
  "finalModelArtifactId": "uuid | null",// content-addressed final model in S3/MinIO
  "finalModelSha256": "hex64 | null",
  "totalRounds":      20,
  "errorMessage":     "null"            // set only when finalStatus=FAILED
}

// CheckpointReportDto  (POST .../checkpoint) -- per-round durable checkpoint pointer
{
  "round":      7,
  "artifactId": "uuid",                 // already uploaded to S3 by the FL server, content-addressed
  "sha256":     "hex64",
  "sizeBytes":  104857600
}

// RunStatusReportDto  (POST .../status) -- executor self-report between rounds
{
  "status":        "RUNNING",           // STARTING | RUNNING — terminal states use /finished
  "currentRound":  7,
  "activeClients": 3,
  "grpcEndpoint":  "host:port | null"   // first report carries the reachable endpoint
}
```

**Reasoning — `RoundResultDto` gains `uplinkBytes`/`downlinkBytes`/`scalarsTransmitted`/`modelParamCount`:** DeComFL's entire thesis is O(K·P) communication independent of model dimension `d`; v1's schema had no communication-cost column so the platform could not demonstrate its own differentiator (`B3` §6.2). These four fields back the Grafana/recharts "communication-cost panel" that visualizes "bytes-per-round vs equivalent FedAvg full-model bytes" — the customer-facing proof of the wedge. They are nullable so FedAvg runs (which don't transmit scalars) can omit `scalarsTransmitted`. **Schema impact:** the `V7__fl_runs_and_artifacts.sql` migration creates the `round_results` table (plural; it replaces the v1 `round_result` table) with these as nullable columns named `uplink_bytes`, `downlink_bytes`, `scalars_transmitted` (and `round_results.round_idx` is the `serverRound` column on the wire) — see `03-DATA-MODEL.md §5.2`.

---

## 6. Users API — `/api/users/*`

v1 leaked the JPA `User` entity (PII) from `/api/users` (`A1-F3`). v2 returns DTOs only and scopes everything to the caller.

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `GET` | `/api/users/me` | `AUTH` | — | `200` `MeResponse` (same shape as `/api/auth/me`) | `401` |
| `PATCH` | `/api/users/me` | `AUTH` | `UpdateProfileRequest` | `200` `MeResponse` | `400`, `409 USERNAME_TAKEN` |
| `POST` | `/api/users/me/password` | `AUTH` | `ChangePasswordRequest` | `204` | `400`, `401 WRONG_CURRENT_PASSWORD` |

```jsonc
// UpdateProfileRequest  (all optional)
{ "username": "string, 3..50, optional", "displayName": "string, optional" }

// ChangePasswordRequest
{ "currentPassword": "string, required", "newPassword": "string, 8..100, required" }
```

**Reasoning — `/api/users` (list-all) is deleted:** it duplicated `/api/admin/users` and returned raw entities (`A1-F3`). The only self-service user surface is `/api/users/me`; bulk user listing is admin-only (§7).

---

## 7. Admin API — `/api/admin/*`

Platform-administration surface. **Gated on the `PlatformRole=PLATFORM_ADMIN` enum** (the v1 bug was gating on the literal string `ADMIN` that production never produced — `A1-F1`).

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `GET` | `/api/admin/users` | `PLATFORM_ADMIN` | `?page&size&q` | `200` `AdminUserDto[]` | `403` |
| `GET` | `/api/admin/users/{userId}` | `PLATFORM_ADMIN` | — | `200` `AdminUserDto` | `403`, `404` |
| `PATCH` | `/api/admin/users/{userId}/role` | `PLATFORM_ADMIN` | `UpdateUserRoleRequest` | `200` `AdminUserDto` | `400`, `403`, `404`, `409 LAST_ADMIN` |
| `PATCH` | `/api/admin/users/{userId}/status` | `PLATFORM_ADMIN` | `UpdateUserStatusRequest` | `200` `AdminUserDto` | `400`, `403`, `404` |
| `GET` | `/api/admin/projects` | `PLATFORM_ADMIN` | `?page&size` | `200` `ProjectResponseDto[]` (all orgs) | `403` |
| `GET` | `/api/admin/audit` | `PLATFORM_ADMIN` | `?page&size&action&targetType&from&to` | `200` `AuditEventDto[]` | `403` |

```jsonc
// AdminUserDto  (no password, but more than MeResponse — admin context)
{
  "userId":        1234,
  "username":      "alice",
  "email":         "alice@example.com",
  "platformRole":  "USER",         // PlatformRole enum
  "status":        "ACTIVE",       // ACTIVE | PENDING | DISABLED
  "emailVerified": true,
  "lastLoginAt":   "2026-05-29T15:00:00Z | null",
  "createdAt":     "2026-05-01T00:00:00Z"
}

// UpdateUserRoleRequest
{ "platformRole": "PLATFORM_ADMIN" }   // enum-constrained: USER | PLATFORM_ADMIN; valueOf rejects typos -> 400

// UpdateUserStatusRequest
{ "status": "DISABLED" }               // ACTIVE | PENDING | DISABLED

// AuditEventDto
{
  "id":         "uuid",
  "action":     "RUN_START",      // AuditAction enum
  "actorUserId": 1234,
  "orgId":      "uuid | null",
  "targetType": "FlRun",
  "targetId":   "uuid | null",
  "metadata":   { },              // JSONB, free-form
  "occurredAt": "2026-05-29T16:09:58Z"
}
```

**Reasoning — `409 LAST_ADMIN` guard now actually fires:** v1's "cannot demote the last admin" check compared against the literal `ADMIN` string and so never triggered for a real `PLATFORM_ADMIN` (`A1-F1` co-symptom). v2 counts `PlatformRole=PLATFORM_ADMIN` rows. **Reasoning — `UpdateUserRoleRequest.platformRole` is enum-constrained:** Jackson deserializes to the `PlatformRole` enum so `valueOf` rejects any typo at the wire boundary (the v1 string-drift class can't recur). **Reasoning — `audit_events.metadata` is JSONB, not CLOB:** v1 used `CLOB` (H2-ism); the cutover to managed Postgres requires `TEXT/JSONB` (`README.md` §3 item 7, `A1` audit-log note).

---

## 8. Organisations & datasets

### 8.1 Organisations — `/api/orgs/*`

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/orgs` | `AUTH` (creator becomes `OWNER`) | `CreateOrgRequest` | `201` `OrgDto` | `400`, `409 ORG_NAME_TAKEN` |
| `GET` | `/api/orgs` | `AUTH` | — | `200` `OrgDto[]` (caller's orgs) | — |
| `GET` | `/api/orgs/{orgId}` | `ORG_MEMBER` | — | `200` `OrgDto` | `403`, `404` |
| `GET` | `/api/orgs/{orgId}/members` | `ORG_MEMBER` | — | `200` `OrgMemberDto[]` | `403`, `404` |
| `POST` | `/api/orgs/{orgId}/members` | `ORG_ADMIN` | `AddOrgMemberRequest` | `201` `OrgMemberDto` | `400`, `403`, `404 USER_NOT_FOUND`, `409 ALREADY_MEMBER` |
| `PATCH` | `/api/orgs/{orgId}/members/{userId}` | `ORG_ADMIN` | `UpdateOrgMemberRoleRequest` | `200` `OrgMemberDto` | `400`, `403`, `404`, `409 LAST_OWNER` |
| `DELETE` | `/api/orgs/{orgId}/members/{userId}` | `ORG_ADMIN` | — | `204` | `403`, `404`, `409 LAST_OWNER` |

```jsonc
// CreateOrgRequest
{ "name": "string, non-empty, required" }

// OrgDto
{ "id": "uuid", "name": "Acme Health", "createdAt": "2026-05-01T00:00:00Z" }

// OrgMemberDto
{ "userId": 1234, "username": "alice", "email": "alice@example.com", "orgRole": "ADMIN" }

// AddOrgMemberRequest  (add by email; OrgRole enum)
{ "email": "bob@example.com", "orgRole": "MEMBER" }

// UpdateOrgMemberRoleRequest
{ "orgRole": "ADMIN" }   // enum: OWNER | ADMIN | MEMBER
```

### 8.2 Datasets & partitions — `/api/datasets/*`

The dataset/partition registry is **new in v2** (`C2`, `README.md` §1.1). It replaces ad-hoc pickle caches and the removed `flwr-datasets`. Datasets and versions are content-addressed by sha256; a partition recipe is the deterministic non-IID split (the platform owns its own Dirichlet partitioner — `README.md` §1.1).

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/datasets` | `ORG_MEMBER` | `CreateDatasetRequest` | `201` `DatasetDto` | `400`, `403`, `409 DATASET_NAME_TAKEN` |
| `GET` | `/api/datasets` | `AUTH` | `?orgId` | `200` `DatasetDto[]` (org-scoped) | `403` |
| `GET` | `/api/datasets/{datasetId}` | `ORG_MEMBER` | — | `200` `DatasetDto` | `403`, `404` |
| `POST` | `/api/datasets/{datasetId}/versions` | `ORG_MEMBER` | `CreateDatasetVersionRequest` | `201` `DatasetVersionDto` | `400`, `403`, `404`, `409 VERSION_SHA_EXISTS` |
| `GET` | `/api/datasets/{datasetId}/versions` | `ORG_MEMBER` | — | `200` `DatasetVersionDto[]` | `403`, `404` |
| `POST` | `/api/datasets/{datasetId}/versions/{versionId}/partitions` | `ORG_MEMBER` | `CreatePartitionRecipeRequest` | `201` `PartitionRecipeDto` | `400`, `403`, `404` |
| `GET` | `/api/datasets/{datasetId}/versions/{versionId}/partitions` | `ORG_MEMBER` | — | `200` `PartitionRecipeDto[]` | `403`, `404` |

```jsonc
// CreateDatasetRequest
{ "name": "Pneumonia-CXR", "orgId": "uuid", "description": "string, optional" }

// DatasetDto
{ "id": "uuid", "name": "Pneumonia-CXR", "orgId": "uuid",
  "description": "...", "createdAt": "2026-05-29T16:00:00Z" }

// CreateDatasetVersionRequest  (registers an already-uploaded, content-addressed blob)
{ "sha256": "hex64, required",          // content hash; uniqueness enforced
  "sizeBytes": 524288000,
  "uri": "s3://fedlearn-data/<sha256>", // where the content-addressed blob lives
  "label": "v1-2026-05" }

// DatasetVersionDto
{ "id": "uuid", "datasetId": "uuid", "sha256": "hex64",
  "sizeBytes": 524288000, "uri": "s3://...", "label": "v1-2026-05",
  "createdAt": "2026-05-29T16:01:00Z" }

// CreatePartitionRecipeRequest  (the deterministic non-IID split spec)
{ "method": "DIRICHLET",        // enum: "DIRICHLET" | "IID" | "EXPLICIT"
  "numClients": 10,
  "alpha": 0.5,                 // Dirichlet concentration; required iff method=DIRICHLET
  "seed": 42,
  "explicitMap": null }         // required iff method=EXPLICIT: { "clientId": [sampleIndices...] }

// PartitionRecipeDto
{ "id": "uuid", "datasetVersionId": "uuid", "method": "DIRICHLET",
  "numClients": 10, "alpha": 0.5, "seed": 42,
  "contentHash": "hex64",       // hash of the recipe params -> deterministic, cache-keyable
  "createdAt": "2026-05-29T16:02:00Z" }
```

**Reasoning — datasets are registered by content hash, not uploaded through the API:** the blob lives in S3/MinIO content-addressed by sha256; the REST API registers metadata + the `uri` + `sha256`. This keeps large medical-imaging blobs out of the JVM request path (v1 shipped a 5.7 MB ECG CSV inside the JAR — `C2`, `README.md` §3 item 5) and makes the dataset version immutable and reproducible (the determinism manifest references `datasetSha256`). The partition recipe's `contentHash` is a deterministic function of its params, so the partitioner output is cacheable and a re-run with the same recipe reproduces the same split (closes the v1 stale-split pickle trap — `C2`, `README.md` R15).

---

## 9. Artifacts API — `/api/artifacts/*`

Content-addressed model/checkpoint store. The API brokers metadata + pre-signed S3/MinIO URLs; **blob bytes never transit the JVM** (avoids the v1 in-JVM streaming memory blowups noted for large models — `A3` §4).

| Method | Path | Auth | Request body | Success response | Errors |
|---|---|---|---|---|---|
| `POST` | `/api/artifacts/upload-url` | `ORG_MEMBER` | `ArtifactUploadUrlRequest` | `200` `ArtifactUploadUrlResponse` | `400`, `403` |
| `POST` | `/api/artifacts` | `ORG_MEMBER` | `RegisterArtifactRequest` | `201` `ArtifactDto` | `400`, `403`, `409 SHA_EXISTS` |
| `GET` | `/api/artifacts/{artifactId}` | `ORG_MEMBER(artifact.org)` | — | `200` `ArtifactDto` | `403`, `404` |
| `GET` | `/api/artifacts/{artifactId}/download-url` | `ORG_MEMBER` | — | `200` `{ "url": "https://...", "expiresAt": "..." }` | `403`, `404` |

```jsonc
// ArtifactUploadUrlRequest
{ "sha256": "hex64, required", "sizeBytes": 104857600, "orgId": "uuid", "kind": "MODEL" }
// kind enum: MODEL | CHECKPOINT | DATASET

// ArtifactUploadUrlResponse
{ "uploadUrl": "https://minio.../bucket/<sha256>?X-Amz-...",  // pre-signed PUT
  "expiresAt": "2026-05-29T16:35:00Z",
  "objectKey": "<sha256>" }

// RegisterArtifactRequest  (after the client PUTs the blob to the pre-signed URL)
{ "sha256": "hex64", "sizeBytes": 104857600, "orgId": "uuid", "kind": "MODEL",
  "uri": "s3://fedlearn-artifacts/<sha256>" }

// ArtifactDto
{ "id": "uuid", "sha256": "hex64", "sizeBytes": 104857600, "orgId": "uuid",
  "kind": "MODEL", "uri": "s3://...", "createdAt": "2026-05-29T16:30:00Z" }
```

**Reasoning — pre-signed URL pattern, content-addressed by sha256:** v1 had no artifact store (only S3 TODOs — `README.md` §2 "Artifact/model store"). Content-addressing makes checkpoints deduplicated, immutable, and integrity-checkable; brokering pre-signed URLs keeps multi-GB models off the JVM heap (the v1 `getvalue()`/slice 2× memory problem — `A3` M4). MLflow self-hosted is the model registry layered on top of this store (`README.md` §1.1, `B3` §4).

---

## 10. gRPC contract — `fedlearn.v2`

This is the FL client ⇄ FL server wire protocol. It replaces `fedlearn.v1` (the existing proto at `framework/src/fedlearn/communication/protos/fedlearn.proto`). `buf` is the single source of truth with a breaking-change gate; codegen targets Python, Java, TypeScript, and C++ (mobile).

### 10.1 Design decisions baked into v2 (with reasoning)

1. **`package fedlearn.v2`** (not `v1`). The Java option becomes `com.fedlearn.v2`. Reasoning: a hard package bump signals the contract break and lets `buf` enforce no-silent-drift. The v1 audit found the mobile branch had a malformed `SubmitModelUpdate` RPC and a message-type typo (`A3` §5, observations 1438/1523) — a single buf-governed source kills that drift class.

2. **Add `protocol_version` to handshake + a `run_id` to every client-initiated RPC.** Reasoning: v1 had no protocol version field and self-asserted `client_id` only (Sybil risk — `B4`/`README.md` R6). v2 binds every call to a `run_id` (so the server routes to the correct long-running run keyed on `run_id`) and carries `protocol_version` so mismatched clients are rejected at registration, not mid-round.

3. **Fix the chunked-upload asymmetry at the contract level.** v1's `SubmitModelUpdateStream` shipped an opaque `torch.save` blob that the server read expecting a wrapped dict → `KeyError: 'parameters'` on every LLM upload (`A3-C1`, `B1` Bug 3, spec §3 Bug 3). v2 keeps the chunk message but **adds explicit framing fields** (`sha256`, `compressed`, `codec`, `total_bytes`) so the receiver validates symmetry and never infers anything from env vars (v1 inferred `compressed` from `FEDLEARN_USE_COMPRESSION` — `A3-C3`). The blob payload is `safetensors` (typed, no pickle), not `torch.save`.

4. **Add `bytes_transmitted` accounting to upload/download responses** so the FL server can populate the DeComFL communication-cost telemetry (§5.1, `B3` §6.2).

5. **DeComFL config carries CPU-canonical RNG metadata.** `GetDeComFLConfigResponse` includes `torch_version` and `grad_estimate_method` so the client generates perturbations with the identical CPU-canonical RNG contract (`B1-C2`, spec §6). Perturbations are **never** transmitted — only seeds (the O(K·P) wedge).

6. **Dual-heartbeat preserved.** The training stub blocks during `fit()`; the heartbeat stub runs on a parallel thread so the server doesn't time the client out during long rounds (the project conventions "Parallel heartbeat"). `Heartbeat` carries `run_id` and the server's `HeartbeatResponse.should_stop` is now **wired** (v1 hard-coded it `False` and never consumed `is_client_alive` — `A3-H1`).

### 10.2 The full `.proto` (authoritative — generate from this)

```protobuf
syntax = "proto3";

package fedlearn.v2;

option java_package = "com.fedlearn.v2";
option java_multiple_files = true;

// ============================================================================
// SERVICE
// ============================================================================
service FederatedLearningService {
  // --- lifecycle / control ---
  rpc RegisterClient        (RegisterClientRequest)        returns (RegisterClientResponse);
  rpc GetServerStatus       (GetServerStatusRequest)       returns (GetServerStatusResponse);
  rpc Heartbeat             (HeartbeatRequest)             returns (HeartbeatResponse);

  // --- model transfer (FedAvg path) ---
  rpc GetGlobalModel        (GetGlobalModelRequest)        returns (GetGlobalModelResponse);
  rpc GetGlobalModelStream  (GetGlobalModelRequest)        returns (stream ModelChunk);
  rpc SubmitModelUpdate     (SubmitModelUpdateRequest)     returns (SubmitModelUpdateResponse);
  rpc SubmitModelUpdateStream(stream ModelUpdateChunk)     returns (SubmitModelUpdateResponse);

  // --- DeComFL path (scalars + seeds only; no weights on the wire) ---
  rpc GetDeComFLConfig      (GetDeComFLConfigRequest)      returns (GetDeComFLConfigResponse);
  rpc SubmitGradientScalars (SubmitGradientScalarsRequest) returns (SubmitGradientScalarsResponse);

  // --- telemetry (closes the mobile observability island, B3 §6.4) ---
  rpc ReportClientMetrics   (ReportClientMetricsRequest)   returns (ReportClientMetricsResponse);
}

// ============================================================================
// CORE MESSAGES
// ============================================================================

// A single typed tensor. The ONLY weight-bearing wire type; no torch.save blobs.
message Tensor {
  bytes          data  = 1;   // raw bytes, dtype+dims interpret them
  repeated int64 dims  = 2;
  string         dtype = 3;   // whitelist: "float32","float64","int32","int64","uint8","bool"
}

message ModelParameters {
  map<string, Tensor> tensors              = 1;
  int64               num_examples_trained = 2;
}

// ============================================================================
// REGISTRATION  (binds the client to a run + enrollment identity)
// ============================================================================
message RegisterClientRequest {
  string client_id        = 1;   // client-chosen handle (display only; NOT trusted for authz)
  string run_id           = 2;   // UUID string of the fl_runs row this client joins
  int32  protocol_version = 3;   // MUST equal server's; mismatch -> REJECTED
  string enrollment_token = 4;   // backend-minted token binding identity (anti-Sybil, B4/R6)
}

message RegisterClientResponse {
  enum Status {
    STATUS_UNSPECIFIED = 0;
    ACCEPTED           = 1;
    REJECTED           = 2;
  }
  Status status            = 1;
  string message           = 2;
  int32  assigned_round    = 3;   // round the client should start training on (for late joiners)
  int32  protocol_version  = 4;   // server's version (client logs on mismatch)
}

// ============================================================================
// SERVER STATUS
// ============================================================================
message GetServerStatusRequest {
  string run_id = 1;
}

message GetServerStatusResponse {
  enum ServerState {
    STATE_UNSPECIFIED   = 0;
    INITIALIZING        = 1;
    WAITING_FOR_CLIENTS = 2;
    TRAINING            = 3;
    AGGREGATING         = 4;
    TRAINING_COMPLETE   = 5;
    FAILED              = 6;
  }
  ServerState server_state               = 1;
  int32       current_round              = 2;
  int32       required_clients_for_round = 3;  // the minimum quorum
  int32       received_updates_this_round= 4;
  int32       active_clients             = 5;
  int64       round_deadline_unix_ms     = 6;  // when the current round hard-stops (NO infinite hang)
}

// ============================================================================
// HEARTBEAT  (parallel stub; runs while fit() blocks the training stub)
// ============================================================================
message HeartbeatRequest {
  string client_id     = 1;
  string run_id        = 2;
  string status        = 3;   // free-text client phase, e.g. "TRAINING","IDLE"
  int32  current_step  = 4;
  int32  total_steps   = 5;
  int32  current_round = 6;
}

message HeartbeatResponse {
  bool   acknowledged = 1;
  bool   should_stop  = 2;   // NOW WIRED: server tells client to abort (deadline/quorum-lost/stopped)
  string message      = 3;
}

// ============================================================================
// MODEL TRANSFER — FedAvg path (typed framing both directions)
// ============================================================================
message GetGlobalModelRequest {
  string client_id = 1;
  string run_id    = 2;
}

message GetGlobalModelResponse {
  ModelParameters    parameters    = 1;
  int32              current_round = 2;
  map<string,string> config        = 3;
  int64              total_bytes   = 4;   // for comm-cost accounting
}

// Streaming chunk — used for models > 300 MB (parameter chunking, the project conventions).
// v2 fix: explicit framing so save/load are symmetric (no env-inferred compression, A3-C3).
message ModelChunk {
  int32  chunk_index    = 1;
  int32  total_chunks   = 2;
  bytes  chunk_data     = 3;
  bool   is_final_chunk = 4;
  int32  current_round  = 5;
  map<string,string> config = 6;
  // --- v2 framing fields ---
  string codec          = 7;   // "safetensors" (typed; NOT torch.save) — required, validated
  bool   compressed     = 8;   // on the wire, not inferred from env (A3-C3); codec="lz4+safetensors" if true
  int64  total_bytes    = 9;   // full reassembled size; receiver bounds-checks cumulative (H5)
  string sha256         = 10;  // hash of the full reassembled blob; receiver verifies (integrity)
}

message SubmitModelUpdateRequest {
  string          client_id       = 1;
  string          run_id          = 2;
  ModelParameters parameters      = 3;
  int32           trained_on_round= 4;
}

message SubmitModelUpdateResponse {
  bool  received        = 1;
  int64 bytes_received  = 2;   // comm-cost accounting
}

// Streaming upload chunk. v2 fix for A3-C1: same explicit framing as ModelChunk,
// so chunks_to_parameters reads a wrapped {parameters, num_examples} blob symmetrically.
message ModelUpdateChunk {
  string client_id        = 1;
  string run_id           = 2;
  int32  trained_on_round = 3;
  int32  chunk_index      = 4;
  int32  total_chunks     = 5;
  bytes  chunk_data       = 6;
  bool   is_final_chunk   = 7;
  int64  num_examples     = 8;
  // --- v2 framing fields ---
  string codec       = 9;    // "safetensors"
  bool   compressed  = 10;
  int64  total_bytes = 11;   // server bounds-checks cumulative against max_payload_bytes (H5)
  string sha256      = 12;   // verified on reassembly
}

// ============================================================================
// DeComFL — seeds + scalars only (the dimension-free / DLG-resistant path)
// ============================================================================

// Seeds organized as [local_step][perturbation_index].
message PerturbationSeeds {
  repeated LocalStepSeeds local_steps = 1;
}
message LocalStepSeeds {
  repeated int64 seeds = 1;   // P seeds for this local step (int64 to match C++ int64_t, B1 Low note)
}

// Gradient scalars organized as [local_step][perturbation_index].
message GradientScalars {
  repeated LocalStepGradients local_steps = 1;
}
message LocalStepGradients {
  repeated double scalars = 1;   // P gradient scalars for this local step
}

// Server-maintained history for rebuilding the model after missed rounds (Alg.2 rebuild).
message RebuildHistory {
  repeated RoundHistory rounds = 1;
}
message RoundHistory {
  int32             round_number      = 1;
  PerturbationSeeds seeds             = 2;
  GradientScalars   average_gradients = 3;   // server's AVERAGED (1/N) gradients (B1-C1 fixed)
}

message GetDeComFLConfigRequest {
  string client_id = 1;
  string run_id    = 2;
}

message GetDeComFLConfigResponse {
  int32             current_round   = 1;
  PerturbationSeeds current_seeds   = 2;   // seeds for the current round
  RebuildHistory    rebuild_history = 3;   // history for the rounds this client missed
  map<string,string> config         = 4;   // lr, mu, P (num_perturbations), K (num_local_steps)
  // --- v2 determinism contract (B1-C2 / spec §6) ---
  string torch_version       = 5;   // client MUST match for CPU-canonical RNG parity (else WARN/reject)
  string grad_estimate_method= 6;   // "forward" | "central" (B1-H2)
  string golden_vector_sha256= 7;   // RNG golden-vector fixture the client validates against
}

message SubmitGradientScalarsRequest {
  string          client_id        = 1;
  string          run_id           = 2;
  int32           trained_on_round = 3;
  GradientScalars gradients        = 4;
  int64           num_examples     = 5;   // collected; DeComFL aggregation is UNWEIGHTED (B1 Low note)
}

message SubmitGradientScalarsResponse {
  bool  received       = 1;
  int64 bytes_received = 2;   // K*P*8 bytes typically — the O(K*P) comm-cost number (B3 §6.2)
}

// ============================================================================
// CLIENT METRICS  (closes the mobile observability island, B3 §6.4)
// ============================================================================
message ReportClientMetricsRequest {
  string client_id     = 1;
  string run_id        = 2;
  int32  round         = 3;
  double loss          = 4;
  double accuracy      = 5;
  int32  current_step  = 6;
  int32  total_steps   = 7;
  string client_type   = 8;   // "desktop" | "docker" | "mobile" — for the per-type split panel
  int64  compute_ms    = 9;   // local compute time this round
}
message ReportClientMetricsResponse {
  bool acknowledged = 1;
}
```

### 10.3 gRPC framing rules the implementer MUST enforce (not expressible in proto)

| Rule | Value | Reasoning |
|---|---|---|
| Channel security | `grpc.secure_channel` with TLS server cert + mTLS client cert by default; plaintext only when profile is `dev`. | v1 defaulted to insecure with a log warning (`A3-C4`, `B4`/R6). The CN of the client cert binds identity (`README.md` §1.2). |
| Identity authz | The server trusts the **cert CN + `enrollment_token`**, never the self-asserted `client_id`. | Anti-Sybil (R6). |
| `codec` whitelist | `{"safetensors", "lz4+safetensors"}`. Reject any other value → `INVALID_ARGUMENT`. | No `torch.save`/pickle on the wire; removes the `weights_only` foot-gun (`A3` §5). |
| Chunk symmetry | The reassembled blob is a `safetensors` dict; sender wraps, receiver unwraps; `sha256` must match. | Fixes `A3-C1` at the protocol level (the v1 `KeyError`). |
| `max_payload_bytes` | Server-enforced cumulative cap (config, default 2 GB). Exceed → `RESOURCE_EXHAUSTED`. | v1 had no cumulative guard (`A3-H5`). |
| Perturbation transport | **Never** transmit the perturbation vector `z`; transmit only `seeds`. Both sides regenerate `z` via CPU-canonical RNG (`canonical_perturbation`, spec §3 Bug 2). | The O(K·P) dimension-free wedge; transmitting `z` would defeat it and re-expose DLG. |
| Round deadline + quorum | The server enforces `round_deadline_unix_ms`; if fewer than `required_clients_for_round` submit by the deadline, the round proceeds with what it has (≥ min-quorum) or is marked late; it never hangs forever. | v1 hung on one straggler (`README.md` R9, `C1`). |
| gRPC status mapping | `UNAUTHENTICATED` (bad cert/token), `INVALID_ARGUMENT` (bad codec/protocol_version), `RESOURCE_EXHAUSTED` (payload cap), `FAILED_PRECONDITION` (run terminal/not RUNNING), `NOT_FOUND` (unknown run_id). | One consistent mapping for all four codegen targets. |

---

## 11. STOMP-over-WebSocket topics

Endpoint: `/ws-logs` (SockJS-free raw WS, JWT via `JwtHandshakeInterceptor` cookie at handshake). The broker is the in-memory simple broker for single-replica; swap to a Redis/RabbitMQ relay (`enableStompBrokerRelay`) when multi-replica (`README.md` §1.1, one-line change). Application destination prefix `/app`; broker prefix `/topic`.

**Topic-level authorization (NEW, closes `A1` WS gap):** `JwtChannelInterceptor` must parse the destination on every `SUBSCRIBE` frame and call `AuthorizationService.requireParticipant(projectId)` (or the run/org equivalent). v1 let any authenticated user subscribe to any project's live telemetry — a cross-tenant leak. A `SUBSCRIBE` to a project the user does not participate in is rejected at the frame level.

| Destination | Direction | Payload | Auth to SUBSCRIBE | Reasoning |
|---|---|---|---|---|
| `/topic/logs/{projectId}` | server → browser | `LogLinePayload` (JSON) | `PROJECT_PARTICIPANT(projectId)` | Live training-log stream. v1 sent a raw string; v2 sends structured JSON so the frontend can color by level and carry `traceId`. |
| `/topic/results/{projectId}` | server → browser | `RoundResultPayload` (JSON) | `PROJECT_PARTICIPANT(projectId)` | Per-round metrics for the recharts chart; fired **per round** (incremental, `B3`). |
| `/topic/status/{projectId}` | server → browser | `ProjectStatusUpdatePayload` (JSON) | `PROJECT_PARTICIPANT(projectId)` | Run lifecycle transitions (STARTING→RUNNING→…). |
| `/topic/runs/{projectId}` | server → browser | `RunEventPayload` (JSON) | `PROJECT_PARTICIPANT(projectId)` | NEW: granular run events (round started, client joined/left, checkpoint written) for the live run view. |

**Reasoning — topics are keyed on `projectId`, not `runId`:** the dashboard subscribes when a user opens a project, before a run id exists; the payload carries `runId` so the frontend can filter to the active run. This matches the existing frontend subscription (`DashboardV2.tsx:153` subscribes `/topic/results/*`) and avoids a re-subscribe on every run.

### 11.1 STOMP payload JSON shapes (exact)

```jsonc
// /topic/logs/{projectId}  -> LogLinePayload  (v2: structured, was a raw string in v1)
{
  "projectId": "uuid",
  "runId":     "uuid | null",
  "level":     "INFO",          // INFO | WARN | ERROR | DEBUG
  "message":   "Round 7 aggregation complete",
  "stackTrace":"null",
  "roundIdx":  7,               // null when not round-scoped
  "traceId":   "0af7651916cd43dd8448eb211c80319c",  // W3C trace id (§14), for cross-hop correlation
  "timestamp": "2026-05-29T16:18:02.123Z"
}

// /topic/results/{projectId}  -> RoundResultPayload  (mirrors RoundResultDto + ids)
{
  "id":          "uuid",
  "projectId":   "uuid",
  "runId":       "uuid",
  "serverRound": 7,
  "loss":        0.2314,
  "accuracy":    0.9012,
  "gpuUtilization": 0.0,
  "uplinkBytes": 240,
  "downlinkBytes": 240,
  "scalarsTransmitted": 50,
  "modelParamCount": 66000000,
  "roundDurationSeconds": 4.2,
  "aggregationSeconds": 0.1,
  "activeClients": 3,
  "timestamp":   "2026-05-29T16:18:02Z"
}

// /topic/status/{projectId}  -> ProjectStatusUpdatePayload
{
  "projectId": "uuid",
  "runId":     "uuid",
  "status":    "RUNNING",        // run state-machine value (§4.3)
  "currentRound": 7,
  "numRounds": 20,
  "message":   "Run is RUNNING",
  "timestamp": "2026-05-29T16:18:02Z"
}

// /topic/runs/{projectId}  -> RunEventPayload
{
  "projectId": "uuid",
  "runId":     "uuid",
  "eventType": "ROUND_STARTED",  // ROUND_STARTED | ROUND_COMPLETED | CLIENT_JOINED | CLIENT_LEFT | CHECKPOINT_WRITTEN | DEADLINE_HIT
  "round":     7,
  "clientId":  "string | null",  // set on CLIENT_JOINED/LEFT
  "detail":    { },              // event-specific JSON object
  "traceId":   "0af7651916cd43dd8448eb211c80319c",
  "timestamp": "2026-05-29T16:18:02Z"
}
```

---

## 12. Standard error envelope

**Every** non-2xx REST response uses this exact shape. v1 had two error contracts coexisting (typed exceptions + raw `ResponseStatusException`); v2 has one, emitted by a single `GlobalExceptionHandler`. No raw stack traces, no bare strings.

```jsonc
// Error envelope (Content-Type: application/json)
{
  "timestamp": "2026-05-29T16:18:02.123Z",
  "status":    409,                       // HTTP status code (mirrors the response code)
  "code":      "RUN_ALREADY_ACTIVE",      // STABLE machine-readable enum (see table below)
  "message":   "A run is already active for this project.",  // human-readable, safe to display
  "path":      "/api/projects/7a3c.../runs",
  "traceId":   "0af7651916cd43dd8448eb211c80319c",  // W3C trace id for support correlation (§14)
  "fieldErrors": [                        // present ONLY when code=VALIDATION_FAILED
    { "field": "numRounds", "message": "numRounds must be at most 1000" }
  ]
}
```

**Reasoning — `code` is a stable enum, `message` is for humans:** the frontend switches on `code` (machine-readable, never localized/changed), and shows `message` to the user. `traceId` lets a support engineer jump from a user-reported error straight to the Tempo trace (`B3` §5). `fieldErrors` is present only for validation failures so the form can highlight the offending field (the frontend's Zod layer and the backend's Bean Validation produce the same field names).

### 12.1 Error `code` registry (stable enum — implement exactly; never reuse a code for a different meaning)

| `code` | HTTP | Where |
|---|---|---|
| `VALIDATION_FAILED` | 400 | Any DTO bean-validation failure; includes `fieldErrors`. |
| `TOKEN_INVALID` | 400 | Email-verify / password-reset token malformed. |
| `BAD_CREDENTIALS` | 401 | Login failure. |
| `NOT_AUTHENTICATED` | 401 | `/api/auth/me` with no/invalid session. |
| `RUN_TOKEN_INVALID` | 401 | Internal callback with a bad per-run token. |
| `ACCOUNT_NOT_VERIFIED` | 403 | Login by a `PENDING` account. |
| `FORBIDDEN` | 403 | Authn ok, authz denied (org/project/platform scope). |
| `RUN_TOKEN_MISMATCH` | 403 | Per-run token's `runId` ≠ path `runId`. |
| `*_NOT_FOUND` | 404 | `PROJECT_NOT_FOUND`, `RUN_NOT_FOUND`, `ORG_NOT_FOUND`, `USER_NOT_FOUND`, `DATASET_NOT_FOUND`, `ARTIFACT_NOT_FOUND`. |
| `USER_ALREADY_EXISTS` | 409 | Register with a taken username/email. |
| `PROJECT_NAME_TAKEN` | 409 | Duplicate project name within an org. |
| `ORG_NAME_TAKEN` | 409 | Duplicate org name. |
| `RUN_ALREADY_ACTIVE` | 409 | Partial-unique-index violation on start. |
| `RUN_ACTIVE` | 409 | Mutating a project that has an active run. |
| `RUN_NOT_STOPPABLE` | 409 | Stop on an already-terminal run. |
| `RUN_TERMINAL` | 409 | Internal callback for a run already in a terminal state. |
| `ORG_QUOTA_EXCEEDED` | 409 | Org concurrent-run quota reached. |
| `LAST_ADMIN` | 409 | Demoting/removing the last `PLATFORM_ADMIN`. |
| `LAST_OWNER` | 409 | Demoting/removing the last org `OWNER`. |
| `ALREADY_MEMBER` | 409 | Adding an existing org member. |
| `SHA_EXISTS` / `VERSION_SHA_EXISTS` | 409 | Registering a content hash that already exists. |
| `TOKEN_EXPIRED` | 410 | Email-verify / reset token expired. |
| `NO_DATASET_VERSION` | 422 | Start with no dataset version and no project default. |
| `UNSUPPORTED_LAUNCHER` | 422 | `LOCAL_PROCESS` requested outside `dev`. |
| `MANIFEST_NOT_READY` | 409 | Manifest requested before the run started. |
| `RATE_LIMITED` | 429 | Bucket4j throttle on auth endpoints. |
| `INTERNAL_ERROR` | 500 | Catch-all; message is generic, details only in logs. |

---

## 13. Per-run scoped result token (the internal-callback auth contract)

Replaces v1's single global `APP_INTERNAL_API_KEY` (which let any task write any project's results — `A1-F6`). One token per run, minted by the backend at launch, injected into the executor's environment, validated on every `/api/internal/runs/{runId}/**` call.

**Token format (locked):** an opaque string `flrun_<base64url(payload)>.<base64url(hmac)>` — a compact signed token (not a full JWT, to keep it env-var-sized). The signature is HMAC-SHA256 over the payload using a server-side secret `app.internal.run-token-secret` (from the secrets manager).

**Payload (the signed claims):**

```jsonc
{
  "runId":     "uuid",          // the ONLY run this token may write to
  "projectId": "uuid",
  "orgId":     "uuid",
  "issuedAt":  1748534400,      // unix seconds
  "expiresAt": 1748620800,      // unix seconds; <= run's max lifetime + grace
  "nonce":     "random-128-bit" // ties to one launch; rotated per run
}
```

**Validation algorithm the backend MUST implement (pseudocode):**

```text
function validateRunToken(authHeader, pathRunId):
    if not authHeader.startsWith("Bearer flrun_"):  -> 401 RUN_TOKEN_INVALID
    raw       = authHeader.removePrefix("Bearer ")
    payloadB64, sigB64 = raw.removePrefix("flrun_").splitOnLast(".")
    expectedSig = HMAC_SHA256(secret, payloadB64)             // constant-time compare
    if not constantTimeEquals(expectedSig, base64urlDecode(sigB64)): -> 401 RUN_TOKEN_INVALID
    claims = json.parse(base64urlDecode(payloadB64))
    if now() > claims.expiresAt:                             -> 401 RUN_TOKEN_INVALID
    if claims.runId != pathRunId:                            -> 403 RUN_TOKEN_MISMATCH
    run = flRunRepository.findById(claims.runId)
    if run == null:                                          -> 404 RUN_NOT_FOUND
    if run.status in TERMINAL_STATES:                        -> 409 RUN_TERMINAL
    // success: request context now carries (runId, projectId, orgId) from the TOKEN, not the body
    return RunContext(claims.runId, claims.projectId, claims.orgId)
```

**Injection into the executor (locked env var names):** the launcher sets, in the spawned environment (k8s Job env / ECS task override / dev process env):

| Env var | Value |
|---|---|
| `FEDLEARN_RUN_ID` | the run UUID |
| `FEDLEARN_RUN_TOKEN` | `flrun_<...>` (the signed token above) |
| `FEDLEARN_BACKEND_URL` | base URL for `/api/internal/...` (VPC-internal/HTTPS outside dev) |
| `FEDLEARN_PROJECT_ID` | project UUID (display/log convenience) |
| `TRACEPARENT` | the W3C trace context of the launch span (§14) |

The FL server reads `FEDLEARN_RUN_TOKEN` and sets `Authorization: Bearer ${FEDLEARN_RUN_TOKEN}` on every callback. Reasoning: the token is short-lived, run-scoped, and signed — a compromised or curious task that extracts its own token still cannot write any other run's telemetry (the `runId` mismatch and HMAC both block it). HTTPS/VPC-internal transport is mandatory outside `dev` (the token is a bearer credential).

---

## 14. W3C `traceparent` propagation contract (REST → process-env → gRPC-metadata)

One `trace_id` must join: browser click → Spring Boot span → spawned FL-server process → every client gRPC call → mobile. v1 had no correlation id anywhere (`B3` §5). The carrier at each hop is locked below.

**Format (W3C Trace Context, locked):** the `traceparent` header value is
`version "-" trace-id "-" parent-id "-" trace-flags`, e.g.
`00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01`
where `trace-id` is 16 bytes (32 hex), `parent-id` is 8 bytes (16 hex), `trace-flags` is 1 byte (2 hex, `01` = sampled).

| Hop | Carrier | Mechanism |
|---|---|---|
| Browser → Spring Boot | HTTP header `traceparent` | The frontend OTel web SDK (or the OTel Java agent originating a root span if absent) sets it on the `/api/projects/{id}/runs` request. |
| Spring Boot span → FL-server process | **environment variable `TRACEPARENT`** | At launch, `FlServerLauncher` serializes the active span context via the OTel `TextMapPropagator` into the executor env (`TRACEPARENT=00-...-...-01`) alongside the run token (§13). |
| FL-server process (root span) | OTel SDK `extract()` from `os.environ["TRACEPARENT"]` | The Python server parents its run root-span "fl-run {run_id}" to the JVM span. |
| FL-server → client (gRPC) | **gRPC metadata key `traceparent`** | The gRPC server/client OTel interceptor injects/continues `traceparent` in outgoing metadata (the W3C-standard carrier for gRPC). |
| Client → mobile | gRPC metadata key `traceparent` | The mobile C++ gRPC client interceptor continues the same context. |
| Any hop → logs | structlog/MDC field `trace_id` | Every Python log line and every Java log line binds `trace_id` (+ `project_id`, `round_idx`); STOMP `LogLinePayload`/`RunEventPayload` carry `traceId` (§11). |
| Any error response | error envelope `traceId` (§12) | So a user-reported error maps to a Tempo trace. |

**Caveat (locked, must be honored):** gRPC is plaintext only in `dev`; in all other profiles `traceparent` and baggage travel over TLS+mTLS. **Never put PII in baggage** (the trace context crosses the FL transport that also carries scalars/seeds — `B3` risk #7, `README.md` audit item #37). `traceparent` carries only opaque ids.

**Reasoning — env var for the JVM→process hop (not a CLI flag, not a file):** the FL server is launched by three different backends (k8s Job, ECS RunTask, dev process); an environment variable is the one carrier all three support uniformly and is exactly how the run token and `FEDLEARN_BACKEND_URL` already flow. The OTel SDK's standard env-extraction path (`TRACEPARENT`) makes the Python side a two-line `extract()` with no custom parsing.

---

## 15. Contract summary (what decouples each unit)

| Boundary | Contract | Section |
|---|---|---|
| Browser/desktop/mobile ⇄ control plane | REST + cookie JWT + standard error envelope | §2–§9, §12 |
| Browser ⇄ control plane (live) | STOMP topics, structured JSON payloads, topic-level authz | §11 |
| FL server ⇄ control plane (callbacks) | `/api/internal/runs/{runId}/**` + per-run scoped token | §5, §13 |
| FL clients ⇄ FL server | gRPC `fedlearn.v2`, TLS+mTLS, cert-CN identity, safetensors framing | §10 |
| Everything ⇄ everything (telemetry) | W3C `traceparent` across REST → env → gRPC metadata | §14 |
| Reproducibility | Determinism manifest + content-addressed artifacts/datasets | §4.4, §8.2, §9 |

**Closing reasoning — why this contract surface and not a thinner one:** each addition over v1 closes a specific verified audit finding rather than adding speculative surface. The run/manifest/checkpoint objects exist because v1 had no durable run entity and could not reproduce or resume (`R9`/`R14`). The per-run token exists because the global key was a multi-tenant integrity break (`F6`). The `org_id` scoping and topic-level authz exist because v1 leaked cross-tenant data (`F9`, WS gap). The DeComFL comm-cost fields and the `fedlearn.v2` framing fixes exist because the platform's one differentiator was both unmeasured and broken on its live path (`B3` §6.2, `A3-C1`, `B1-C1/C2`). Nothing here is "while we're at it" — every field traces to a finding.
