# A1 — Backend Audit (v2 Greenfield): `backend/fl-platform-api/`

**Date:** 2026-05-29
**Scope:** Spring Boot 3.4.5, Java 21, Gradle. 132 Java sources (main + test), 29 test files. Flyway V1–V5.
**Branch:** `main-clean`
**Builds on:** `docs/audit/2026-05-27/01-backend.md` (cited inline as *[2026-05-27 Cx/Hx/Mx]*). This report **verifies what was fixed, what regressed, and what is still open**, then calibrates a v2 architecture for a production startup.

---

## Executive summary

The backend is a competent POC with surprisingly good security *intent* (cookie-only JWT, fail-fast secrets, internal-key filter, handshake-time WS auth) but three classes of production blocker: (1) a **dead admin RBAC path** that the test suite actively masks; (2) the **ProcessBuilder-spawns-Python FL orchestration model**, which is a hard scaling cliff (~11 concurrent runs, no isolation, no HA, control state lost on JVM restart) — the single most important thing to rebuild; and (3) **multi-tenant integrity gaps** (mixed BIGINT/UUID keys, a global internal API key that lets any FL task write results for any project, near-zero `@Auditable` coverage). The ECS Fargate path added since the prior audit is the right *direction* but is wired as a fork inside `FlowerServerManager` with no task tracking, no result authentication, and no lifecycle reconciliation.

Verdict at a glance: **keep Spring Boot**, **salvage** the auth/data layers with surgical fixes, **rebuild** FL-server orchestration as a proper control plane (k8s Jobs or ECS RunTask + a reconciler + a tasks table), and **kill** the local-ProcessBuilder path for any non-dev deployment.

---

## What changed since 2026-05-27 (verification pass)

| Prior finding | Status today | Evidence |
|---|---|---|
| **C1** `hasRole('ADMIN')` dead vs `PLATFORM_ADMIN` | **STILL BROKEN** + now *masked by a test* | `AdminController.java:17`, `AdminService.java:41-42`, `AuthorizationService.java:42`, `UserController.java:32,54` all still say `ADMIN`; `CustomUserDetailsService.java:53-54` emits `ROLE_PLATFORM_ADMIN`; `BootstrapRunner.java:122` writes `PLATFORM_ADMIN`. See F1. |
| **C3** `UserController` returns JPA entity (PII) | **STILL OPEN** | `UserController.java:33` returns `ResponseEntity<List<User>>`; `:42-50` returns raw `User`. `AdminController` *did* move to `AdminUserDto` (good), but `/api/users` was left behind. |
| **C4** `/start` race → duplicate servers | **Partially fixed** | Port collision closed via `reservedPorts`+`portReservationLock` (`FlowerServerManager.java:91-92,337-353`). But `ProjectService.startServerForProject():147` is **still not `@Transactional`** and still does check-then-act (`isServerRunning():172` → `startServerForProject():179`) with **no row lock**. See F4. |
| **C2** unauthenticated/unthrottled register, no email gate | **STILL OPEN** | `AuthController.java:70-94` (permitAll via `SecurityConfig:118`), `UserService.registerUser():24-44` never sets `emailVerified`/`status`; default `emailVerified=false` (`User.java:44`) is **never enforced at login** (`CustomUserDetailsService.java:45` gates only on `status==ACTIVE`). No rate limit. See F5. |
| **H2** JWT missing issuer/aud/jti/skew | **STILL OPEN** | `JwtTokenProvider.validateToken():54-57` checks subject+expiry only; no `requireIssuer`/`requireAudience`/`allowedClockSkewSeconds`/`jti`. |
| **H3** `@Auditable` coverage ~zero | **STILL OPEN** | Only 2 annotations exist: `AuthController.java:76,189`. Project create/start/stop/delete, membership, admin role-change, access-request decide all unaudited. |
| **C7** anonymous `AuthenticationException` subclasses in STOMP | **STILL OPEN** | `JwtChannelInterceptor.java:71,78` still `throw new AuthenticationException(...) {}`. |
| **M2** 11-port concurrency cap | **STILL OPEN** by design | `application.properties:125-126` (50000–50010). |
| ECS Fargate orchestration | **NEW since prior audit** | `FlowerServerManager.startEcsFargateServer():103-148` (`software.amazon.awssdk:ecs:2.25.11`, `build.gradle:42`). Right direction, incomplete — see F2. |
| Micrometer/Prometheus | **STILL ABSENT** | `build.gradle` has no micrometer/prometheus; actuator exposes `health,loggers` only (`application.properties:53`). FL-run observability is invisible. |

Net: the prior audit's *quick wins* (rename role, DTO-ify `/api/users`) were **not landed**, while real engineering effort went into the ECS path and port-reservation. The cheapest, highest-impact P0s are still on the floor.

---

## Architecture & layering

Standard Spring layered MVC: `controller → service → repository`, JPA entities, DTOs in `dto/`, cross-cutting `audit/`, `security/`, `email/`, `bootstrap/`. Generally sound. Concrete defects:

- **Controllers reach past services into repositories.** `UserController.java:38,57,60` calls `userRepository.findAll()/existsById/deleteById` directly; `ResultsController.java:44-53` uses `projectRepository`/`roundResultRepository` directly. This is why the PII leak (F3) and the unauthenticated result-write (F6) live in controllers — there's no service seam to enforce invariants. An ArchUnit rule ("controllers never touch repositories") would have caught both.
- **Injection style is inconsistent.** `ProjectService` field-injects 11 collaborators (`:41-62`); `AuthorizationService`/`AdminService` field-inject; `AuthController`/`AuditAspect`/`BootstrapRunner` use constructor injection. Field injection blocks `final`, hides the dependency explosion in `ProjectService` (a god-service), and makes unit testing require Spring.
- **`ProjectService` is a god object** (438 lines): project CRUD, FL lifecycle, log paging, discovery, notifications, DTO mapping. v2 should split: `ProjectService` (CRUD/visibility), `FlRunService` (lifecycle), `ProjectLogService` (paging/export), `DiscoveryService`.
- **Two error-handling contracts coexist.** Typed exceptions + `GlobalExceptionHandler` *and* raw `ResponseStatusException` (`AdminService.java:44`, prior *[M5]*). Pick one.
- **Lombok carries its weight for ~2 builders** (`build.gradle:48,51`) — drop for Java 21 records, simpler build.

**Verdict: salvage** (layering is fine; enforce the seam, decompose `ProjectService`, standardize injection + error contract).

---

## Auth / RBAC — the admin path is dead, and a test hides it

### F1 (CRITICAL, MAJOR) — The bootstrap admin gets 403 on every `/api/admin/**` route; the integration test masks it

End-to-end trace, verified:

1. `BootstrapRunner.java:122` → `admin.setPlatformRole("PLATFORM_ADMIN")`; existence check `:92` is `existsByPlatformRole("PLATFORM_ADMIN")`.
2. `CustomUserDetailsService.java:53-54` → authority = `"ROLE_" + platformRole` = **`ROLE_PLATFORM_ADMIN`**.
3. `AdminController.java:17` → `@PreAuthorize("hasRole('ADMIN')")` requires **`ROLE_ADMIN`**, which is never granted.
4. **Result: the only admin the system can self-provision is locked out of the entire admin API (403).**

Co-symptoms (all confirmed unfixed):
- `AdminService.updateRole():41-42` guards `"USER".equals(newRole) && "ADMIN".equals(target.getPlatformRole())` and counts `countByPlatformRole("ADMIN")` — so "cannot demote the last admin" **never fires** for a real `PLATFORM_ADMIN`.
- `AuthorizationService.isAdmin():42` matches `"ROLE_ADMIN"` → always false for real admins. This silently downgrades `requireOwnerOrAdmin`/`requireParticipant` (`:63-84`) and `ProjectService.getProject():378` so a platform admin is treated as a non-privileged stranger on projects they don't own.
- `UserController.java:32,54` `hasRole('ADMIN')` → dead.

**Why this passed review twice:** `AdminControllerIntegrationTest.java:30,66,81,95` seeds users with `setPlatformRole("ADMIN")` — the literal string `"ADMIN"`, which the production bootstrap path **never produces**. The test promotes to `"ADMIN"` (`:103`) and asserts `"ADMIN"` (`:105`). So `CustomUserDetailsService` emits `ROLE_ADMIN`, `hasRole('ADMIN')` matches, the test is green — while exercising a role value that no real deployment ever has. This is worse than no test: **green CI is actively asserting that a broken production path works.**

**Fix (P0, <1h):** Make `PLATFORM_ADMIN` the single source of truth. Replace `ADMIN` → `PLATFORM_ADMIN` at the 5 sites; constrain `UpdateUserRoleRequest.role` to `^(USER|PLATFORM_ADMIN)$`; **and fix the test to seed `PLATFORM_ADMIN`** so it tests reality. Better v2: replace string roles with an `enum PlatformRole { USER, PLATFORM_ADMIN }` so `valueOf` rejects typos at the boundary and `hasRole`/entity/test can't drift.

### F7 (HIGH) — JWT lacks issuer/audience/jti/clock-skew; no revocation

`JwtTokenProvider.generateToken():41-46` emits only sub/iat/exp; `validateToken():54-57` checks subject + expiry. No `iss`/`aud` binding, no `jti` (so no server-side revocation/blacklist — logout is purely client-cookie-clearing, `AuthController.logout():188-201`), no clock-skew allowance. For a multi-tenant SaaS, add `iss`, `aud`, `jti`, a `tokenVersion` claim (bump on password change/forced logout), and a short access-token + refresh model.

### WS auth — solid handshake, but no topic-level authorization
`JwtHandshakeInterceptor` rejects unauthenticated upgrades at handshake (`:58-88`) — good, no "subscribe-then-check" window. `JwtChannelInterceptor` re-validates on CONNECT (`:44-88`). **Gap:** there is **no `SUBSCRIBE`-frame check** that the connected user is a participant of `projectId` in `/topic/logs/{projectId}` / `/topic/results/{projectId}`. Any authenticated user can subscribe to any project's live logs/metrics — a cross-tenant leak of FL run telemetry. Add a `SUBSCRIBE` branch in `JwtChannelInterceptor` that parses the destination and calls `AuthorizationService.requireParticipant`. Also fix C7's anonymous `AuthenticationException` subclasses (`:71,78`).

**Verdict (auth/RBAC): refactor.** The plumbing is good; the role model is the defect. Collapse to an enum, fix the masked test, add topic authz, harden JWT claims.

---

## FL-server lifecycle — the scaling cliff (the headline rebuild)

`FlowerServerManager` runs two mutually exclusive paths chosen by whether `ecs.cluster-name` is set (`startServerForProject():94-101`):

- **Local:** `startLocalServer():150-276` spawns `bash run_fl_server.sh ...` via `ProcessBuilder`, tracks the `Process` in `ConcurrentHashMap<UUID,Process> runningServers` (`:85`), streams merged stdout to STOMP on a daemon thread (`:206-227`), 3s startup probe (`:229`).
- **ECS Fargate:** `startEcsFargateServer():103-148` `RunTask` with env overrides; returns `Optional.empty()` (port managed externally).

### F2 (CRITICAL, MAJOR) — The local ProcessBuilder model cannot go to production; the ECS path is half a control plane

The local model's structural limits (each independently fatal for a startup that wants paying tenants running concurrent FL jobs):

1. **Concurrency ceiling ~11.** `findFreePort()` scans `50000–50010` (`application.properties:125-126`, `FlowerServerManager:337-352`). The 12th concurrent run throws `IllegalStateException` ("No free port in range"). One JVM = ≤11 FL runs, period.
2. **No isolation.** Every FL server is a child of the API JVM sharing its CPU/RAM/filesystem/network namespace. A run that OOMs or forks a runaway PyTorch process degrades the API itself. On an M4 Max (36 GB unified) dev box, a few transformer federations exhaust memory and the API becomes unresponsive — there is no cgroup/quota boundary.
3. **No HA, and control state is volatile.** `runningServers` is an in-process map. A JVM restart/crash/deploy **orphans every running Python child** (they keep their gRPC ports bound but are now unreachable from the backend) and the backend has **no record they ever existed** — `ProjectService.stopServerForProject` finds nothing in the map (`FlowerServerManager.stopServerForProject():278-294`). `@PreDestroy stopAllOnShutdown()` (`:302-323`) only helps on *graceful* shutdown, and it `destroyForcibly()` (SIGKILL) with no SIGTERM grace (prior *[H5]*), so even graceful shutdown can't let Python flush model state.
4. **stdout pipe wedge.** Reader on a daemon thread (`:206-227`); if it dies, the child's ~64KB stdout pipe fills and the FL server blocks on `write()` (prior *[H4]*). Logs are the *only* progress signal, so a wedged reader silently freezes the run.
5. **cwd-dependent paths.** `pb.directory(new File("."))` (`:197`) + relative `flServerWrapperPath` default (`application.properties:115`) — behavior depends on where the JVM was launched.

The ECS path fixes isolation and the port cap, but is **not yet a control plane**:
- **No task tracking.** `startEcsFargateServer` logs the `taskArn` and returns `Optional.empty()` (`:141-143`) — the ARN is **never persisted**. So `/stop` cannot `StopTask`, `/delete` cannot reclaim, and there is no reconciliation of "task says RUNNING but DB says STOPPED" or vice-versa. `Project.serverPort` is set to `null` for ECS (`ProjectService:181`) so even the connection endpoint can't tell clients where to dial.
- **Per-call `EcsClient`.** `EcsClient.builder().build()` inside a try-with-resources per request (`:132`) — rebuilds HTTP client + credential chain every start. Make it a singleton bean.
- **No backpressure / quota.** Nothing caps how many Fargate tasks a tenant (or a runaway loop) can launch. That's an unbounded AWS bill.
- **Result authenticity gap (see F6).**

### F4 (HIGH) — `/start` is still not transactional; check-then-act race survives at the DB layer
`ProjectService.startServerForProject():147` has no `@Transactional` and no row lock. Two concurrent `/start` for the same project both pass `isServerRunning():172` (process not yet in the map), both call `startServerForProject():179`. The port-reservation lock prevents the *same port* being chosen, but you still get **two FL servers for one project** on two different ports; the second's `startLocalServer:154` `stopServerForProject` then races the first's map entry. Fix: `@Transactional` + `projectRepository.findById(...)` under `PESSIMISTIC_WRITE`, or a DB unique constraint on "one active run per project" backed by a `fl_runs` table (see v2).

### v2 FL orchestration — recommended target

Keep Spring Boot as the **control plane API**; stop treating it as the FL-server's parent process. Introduce an explicit run model and an external executor:

1. **`fl_runs` table** (new V6 migration): `id UUID, project_id UUID, org_id UUID, status (PENDING|STARTING|RUNNING|SUCCEEDED|FAILED|STOPPED), executor_ref (k8s job name / ECS task ARN), grpc_endpoint, requested_by, started_at, finished_at, num_rounds, strategy, min_clients`. This is the durable source of truth that survives JVM restarts and enables reconciliation. Add a **partial unique index** `(project_id) WHERE status IN ('PENDING','STARTING','RUNNING')` — this closes F4 declaratively (the second concurrent `/start` gets a constraint violation → 409).
2. **Executor abstraction** `FlServerLauncher { LaunchResult launch(FlRunSpec); void stop(String executorRef); RunState poll(String executorRef); }` with implementations:
   - `LocalProcessLauncher` — **dev/test only**, the current ProcessBuilder path, hard-capped and never enabled outside `dev`.
   - `KubernetesJobLauncher` — **recommended primary.** One **k8s Job per FL run**: native resource requests/limits (cgroup isolation), `activeDeadlineSeconds` (auto-kill stuck runs), `ttlSecondsAfterFinished` (auto-GC), per-job ServiceAccount, NetworkPolicy, and a `Service` for the gRPC endpoint. The fabric8 kubernetes-client is a mature JVM dependency. This is the cleanest fit for "N concurrent isolated, observable, self-cleaning FL runs."
   - `EcsRunTaskLauncher` — the existing path, completed: persist the ARN to `fl_runs.executor_ref`, implement `stop`/`poll` via `StopTask`/`DescribeTasks`, singleton `EcsClient` bean.
3. **A reconciler** (`@Scheduled` or k8s informer): periodically `poll()` each non-terminal `fl_run`, reconcile executor state → DB → STOMP status, and reap orphans on startup (close F2.3). This is what makes restarts safe.
4. **Logs off the stdout pipe.** Don't stream logs through the backend JVM at all in v2. Have the FL server emit structured logs to stdout, collected by the platform log agent (Fluent Bit/Loki/CloudWatch), and have the run publish *progress events* (round complete, metric) to the backend via the authenticated internal API → STOMP. This kills F2.4 and gives real FL-run observability (loss/accuracy/round latency per run) instead of raw text.
5. **Quota** per org (max concurrent runs, max rounds) enforced in `FlRunService` before launch.

**Verdict (FL orchestration): rebuild.** Local ProcessBuilder = **kill** for any non-dev profile (keep only as `LocalProcessLauncher` behind the abstraction). ECS path = **salvage into** the launcher abstraction with task tracking + reconciliation. k8s Jobs are the recommended production substrate.

---

## Multi-tenancy & data layer

### F8 (HIGH, MAJOR) — Mixed BIGINT/UUID key strategy will calcify into a sharding/migration tax
- `users.id BIGSERIAL` (Long) — `V1__init.sql:6`, `User.java:13-15` `@GeneratedValue(IDENTITY)`.
- `organizations.id UUID`, `projects.id UUID` — `V1:15`, `V5:8`.
- Composite keys mix both: `organization_memberships(org_id UUID, user_id BIGINT)` (`V5:18-19`), mirrored in `OrganizationMembershipId(UUID, Long)` and `ProjectMembershipId`.

This works today but is a real tax for a multi-tenant SaaS: (a) sequential `BIGINT` user ids are enumerable and leak user count/growth (any endpoint that echoes a user id); (b) `IDENTITY` generation is a per-insert round-trip and blocks Hibernate batch inserts; (c) you can't pre-generate ids client-side; (d) future tenant-sharding/data-export is harder with two id schemes. **v2 recommendation:** standardize on UUIDv7 (time-ordered, index-friendly) for all top-level entities including `users`, OR commit to BIGINT everywhere — but not the current split. If migrating `users.id`, do it now while the user table is tiny; it's a one-time pain that compounds with every new FK.

### F9 (HIGH) — Tenant isolation is enforced in application code only; no DB-level guarantee
`projects.org_id` is `NOT NULL` (`V5:78`) — good — but nothing scopes queries by org. `AdminService.listUsers()` / `listAllProjects()` (`:26-28,52`) and `ProjectService.getProjectsForCurrentUser()` filter by *user*, not *org*. There is no row-level-security and no org-scoped repository layer. A single missing `where org_id = ?` (or the F1 admin downgrade) becomes a cross-tenant data leak. For production multi-tenancy, add an org-scoping aspect/`@Filter` or Postgres RLS keyed on a request-scoped current-org, and bake `org_id` into every tenant-owned query.

### F10 (HIGH) — H2-file-mode is not Postgres; V1 ships Postgres-specific DDL that H2 only approximates
`ec2demo` runs H2 file-mode with `MODE=PostgreSQL` (`application.properties:11`, `application-ec2demo.properties:4-6`); `production` uses real Postgres (`application-production.properties:22`). But the **`test` profile disables Flyway and uses `create-drop`** (per `CLAUDE.md`), so the V1–V5 SQL — `BIGSERIAL`, `TIMESTAMP WITH TIME ZONE` (`V1:10`), `CLOB` (`V5:51`), `ALTER COLUMN ... RENAME TO` (`V5:28`) — is **never validated against real Postgres in CI**. `MODE=PostgreSQL` is an emulation, not a guarantee; the V4 migration already had an H2-specific multi-clause-`ALTER` failure historically. **v2:** add a Testcontainers-Postgres test profile that runs the real migrations; make Postgres the only datasource for `ec2demo`/`production` (drop H2 outside `dev`/`test`). H2-file-mode is also single-writer and not a real concurrent DB — it caps the platform at one JVM regardless of the FL-orchestration fix.

### F3 (HIGH, MAJOR) — `UserController` still returns the JPA `User` entity (PII leak), and a stale remediation comment misleads
`UserController.getAllUsers():33` returns `ResponseEntity<List<User>>`; `createUser():42-50` returns raw `User`. `@JsonIgnore` hides the password (`User.java:23`) but **email, displayName, lastLoginAt, status, emailVerified, timestamps all serialize**. `AdminController` was migrated to `AdminUserDto` — `/api/users` was simply left behind. Worse, the in-code "how to bootstrap admin" comment (`UserController.java:36-37`) says `UPDATE users SET role = 'ADMIN'` — but the column is `platform_role` and the working value is `PLATFORM_ADMIN`; the remediation hint is itself wrong (and reinforces F1). Fix: delete `/api/users` (it duplicates `/api/admin/users`) or return `AdminUserDto`; remove the stale comment.

### F6 (HIGH, MAJOR) — Any FL task can write results/finish ANY project (broken object-level auth on internal callbacks)
`ResultsController` (`/api/internal/results/{projectId}`) is gated only by the **single global** `APP_INTERNAL_API_KEY` (`InternalApiKeyFilter`). `reportRoundResult():38-58` and `markProjectAsFinished():61-65` take `projectId` from the path and **never verify the calling task owns that project**. Every FL-server task shares the one key (`FlowerServerManager:189,386`). So a compromised or buggy task (or a curious tenant who extracts the key from their own container) can POST fabricated loss/accuracy for, or force-complete, **any other tenant's run**. For a multi-tenant platform this is an integrity break. v2: mint a **per-run scoped token** (short-lived, bound to `fl_run_id`+`project_id`+`org_id`, signed by the backend) injected into the launcher env, and validate it server-side instead of (or in addition to) the shared key. Also note gRPC is plaintext over WAN (audit #37) — these callbacks must be HTTPS/VPC-internal in production (`application-production.properties:76` mandates a VPC-internal URL, good).

### Audit log
`@Auditable` is wired correctly (`AuditAspect` proceeds-then-writes so caller rollback rolls back the row, `:51-82`) but applied to only **2 methods** (`AuthController:76,189`). Project lifecycle, membership, role change, access-request decisions are unaudited — for a platform that will need SOC2/tenant audit trails, this is a compliance gap, not a nicety. The hand-rolled JSON serializer (`AuditAspect:131-136`) should use Jackson (prior *[M10]*). `audit_events.metadata CLOB` (`V5:51`) is unbounded and unpartitioned — add retention/partitioning before it bloats.

**Verdict (data/multi-tenancy): refactor** the key strategy + add org-scoping + Postgres-in-CI; **salvage** the Flyway-owns-schema discipline (it's correct and worth keeping).

---

## Secrets & profiles
Good: base profile fails fast without `APP_JWT_SECRET`/`APP_INTERNAL_API_KEY`/`CORS_ALLOWED_ORIGINS` (`application.properties:62,78,84`); H2 console gated to `dev` (`SecurityConfig:123-126`); internal key uses constant-time compare (`InternalApiKeyFilter:69-72`). Gaps: (1) `BootstrapRunner.isDevProfile():158-160` passes if `dev` is *among* active profiles, so `dev,ec2demo` would generate+WARN-log a password (prior *[C6]* unfixed — move to a 0600 sidecar and require profiles == exactly `[dev]`); (2) no secrets manager integration — env-var injection only, no rotation story; (3) the single shared internal key (F6) is itself a secrets-design smell. v2: AWS Secrets Manager / Vault with rotation; per-run scoped tokens; drop generated-password logging entirely.

## Observability of FL runs (the product surface)
There is essentially none. No Micrometer/Prometheus on the classpath (`build.gradle`), actuator exposes `health,loggers` only (`application.properties:53`). The only run-level signal is raw stdout text relayed to `/topic/logs/{projectId}` and `RoundResult` rows POSTed back. For a startup whose *product* is "run FL and see how it's going," you need: per-run metrics (round latency, loss/accuracy trajectory, client count, dropouts), platform metrics (runs launched/failed, time-to-first-round, executor queue depth), and correlation IDs crossing API → executor → client. Add `micrometer-registry-prometheus`, emit FL-run metrics, and (cross-ref `2026-05-27/04-observability.md`) wire OTel traceparent through the launcher env. The in-memory STOMP broker (`WebSocketConfig:34-42`) is a single-replica cap — once the API is HA, move to a Redis/RabbitMQ relay.

---

## Decision table (per subsystem)

| Subsystem | Verdict | One-line rationale |
|---|---|---|
| Spring Boot framework choice | **salvage** | Mature, fits the team and the cookie-JWT/STOMP/Flyway model; no reason to rewrite the API layer. |
| Layering (controller/service/repo) | **salvage** | Sound shape; enforce the seam (ArchUnit), decompose the `ProjectService` god-object. |
| Auth / RBAC role model | **refactor** | String roles drift (`ADMIN` vs `PLATFORM_ADMIN`); collapse to an enum, fix the masked test, add topic-level WS authz. |
| JWT / session handling | **refactor** | Add iss/aud/jti/skew + revocation (tokenVersion) + refresh-token model. |
| **FL-server orchestration (local ProcessBuilder)** | **kill** (keep as dev-only launcher) | 11-run cap, no isolation, no HA, control state lost on restart — unfit for production. |
| **FL-server orchestration (v2 target)** | **rebuild** | `fl_runs` table + launcher abstraction (k8s Jobs primary, ECS RunTask salvaged) + reconciler. |
| ECS Fargate path (current) | **salvage** | Right direction; complete with ARN persistence, stop/poll, singleton client, quota. |
| Multi-tenant data model (BIGINT/UUID split) | **refactor** | Standardize ids (UUIDv7) + add org-scoped queries / RLS before more FKs calcify it. |
| Schema migrations (Flyway discipline) | **salvage** | Flyway-owns-schema + validate-only is correct; add Postgres-in-CI, drop H2 outside dev/test. |
| `UserController` entity exposure | **refactor** | Delete/DTO-ify `/api/users`; remove stale `role='ADMIN'` comment. |
| Internal result callbacks (`ResultsController`) | **refactor** | Add per-run scoped token + object-level ownership check; one global key is an integrity break. |
| Registration / account lifecycle | **refactor** | Rate-limit + enforce `emailVerified` at login + verification flow. |
| Audit logging | **salvage** | Aspect is correct; expand coverage to all mutations, Jackson-serialize, add retention/partitioning. |
| Observability | **rebuild** | None today; add Micrometer + FL-run metrics + correlation IDs + STOMP relay for HA. |
| Secrets / profile handling | **salvage** | Fail-fast + H2 gating are good; fix dev-profile password leak, add a secrets manager. |

---

## Prioritized recommendations

### P0 — correctness/security, land this week (each small)
1. **Fix F1 end-to-end:** `ADMIN` → `PLATFORM_ADMIN` at the 5 sites, **and fix `AdminControllerIntegrationTest` to seed `PLATFORM_ADMIN`** so the test stops masking the bug. Convert `platformRole` to an enum.
2. **Fix F3:** delete or DTO-ify `/api/users`; remove the wrong `role='ADMIN'` comment.
3. **Fix F4:** `@Transactional` + pessimistic lock on `/start` now; add the `fl_runs` partial-unique-index in v2.
4. **Fix F6:** at minimum, verify the calling context owns `projectId` on internal result callbacks; design per-run tokens for v2.
5. **Fix F5:** rate-limit `/api/auth/register` + `/login` (Bucket4j); enforce `emailVerified` at login or set `status=PENDING` on register.
6. Drop anonymous `AuthenticationException` subclasses (`JwtChannelInterceptor:71,78`); SIGTERM-before-SIGKILL on shutdown.

### P1 — production-readiness (weeks)
7. **Rebuild FL orchestration:** `fl_runs` table (V6) + `FlServerLauncher` abstraction + reconciler; k8s Jobs as primary executor, complete the ECS path, hard-gate `LocalProcessLauncher` to `dev`.
8. **Postgres everywhere outside dev/test** + Testcontainers-Postgres CI that runs the real migrations.
9. **WS topic-level authz** (`SUBSCRIBE` → `requireParticipant`); STOMP relay (Redis/RabbitMQ) for HA.
10. **Observability:** Micrometer/Prometheus, FL-run metrics, correlation IDs through the launcher.
11. Harden JWT (iss/aud/jti/skew + revocation); per-run scoped internal tokens; secrets manager + rotation.

### P2 — hardening & maintainability
12. Org-scoped query layer or Postgres RLS; standardize id strategy (UUIDv7).
13. Expand `@Auditable` to all mutations + Jackson serialize + retention/partitioning.
14. Decompose `ProjectService`; constructor injection everywhere; ArchUnit rule "controllers never touch repositories"; drop Lombok.

---

## Key files
- `flower/FlowerServerManager.java` — FL lifecycle, the scaling cliff (F2), the narrowed-but-open race (F4), ECS path gaps.
- `controller/AdminController.java` + `service/AdminService.java` + `service/AuthorizationService.java` + `service/CustomUserDetailsService.java` + `bootstrap/BootstrapRunner.java` — the dead-admin chain (F1).
- `src/test/.../admin/AdminControllerIntegrationTest.java` — the test that masks F1.
- `controller/UserController.java` — PII leak (F3) + stale comment.
- `controller/ResultsController.java` + `security/InternalApiKeyFilter.java` — internal-callback object-auth gap (F6).
- `security/{JwtTokenProvider,JwtHandshakeInterceptor,JwtChannelInterceptor}.java` — JWT hardening (F7), WS topic authz gap.
- `resources/db/migration/V1__init.sql`, `V5__identity_foundations.sql` — id strategy (F8), tenant isolation (F9), Postgres-vs-H2 (F10).
- `resources/application*.properties` — profiles/secrets, 11-port cap, actuator exposure.
