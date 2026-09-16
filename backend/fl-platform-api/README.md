# FedLearn Platform — Backend API

Spring Boot 3 / Java 21 / Gradle. REST + STOMP-over-WebSocket. Manages users, projects, training results, and the lifecycle of Python FL-server processes.

For Spring profile semantics, deployment topology, and the cookie auth model, see [Spring profiles](#spring-profiles), [Auth contract](#auth-contract), and [Deployment](#deployment) below.

## Stack

- **Spring Boot 3.4.5** + **Spring Security**
- **Java 21** + **Gradle** (wrapper committed; do not switch to Maven)
- **JWT** delivered to browsers as an **HttpOnly cookie** — no JS-readable token in the body
- **WebSocket / STOMP** for real-time logs, status, round results and inference tokens
- **JPA / Hibernate** in `validate`-only mode — schema is owned by **Flyway** (highest migration: `V23`)
- **PostgreSQL** in every profile — H2 has been retired. Local dev runs against `docker compose up -d` (`postgres:16.6-alpine`); tests use **Testcontainers**

## Spring profiles

| Profile | Purpose | Activation |
|---|---|---|
| (base) | `application.properties`. Refuses to boot without `APP_JWT_SECRET`, `APP_INTERNAL_API_KEY`, `CORS_ALLOWED_ORIGINS` — these have **no fallback**, deliberately. Sets `ddl-auto=validate` + the Postgres datasource for everyone. | always loaded |
| `dev` | Public dev secrets, permissive CORS (`http://localhost:*`, `http://127.0.0.1:*`, `http://[::1]:*`, `file://`), `cookie.secure=false`. Inherits the base Postgres datasource — start one with `docker compose up -d` first. **Never activate in any deployed env.** | `SPRING_PROFILES_ACTIVE=dev` / `./launch_all.sh` |
| `test` | Testcontainers Postgres (`jdbc:tc:postgresql:16.6-alpine:///fedlearn_test`), Flyway disabled, Hibernate `create-drop`, public test secrets, Hikari pool capped at 4. Needs a running Docker daemon. | `@ActiveProfiles("test")` (per test class) |
| `ec2demo` | Single-EC2 demo. Postgres (local container or RDS via `SPRING_DATASOURCE_URL`), FL servers as local Python processes, `RemoteIpValve` on (nginx forwards `X-Forwarded-*`), `cookie.secure` env-driven (`APP_AUTH_COOKIE_SECURE`, default `false`), `SameSite=Lax`, **`app.fl.require-tls=true`**. | `SPRING_PROFILES_ACTIVE=ec2demo` (set by `scripts/deploy-to-aws.sh`) |
| `production` | Hardened single-VM: PostgreSQL required from env (no fallbacks), FL servers as local processes, graceful shutdown, cookie `Secure=true` + `SameSite=Strict` (SE-21), `/actuator/health` details admin-only (SE-24), **`app.fl.require-tls=true`**. ECS/Fargate is **not** implemented — see [Deployment](#hardened-single-vm-production). | `SPRING_PROFILES_ACTIVE=production` |

Three `@PostConstruct` validators fail the boot rather than let a misconfiguration surface mid-federation:

- `FlOrchestrationModeValidator` — **every profile**: a non-blank `ecs.cluster-name` (see [production](#hardened-single-vm-production)).
- `FlBoundaryAuthPolicyValidator` (SE-14) — **`ec2demo`/`production` only**: `require-client-auth=true` with `require-tls=false`, which would ride connection tokens over plaintext gRPC. Override with `app.fl.allow-plaintext-client-auth=true` when the network is already encrypted (Tailscale / loopback / private VPC).
- `FlSecretDistinctnessValidator` (SE-20) — **`ec2demo`/`production` only**: `app.fl.token-secret` resolving equal to `app.jwt.secret`, which would let a compromise of the network-facing FL server mint web/admin sessions. Local profiles keep the convenience fallback.

## Quick start

```bash
cd backend/fl-platform-api
docker compose up -d                          # PostgreSQL on localhost:5432 (db/user/pass: federance)
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun
```

The app starts on `:8081`. The datasource defaults to `jdbc:postgresql://localhost:5432/federance`; override anywhere with `SPRING_DATASOURCE_URL` / `_USERNAME` / `_PASSWORD`.

> **Run it from `backend/fl-platform-api/`.** The FL-runtime script paths default to `../../fl-runtime/…` and are resolved against the JVM's working directory — see [Where the Python lives](#where-the-python-lives).

### Tests

```bash
./gradlew test
./gradlew test --tests "com.federated.fl_platform_api.SomeTest"
./gradlew test jacocoTestCoverageVerification   # what CI actually gates on
```

Tests run against **real PostgreSQL via Testcontainers** — `jdbc:tc:postgresql:16.6-alpine:///fedlearn_test` starts/reuses a throwaway container for the JVM run, so a working **Docker daemon** is required. `docker compose up -d` is *not* needed for tests. The profile is activated per test class by `@ActiveProfiles("test")`, not forced by `build.gradle`.

`spring.datasource.hikari.maximum-pool-size=4` (with `minimum-idle=0`) is **deliberate, not a leftover**: ~20 cached `@SpringBootTest` contexts all pool against the one shared container, and at Hikari's default of 10 the summed pools overshoot Postgres's `max_connections` — a context then fails to start with "too many clients" under full-suite load while every test still passes in isolation.

The bulk suite runs Hibernate `create-drop` with Flyway disabled; **never** change this — Flyway migrations must validate against `dev`/`ec2demo`/`production` only. The twelve `V*MigrationTest` classes flip Flyway back on (with their own per-test Testcontainers database) to exercise the real migrations against real Postgres.

**The CI gate is `./gradlew test jacocoTestCoverageVerification` with `SPRING_PROFILES_ACTIVE=test`, not bare `./gradlew test`.** `build.gradle` sets a JaCoCo bundle **line-coverage floor of 0.70** (TE-11) — a coverage regression fails the job even when every test passes.

### Build

```bash
./gradlew build           # full build with tests
./gradlew bootJar         # fat JAR only
```

## Project structure

```
backend/fl-platform-api/
├── src/main/java/com/federated/fl_platform_api/
│   ├── audit/          # @Auditable + AuditAspect — writes audit_events rows
│   ├── bootstrap/      # BootstrapRunner (seeds first admin + org), StartupReconciler, ProcessProbe
│   ├── config/         # SecurityConfig, WebSocketConfig, and the three fail-fast startup validators
│   ├── controller/     # 24 @RestControllers — see the API surface table below
│   ├── dto/            # Request/response DTOs (LoginRequest, CreateProjectRequest, StartProject, ...)
│   ├── email/          # EmailService + LoggingEmailService / SmtpEmailService
│   ├── exception/      # GlobalExceptionHandler + typed exceptions
│   ├── model/          # JPA entities + enums (Project, Run, TrainingArm, PlatformRole, ...)
│   ├── orchestration/  # FlServerManager + the FlServerProcessRunner seam — spawns/manages FL servers
│   ├── repository/     # Spring Data JPA repositories
│   ├── security/       # JwtTokenProvider, the JWT/STOMP filters, OrgScopeFilter, ConnectionTokenService
│   ├── service/        # Business logic (ProjectService, RunService, ModelRecipeService, ...)
│   ├── validation/     # @ValueOfEnum constraint
│   └── FlPlatformApiApplication.java
├── src/main/resources/
│   ├── application.properties               # base — env-var-driven; three vars have NO fallback
│   ├── application-dev.properties           # local-dev convenience
│   ├── application-ec2demo.properties       # single-EC2 demo
│   ├── application-production.properties    # hardened single-VM
│   └── db/migration/                        # Flyway versioned migrations (V1 … V23)
├── src/test/resources/
│   └── application-test.properties          # Testcontainers Postgres, public test secrets
├── docker-compose.yml  # local-dev PostgreSQL (postgres:16.6-alpine)
├── DEVELOPMENT.md      # Deeper backend dev notes
└── build.gradle
```

> `ls` sorts migrations lexicographically, so `V5`–`V9` print *after* `V21` — don't read the last line as the highest version. Recent ones: `V20__project_derivation.sql` (opt-in pretrained-base derivation columns), `V21__base_ref_unique_index.sql` (one `BASE_REF` per org+model, backing a race-safe find-or-create), `V22__project_training_arm.sql` and `V23__training_arm_ova_lp.sql` (see [Training arms](#training-arms)).

## Auth contract

| Cookie | Set by | Attributes |
|---|---|---|
| `jwtToken` | `POST /api/auth/login` | `HttpOnly`, `Secure` (`app.auth.cookie.secure`), `SameSite` (`app.auth.cookie.same-site` — `Lax` locally, `Strict` on `production`), `Max-Age` = the JWT's own lifetime (SE-8) |

The frontend sends `withCredentials: true` on every Axios call; the cookie flows automatically. **The browser never receives a JS-readable token** — a browser login response carries identity only.

Three details worth knowing before you touch the security package:

- **`POST /api/auth/register` does not log you in.** It returns `201` with `{"message": ..., "userId": ...}` and sets **no** cookie. Call `/login` afterwards.
- **Audience scoping (SE-20).** The web JWT carries `aud: fedlearn-web`, and verification *requires* it — so an FL connection token (`aud: fedlearn-fl-server`) or a legacy audience-less token signed with the same HMAC key cannot be replayed against the web surface. Preserve that check when editing `JwtTokenProvider`.
- **`Authorization: Bearer` is accepted from native clients only (SE-9).** Desktop/mobile self-identify with a non-blank `X-FedLearn-Client` header; only then does the login response include `accessToken` and only then does the filter honour a Bearer header. Absent the marker, Bearer does nothing — the browser path is strictly cookie-only.

`POST /api/auth/login` response body (identity only; the JWT is in the `Set-Cookie` header):

```json
{
  "username": "anurag",
  "email": "anurag@example.com",
  "role": "PROJECT_OWNER"
}
```

`GET /api/auth/me` returns the same three fields, and 401s (not 403s) when unauthenticated so the SPA can probe session validity without triggering its logout redirect. `POST /api/auth/logout` revokes the token's `jti` as well as clearing the cookie.

### Roles and tenancy

Three independent layers, all wired:

| Layer | Column / enum | Values |
|---|---|---|
| Platform role | `users.platform_role` → `PlatformRole` | `USER` (default; may join/train) → `PROJECT_OWNER` (may create projects; admin-granted) → `PLATFORM_ADMIN` |
| Org role | `organization_memberships.org_role` → `OrgRole` | `OWNER`, `ADMIN`, `MEMBER` |
| Project membership | `project_memberships.role` → `MembershipRole` | `OWNER`, `MEMBER`, `CLIENT` |

Platform roles map to `ROLE_*` authorities and are reloaded from the database on every request, so a role change takes effect without re-login. Project visibility is three-tier (`ProjectVisibility`): `PUBLIC` (auto-join), `RESTRICTED` (owner-approved access request), `PRIVATE` (invite-only).

**`OrgScopeFilter` only *populates* the request-scoped `OrgScope` — it is not a security boundary.** It resolves the caller's visible org ids (platform admins are marked unrestricted; a user with no memberships falls back to the single bootstrap org) and always calls `filterChain.doFilter`. It never denies a request. Actual tenant enforcement happens downstream in `AuthorizationService` / the query layer — don't mistake the filter for the gate.

## API surface

24 `@RestController`s. The highlights, grouped:

| Method | Path | Notes |
|---|---|---|
| POST | `/api/auth/register` | Create account → `201 {message, userId}`. No cookie. |
| POST | `/api/auth/login` | Authenticate, set `jwtToken` cookie |
| GET | `/api/auth/me` | Bootstrap probe — 401 with no logout side-effect |
| POST | `/api/auth/logout` | Clear cookie + revoke the token's `jti` |
| GET / POST | `/api/projects` | List the caller's projects / create one |
| GET | `/api/projects/discover` | Discoverable (`PUBLIC` + `RESTRICTED`) projects |
| GET / **PATCH** | `/api/projects/{id}` | Read / partial update (**there is no `PUT`**) |
| POST | `/api/projects/{id}/start` \| `/stop` | Spawn / kill the FL server |
| GET | `/api/projects/{id}/results` | Round-by-round metrics |
| GET | `/api/projects/{id}/logs` | Paged log history (`?page=&size=`, default 200); live stream is over WebSocket |
| GET | `/api/projects/{id}/logs/export` | Whole log as a `text/plain` attachment |
| POST / GET | `/api/projects/{id}/deletion-request` | Owner files for deletion / reads its status |
| DELETE | `/api/projects/{id}` | **Platform-admin only.** Owners go through the deletion-request workflow. |
| GET / POST / DELETE | `/api/projects/{id}/memberships[/{userId}]` | List / add / remove participants |
| POST / GET / PUT | `/api/projects/{id}/access-requests[/{requestId}]` | Join-request workflow for `RESTRICTED` projects (mirrored by `GET /api/my/access-requests`) |
| GET | `/api/model-recipes` | The recipe catalog, read from `fl-runtime/recipes.py --describe` |
| GET / POST | `/api/client/projects`, `/{id}/join`, `/{id}/connection` | Drives the desktop/mobile "models I can train" flow; `/connection` mints the `FEDLEARN_CONNECTION_TOKEN` (SE-14) |
| GET / POST | `/api/runs/{runId}/status` \| `/manifest` \| `/enroll` \| `/model-bundle` \| `/files/{name}` | Run lifecycle + on-device training bundle delivery |
| GET | `/api/artifacts`, `/{id}`, `/{id}/blob`, `/{id}/lineage`, `/latest` | Content-addressed model registry + lineage |
| GET / POST / DELETE | `/api/marketplace/adapters[/{id}/publish]` | Browse published adapters; publish / unpublish one |
| GET / POST | `/api/inference/models`, `/api/inference/{id}` \| `/generate` \| `/generate/stop` | "Use a model" |
| GET / PUT / POST | `/api/admin/**` | Admin surface (`@PreAuthorize("hasRole('PLATFORM_ADMIN')")`): overview, user + project search, role and status changes, audit events, owner-promotion and deletion approvals, benchmarks, SMTP smoke test |
| POST | `/api/owner-requests`, GET `/mine` | Owner-promotion request workflow |
| GET / PATCH | `/api/users/me/profile` | Self-service profile (chain-level `permitAll`; the controller 401s anonymous callers itself) |
| POST | `/api/internal/**` | **Service-to-service only.** `InternalApiKeyFilter` requires `X-Internal-Key` before Spring Security sees the request. Round results, benchmark ingest, final-model artifact upload, connection-token verification. |

`SecurityConfig`'s `publicPaths` allowlist is exactly `/api/auth/**`, `/ws-logs/**`, `/error`, `/actuator/health`. Two more matchers are `permitAll` at the chain level and carry their own gate instead — `/api/internal/**` (behind `InternalApiKeyFilter`) and `/api/users/me/profile` (the controller 401s anonymous callers), both as noted above. `/actuator/**` beyond health is admin-only (SE-5); everything else requires authentication. For the canonical list, browse `controller/`.

## WebSocket / STOMP

| | |
|---|---|
| Endpoint | `/ws-logs` (allowed origins share the REST CORS allowlist) |
| Auth | Same `jwtToken` cookie — `JwtHandshakeInterceptor` at handshake, `JwtChannelInterceptor` at CONNECT |
| Authorization | `StompSubscriptionInterceptor` (BA-5) re-runs the org-scope + participant check on every `SUBSCRIBE`, so an authenticated user cannot subscribe to another tenant's project stream |
| Topics | `/topic/logs/{projectId}`, `/topic/status/{projectId}`, `/topic/results/{projectId}`, `/topic/inference/{projectId}`, plus per-user `/user/{username}/queue/notifications` |

The broker is Spring's in-memory `SimpleBroker` — fine for a single replica; a multi-instance deploy needs a relay (RabbitMQ/Redis).

Client side, the SPA owns one STOMP lifecycle in `frontend/src/hooks/useStompClient.ts`:

```javascript
const client = new Client({
  brokerURL: `${wsBase}/ws-logs`,
  reconnectDelay: 5000,
});
client.onConnect = () => {
  client.subscribe(`/topic/logs/${projectId}`, (msg) => { /* ... */ });
};
client.activate();
```

## FL-server orchestration

`FlServerManager` (package `orchestration`) is the entry point. On `POST /api/projects/{id}/start` it:

1. Gates the start: the SE-11 DP policy check (a `regulated` project must have a complete DP config) and the SE-10 catalog check (`modelType` must be an exact-case key in `GET /api/model-recipes`) both run **before** any spawn.
2. Reserves a port from `fl.server.port-range.start..end` (default `50000-50010`), holding the reservation for the child's whole life rather than just the startup probe (BA-13).
3. Builds the argv — every project-derived string is allowlisted against a character class first (SE-10), so nothing can option-inject or path-traverse into `fl_server.py`.
4. Delegates the launch to the **`FlServerProcessRunner`** seam (DA-8). The default `LocalProcessFlServerRunner` runs `bash ../../fl-runtime/run_fl_server.sh ...`; the `FoT` strategy selects `python.script.fot-server.path` (`run_fot_server.sh`) instead.
5. Rebuilds the child's environment **from an allowlist** (SE-17) rather than inheriting-then-subtracting, so the FL server — which is network-facing and loads datasets — inherits *no* backend secret: the DB password, the `APP_*` vars, cloud creds and CORS origins are all dropped, and the web JWT secret with them, so a compromise of the FL boundary cannot forge web/admin sessions. It is then handed three secrets explicitly, and only these three: `FEDLEARN_INTERNAL_API_KEY` (the value of `app.internal.api-key`, which its `/api/internal/**` callbacks need), `FEDLEARN_FL_TOKEN_SECRET` (the connection-token verify secret, deliberately distinct from the JWT secret), and its own per-run `FEDLEARN_INTERNAL_RUN_TOKEN` (SE-7), which scopes those callbacks to this run's project.
6. Captures merged stdout+stderr line-by-line and broadcasts via `WebSocketService` to `/topic/logs/{projectId}`, then probes for `fl.server.startup-probe-seconds` (default 3s). A **non-zero** in-window exit is a startup crash and surfaces with the captured output; a **zero** exit is a legitimately fast completed run and is not treated as a failure.
7. Tracks each child as a `ProcessHandle` in a `ConcurrentHashMap<UUID, ProcessHandle>` — a handle, not a `Process`, so a restarted JVM can re-adopt an orphan (BA-3, via `StartupReconciler`). `/stop` calls `destroyForcibly()`.

Keeping the raw `ProcessBuilder` mechanics behind `FlServerProcessRunner` is what makes the spawn path unit-testable with a fake runner (no real process), and it is the extension point where a future managed-task runner would slot in. All policy — argv building, the env scrub, port reservation, run-state persistence, the startup probe, log broadcasting — stays in the manager.

### Where the Python lives

The backend never runs the `framework/` library directly. It shells out to the repo-root **`fl-runtime/`** scripts, resolved from `application.properties:149-154`:

```properties
python.executable.path=${PYTHON_EXECUTABLE_PATH:../../fl-runtime/run_init_model.sh}
python.script.fl-server.path=${PYTHON_SCRIPT_FL_SERVER_PATH:../../fl-runtime/run_fl_server.sh}
python.script.infer.path=${PYTHON_SCRIPT_INFER_PATH:../../fl-runtime/run_infer.sh}
python.script.recipes.path=${PYTHON_SCRIPT_RECIPES_PATH:../../fl-runtime/run_recipes.sh}
# and, on FlServerManager: python.script.fot-server.path → ../../fl-runtime/run_fot_server.sh
```

(`python.executable.path` is a legacy name — it holds the **init-model wrapper** path that `ModelInitializer` runs, not a Python binary.)

**Those `../../` defaults are resolved against the JVM's working directory**, and every spawn sets the child's working dir to `new File(".")` — which is why the backend is normally launched from `backend/fl-platform-api/`. Start it from anywhere else and recipe loading, model init, inference and FL-server spawn all fail to find their scripts. A deployed host sidesteps this by overriding the `PYTHON_*` env vars with absolute paths (`scripts/ec2-bootstrap.sh` does exactly that in the systemd unit).

The wrapper is a shell script so the same backend code works on Mac/Linux/Windows; there's a parallel `.bat` for Windows local dev.

### Training arms

A project stores **which parameters it trains and federates**, and an arm carries an *objective*, not just a parameter subset:

| Arm | Meaning |
|---|---|
| `FULL` | Every parameter is trainable; the whole model rides the wire. The default. |
| `FROZEN_HEAD` | Backbone frozen, head only — dramatically cheaper per round. |
| `OVA_LP` | Same frozen encoder as `FROZEN_HEAD`, but trained under C independent one-vs-all binary classifiers instead of one softmax (arXiv:2511.05028). Federates the same parameters as `FROZEN_HEAD`; only the objective distinguishes the two. |

The chain: frontend picker → `CreateProjectRequest` / `StartProject` (regex-validated) → `TrainingArm` enum → `projects.training_arm` → `--training-arm` on the spawned `fl_server.py` and on `client.py` (delivered to clients via `GET /api/client/projects/{id}/connection`). The flag is emitted only when the arm is **not** `FULL`, so an existing project's argv is byte-identical to before — `recipes.validate_arm()` resolves an omitted arm to `FULL` whenever the recipe supports it (all seven currently do).

Two authorities, deliberately split:

- **This enum is the vocabulary**, and `V22`'s `chk_projects_training_arm` CHECK constraint is its last line of defence — DTO validation can be bypassed by any direct writer (a migration, an ops script), and an unrecognised arm would otherwise fail at FL-server spawn rather than at write time. `V23` widened that CHECK for `OVA_LP`. **Adding a `TrainingArm` constant therefore requires a new migration widening the CHECK** — `V22TrainingArmMigrationTest` asserts every enum constant is accepted, so the split-brain fails a test instead of a user's federation.
- **Which arms a recipe actually supports is declared in `fl-runtime/recipes.py`** (`supported_arms` / `trainable_spec`) and validated there. `CIFAR_RESNET18` is currently the only recipe offering all three; `PNEUMONIA_CNN` and `CNN` offer `FULL` + `FROZEN_HEAD`; the rest are `FULL`-only. The picker's per-recipe trade-off copy comes from measured results, not from this layer.

> **On Flower/`flwr`:** the FL framework is genuinely custom — no Flower server, client, or strategy semantics anywhere, and its own protobuf contract. Don't add Flower FL abstractions. **The dependency itself is gone too**: this unit's `requirements.txt` no longer pins `flwr` or `flwr-datasets` (the CIFAR-10 IID shard they were used for is reproduced natively in `fl-runtime/recipes.py`, byte-identical per partition). Dropping them cleared both caps they dragged in — `cryptography` is now pinned at `46.0.7`, at the framework's security floor, and `protobuf` at `5.29.5`, which is what the FoT gencode actually requires. Any doc describing the `cryptography<45.0.0` constraint as an open "SE-22 residual" is stale.

## Deployment

### Local

`./launch_all.sh` (root) starts backend + frontend + Electron + FL-client launcher. Or run components individually as described above. macOS-only — it drives Terminal via AppleScript.

### EC2 demo (`ec2demo`)

The demo host defaults to `fedlearn.duckdns.org` (`FEDLEARN_DOMAIN` in [`scripts/deploy-to-aws.sh`](../../scripts/deploy-to-aws.sh) and [`scripts/ec2-bootstrap.sh`](../../scripts/ec2-bootstrap.sh)). TLS / reverse-proxy setup: [`deploy/TLS.md`](../../deploy/TLS.md) and [`deploy/nginx/fedlearn.conf`](../../deploy/nginx/fedlearn.conf).

`ec2-bootstrap.sh` installs a `fedlearn.service` systemd unit. **Secrets are never inline in the unit** — unit files are world-readable, so they live in a `0600 root:root` env file loaded by systemd:

```ini
# /etc/systemd/system/fedlearn.service (excerpt)
EnvironmentFile=/etc/fedlearn/secrets.env       # APP_JWT_SECRET, APP_INTERNAL_API_KEY,
                                                # APP_FL_TOKEN_SECRET, SPRING_DATASOURCE_PASSWORD
Environment="SPRING_PROFILES_ACTIVE=ec2demo"
Environment="PYTHON_SCRIPT_FL_SERVER_PATH=/…/fl-runtime/run_fl_server.sh"   # absolute — no CWD dependency
Environment="APP_AUTH_COOKIE_SECURE=true"       # emitted once a Let's Encrypt cert exists
Environment="FEDLEARN_GRPC_SERVER_KEY=/etc/fedlearn/grpc/server.key"
Environment="FEDLEARN_GRPC_SERVER_CERT=/etc/fedlearn/grpc/server.crt"
# CORS_ALLOWED_ORIGINS is left commented out — set it to your frontend origin(s) before first start.
```

The secrets file is generated once and never clobbered by a re-bootstrap; delete it to rotate. `APP_FL_TOKEN_SECRET` is deliberately distinct from `APP_JWT_SECRET` (SE-7) so a compromise of the network-facing FL boundary cannot forge web sessions.

nginx terminates TLS on `:443` and proxies to `127.0.0.1:8081`. Nothing in this repo firewalls that port — the app binds `0.0.0.0:8081` (`server.address`) and the bootstrap configures no `ufw`/iptables rule, so **keeping `:8081` closed in the EC2 security group is an operator step**, recorded as an accepted risk conditional on exactly that in [`deploy/TLS.md`](../../deploy/TLS.md). The `/ws-logs` location carries the STOMP upgrade headers and `proxy_read_timeout 3600s` for long-lived training rounds. FL-server gRPC ports (`50000-50010`) bypass nginx entirely, which is why `ec2demo` sets `app.fl.require-tls=true` and the bootstrap provisions a gRPC keypair.

### Hardened single-VM (`production`)

FL servers run as local processes — the only supported deployed architecture.

**ECS/Fargate orchestration is not implemented** (OP-14). `FlOrchestrationModeValidator` **fails the boot** in every profile when `ecs.cluster-name` (`ECS_CLUSTER_NAME`) is non-blank, so the gap surfaces at startup rather than on the first federation; `FlServerManager` also guards the path with an `UnsupportedOperationException` as a backstop. Leave it unset. Remaining work for the managed-task path — S3 model storage, FL servers as `RunTask` invocations, multi-replica safety, ALB target-group idle timeouts — is deferred to **OP-12**.

## Adjacent docs

- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** — deeper architectural walkthroughs and contribution patterns
- **[`wikis/backend/`](../../wikis/backend/)** — long-form wiki: architecture, security, project lifecycle, FL orchestration, WebSocket streaming, identity/multitenancy, artifact registry
