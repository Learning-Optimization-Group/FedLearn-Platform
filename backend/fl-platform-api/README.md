# FedLearn Platform — Backend API

Spring Boot 3 / Java 21 / Gradle. REST + STOMP-over-WebSocket. Manages users, projects, training results, and the lifecycle of Python FL-server processes.

For Spring profile semantics, deployment topology, and the cookie auth model, see [Spring profiles](#spring-profiles), [Cookie auth contract](#cookie-auth-contract), and [Deployment](#deployment) below.

## Stack

- **Spring Boot 3** + **Spring Security**
- **Java 21** + **Gradle** (wrapper committed; do not switch to Maven)
- **JWT** delivered as **HttpOnly cookies** — no Bearer headers, no JS-readable token
- **WebSocket / STOMP** for real-time logs and round-result telemetry
- **JPA / Hibernate** in `validate`-only mode — schema is owned by **Flyway**
- **PostgreSQL** in every profile — H2 has been retired. Local dev runs against `docker compose up -d` (`postgres:16.6-alpine`); tests use **Testcontainers**

## Spring profiles

| Profile | Purpose | Activation |
|---|---|---|
| (base) | `application.properties`. Refuses to boot without `APP_JWT_SECRET`, `APP_INTERNAL_API_KEY`, `CORS_ALLOWED_ORIGINS`. | always loaded |
| `dev` | Public dev secrets, permissive CORS (`http://localhost:*`), `cookie.secure=false`. Inherits the base Postgres datasource — start one with `docker compose up -d` first. **Never activate in any deployed env.** | `SPRING_PROFILES_ACTIVE=dev` / `./launch_all.sh` |
| `test` | Testcontainers Postgres (`jdbc:tc:postgresql:16.6-alpine`), Flyway disabled, Hibernate `create-drop`, public test secrets, quiet logging. Needs a running Docker daemon. | `@ActiveProfiles("test")` (per test class) |
| `ec2demo` | Single-EC2 demo at `https://fedlearn.duckdns.org`. Postgres (local container or RDS via `SPRING_DATASOURCE_URL`), FL servers as local Python processes, `cookie.secure` env-driven (`APP_AUTH_COOKIE_SECURE`, default `false`). | `SPRING_PROFILES_ACTIVE=ec2demo` (set by `scripts/deploy-to-aws.sh`) |
| `production` | Hardened single-VM: PostgreSQL (required from env, no fallbacks), FL servers as local processes, graceful shutdown. ECS/Fargate is **not** implemented — see [Deployment](#hardened-single-vm-production). | `SPRING_PROFILES_ACTIVE=production` |

## Quick start

```bash
cd backend/fl-platform-api
docker compose up -d                          # PostgreSQL on localhost:5432 (db/user/pass: federance)
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun
```

The app starts on `:8081`. The datasource defaults to `jdbc:postgresql://localhost:5432/federance`; override anywhere with `SPRING_DATASOURCE_URL` / `_USERNAME` / `_PASSWORD`.

### Tests

```bash
./gradlew test
./gradlew test --tests "com.federated.fl_platform_api.SomeTest"
```

Tests run against **real PostgreSQL via Testcontainers** — `jdbc:tc:postgresql:16.6-alpine:///fedlearn_test` starts/reuses a throwaway container for the JVM run, so a working **Docker daemon** is required. `docker compose up -d` is *not* needed for tests. The profile is activated per test class by `@ActiveProfiles("test")`, not forced by `build.gradle`.

The bulk suite runs Hibernate `create-drop` with Flyway disabled; **never** change this — Flyway migrations must validate against `dev`/`ec2demo`/`production` only. The dedicated `V*MigrationTest` classes flip Flyway back on to exercise the real migrations against real Postgres.

### Build

```bash
./gradlew build           # full build with tests
./gradlew bootJar         # fat JAR only
```

## Project structure

```
backend/fl-platform-api/
├── src/main/java/com/federated/fl_platform_api/
│   ├── config/         # SecurityConfig, WebSocketConfig, CorsConfig
│   ├── controller/     # AuthController, ProjectController, ResultsController, UserController
│   ├── dto/            # Request/response DTOs (LoginRequest, RegisterRequest, ProjectResponseDto, ...)
│   ├── exception/      # GlobalExceptionHandler + typed exceptions
│   ├── model/          # JPA entities (User, Project, RoundResult)
│   ├── orchestration/  # FlServerManager + the FlServerProcessRunner seam — spawns/manages FL-server processes
│   ├── repository/     # Spring Data JPA repositories
│   ├── security/       # JwtTokenProvider, JwtAuthenticationFilter, JwtHandshakeInterceptor (for STOMP)
│   ├── service/        # Business logic
│   └── FlPlatformApiApplication.java
├── src/main/resources/
│   ├── application.properties               # base — env-var-driven, no fallbacks
│   ├── application-dev.properties           # local-dev convenience
│   ├── application-ec2demo.properties       # single-EC2 demo
│   ├── application-production.properties    # hardened single-VM
│   └── db/migration/                        # Flyway versioned migrations (highest: V19)
├── src/test/resources/
│   └── application-test.properties          # Testcontainers Postgres, public test secrets
├── docker-compose.yml  # local-dev PostgreSQL (postgres:16.6-alpine)
├── DEVELOPMENT.md      # Deeper backend dev notes
└── build.gradle
```

## Cookie auth contract

| Cookie | Set by | Attributes |
|---|---|---|
| `jwtToken` | `/api/auth/login`, `/api/auth/register` | `HttpOnly`, `SameSite=Lax` (or `Strict` in prod), `Secure` flag controlled by `app.auth.cookie.secure` |

The frontend sends `withCredentials: true` on every Axios call; the cookie flows automatically. No token appears in any response body.

`POST /api/auth/login` returns the user's profile only — the JWT lives entirely in the `Set-Cookie` response header:

```json
{
  "id": 1,
  "username": "anurag",
  "email": "anurag@example.com",
  "roles": ["USER"]
}
```

## REST endpoints (overview)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register` | – | Create account, set `jwtToken` cookie |
| POST | `/api/auth/login` | – | Authenticate, set `jwtToken` cookie |
| POST | `/api/auth/logout` | ✓ | Clear cookie |
| GET | `/api/auth/me` | silent 401 | Bootstrap probe — returns user or 401 with no logout side-effect |
| GET | `/api/projects` | ✓ | List the caller's projects |
| POST | `/api/projects` | ✓ | Create a project |
| GET / PUT / DELETE | `/api/projects/{id}` | ✓ | CRUD |
| POST | `/api/projects/{id}/start` | ✓ | Spawn FL server for project |
| POST | `/api/projects/{id}/stop` | ✓ | Kill FL server |
| GET | `/api/results/{projectId}` | ✓ | Round-by-round metrics |
| GET | `/api/projects/{id}/logs` | ✓ | Persisted log history (the live stream is over WebSocket) |

For the canonical list, browse `controller/`.

## WebSocket / STOMP

| | |
|---|---|
| Endpoint | `/ws-logs` (relative to backend host) |
| Auth | Same `jwtToken` cookie — validated by `JwtHandshakeInterceptor` |
| Topics | `/topic/logs/{projectId}` (training output), `/topic/status/{projectId}` (server lifecycle), `/topic/results/{projectId}` (round results) |

Client example (the SPA does this in `services/logStore.ts`):

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

1. Reserves a port from `fl.server.port-range.start..end` (default `50000-50010`).
2. Builds the argv and delegates the launch to the **`FlServerProcessRunner`** seam (DA-8) — the default `LocalProcessFlServerRunner` runs `bash ../../fl-runtime/run_fl_server.sh ...` (`python.script.fl-server.path`). The `FoT` strategy selects `python.script.fot-server.path` (`run_fot_server.sh`) instead.
3. Captures merged stdout+stderr line-by-line and broadcasts via `WebSocketService` to `/topic/logs/{projectId}`.
4. Tracks each child as a `ProcessHandle` in a `ConcurrentHashMap<UUID, ProcessHandle>` — a handle, not a `Process`, so a restarted JVM can re-adopt an orphan (BA-3). `/stop` calls `destroyForcibly()`.

The Python lives in the repo-root `fl-runtime/` directory, not under `src/main/resources/`. The shell wrapper exists so the entry point is portable across local dev (Mac, Linux) and the EC2 host; there's a parallel `.bat` for Windows local dev.

Keeping the raw `ProcessBuilder` mechanics behind `FlServerProcessRunner` makes the spawn path unit-testable with a fake runner, and is the extension point where a future managed-task runner would slot in.

> **On Flower/`flwr`:** the FL framework is genuinely custom — no Flower server, client, or strategy semantics, and its own protobuf contract. But this unit's `requirements.txt` *does* pin `flwr==1.20.0` + `flwr-datasets==0.5.0`, used **only** for dataset partitioning (`FederatedDataset` in `fl_server.py` / `client.py`). Known wart: that pulls in a `cryptography<45.0.0` constraint, which makes the framework's `>=46.0.6` floor unreachable here — tracked as the SE-22 residual and documented at `requirements.txt:16`.

## Deployment

### Local

`./launch_all.sh` (root) starts backend + frontend + Electron + FL-client launcher. Or run components individually as described above.

### EC2 demo (`ec2demo`)

Live at **https://fedlearn.duckdns.org**. TLS / reverse-proxy setup: [`deploy/TLS.md`](../../deploy/TLS.md) and [`deploy/nginx/fedlearn.conf`](../../deploy/nginx/fedlearn.conf).

The EC2 instance runs `fedlearn.service` (systemd) which sources its env from `/etc/systemd/system/fedlearn.service`:

```ini
Environment="SPRING_PROFILES_ACTIVE=ec2demo"
Environment="APP_JWT_SECRET=..."           # openssl rand -base64 64
Environment="APP_INTERNAL_API_KEY=..."     # openssl rand -hex 32
Environment="CORS_ALLOWED_ORIGINS=https://fedlearn.duckdns.org,http://localhost:5173"
Environment="APP_AUTH_COOKIE_SECURE=true"
```

nginx terminates TLS on `:443` and proxies to `127.0.0.1:8081` — port 8081 is **not** publicly exposed. STOMP upgrade headers and a `proxy_read_timeout 3600s` are required for long-lived training rounds.

### Hardened single-VM (`production`)

FL servers run as local processes — the only supported deployed architecture.

**ECS/Fargate orchestration is not implemented** (OP-14). `FlOrchestrationModeValidator` **fails the boot** in every profile when `ecs.cluster-name` (`ECS_CLUSTER_NAME`) is set, so the gap surfaces at startup rather than on the first federation; `FlServerManager` also guards the path with an `UnsupportedOperationException`. Leave it unset. Remaining work for the managed-task path — S3 model storage, FL servers as `RunTask` invocations, multi-replica safety, ALB target-group idle timeouts — is deferred to **OP-12**.

## Adjacent docs

- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** — deeper architectural walkthroughs and contribution patterns
- **[`wikis/backend/`](../../wikis/backend/)** — long-form wiki: architecture, security, project lifecycle, FL orchestration, WebSocket streaming
