# FedLearn Platform — Backend API

Spring Boot 3 / Java 21 / Gradle. REST + STOMP-over-WebSocket. Manages users, projects, training results, and the lifecycle of Python FL-server processes.

For Spring profile semantics, deployment topology, and the cookie auth model, see `DEVELOPMENT.md` in this directory.

## Stack

- **Spring Boot 3** + **Spring Security**
- **Java 21** + **Gradle** (wrapper committed; do not switch to Maven)
- **JWT** delivered as **HttpOnly cookies** — no Bearer headers, no JS-readable token
- **WebSocket / STOMP** for real-time logs and round-result telemetry
- **JPA / Hibernate** in `validate`-only mode — schema is owned by **Flyway**
- **H2** (file-mode) on the EC2 demo and locally; **PostgreSQL** wired in the `production` profile (unfinished)

## Spring profiles

| Profile | Purpose | Activation |
|---|---|---|
| (base) | `application.properties`. Refuses to boot without `APP_JWT_SECRET`, `APP_INTERNAL_API_KEY`, `CORS_ALLOWED_ORIGINS`. | always loaded |
| `dev` | Public dev secrets, permissive CORS (`http://localhost:*`), H2 console enabled, `cookie.secure=false`. **Never activate in any deployed env.** | `SPRING_PROFILES_ACTIVE=dev` / `./launch_all.sh` |
| `test` | In-memory H2, Flyway disabled, Hibernate `create-drop`, public test secrets, quiet logging. | `@ActiveProfiles("test")` or `SPRING_PROFILES_ACTIVE=test ./gradlew test` |
| `ec2demo` | Single-EC2 demo at `https://fedlearn.duckdns.org`. H2 file-mode, FL servers as local Python processes, `cookie.secure=true`. | `SPRING_PROFILES_ACTIVE=ec2demo` (set by `scripts/deploy-to-aws.sh`) |
| `production` | ECS Fargate, PostgreSQL, FL servers as ECS tasks. **Unfinished — do not activate.** | – |

## Quick start

```bash
cd backend/fl-platform-api
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun
```

H2 file at `./data/federated_platform_db`. H2 console at `http://localhost:8081/h2-console` (gated to the `dev` profile by SecurityConfig). The app starts on `:8081`.

### Tests

```bash
SPRING_PROFILES_ACTIVE=test ./gradlew test
./gradlew test --tests "com.federated.fl_platform_api.SomeTest"
```

The `test` profile uses an in-memory H2 with Hibernate `create-drop`. Flyway is disabled for tests; **never** change this — Flyway migrations must validate against `dev`/`ec2demo`/`production` only.

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
│   ├── flower/         # FlowerServerManager — spawns/manages Python FL-server processes (legacy package name; we do NOT depend on Flower/flwr)
│   ├── model/          # JPA entities (User, Project, RoundResult)
│   ├── repository/     # Spring Data JPA repositories
│   ├── security/       # JwtTokenProvider, JwtAuthenticationFilter, JwtHandshakeInterceptor (for STOMP)
│   ├── service/        # Business logic
│   └── FlPlatformApiApplication.java
├── src/main/resources/
│   ├── application.properties               # base — env-var-driven, no fallbacks
│   ├── application-dev.properties           # local-dev convenience
│   ├── application-test.properties          # in-memory H2, public test secrets
│   ├── application-ec2demo.properties       # single-EC2 demo
│   ├── application-production.properties    # ECS Fargate path (unfinished)
│   ├── db/migration/                        # Flyway versioned migrations
│   └── scripts/                             # Python FL-server scripts (init_model.py, fl_server.py, ...)
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

`FlowerServerManager` is the entry point. On `POST /api/projects/{id}/start` it:

1. Reserves a port from `fl.server.port-range.start..end` (default `50000-50010`).
2. Spawns `python src/main/resources/scripts/run_fl_server.sh ...` with project config.
3. Captures stdout/stderr line-by-line and broadcasts via `WebSocketService` to `/topic/logs/{projectId}`.
4. Tracks PID + process handle so `/stop` can terminate cleanly.

The shell wrapper exists so the Python entry point is portable across local dev (Mac, Linux) and the EC2 host. There's a parallel `.bat` for Windows local dev.

## Deployment

### Local

`./launch_all.sh` (root) starts backend + frontend + Electron + FL-client launcher. Or run components individually as described above.

### EC2 demo (`ec2demo`)

Live at **https://fedlearn.duckdns.org**. Procedure: [`docs/guides/aws_deployment_guide.md`](../../docs/guides/aws_deployment_guide.md).

The EC2 instance runs `fedlearn.service` (systemd) which sources its env from `/etc/systemd/system/fedlearn.service`:

```ini
Environment="SPRING_PROFILES_ACTIVE=ec2demo"
Environment="APP_JWT_SECRET=..."           # openssl rand -base64 64
Environment="APP_INTERNAL_API_KEY=..."     # openssl rand -hex 32
Environment="CORS_ALLOWED_ORIGINS=https://fedlearn.duckdns.org,http://localhost:5173"
Environment="APP_AUTH_COOKIE_SECURE=true"
```

nginx terminates TLS on `:443` and proxies to `127.0.0.1:8081` — port 8081 is **not** publicly exposed. STOMP upgrade headers and a `proxy_read_timeout 3600s` are required for long-lived training rounds.

### ECS Fargate (`production`)

Unfinished. See [`docs/guides/AWS_AUDIT.md`](../../docs/guides/AWS_AUDIT.md) Tier 2 items 10–17 for the remaining work (S3 model storage, FL servers as `RunTask` invocations, multi-replica safety, ALB target-group idle timeouts).

## Adjacent docs

- **`DEVELOPMENT.md`** — deeper architectural walkthroughs and contribution patterns
- **`docs/wikis/backend/`** — long-form wiki: architecture, security, project lifecycle, FL orchestration, WebSocket streaming
