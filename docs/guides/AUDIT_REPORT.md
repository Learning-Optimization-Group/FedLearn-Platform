# FedLearn Platform — AWS Readiness Audit

**Scope:** Backend (Spring Boot), Framework (Python/gRPC), Electron Desktop, Frontend (React), Client Docker image.
**Verdict:** **Not ready to deploy to AWS.** Three hard blockers plus several high-severity issues.

---

## Executive summary

| Blocker                                                                                   | Component        | File                                                                                                    |
| ----------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| ECS `RunTaskRequest` missing `networkConfiguration` (Fargate requires it)             | Backend          | `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/flower/FlowerServerManager.java` |
| Hardcoded DB password + JWT secret committed to git (both dev and prod configs)           | Backend          | `backend/fl-platform-api/src/main/resources/application*.properties`                                  |
| Proto typo crashes every unary model upload at runtime                                    | Framework        | `framework/src/fedlearn/client/grpc_client.py:128`                                                    |
| Login stores literal `"undefined"` in localStorage (frontend/backend contract mismatch) | Frontend/Backend | `frontend/src/pages/LoginPage.tsx:50` + `AuthController.java:122-128`                               |

---

## Backend (Spring Boot)

### B1. ECS integration incomplete — CRITICAL

**File:** `FlowerServerManager.java:49-84`

- No `networkConfiguration` on the `RunTaskRequest`. Fargate tasks in `awsvpc` mode require subnet IDs and security groups. The current request will be rejected with `InvalidParameterException`.
- Cluster name `"fedlearn-production-cluster"` and task definition `"fl-server-task"` are hardcoded.
- `ecsClient` is instantiated inline and never closed.
- No exception handling around `ecsClient.runTask()` and no check on `response.failures()` — a failed launch is silently swallowed.

**Fix:** Load cluster, task, subnets, security groups from env vars. Use try-with-resources. Inspect `response.failures()` and throw on non-empty.

### B2. Hardcoded secrets committed to git — CRITICAL

**Files:** `application.properties:10,34` and `application-production.properties:10,32`

- Dev profile: plaintext DB password and hardcoded JWT secret (not even env-var-driven).
- Prod profile: env vars with hardcoded **fallback defaults**, defeating env-based configuration.

**Fix:** Rotate both secrets immediately. Remove all hardcoded fallbacks — fail fast if env vars are missing. Inject via AWS Secrets Manager in the ECS task definition.

### B3. CORS wildcard with credentials — HIGH

**File:** `SecurityConfig.java:56-64`

`setAllowedOriginPatterns("*")` with `setAllowCredentials(true)` allows any origin to make credentialed requests.

**Fix:** Explicit origin allowlist from `CORS_ALLOWED_ORIGINS` env var.

### B4. Debug CORS controller reflects any Origin — HIGH

**File:** `CorsTestController.java`

Echoes the `Origin` and `Access-Control-Request-Headers` request headers back verbatim, bypassing the central CORS config.

**Fix:** Delete the file.

### B5. JWT cookie not `secure`, wrong `SameSite`, 24h TTL — HIGH

**File:** `AuthController.java:114-120`

`.secure(false)` was chosen for "local network" but applies in production too. `SameSite=Lax` allows CSRF on form submits. 24-hour TTL is too long.

**Fix:** `secure(true)`, `sameSite("Strict")`, `maxAge(3600)`.

### B6. Unauthenticated project and internal routes — HIGH

**File:** `SecurityConfig.java:76`

```java
.requestMatchers("/api/auth/**", "/ws-logs/**", "/error",
                 "/api/projects/**", "/api/internal/**").permitAll()
```

`/api/projects/**` and `/api/internal/**` are fully open.

**Fix:** Require authentication on both; keep only auth endpoints and WebSocket registration public.

### B7. Sensitive debug logging in auth flow — HIGH

**File:** `AuthController.java:80-112,131`

`System.out.println` dumps the login identifier, `CRITICAL_ERROR` messages echo user input, and `UserDetails` are printed to stderr on lookup failure.

**Fix:** Replace with SLF4J `log.debug/error`. Never include raw user input in error messages.

### B8. Generic exception handler leaks internals — MEDIUM

**File:** `GlobalExceptionHandler.java:63-70`

Returns `"An unexpected error occurred: " + ex.getMessage()` to the client and calls `ex.printStackTrace()`.

**Fix:** Log full trace server-side with a correlation ID; return only the correlation ID to the client.

### B9. TRACE-level logging in production properties — MEDIUM

**File:** `application-production.properties:22-24`

`org.springframework.security=DEBUG` and CORS filter set to `TRACE` will flood CloudWatch and inflate costs.

**Fix:** Drop to `WARN` or `INFO` for production.

### B10. Health endpoint leaks component details — MEDIUM

**File:** `application.properties:37` and `application-production.properties:35`

`management.endpoint.health.show-details=always` exposes DB connection and component states unauthenticated.

**Fix:** Use `when-authorized`.

### B11. Stale EC2 script paths in prod properties — MEDIUM

**File:** `application-production.properties:38-39`

`FlowerServerManager` no longer invokes shell scripts — it dispatches ECS tasks. The script paths are dead config.

**Fix:** Delete the properties.

### B12. `spring.jpa.show-sql=true` in both profiles — LOW

Every SQL statement is logged.

**Fix:** Disable in production.

### B13. Dockerfile entrypoint lacks secret validation — MEDIUM

**File:** `backend/fl-platform-api/Dockerfile:62-68`

Does not fail fast when `APP_JWT_SECRET`, `SPRING_DATASOURCE_USERNAME`, or `SPRING_DATASOURCE_PASSWORD` are missing.

**Fix:** Validate required env vars in the entrypoint; exit 1 on missing.

---

## Framework (Python / gRPC) — for user review

### F1. Proto typo crashes unary upload — CRITICAL

**File:** `grpc_client.py:128`

`fedlearn_pb2.SubmitModelUpdateReque(...)` — missing `st`. Every small-model (< 100MB) upload throws `AttributeError`.

### F2. Insecure gRPC channel — HIGH

**File:** `grpc_client.py:38,45`

`grpc.insecure_channel(...)` — no TLS. Unacceptable for public AWS traffic.

### F3. LZ4 import branch sets `LZ4_AVAILABLE = False` on success — HIGH

**File:** `serializer.py:6-14`

```python
try:
    import lz4.frame
    LZ4_AVAILABLE = False  # success branch never enables compression
    USE_COMPRESSION = False
except ImportError:
    ...
```

Compression is permanently disabled regardless of whether `lz4` is installed.

### F4. No retry/backoff on retryable gRPC errors — MEDIUM

**File:** `grpc_client.py` throughout

Handlers swallow `RpcError` without distinguishing `UNAVAILABLE` / `DEADLINE_EXCEEDED` from terminal codes. Single WAN blip kills a round.

### F5. `print()` instead of `logging` — MEDIUM

Unstructured CloudWatch output, no level filtering.

### F6. 50MB chunk size — LOW

**File:** `serializer.py:20`

Fine for LAN; aggressive for WAN/AWS ALB. Consider 4–8MB for cloud deployments.

---

## Electron Desktop

### D1. CSP blocks API calls in packaged mode — HIGH

**File:** `src/main/main.ts:60-74`

`connect-src 'self'` with a packaged `file://` origin blocks all HTTP/HTTPS calls to the backend API.

**Fix:** Add explicit backend host(s) to `connect-src`.

### D2. `fedlearn-client:latest` tag — LOW

**File:** `src/main/docker.service.ts:37`

Unpinned image tag. Use semver or digest for reproducibility.

The rest of the Electron shell is well-built: `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false`, `setWindowOpenHandler` deny, `will-navigate` allowlist.

---

## Frontend (React / Vite / TS)

### FE1. Broken login contract — CRITICAL

**Files:** `frontend/src/pages/LoginPage.tsx:50` and `backend/.../AuthController.java:122-128`

Frontend reads `responseData.accessToken`, but backend response body contains only `username` and `email`. `localStorage.setItem('jwtToken', undefined)` stores the literal string `"undefined"`.

**Fix:** Add `accessToken` to the login response body so the frontend can store the real JWT. Keep the `httpOnly` cookie for session auth if desired.

### FE2. `axiosConfig` silent fallback — MEDIUM

**File:** `frontend/src/api/axiosConfig.ts:4`

`baseURL` falls back to `http://${hostname}:8081/api` when `VITE_API_BASE_URL` is unset. In production builds this silently fails.

**Fix:** Throw if `VITE_API_BASE_URL` is unset in `import.meta.env.PROD`.

### FE3. Duplicate API URL logic in `LoginPage.tsx` — LOW

Same fallback duplicated in the login page. Move all requests through `axiosConfig`.

---

## Client Docker

### C1. Base image is old — LOW

**File:** `client-docker/Dockerfile:8`

`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` — CUDA 11.7 is stale relative to current GPU driver baselines (mismatch with newer hosts can cause subtle failures). Not a deployment blocker for the demo, but worth bumping.

### C2. Container runs as root — LOW

No `USER` directive. For client containers on user hardware this is acceptable but not ideal.

---

## Cross-cutting

### X1. `README.md` documents obsolete architecture — MEDIUM

**File:** `README.md:52-74,165-176,509-511`

Documents `ProcessBuilder` spawning Python scripts, `FL_SCRIPTS_PATH`, and `BufferedReader` → WebSocket streaming. None of this matches the current `FlowerServerManager` (ECS Fargate). The stale README is the source of several bogus "audit findings" that have been surfaced in the past.

**Fix:** Update the README to reflect the ECS architecture.

### X2. No IaC — HIGH

No ECS task definition, IAM policies, VPC wiring, or security-group config in the repo. Required before any deploy:

- `ecsTaskExecutionRole` with Secrets Manager read + ECR pull
- FL server task role (CloudWatch Logs, any S3 buckets)
- Security groups allowing gRPC (50051) between API service and FL tasks
- `awslogs` driver config in task definition

---

## Required environment variables after fixes

| Variable                       | Purpose                                            |
| ------------------------------ | -------------------------------------------------- |
| `SPRING_DATASOURCE_URL`      | JDBC URL                                           |
| `SPRING_DATASOURCE_USERNAME` | DB user                                            |
| `SPRING_DATASOURCE_PASSWORD` | DB password (from Secrets Manager)                 |
| `APP_JWT_SECRET`             | JWT signing key (from Secrets Manager)             |
| `APP_JWT_EXPIRATION_MS`      | Token TTL                                          |
| `CORS_ALLOWED_ORIGINS`       | Comma-separated list of allowed origins            |
| `ECS_CLUSTER_NAME`           | Fargate cluster name                               |
| `ECS_TASK_DEFINITION`        | FL server task definition name                     |
| `ECS_SUBNETS`                | Comma-separated subnet IDs                         |
| `ECS_SECURITY_GROUPS`        | Comma-separated security group IDs                 |
| `ECS_ASSIGN_PUBLIC_IP`       | `ENABLED` or `DISABLED` (default `DISABLED`) |
| `VITE_API_BASE_URL`          | Frontend API origin (required in prod builds)      |

---

## Minimum fix order before AWS deploy

1. Fix `grpc_client.py:128` proto typo.
2. Fix `LoginPage.tsx` ↔ `AuthController.java` login contract (add `accessToken` to response body).
3. Rotate secrets; remove all hardcoded values and fallback defaults.
4. Complete `FlowerServerManager` (network config, error handling, env vars, try-with-resources).
5. Lock CORS; delete `CorsTestController`.
6. Secure the JWT cookie (`secure=true`, `SameSite=Strict`, 1h TTL).
7. Authenticate `/api/projects/**` and `/api/internal/**`.
8. Strip debug logging in auth flow and production properties.
9. Electron `connect-src` must list backend hosts.
10. Provision IaC: ECS task def, IAM, VPC, security groups, CloudWatch.
