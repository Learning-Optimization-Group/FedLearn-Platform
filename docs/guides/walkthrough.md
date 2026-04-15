# FedLearn-Platform: Comprehensive Codebase Audit Report

**Date**: April 1, 2026
**Scope**: Full-stack analysis — Python FL Framework, Spring Boot API, React Frontend
**Codebase Version**: `0.0.1-SNAPSHOT` / `0.1.0` (framework)

---

## Executive Summary

FedLearn-Platform is an ambitious multi-component federated learning system with a **Python gRPC framework** (FL coordination, aggregation, DeComFL), a **Spring Boot 3.4 REST API** (project management, auth, process orchestration), and a **React 19 + Vite frontend** (dashboard UI). The architecture shows thoughtful design — separate gRPC channels for heartbeat/data, streaming for large models, proper strategy patterns — but has accumulated significant technical debt during rapid development.

### Health Score: **5.5 / 10** — Functional but carries critical risks

| Category                  | Rating      | Summary                                                                                 |
| ------------------------- | ----------- | --------------------------------------------------------------------------------------- |
| **Security**        | 🔴 Critical | Hardcoded secrets committed to VCS, insecure deserialization, over-permissive endpoints |
| **Correctness**     | 🔴 Critical | Double round-increment bug will corrupt FL training, broken frontend API call           |
| **Performance**     | 🟡 Moderate | Adequate for current scale, but missing connection pooling, no async I/O in framework   |
| **Maintainability** | 🟡 Moderate | Reasonable structure, but dead code, dual build systems, zero framework tests           |
| **Architecture**    | 🟢 Good     | Clean separation of concerns, well-designed strategy pattern, proper streaming          |

---

## 🔴 Critical / Immediate Fixes (High Priority)

### C1. Double Round-Increment Bug — Data Corruption Risk

> [!CAUTION]
> This **will silently corrupt federated learning** by skipping every other round's data.

The coordinator increments `current_round` in `_trigger_aggregation_and_evaluation()` at L89, and then the server loop **increments it again** at L116.

```diff
# server.py L89 (inside coordinator._trigger_aggregation_and_evaluation)
  self.current_round += 1  # Increment #1

# server.py L116 (in the training loop)
  coordinator.current_round += 1  # Increment #2 — DOUBLE INCREMENT
```

**Impact**: After round 1 completes, `current_round` jumps to **3** instead of 2. Clients submitting updates for round 2 are silently ignored as "stale" (coordinator L52-53). Effectively halves the number of actual training rounds.

**Files**: [server.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/server.py#L116) L116, [coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/coordinator.py#L89) L89

**Fix**: Remove the redundant increment in `server.py` L116.

---

### C2. Hardcoded Secrets Committed to Version Control

> [!CAUTION]
> Production database passwords, JWT secrets, and an AWS host IP are committed in plaintext to the repository.

| File                                                                                                                                                          | Secret Type                                | Line   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------ |
| [env.txt](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/env.txt#L1-L5)                                                | AWS host, DB password, JWT secret          | L1-5   |
| [application.properties](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/resources/application.properties#L10) | Default DB password `Coloreal@1`         | L10    |
| [application.properties](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/resources/application.properties#L34) | 512-bit JWT secret as default fallback     | L34    |
| `application-production.properties`                                                                                                                         | Production DB password `Postgres@272025` | L10    |
| [async_coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/async_coordinator.py#L13-L24)              | RabbitMQ creds**printed to stdout**  | L13-24 |

**Remediation**:

1. Add `env.txt`, `*.properties` to `.gitignore` immediately
2. Use `git filter-branch` or BFG Repo-Cleaner to scrub history
3. Rotate ALL exposed credentials (DB passwords, JWT secret, AWS access)
4. Use proper secret management (env vars only, no defaults for secrets)

---

### C3. Insecure Deserialization — Remote Code Execution Vector

> [!WARNING]
> `torch.load(..., weights_only=False)` and `pickle.loads()` of network data enable arbitrary code execution.

**File**: [grpc_client.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/client/grpc_client.py#L112) L112

```python
model_data = torch.load(buffer, map_location='cpu', weights_only=False)
```

A compromised server can send a malicious pickle payload that executes arbitrary code on the client. Similarly, the server's `serializer.py` L123 uses `torch.load` without `weights_only=True`.

**File**: [async_coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/async_coordinator.py#L116) L116

```python
result = pickle.loads(body)  # RabbitMQ message deserialized as pickle
```

Any entity with queue access can execute arbitrary code on the server.

**Remediation**:

- Add `weights_only=True` to all `torch.load` calls
- Replace `pickle.loads` with a safe serialization format (MessagePack, JSON, protobuf)
- Consider using `safetensors` (already in requirements) for model serialization

---

### C4. Broken Frontend API Call — `startProjectServer` Sends Wrong Payload

> [!CAUTION]
> The start-server API call wraps the body in an extra `{body: {...}}` wrapper, causing the backend to receive null values for all parameters.

**File**: [apiServices.jsx](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/frontend/src/services/apiServices.jsx#L66) L66

```javascript
// BROKEN: Wraps body in another object
return api.post(`/projects/${projectId}/start`, { body }, null);

// SHOULD BE:
return api.post(`/projects/${projectId}/start`, body);
```

**Impact**: Backend receives `{"body": {"strategy": "FedAvg", ...}}` instead of `{"strategy": "FedAvg", ...}`. The `@RequestBody StartProject` will deserialize with null fields, falling back to hardcoded defaults every time. The user's chosen strategy, round count, and client count are silently ignored.

---

### C5. Exposed Project Endpoints — No Authentication Required

> [!WARNING]
> All `/api/projects/**` endpoints are publicly accessible without authentication.

**File**: [SecurityConfig.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/config/SecurityConfig.java#L83) L83

```java
.requestMatchers("/api/auth/**", "/ws-logs/**", "/error", "/api/projects/**").permitAll()
```

**Impact**: Any unauthenticated user can:

- List ALL projects for ALL users
- Create, start, stop, and **delete** any project
- Access training results
- Launch arbitrary Python processes on the server via the start endpoint

**Fix**: Remove `/api/projects/**` from `permitAll()` and let it fall through to `.anyRequest().authenticated()`.

---

### C6. JWT Key Initialization Mismatch

> [!WARNING]
> The JWT signing key is created from the raw string bytes, making the Base64-decoded bytes on L37 dead code. This can cause signature verification failures if the environment provides a properly Base64-encoded key.

**File**: [JwtTokenProvider.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/security/JwtTokenProvider.java#L36-L38) L36-38

```java
@PostConstruct
public void init() {
    byte[] keyBytes = Decoders.BASE64.decode(jwtSecretString);    // ← decoded but never used
    this.jwtSecretKey = Keys.hmacShaKeyFor(jwtSecretString.getBytes()); // ← uses raw string
}
```

**Fix**: Use the decoded bytes: `this.jwtSecretKey = Keys.hmacShaKeyFor(keyBytes);`

---

### C7. Dual Build System Conflict — Maven + Gradle

Both `pom.xml` (Maven) and `build.gradle` (Gradle) exist in [`fl-platform-api/`](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api), with **conflicting Java version targets**:

| Setting      | Gradle    | Maven                                     |
| ------------ | --------- | ----------------------------------------- |
| Java version | `21`    | `23` (`maven.compiler.source/target`) |
| Spring Boot  | `3.4.5` | `3.4.5`                                 |

The Dockerfile ([L16](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/Dockerfile#L16)) uses Maven (`mvn package`), while local dev uses Gradle (`./gradlew bootRun`). This creates divergent build behavior between local and deployed environments.

**Fix**: Delete `pom.xml` and update Dockerfile to use `./gradlew build`:

```dockerfile
FROM gradle:8-jdk21 AS build
COPY . .
RUN gradle bootJar --no-daemon
```

---

## 🟡 Suggested Upgrades (Medium / Low Priority)

### Architecture & Design

#### U1. Async Coordinator Has a Double-Append Bug

**File**: [async_coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/async_coordinator.py#L59-L60) L59-60

```python
self._round_updates[trained_on_round].append((client_id, params, num_examples))
self._round_updates[trained_on_round].append((params, num_examples))  # DUPLICATE
```

Every client update is stored twice with different tuple shapes, doubling memory usage and corrupting the aggregation count check on L62.

#### U2. DeComFL `_trigger_decomfl_aggregation_and_evaluation` Missing Round Increment

**File**: [coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/coordinator.py#L216-L250) L216-250

The regular aggregation path increments `current_round` (L89), but `_trigger_decomfl_aggregation_and_evaluation` only signals `_round_complete_event.set()` (L250) without incrementing. If the server loop also has the double-increment issue from C1, it may accidentally mask this bug — but if C1 is fixed, DeComFL rounds will stall.

#### U3. `FedAvgAggregator` Has Fragile Tuple Unpacking

**File**: [strategy.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/strategy.py#L82-L98) L82-98

The aggregator tries to handle both 2-tuple and 3-tuple formats with `if len(entry) == 3`, then forces everything to 3-tuples. But the regular FedAvg path sends 2-tuples `(params, num_examples)` while DeComFL sends 3-tuples `(client_id, gradient_scalars, num_examples)`. This is a data-shape landmine waiting to explode.

**Recommendation**: Use typed dataclasses or named tuples instead of raw tuples.

#### U4. No gRPC TLS — All Model Data Transmitted in Plaintext

The gRPC server uses `add_insecure_port()` ([server.py L84](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/server.py#L84)) and clients use `insecure_channel()` ([grpc_client.py L41](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/client/grpc_client.py#L41)). All model parameters, gradient scalars, and heartbeats are transmitted unencrypted. For a federated learning system, this is a significant privacy concern.

#### U5. Process Management via `ProcessBuilder` is Fragile

**File**: [FlowerServerManager.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/flower/FlowerServerManager.java)

- `ConcurrentHashMap<UUID, Process>` is lost on JVM restart (no persistence)
- `destroyForcibly()` won't clean up grandchild processes
- `Thread.sleep(2000)` as a synchronization primitive (L40)
- No health monitoring after startup beyond the initial 3-second check

**Recommendation**: Consider Docker-based process isolation or at minimum persistent state tracking in the database.

---

### Dependency & Modernization

#### U6. Deprecated JJWT API Usage

**File**: [JwtTokenProvider.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/security/JwtTokenProvider.java#L47-L51)

Uses deprecated `Jwts.builder().setSubject()`, `.setIssuedAt()`, `.signWith()` API. JJWT 0.12.x recommends the fluent builder:

```java
Jwts.builder()
    .subject(username)
    .issuedAt(now)
    .expiration(expiryDate)
    .signWith(key)
    .compact();
```

#### U7. `sklearn` Deprecated Package Name

**File**: [requirements.txt](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/requirements.txt#L110) L110

`sklearn` is a deprecated stub package. Use `scikit-learn` instead.

#### U8. Compression Hardcoded Off

**File**: [serializer.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/communication/serializer.py#L7-L22)

Even when LZ4 is successfully imported, the flag is immediately overridden:

```python
LZ4_AVAILABLE = False  # L9 — even though import succeeded
USE_COMPRESSION = False  # L10 — forced off
# ...
USE_COMPRESSION = False  # L22 — overridden again at module level
```

This means compression is permanently disabled despite the code being fully written.

#### U9. Frontend User-Agent Spoofing

**File**: [axiosConfig.jsx](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/frontend/src/api/axiosConfig.jsx#L10) L10

```javascript
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
```

Hardcoded fake User-Agent header. Browsers will override this anyway; this adds no value and suggests bypassing server-side UA validation.

---

### Testing & Quality

#### U10. Zero Tests for the FL Framework

The entire `framework/` Python module — the core value proposition of the platform — has **0 test files**. The `test.desc` file exists but is not a test suite.

Critical paths that need test coverage:

- `FedAvgAggregator.aggregate()` with edge cases (empty, single client, float precision)
- `FLCoordinator` round management and thread safety
- `serializer.py` round-trip serialization
- `ZerothOrderEstimator` gradient computation

#### U11. Backend Tests Exist but Are Likely Stale

Three test files exist:

- `FlPlatformApiApplicationTests.java` — context load smoke test
- `ProjectServiceTest.java`
- `ProjectControllerTest.java`

These should be reviewed for coverage against the current codebase, especially the new WebSocket and DeComFL-related endpoints.

#### U12. Console Logging Instead of Structured Logging

**Across all components**, critical information is logged via `print()` (Python) and `System.out.println()` (Java) rather than structured logging. Examples:

- [coordinator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/coordinator.py#L59): `print(f"[Coordinator] Received update...")`
- [ProjectController.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/controller/ProjectController.java#L61-L68): Debug request logging with `System.out.println`
- [AuthController.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/controller/AuthController.java#L78-L91): Login flow logging with `System.out.println`

**Impact**: No log levels, no structured fields, no correlation IDs. Makes production debugging and monitoring effectively impossible.

---

### Best Practices

#### U13. Dead Code and Commented-Out Code

| Location                                                                                                                                                                                      | Description                                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| [core/aggregator.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/core/aggregator.py)                                                                 | Entire file is commented out (44 lines)                  |
| [server.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/server/server.py#L8-L10) L8-10                                                               | Commented-out pika/async imports                         |
| [client.py](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/framework/src/fedlearn/client/client.py#L4-L13)                                                                     | Unused `pickle`, `pika` imports                      |
| [UserService.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/UserService.java#L49-L57) L49-57 | Debug `main()` method with hardcoded password hash     |
| [SecurityConfig.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/config/SecurityConfig.java#L12) L12   | Duplicate import `org.springframework.http.HttpMethod` |
| [DashboardPage.jsx](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/frontend/src/pages/DashboardPage.jsx#L175-L181) L175-181                                                    | Commented-out ResultsModal component                     |

#### U14. Missing `DELETE` HTTP Method for Project Deletion

**File**: [ProjectController.java](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/controller/ProjectController.java#L99) L99

```java
@PostMapping("/{projectId}/delete")  // Should be @DeleteMapping
```

Uses POST for a destructive delete operation. Also has no authorization check — any user can delete any other user's project by ID.

#### U15. `.gitignore` is Incomplete

The root [.gitignore](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/.gitignore) doesn't cover:

- `env.txt`, `.env`, `*.properties` with secrets
- `node_modules/` (frontend has it committed potentially)
- `.idea/` (JetBrains IDE configs, present in `framework/` and `client-docker/`)
- `build/`, `target/` (Java build artifacts)
- `Logs.txt` (240KB log file in backend/)
- `__pycache__/` only at root, not recursively
- `*.egg-info/` only at root

#### U16. No CI/CD Pipeline, Linting, or Pre-commit Hooks

- No `.github/workflows/` CI configuration (`.github/` exists but appears empty or non-functional)
- No `pre-commit-config.yaml` or equivalent
- ESLint is configured for frontend but no evidence of enforcement
- `pyproject.toml` lists dev tools (`pytest`, `black`, `isort`, `mypy`) but none appear to be used
- No Dockerfile for the frontend (only backend)

---

### Performance Notes

#### P1. LoginPage Uses Raw `fetch` While Rest of App Uses Axios

**File**: [LoginPage.jsx](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/frontend/src/pages/LoginPage.jsx#L38-L64) L38-64

The login page uses raw `fetch()` with hardcoded URLs instead of the centralized `api` axios instance. This bypasses the interceptor for response error handling, and has a subtle bug on L48:

```javascript
const responseData = response.json;  // Missing () — this is the Function, not the result
```

This means on non-OK responses (L63), `responseData.error` will be `undefined`, always showing a generic error.

#### P2. WebSocket Topic Subscription Uses Literal Wildcard

**File**: [DashboardPage.jsx](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/frontend/src/pages/DashboardPage.jsx#L44) L44

```javascript
client.subscribe('/topic/status/*', ...);
```

STOMP wildcards use `.*` not `*`. This subscription likely works by accident due to Spring's default STOMP broker behavior, but is technically incorrect and may fail with a full-featured STOMP broker.

---

## Summary of Findings

| Priority    | ID  | Category | Component | Description                                                            |
| ----------- | --- | -------- | --------- | ---------------------------------------------------------------------- |
| 🔴 Critical | C1  | Bug      | Framework | Double round-increment corrupts FL training                            |
| 🔴 Critical | C2  | Security | Backend   | Hardcoded secrets in VCS (DB passwords, JWT, AWS IP)                   |
| 🔴 Critical | C3  | Security | Framework | `torch.load(weights_only=False)` + `pickle.loads` on network data  |
| 🔴 Critical | C4  | Bug      | Frontend  | `startProjectServer` wraps body incorrectly, params always null      |
| 🔴 Critical | C5  | Security | Backend   | `/api/projects/**` bypasses auth — any user can control any project |
| 🔴 Critical | C6  | Bug      | Backend   | JWT key init uses raw bytes instead of decoded Base64                  |
| 🔴 Critical | C7  | Build    | Backend   | Maven + Gradle conflict with different Java version targets            |
| 🟡 Medium   | U1  | Bug      | Framework | Async coordinator double-appends updates                               |
| 🟡 Medium   | U2  | Bug      | Framework | DeComFL aggregation path missing round increment                       |
| 🟡 Medium   | U3  | Design   | Framework | Fragile tuple unpacking in aggregator                                  |
| 🟡 Medium   | U4  | Security | Framework | No TLS on gRPC — plaintext model transmission                         |
| 🟡 Medium   | U5  | Design   | Backend   | Fragile process management via `ProcessBuilder`                      |
| 🟡 Medium   | U6  | Deps     | Backend   | Deprecated JJWT API methods                                            |
| 🟢 Low      | U7  | Deps     | Framework | `sklearn` → `scikit-learn`                                        |
| 🟢 Low      | U8  | Design   | Framework | Compression permanently disabled                                       |
| 🟢 Low      | U9  | Hygiene  | Frontend  | Fake User-Agent header in axios config                                 |
| 🟢 Low      | U10 | Testing  | Framework | Zero test coverage for core FL logic                                   |
| 🟢 Low      | U11 | Testing  | Backend   | Test staleness risk                                                    |
| 🟢 Low      | U12 | Ops      | All       | `print()`/`System.out.println` instead of structured logging       |
| 🟢 Low      | U13 | Hygiene  | All       | Dead/commented-out code throughout                                     |
| 🟢 Low      | U14 | Design   | Backend   | POST used for DELETE operation, no RBAC                                |
| 🟢 Low      | U15 | Ops      | Root      | Incomplete `.gitignore`                                              |
| 🟢 Low      | U16 | Ops      | Root      | No CI/CD, no pre-commit hooks                                          |

---

## Recommended Action Plan

### Phase 1 — Immediate (This Week)

1. **Fix C1**: Remove duplicate `coordinator.current_round += 1` in `server.py` L116
2. **Fix C2**: Rotate all credentials, add secrets to `.gitignore`, scrub git history
3. **Fix C4**: Change `apiServices.jsx` L66 to `api.post(..., body)`
4. **Fix C5**: Remove `/api/projects/**` from `permitAll()` in `SecurityConfig.java`
5. **Fix C6**: Use `Keys.hmacShaKeyFor(keyBytes)` in `JwtTokenProvider.init()`
6. **Fix C7**: Delete `pom.xml`, standardize on Gradle

### Phase 2 — Short Term (2-3 Weeks)

7. Fix C3 — add `weights_only=True` to all `torch.load` calls
8. Fix U1 — remove duplicate append in `async_coordinator.py`
9. Fix U2 — add `self.current_round += 1` to DeComFL aggregation path
10. Add basic test suite for `FedAvgAggregator` and `FLCoordinator`
11. Replace `print()` with Python `logging` and use SLF4J properly in Java

### Phase 3 — Medium Term (1-2 Months)

12. Add gRPC TLS support with configurable certificates
13. Set up GitHub Actions CI/CD pipeline
14. Migrate to typed dataclasses for FL update payloads
15. Add RBAC to project endpoints (users can only manage their own projects)
16. Enable LZ4 compression and benchmark transfer speedup
