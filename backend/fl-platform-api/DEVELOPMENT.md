# Backend Development Guide

Complete guide for developers working on the Spring Boot backend.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Components](#key-components)
- [Database Layer](#database-layer)
- [Service Layer](#service-layer)
- [Controller Layer](#controller-layer)
- [Security](#security)
- [FL Server Management](#fl-server-management)
- [WebSocket Integration](#websocket-integration)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Best Practices](#best-practices)

---

## Architecture Overview

### Three-Tier Architecture

```
┌─────────────────────────────────────────────┐
│  Controller Layer (REST API)                │
│  - AuthController                           │
│  - ProjectController                        │
│  - ResultsController                        │
└──────────────┬──────────────────────────────┘
               │ DTOs (Request/Response)
┌──────────────▼──────────────────────────────┐
│  Service Layer (Business Logic)             │
│  - ProjectService                           │
│  - UserService                              │
│  - FlServerManager                          │
│  - WebSocketService                         │
└──────────────┬──────────────────────────────┘
               │ Entities
┌──────────────▼──────────────────────────────┐
│  Repository Layer (Data Access - JPA)       │
│  - ProjectRepository                        │
│  - UserRepository                           │
│  - RoundResultRepository                    │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  PostgreSQL Database                        │
└─────────────────────────────────────────────┘
```

### Request Flow Example

**User clicks "Start Server" in Frontend:**

```
1. Frontend: POST /api/projects/{id}/start
              ↓
2. Controller: ProjectController.startProject()
              ↓
3. Service: ProjectService.startServer(projectId)
              ↓
4. Manager: FlServerManager.startServerForProject()
              ↓
5. Runner: FlServerProcessRunner spawns the fl-runtime wrapper script
              ↓
6. WebSocket: Streams logs to /topic/logs/{projectId}
              ↓
7. Frontend: Receives real-time logs via WebSocket
```

---

## Key Components

### 1. FlServerManager

**Location**: `src/main/java/com/federated/fl_platform_api/orchestration/FlServerManager.java`

**Purpose**: Manages FL server processes (start, stop, monitor). Renamed from the historical `FlowerServerManager` in the `flower/` package (DA-12) — the platform has never used Flower's FL semantics.

**Key Features**:
- Policy gates before any spawn (SE-10 argv validation, SE-11 DP policy)
- Range-based port reservation, race-safe across concurrent starts
- Process launch delegated to the `FlServerProcessRunner` seam (DA-8)
- Process output streaming to WebSocket
- Concurrent server tracking by `ProcessHandle` (BA-3 orphan reconciliation)

#### The FlServerProcessRunner seam (DA-8)

The manager does **not** call `ProcessBuilder` directly. It builds the argv and the environment
customizer, then hands both to an injectable runner:

```java
public interface FlServerProcessRunner {
    SpawnedFlProcess start(List<String> command, Consumer<Map<String, String>> envCustomizer,
                           File workingDir) throws IOException;
}
```

- **`LocalProcessFlServerRunner`** is the default — a local child process. It applies the env
  customizer, merges stderr into stdout, sets the working directory, and starts the process.
  Nothing else: all policy (argv building, secret scrubbing, port reservation, run-state
  persistence, the startup probe, log broadcasting) stays in the manager.
- A Spring bean of type `FlServerProcessRunner`, if defined, overrides the default via
  `setProcessRunner`. Tests inject a fake with `ReflectionTestUtils` — this is what makes the
  spawn orchestration unit-testable without a real process.
- `SpawnedFlProcess` abstracts `java.lang.Process`, exposing only what the manager needs:
  `pid()`, `startInstant()`, `toHandle()`, `getInputStream()`, `waitFor()`, `exitValue()`,
  `isAlive()`, `destroyForcibly()`.

> **ECS/Fargate is not implemented.** `startServerForProject` fails closed with an
> `UnsupportedOperationException` when `ecs.cluster-name` is set, and `FlOrchestrationModeValidator`
> already fails the **boot** on the same condition (OP-14). The managed-task runner is tracked as
> OP-12; the seam above is where it would slot in.

#### How It Works

**1. Gate, then reserve a port**:
```java
requireDpPolicySatisfied(project);              // SE-11: regulated projects need complete DP config
requireModelTypeInCatalog(project, strategy);   // SE-10: unknown modelType -> 400 before spawn
stopServerForProject(project.getId());          // Stop any existing server
int freePort = findFreePort();                  // Reserve from the configured range
```

**2. Select the wrapper and build the command**:
```java
boolean isFoT = "FoT".equalsIgnoreCase(strategy);
String wrapperPath = isFoT ? fotServerWrapperPath : flServerWrapperPath;
String absoluteScriptPath = new File(wrapperPath).getAbsolutePath();
```
Federation over Text (`FoT`) is a separate text-federation server spawned through the same seam;
the gradient strategies (FedAvg, DeComFL) take the other branch. The configured wrapper path is used
verbatim: on Linux/Mac it is invoked as `bash <script> ...`, and on Windows the `bash` prefix is
dropped and the script is executed directly (point `PYTHON_SCRIPT_FL_SERVER_PATH` at
`fl-runtime/run_fl_server.bat` there — the manager does not rewrite the extension for you).

**3. Spawn via the runner**:
```java
process = processRunner.start(command,
        env -> configureChildEnv(env, internalApiKey, backendInternalUrl,
                flTokenSecret, requireClientAuth, runIdArg, requireTls, internalRunToken),
        new File("."));
runningServers.put(project.getId(), process.toHandle());
recordProcessIdentity(project.getActiveRunId(), process.pid(),
        process.startInstant().orElse(null), freePort, internalTokenHash);
```
SE-7: a random per-run token scoped to `(projectId, runId)` is minted and handed to the child —
never a secret it could use to forge another project's token. BA-3: if the identity can't be
persisted, the child is killed rather than leaked as an unreconcilable orphan.

**4. Stream Output**:
```java
Thread outputReaderThread = new Thread(() -> {
    try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(readerProcess.getInputStream()))) {
        String line;
        while ((line = reader.readLine()) != null) {
            log.debug("[FL_SERVER {}] {}", project.getId(), line);
            logBroadcaster.sendLogs(project.getId(), line);
            startupOutput.append(line).append('\n');
        }
    } catch (IOException e) { /* errorOccurred[0] = true */ }
}, "fl-server-stdout-" + project.getId());
outputReaderThread.setDaemon(true);
outputReaderThread.start();
```

**5. Startup probe**:
```java
boolean exited = process.waitFor(startupProbeSeconds, TimeUnit.SECONDS);
if (exited) {
    // stdout is buffered — drain before surfacing, or you lose the stack trace
    outputReaderThread.join(stdoutDrainMillis);
    runningServers.remove(project.getId());
    throw new ServerProcessException("FL server exited during startup ...");
}
```
Tunable via `fl.server.startup-probe-seconds` (default `3`) and `fl.server.stdout-drain-millis`
(default `5000`).

#### Configuration

In `application.properties`:
```properties
# Wrapper scripts live in the repo-root fl-runtime/ directory, NOT under src/main/resources.
python.script.fl-server.path=${PYTHON_SCRIPT_FL_SERVER_PATH:../../fl-runtime/run_fl_server.sh}
fl.server.port-range.start=50000
fl.server.port-range.end=50010
```
The FoT wrapper is bound with an inline `@Value` default on the manager
(`python.script.fot-server.path`, default `../../fl-runtime/run_fot_server.sh`) and has no entry in
the properties file.

#### Script Parameters

The gradient (`fl_server`) wrapper receives:
- `--project-id`: UUID of the project
- `--model-path`: Path to model checkpoint
- `--init-model-path`: Registry-resolved warm-start weights (BA-11; omitted on a first run / LoRA)
- `--port`: Port number (reserved from the range)
- `--strategy`: Aggregation strategy (FedAvg, DeComFL)
- `--num-rounds`: Number of training rounds
- `--min-clients`: Minimum clients required
- `--model-type`: Type of model (CNN, Transformer, etc.)
- `--model-name`: Specific model architecture
- `--aggregation FFA_LORA` + `--task-type`: added only for the `LLM_LORA` model type
- `--dp-enabled`, `--dp-clip-norm`, `--dp-target-epsilon`, `--dp-delta`, `--dp-rounds`,
  `--dp-num-clients`: differential-privacy config, when DP is enabled

The FoT wrapper takes a narrower set: `--project-id`, `--port`, `--num-rounds`.

Every attacker-influenceable field is validated before it reaches the argv (SE-10). `ProcessBuilder`
with a `List` never invokes a shell, so the concrete risks are option injection (a value starting
with `-` read as an argparse flag) and path traversal via `--model-path` / `--model-name`. The
checks fail closed and never echo the rejected value.

#### Process Management

**Concurrent Map**:
```java
private final Map<UUID, ProcessHandle> runningServers = new ConcurrentHashMap<>();
```
- Key: Project UUID
- Value: `ProcessHandle` — **not** `Process`. BA-3: a restarted JVM can only recover a *handle* to a
  child that outlived a backend crash, never the original `Process` object, so the tracking map
  stores handles and `StartupReconciler` can re-adopt orphans.

**Stop Server**:
```java
public boolean stopServerForProject(UUID projectId) {
    runTokenRegistry.evictForProject(projectId);   // SE-7: invalidate this run's internal token
    ProcessHandle handle = runningServers.get(projectId);
    if (handle != null && handle.isAlive()) {
        handle.destroyForcibly();
        handle.onExit().get(stopWaitSeconds(), TimeUnit.SECONDS);   // bounded wait
        runningServers.remove(projectId);
        return true;
    }
    return false;
}
```

**Check Status**:
```java
public boolean isServerRunning(UUID projectId) {
    ProcessHandle p = runningServers.get(projectId);
    return (p != null && p.isAlive());
}
```

#### Port Allocation

Ports come from the configured range (`fl.server.port-range.start..end`, default `50000-50010`) —
not from an OS-assigned ephemeral port — because clients must reach the server on a predictable,
firewall-opened range.

```java
private int findFreePort() {
    synchronized (portReservationLock) {
        for (int port = portRangeStart; port <= portRangeEnd; port++) {
            if (reservedPorts.contains(port)) continue;
            try (ServerSocket s = new ServerSocket(port)) {
                reservedPorts.add(port);
                return port;
            } catch (IOException ignored) { /* port in use, try next */ }
        }
        throw new IllegalStateException("No free port in range ...");
    }
}
```
- Scans the range, probing each port with a `ServerSocket`
- The `reservedPorts` set closes a race: without it two concurrent project starts can both probe the
  same port as free, both close their probe socket, and both spawn Python on it
- The reservation is released in a `finally` regardless of outcome — on success the Python child now
  holds the port, so the next probe skips it naturally

---

### 2. WebSocketService

**Location**: `src/main/java/com/federated/fl_platform_api/service/WebSocketService.java`

**Purpose**: Broadcast messages to WebSocket subscribers.

**Key Methods**:
- `sendLogs(UUID projectId, String message)` - Send log to topic
- `sendStatusUpdate(UUID projectId, String status)` - Send status update

**Integration with FlServerManager**:
```java
@Autowired
private WebSocketService logBroadcaster;

// In output reader thread:
logBroadcaster.sendLogs(project.getId(), line);
```

**WebSocket Topic Structure**:
- Logs: `/topic/logs/{projectId}`
- Status: `/topic/status/{projectId}`

---

### 3. ProjectService

**Location**: `src/main/java/com/federated/fl_platform_api/service/ProjectService.java`

**Purpose**: Business logic for project operations.

**Key Responsibilities**:
1. Create/Read/Update/Delete projects
2. Coordinate with FlServerManager for server lifecycle
3. Update project status in database
4. Validate project configuration

**Typical Flow**:

**Create Project**:
```
1. Receive CreateProjectRequest DTO
2. Validate input
3. Create Project entity
4. Save to database via ProjectRepository
5. Return ProjectResponseDto
```

**Start Server**:
```
1. Receive StartProject DTO
2. Get Project from database
3. Call FlServerManager.startServerForProject()
4. Update project.status = RUNNING
5. Update project.port = {assignedPort}
6. Save project
7. Return status
```

**Stop Server**:
```
1. Get Project from database
2. Call FlServerManager.stopServerForProject()
3. Update project.status = STOPPED
4. Save project
5. Return status
```

---

## Database Layer

### Entities

#### Project Entity

**Location**: `src/main/java/com/federated/fl_platform_api/model/Project.java`

**Key Fields**:
```
- id: UUID (Primary Key)
- user: User (Many-to-One, LAZY)
- orgId: UUID (multi-tenant scope)
- name: String
- modelType: String (CNN, TRANSFORMER, LLM_LORA, ...)
- modelName: String
- modelPath: String
- optimizer: String
- taskType: String
- status: String (RUNNING, STOPPED, COMPLETED)
- initStatus: ProjectInitStatus
- serverPort: Integer
- visibility: ProjectVisibility (PUBLIC / RESTRICTED / PRIVATE, default PRIVATE)
- modelPublished / modelDescription / modelTags / modelPublishedAt
- activeRunId: UUID
- regulated: boolean
- dpEnabled: boolean, dpTargetEpsilon / dpDelta / dpClipNorm: Double
```

Note: `strategy`, `rounds` and `minClients` are **not** persisted on the entity — they are
per-start parameters passed into `FlServerManager.startServerForProject(...)`.

**Relationships**:
- Many-to-One with User
- One-to-Many with RoundResult

---

#### User Entity

**Location**: `src/main/java/com/federated/fl_platform_api/model/User.java`

**Key Fields**:
```
- id: Long (Primary Key)
- username: String (Unique, max 50)
- email: String (Unique, max 100)
- password: String (BCrypt hashed)
- platformRole: PlatformRole (USER / PROJECT_OWNER / PLATFORM_ADMIN, default USER)
- status: UserStatus (default ACTIVE)
- emailVerified: Boolean
- displayName / avatarUrl: String
- lastLoginAt / deletedAt: Instant
- createdAt: Instant
- updatedAt: Instant
```

**Relationships**:
- One-to-Many with Project

---

#### RoundResult Entity

**Location**: `src/main/java/com/federated/fl_platform_api/model/RoundResult.java`

**Key Fields**:
```
- id: UUID (Primary Key)
- project: Project (Many-to-One, non-null)
- serverRound: Integer (non-null)
- loss: Double
- accuracy: Double
- gpuUtilization: Double
```

---

### Repositories

#### ProjectRepository

**Location**: `src/main/java/com/federated/fl_platform_api/repository/ProjectRepository.java`

**Extends**: `JpaRepository<Project, UUID>`

**Custom Queries**:
```java
public interface ProjectRepository extends JpaRepository<Project, UUID> {
    List<Project> findByUserId(Long userId);

    // "My Projects": union of owned + joined (any project_memberships role)
    List<Project> findOwnedOrMemberOf(Long userId);

    // Discover feed: every non-PRIVATE project the caller neither owns nor is a member of
    List<Project> findDiscoverable(Long userId);

    // Org-scoped variants of the two above (multi-tenant isolation)
    List<Project> findOwnedOrMemberOfInOrgs(Long userId, Collection<UUID> orgIds);
    List<Project> findDiscoverableInOrgs(Long userId, Collection<UUID> orgIds);

    // PESSIMISTIC_WRITE row lock — serializes concurrent partition assignment for one project
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    Optional<Project> lockById(UUID id);
}
```

**Usage**:
```java
@Autowired
private ProjectRepository projectRepository;

// Find all projects for a user
List<Project> projects = projectRepository.findByUserId(userId);

// Find project by ID
Optional<Project> project = projectRepository.findById(projectId);

// Save project
projectRepository.save(project);

// Delete project
projectRepository.deleteById(projectId);
```

---

## Controller Layer

### ProjectController

**Location**: `src/main/java/com/federated/fl_platform_api/controller/ProjectController.java`

**Base Path**: `/api/projects`

**Endpoints Structure**:

```java
@RestController
@RequestMapping("/api/projects")
public class ProjectController {
    
    @Autowired
    private ProjectService projectService;
    
    @GetMapping
    public ResponseEntity<List<ProjectResponseDto>> getAllProjects() {
        // Implementation
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<ProjectResponseDto> getProjectById(@PathVariable UUID id) {
        // Implementation
    }
    
    @PostMapping
    public ResponseEntity<ProjectResponseDto> createProject(
        @RequestBody CreateProjectRequest request
    ) {
        // Implementation
    }
    
    @PostMapping("/{id}/start")
    public ResponseEntity<ApiResponse> startServer(
        @PathVariable UUID id,
        @RequestBody StartProject request
    ) {
        // Implementation
    }
    
    @PostMapping("/{id}/stop")
    public ResponseEntity<ApiResponse> stopServer(@PathVariable UUID id) {
        // Implementation
    }
}
```

**Authentication**:
- All endpoints require JWT token
- User ID extracted from SecurityContext
- Only project owner can access/modify their projects

---

### AuthController

**Location**: `src/main/java/com/federated/fl_platform_api/controller/AuthController.java`

**Base Path**: `/api/auth`

**Endpoints**:

```java
@RestController
@RequestMapping("/api/auth")
public class AuthController {
    
    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
        // 1. Validate input
        // 2. Check if user exists
        // 3. Hash password (BCrypt)
        // 4. Save user
        // 5. Set the jwtToken cookie, return the profile
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        // 1. Authenticate user
        // 2. Generate JWT token
        // 3. Set it as an HttpOnly jwtToken cookie; return user info ONLY
        //    (the token never appears in a response body)
    }

    @GetMapping("/me")
    public ResponseEntity<?> me() {
        // Silent 401 probe used by the SPA to bootstrap auth state
    }

    @PostMapping("/logout")
    public ResponseEntity<?> logout() {
        // Clear the cookie and revoke the token's jti (SE-8)
    }
}
```

---

## Security

### JWT Authentication Flow

The JWT travels as an **HttpOnly cookie**, not a Bearer header. The browser never sees the token.

```
1. User sends POST /api/auth/login with credentials
              ↓
2. AuthController validates credentials
              ↓
3. JwtTokenProvider generates JWT token
              ↓
4. Token returned as a Set-Cookie header (jwtToken; HttpOnly; SameSite)
   — the response body carries the user profile only
              ↓
5. Browser replays the cookie automatically (frontend sets withCredentials: true)
              ↓
6. JwtAuthenticationFilter intercepts request
              ↓
7. Reads the jwtToken cookie, validates signature + expiry, checks the
   revocation list (SE-8 logout), loads UserDetails, sets SecurityContext
              ↓
8. Request proceeds to controller
```

**SE-9 — Bearer is scoped to native clients only.** The filter always honors a valid `jwtToken`
cookie (the browser path). It falls back to an `Authorization: Bearer <jwt>` header **only** when
the request also carries the native-client marker header — desktop/mobile clients, which hold the
token in the OS keychain, need this. This is deliberately fail-closed: absent the explicit marker,
a Bearer header does nothing, so a browser-origin request cannot use one.

Authorities are reloaded from the database on every request (via `CustomUserDetailsService`), so a
role change takes effect immediately without re-login.

### SecurityConfig

**Location**: `src/main/java/com/federated/fl_platform_api/config/SecurityConfig.java`

**Key Configurations**:
- CORS settings
- JWT filter registration
- Public vs protected endpoints
- Password encoder (BCrypt)

**Public Endpoints** (no auth required):
- `/api/auth/login`
- `/api/auth/register`
- `/ws-logs/**` (WebSocket handshake — authenticated separately by `JwtHandshakeInterceptor`)
- `/api/internal/**` (service-to-service callbacks from FL-server processes — the Spring chain is
  `permitAll` because these authenticate with the internal API key / per-run token, not a JWT)
- `/actuator/health`

**Protected Endpoints** (JWT required):
- `/api/projects/**`
- `/api/results/**`
- `/actuator/**` — `PLATFORM_ADMIN` only (everything except `/actuator/health`)

---

### JwtTokenProvider

**Location**: `src/main/java/com/federated/fl_platform_api/security/JwtTokenProvider.java`

**Key Methods**:

```java
public class JwtTokenProvider {
    
    // Generate token from user details
    public String generateToken(UserDetails userDetails) {
        // Create JWT with claims
        // Set expiration
        // Sign with secret
    }
    
    // Extract username from token
    public String getUsernameFromToken(String token) {
        // Parse JWT
        // Return subject (username)
    }
    
    // Validate token
    public boolean validateToken(String token) {
        // Check signature
        // Check expiration
        // Return true/false
    }
}
```

---

### JwtAuthenticationFilter

**Location**: `src/main/java/com/federated/fl_platform_api/security/JwtAuthenticationFilter.java`

**Filter Chain Position**: Before UsernamePasswordAuthenticationFilter

**Process**:
```java
@Override
protected void doFilterInternal(
    HttpServletRequest request,
    HttpServletResponse response,
    FilterChain filterChain
) {
    // 1. Read the jwtToken cookie; fall back to a Bearer header ONLY for
    //    marked native clients (SE-9)
    String jwt = readJwtCookie(request);
    if (jwt == null && isNativeClient(request)) {
        jwt = readBearerToken(request);
    }

    // 2. Validate token (signature, expiry, and the SE-8 revocation list)
    if (jwt != null && jwtTokenProvider.validateToken(jwt, userDetails)
            && !tokenRevocationService.isRevoked(jwtTokenProvider.getJti(jwt))) {
        // 3. Get username
        String username = jwtTokenProvider.getUsernameFromToken(jwt);

        // 4. Load user details (authorities re-read from the DB every request)
        UserDetails userDetails = userDetailsService.loadUserByUsername(username);

        // 5. Set authentication in SecurityContext
        UsernamePasswordAuthenticationToken auth = 
            new UsernamePasswordAuthenticationToken(
                userDetails, null, userDetails.getAuthorities()
            );
        SecurityContextHolder.getContext().setAuthentication(auth);
    }
    
    // 6. Continue filter chain
    filterChain.doFilter(request, response);
}
```

---

## WebSocket Integration

### WebSocketConfig

**Location**: `src/main/java/com/federated/fl_platform_api/config/WebSocketConfig.java`

**Configuration**:
```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws-logs")
                .setAllowedOrigins(/* the CORS_ALLOWED_ORIGINS allowlist */)
                .addInterceptors(jwtHandshakeInterceptor);
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic", "/queue");
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        // Order matters: jwtChannelInterceptor promotes the handshake principal and
        // rejects unauthenticated CONNECTs; stompSubscriptionInterceptor then
        // authorizes each SUBSCRIBE destination.
        registration.interceptors(jwtChannelInterceptor, stompSubscriptionInterceptor);
    }
}
```

**Endpoints**:
- Connection: `ws://localhost:8081/ws-logs`
- Subscription prefixes: `/topic`, `/queue`
- Application prefix: `/app`

**Three interceptors guard the socket**:
- `JwtHandshakeInterceptor` — authenticates the HTTP upgrade using the same `jwtToken` cookie
- `JwtChannelInterceptor` — promotes the principal onto the STOMP session, rejects unauthenticated `CONNECT`
- `StompSubscriptionInterceptor` — authorizes each `SUBSCRIBE` destination (you cannot subscribe to another project's topic)

---

## Adding New Features

### Adding a New Entity

**1. Create Entity Class**:
```java
@Entity
@Table(name = "new_entity")
public class NewEntity {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    // Fields, getters, setters
}
```

**2. Write a Flyway migration** — **not optional**. Hibernate runs `ddl-auto=validate` in every
profile except `test`, so an entity without a matching migration fails the boot rather than
creating its table. Add `src/main/resources/db/migration/V{n}__add_new_entity.sql` (the highest
committed migration is `V19`):
```sql
CREATE TABLE new_entity (
    id BIGSERIAL PRIMARY KEY
    -- columns matching the entity
);
```

**3. Create Repository**:
```java
public interface NewEntityRepository extends JpaRepository<NewEntity, Long> {
    // Custom queries
}
```

**4. Create Service**:
```java
@Service
public class NewEntityService {
    
    @Autowired
    private NewEntityRepository repository;
    
    // Business logic methods
}
```

**5. Create Controller**:
```java
@RestController
@RequestMapping("/api/new-entity")
public class NewEntityController {
    
    @Autowired
    private NewEntityService service;
    
    // Endpoints
}
```

---

### Adding a New Endpoint

**1. Define DTO** (if needed):
```java
public class NewRequestDto {
    private String field1;
    private Integer field2;
    // Getters, setters, validation
}
```

**2. Add Controller Method**:
```java
@PostMapping("/custom-action")
public ResponseEntity<ApiResponse> customAction(
    @RequestBody NewRequestDto request
) {
    // Call service
    service.performAction(request);
    
    return ResponseEntity.ok(
        new ApiResponse(true, "Action completed")
    );
}
```

**3. Implement Service Logic**:
```java
public void performAction(NewRequestDto request) {
    // Business logic
    // Database operations
    // External calls
}
```

---

## Testing

### Unit Testing Services

```java
@ExtendWith(MockitoExtension.class)
class ProjectServiceTest {
    
    @Mock
    private ProjectRepository projectRepository;
    
    @InjectMocks
    private ProjectService projectService;
    
    @Test
    void testCreateProject() {
        // Arrange
        CreateProjectRequest request = new CreateProjectRequest();
        request.setName("Test Project");
        
        // Act
        ProjectResponseDto response = projectService.createProject(request, userId);
        
        // Assert
        assertNotNull(response);
        assertEquals("Test Project", response.getName());
    }
}
```

### Integration Testing Controllers

Integration tests run against **real PostgreSQL via Testcontainers** — `@ActiveProfiles("test")`
loads `src/test/resources/application-test.properties`, whose `jdbc:tc:postgresql:16.6-alpine:///fedlearn_test`
URL starts (or reuses) one throwaway container for the JVM run. **A working Docker daemon is
required**; there is no in-memory fallback since H2 was retired.

Authenticate with the `jwtToken` **cookie**, not a Bearer header:

```java
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ProjectControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void testGetAllProjects() throws Exception {
        mockMvc.perform(get("/api/projects")
                .cookie(new Cookie("jwtToken", token)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
}
```

The bulk suite builds the schema with Hibernate `create-drop` and keeps Flyway **off**; the
dedicated `V*MigrationTest` classes flip Flyway on to exercise the real migrations against real
Postgres. Note the deliberately small Hikari pool (`maximum-pool-size=4`, `minimum-idle=0`): the
suite caches ~20 Spring contexts that all pool against the single shared container, and at Hikari's
default pool size the summed pools overshoot Postgres's `max_connections` under full-suite load.

---

## Best Practices

### 1. Separation of Concerns
- Controllers handle HTTP
- Services contain business logic
- Repositories handle data access

### 2. DTO Usage
- Never expose entities directly in API
- Use DTOs for request/response
- Map between entities and DTOs in service layer

### 3. Exception Handling
- Use `@ControllerAdvice` for global exception handling
- Return consistent error responses
- Log exceptions appropriately

### 4. Transaction Management
- Use `@Transactional` on service methods
- Let Spring manage transaction boundaries

### 5. Configuration
- Externalize configuration (application.properties)
- Use environment variables for sensitive data
- Different profiles for dev/prod

### 6. Logging
- Use SLF4J/Logback
- Log at appropriate levels (INFO, DEBUG, ERROR)
- Include context in log messages

---

For API endpoint reference, see [REST endpoints](README.md#rest-endpoints-overview) in the README, or the backend wiki: [`wikis/backend/README.md`](../../wikis/backend/README.md).

For deployment guide, see main [README.md](README.md).