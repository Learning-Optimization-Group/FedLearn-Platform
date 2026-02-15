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
│  - FlowerServerManager                      │
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
4. Manager: FlowerServerManager.startServerForProject()
              ↓
5. Process: Spawns Python script with parameters
              ↓
6. WebSocket: Streams logs to /topic/logs/{projectId}
              ↓
7. Frontend: Receives real-time logs via WebSocket
```

---

## Key Components

### 1. FlowerServerManager

**Location**: `src/main/java/com/federated/fl_platform_api/flower/FlowerServerManager.java`

**Purpose**: Manages FL server processes (start, stop, monitor).

**Key Features**:
- Cross-platform support (Windows .bat, Linux/Mac .sh)
- Dynamic port allocation
- Process output streaming to WebSocket
- Concurrent server tracking

#### Full Implementation

```java
package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.WebSocketService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Component
public class FlowerServerManager {

    // This property points to the run_fl_server wrapper script
    @Value("${python.script.fl-server.path}")
    private String flServerWrapperPath;

    // A map to keep track of the running server processes for cleanup
    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    @Autowired
    private WebSocketService logBroadcaster;

    /**
     * Starts a dedicated Flower server process for a given project.
     */
    public int startServerForProject(Project project, boolean isPretrained, String strategy, Integer numRounds, Integer minClients) throws IOException, InterruptedException {

        stopServerForProject(project.getId());
        Thread.sleep(2000);

        int freePort = findFreePort();

        System.out.println("--- Preparing to Start Flower Server ---");

        List<String> command = new ArrayList<>();
        String os = System.getProperty("os.name").toLowerCase();

        // Determine script path based on OS
        String scriptPath;
        if (os.contains("win")) {
            // Windows - use .bat file
            scriptPath = flServerWrapperPath.replace(".sh", ".bat");
            command.add(scriptPath);
        } else {
            // Linux/Mac - use .sh file and call with bash
            scriptPath = flServerWrapperPath.replace(".bat", ".sh");
            File scriptFile = new File(scriptPath);
            String absoluteScriptPath = scriptFile.getAbsolutePath();
            command.add("bash");
            command.add(absoluteScriptPath);
        }

        // Add the arguments for the script
        command.add("--project-id");
        command.add(project.getId().toString());
        command.add("--model-path");
        command.add(project.getModelPath());
        command.add("--port");
        command.add(String.valueOf(freePort));
        command.add("--strategy");
        command.add(strategy);
        command.add("--num-rounds");
        command.add(String.valueOf(numRounds));
        command.add("--min-clients");
        command.add(String.valueOf(minClients));
        command.add("--model-type");
        command.add(project.getModelType());
        command.add("--model-name");
        command.add(project.getModelName());

        if (!isPretrained) {
            command.add("--pretrain");
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        pb.directory(new File("."));

        System.out.println("Executing command: " + String.join(" ", pb.command()));

        Process process = pb.start();
        runningServers.put(project.getId(), process);

        // --- Asynchronous output reader AND process health check ---
        final StringBuilder startupOutput = new StringBuilder();
        final var errorOccurred = new boolean[]{false};

        Thread outputReaderThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("[FL_SERVER_LOG " + project.getId() + "] " + line);
                    logBroadcaster.sendLogs(project.getId(), line);
                    startupOutput.append(line).append("\n");
                }
            } catch (IOException e) {
                System.err.println("Error reading output from Flower server process for project " + project.getId());
                errorOccurred[0] = true;
                logBroadcaster.sendLogs(project.getId(), "ERROR: " + e);
                e.printStackTrace();
            }
        });
        outputReaderThread.setDaemon(true);
        outputReaderThread.start();

        // Wait for a short period to see if the process exits immediately
        boolean exited = process.waitFor(3, TimeUnit.SECONDS);

        if (exited || errorOccurred[0]) {
            outputReaderThread.join(1000);
            throw new RuntimeException("Flower server process failed to start. Exit code: " + process.exitValue() +
                    "\nFull Output:\n" + startupOutput);
        }

        System.out.println("Started Flower server for project " + project.getName() + " on port " + freePort);
        return freePort;
    }

    public boolean stopServerForProject(UUID projectId) {
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            System.out.println("Stopping Flower server for project: " + projectId);
            process.destroyForcibly();
            runningServers.remove(projectId);
            return true;
        }
        System.out.println("No running server found for project: " + projectId);
        return false;
    }

    private int findFreePort() {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            if (serverSocket != null) {
                return serverSocket.getLocalPort();
            }
        } catch (IOException e) {
            throw new IllegalStateException("Could not find a free TCP/IP port", e);
        }
        throw new IllegalStateException("Could not find a free TCP/IP port");
    }

    public boolean isServerRunning(UUID projectId) {
        Process p = runningServers.get(projectId);
        return (p != null && p.isAlive());
    }
}
```

#### How It Works

**1. Start Server**:
```
stopServerForProject(project.getId());  // Stop any existing server
Thread.sleep(2000);                     // Wait for cleanup
int freePort = findFreePort();          // Get available port
```

**2. Build Command** (cross-platform):
```java
// Windows
run_fl_server.bat --project-id {id} --port {port} ...

// Linux/Mac
bash run_fl_server.sh --project-id {id} --port {port} ...
```

**3. Spawn Process**:
```java
ProcessBuilder pb = new ProcessBuilder(command);
pb.redirectErrorStream(true);  // Merge stderr into stdout
Process process = pb.start();
```

**4. Stream Output**:
```java
Thread outputReaderThread = new Thread(() -> {
    try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream()))) {
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println("[FL_SERVER_LOG] " + line);
            logBroadcaster.sendLogs(project.getId(), line);
        }
    }
});
outputReaderThread.setDaemon(true);
outputReaderThread.start();
```

**5. Health Check**:
```java
boolean exited = process.waitFor(3, TimeUnit.SECONDS);
if (exited || errorOccurred[0]) {
    throw new RuntimeException("Server failed to start");
}
```

**6. Track Process**:
```java
runningServers.put(project.getId(), process);
```

#### Configuration

In `application.properties`:
```properties
# Path to the FL server wrapper script
python.script.fl-server.path=src/main/resources/scripts/run_fl_server.sh
```

#### Script Parameters

The wrapper script receives:
- `--project-id`: UUID of the project
- `--model-path`: Path to model checkpoint
- `--port`: Port number (dynamically allocated)
- `--strategy`: Aggregation strategy (FedAvg, DeComFL)
- `--num-rounds`: Number of training rounds
- `--min-clients`: Minimum clients required
- `--model-type`: Type of model (CNN, Transformer, etc.)
- `--model-name`: Specific model architecture
- `--pretrain`: Flag for pre-training (optional)

#### Process Management

**Concurrent Map**:
```java
private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();
```
- Key: Project UUID
- Value: Running Process instance

**Stop Server**:
```java
public boolean stopServerForProject(UUID projectId) {
    Process process = runningServers.get(projectId);
    if (process != null && process.isAlive()) {
        process.destroyForcibly();  // Kill process
        runningServers.remove(projectId);
        return true;
    }
    return false;
}
```

**Check Status**:
```java
public boolean isServerRunning(UUID projectId) {
    Process p = runningServers.get(projectId);
    return (p != null && p.isAlive());
}
```

#### Port Allocation

**Dynamic Port Finding**:
```java
private int findFreePort() {
    try (ServerSocket serverSocket = new ServerSocket(0)) {
        return serverSocket.getLocalPort();
    } catch (IOException e) {
        throw new IllegalStateException("Could not find a free TCP/IP port", e);
    }
}
```
- Creates temporary ServerSocket on port 0
- OS assigns available port
- Returns port number
- Socket closes automatically (try-with-resources)

---

### 2. WebSocketService

**Location**: `src/main/java/com/federated/fl_platform_api/service/WebSocketService.java`

**Purpose**: Broadcast messages to WebSocket subscribers.

**Key Methods**:
- `sendLogs(UUID projectId, String message)` - Send log to topic
- `sendStatusUpdate(UUID projectId, String status)` - Send status update

**Integration with FlowerServerManager**:
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
2. Coordinate with FlowerServerManager for server lifecycle
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
3. Call FlowerServerManager.startServerForProject()
4. Update project.status = RUNNING
5. Update project.port = {assignedPort}
6. Save project
7. Return status
```

**Stop Server**:
```
1. Get Project from database
2. Call FlowerServerManager.stopServerForProject()
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
- user: User (Many-to-One)
- name: String
- type: String (CNN, Transformer, etc.)
- modelName: String
- modelType: String
- modelPath: String
- optimizer: String
- strategy: String
- rounds: Integer
- minClients: Integer
- status: String (RUNNING, STOPPED, COMPLETED)
- port: Integer
- createdAt: LocalDateTime
- updatedAt: LocalDateTime
```

**Relationships**:
- Many-to-One with User
- One-to-Many with RoundResult

---

#### User Entity

**Location**: `src/main/java/com/federated/fl_platform_api/model/User.java`

**Key Fields**:
```
- id: Long (Primary Key)
- username: String (Unique)
- email: String (Unique)
- password: String (BCrypt hashed)
- createdAt: LocalDateTime
```

**Relationships**:
- One-to-Many with Project

---

#### RoundResult Entity

**Location**: `src/main/java/com/federated/fl_platform_api/model/RoundResult.java`

**Key Fields**:
```
- id: Long (Primary Key)
- project: Project (Many-to-One)
- roundNumber: Integer
- accuracy: Double
- loss: Double
- metrics: String (JSON)
- timestamp: LocalDateTime
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
    List<Project> findByStatus(String status);
    Optional<Project> findByIdAndUserId(UUID id, Long userId);
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
        // 5. Return success
    }
    
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        // 1. Authenticate user
        // 2. Generate JWT token
        // 3. Return token + user info
    }
}
```

---

## Security

### JWT Authentication Flow

```
1. User sends POST /api/auth/login with credentials
              ↓
2. AuthController validates credentials
              ↓
3. JwtTokenProvider generates JWT token
              ↓
4. Token returned to client
              ↓
5. Client includes token in Authorization header:
   "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
              ↓
6. JwtAuthenticationFilter intercepts request
              ↓
7. Extracts token, validates, sets SecurityContext
              ↓
8. Request proceeds to controller
```

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
- `/ws/**` (WebSocket)

**Protected Endpoints** (JWT required):
- `/api/projects/**`
- `/api/results/**`

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
    // 1. Extract token from Authorization header
    String token = getTokenFromRequest(request);
    
    // 2. Validate token
    if (token != null && jwtTokenProvider.validateToken(token)) {
        // 3. Get username
        String username = jwtTokenProvider.getUsernameFromToken(token);
        
        // 4. Load user details
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
        registry.addEndpoint("/ws")
                .setAllowedOrigins("http://localhost:5173", "https://your-frontend.com")
                .withSockJS();
    }
    
    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        registry.enableSimpleBroker("/topic");
        registry.setApplicationDestinationPrefixes("/app");
    }
}
```

**Endpoints**:
- Connection: `ws://localhost:8080/ws`
- Subscription prefix: `/topic`
- Application prefix: `/app`

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

**2. Create Repository**:
```java
public interface NewEntityRepository extends JpaRepository<NewEntity, Long> {
    // Custom queries
}
```

**3. Create Service**:
```java
@Service
public class NewEntityService {
    
    @Autowired
    private NewEntityRepository repository;
    
    // Business logic methods
}
```

**4. Create Controller**:
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

```java
@SpringBootTest
@AutoConfigureMockMvc
class ProjectControllerIntegrationTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Test
    void testGetAllProjects() throws Exception {
        mockMvc.perform(get("/api/projects")
                .header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
}
```

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

For API endpoint reference, see [API.md](API.md).

For deployment guide, see main [README.md](README.md).