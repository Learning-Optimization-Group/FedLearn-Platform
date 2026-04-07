# FedLearn Platform - Backend API

Spring Boot REST API for managing federated learning projects, authentication, and server lifecycle management. This backend orchestrates FL experiments by spawning Python FL servers and streaming real-time logs via WebSocket.

## Features

- 🔐 **JWT Authentication** - Secure user authentication and authorization
- 📊 **Project Management** - CRUD operations for FL projects
- 🚀 **Server Lifecycle** - Start/stop FL servers dynamically
- 📡 **WebSocket Streaming** - Real-time server logs
- 💾 **PostgreSQL Database** - Persistent storage with Hibernate/JPA
- 🐍 **Python Integration** - Spawns FL server processes (`fl_server.py`)

## Tech Stack

- **Framework**: Spring Boot 3.x
- **Database**: PostgreSQL with Hibernate/JPA
- **Security**: Spring Security + JWT
- **WebSocket**: STOMP over WebSocket
- **Build Tool**: Maven/Gradle
- **Java Version**: 17+

## Project Structure

```
backend/fl-platform-api/
├── src/main/java/com/federated/fl_platform_api/
│   ├── config/
│   │   ├── SecurityConfig.java          # Spring Security configuration
│   │   └── WebSocketConfig.java         # WebSocket/STOMP configuration
│   ├── controller/
│   │   ├── AuthController.java          # Login/Register endpoints
│   │   ├── CorsTestController.java      # CORS testing
│   │   ├── HomeController.java          # Health check
│   │   ├── ProjectController.java       # Project CRUD + Server control
│   │   └── ResultsController.java       # Training results
│   ├── dto/
│   │   ├── ApiResponse.java             # Standard API response wrapper
│   │   ├── CreateProjectRequest.java    # Project creation payload
│   │   ├── LoginRequest.java            # Login credentials
│   │   ├── ProjectResponseDto.java      # Project response
│   │   ├── ProjectStatusUpdateDto.java  # Status update payload
│   │   ├── RegisterRequest.java         # Registration payload
│   │   ├── RoundResultDto.java          # Round metrics
│   │   └── StartProject.java            # Server start payload
│   ├── exception/
│   │   ├── GlobalExceptionHandler.java  # Centralized error handling
│   │   └── UserAlreadyExistsException.java
│   ├── flower/
│   │   └── FlowerServerManager.java     # Manages FL server processes
│   ├── model/
│   │   ├── Project.java                 # Project entity
│   │   ├── RoundResult.java             # Training round results
│   │   └── User.java                    # User entity
│   ├── repository/
│   │   ├── ProjectRepository.java       # Project data access
│   │   ├── RoundResultRepository.java   # Results data access
│   │   └── UserRepository.java          # User data access
│   ├── security/
│   │   ├── JwtAuthenticationFilter.java # JWT token validation filter
│   │   └── JwtTokenProvider.java        # JWT token generation/validation
│   ├── service/
│   │   ├── CustomUserDetailsService.java # Spring Security user details
│   │   ├── ModelInitializer.java        # Model initialization logic
│   │   ├── ProjectService.java          # Project business logic
│   │   ├── UserService.java             # User business logic
│   │   └── WebSocketService.java        # WebSocket message handling
│   └── FlPlatformApiApplication.java    # Main Spring Boot application
├── src/main/resources/
│   ├── scripts/                         # Python FL server scripts
│   │   ├── architecture.cnn/
│   │   ├── data_loaders/
│   │   ├── data_splits/
│   │   ├── ecg_data/
│   │   ├── models/                      # Saved model checkpoints
│   │   ├── client.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── fl_server.py                 # Python FL server script
│   │   ├── init_model.py
│   │   ├── model_utils.py
│   │   ├── models.py
│   │   ├── run_clients.sh
│   │   ├── run_fl_server.bat
│   │   ├── run_fl_server.sh             # Shell script to start FL server
│   │   ├── run_init_model.bat
│   │   └── run_init_model.sh
│   ├── application.properties           # Main configuration
│   └── application-production.properties # Production config
├── .ebextensions/                       # AWS Elastic Beanstalk config
├── .github/                             # GitHub Actions CI/CD
├── data/                                # Data storage
├── models/                              # Model storage
└── queries/                             # SQL queries/migrations
```

## Quick Start

### Prerequisites

- Java 17+
- PostgreSQL 12+
- Python 3.10+ (for FL servers)
- Maven or Gradle

### Installation

```bash
# Navigate to backend directory
cd backend/fl-platform-api

# Configure database
# Edit src/main/resources/application.properties

# Build the project
mvn clean install
# or
gradle build

# Run the application
mvn spring-boot:run
# or
gradle bootRun
```

The API will be available at `http://localhost:8080`

## Configuration

### Database Configuration

Edit `src/main/resources/application.properties`:

```properties
# PostgreSQL Configuration
spring.datasource.url=jdbc:postgresql://localhost:5432/fedlearn_db
spring.datasource.username=your_username
spring.datasource.password=your_password

# Hibernate
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect

# JWT Configuration
jwt.secret=your-secret-key-here
jwt.expiration=86400000

# Python Scripts Path
fl.scripts.path=src/main/resources/scripts
fl.models.path=models

# WebSocket
spring.websocket.allowed-origins=http://localhost:5173,https://your-frontend-url.com
```

### Production Configuration

For production, use `application-production.properties` with environment variables:

```properties
spring.datasource.url=${DATABASE_URL}
spring.datasource.username=${DB_USERNAME}
spring.datasource.password=${DB_PASSWORD}
jwt.secret=${JWT_SECRET}
```

## Architecture Overview

### Layer Architecture

```
┌─────────────────────────────────────────┐
│         Controller Layer                │  ← REST endpoints
│  (AuthController, ProjectController)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Service Layer                   │  ← Business logic
│  (ProjectService, UserService)          │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Repository Layer                │  ← Data access (JPA)
│  (ProjectRepository, UserRepository)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         PostgreSQL Database             │
└─────────────────────────────────────────┘
```

### FL Server Integration

```
┌──────────────────┐
│  Spring Boot API │
│  ProjectService  │
└────────┬─────────┘
         │
         │ Spawns Process
         ▼
┌──────────────────┐
│ FlowerServerMgr  │──→ Runs: python fl_server.py --port 50051 --project-id {id}
└────────┬─────────┘
         │
         │ Process Output
         ▼
┌──────────────────┐
│ WebSocketService │──→ Streams logs to: /topic/logs/{projectId}
└──────────────────┘
```

## API Endpoints

### Authentication (`/api/auth`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/register` | Register new user | No |
| POST | `/api/auth/login` | Login and get JWT token | No |

**Register Request**:
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securePassword123"
}
```

**Login Request**:
```json
{
  "email": "john@example.com",
  "password": "securePassword123"
}
```

**Login Response**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com"
  }
}
```

---

### Projects (`/api/projects`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/projects` | Get all user projects | Yes |
| GET | `/api/projects/{id}` | Get project by ID | Yes |
| POST | `/api/projects` | Create new project | Yes |
| PUT | `/api/projects/{id}` | Update project | Yes |
| DELETE | `/api/projects/{id}` | Delete project | Yes |
| POST | `/api/projects/{id}/start` | Start FL server | Yes |
| POST | `/api/projects/{id}/stop` | Stop FL server | Yes |
| GET | `/api/projects/{id}/status` | Get server status | Yes |

**Create Project Request**:
```json
{
  "name": "CNN Training v1",
  "type": "CNN",
  "model": "resnet18",
  "optimizer": "Adam",
  "strategy": "FedAvg",
  "rounds": 10,
  "minClients": 2,
  "clientsPerRound": 5
}
```

**Project Response**:
```json
{
  "id": "uuid-here",
  "name": "CNN Training v1",
  "type": "CNN",
  "model": "resnet18",
  "status": "STOPPED",
  "strategy": "FedAvg",
  "rounds": 10,
  "minClients": 2,
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T10:30:00Z"
}
```

---

### Results (`/api/results`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/results/{projectId}` | Get training results | Yes |
| GET | `/api/results/{projectId}/round/{roundNum}` | Get specific round results | Yes |

**Results Response**:
```json
{
  "projectId": "uuid-here",
  "totalRounds": 10,
  "currentRound": 5,
  "rounds": [
    {
      "roundNumber": 1,
      "accuracy": 0.75,
      "loss": 0.45,
      "timestamp": "2024-01-15T10:35:00Z"
    }
  ]
}
```

---

## WebSocket Endpoints

### Log Streaming

**Connect**: `ws://localhost:8080/ws`

**Subscribe**: `/topic/logs/{projectId}`

**Message Format**:
```json
{
  "timestamp": "2024-01-15T10:35:00Z",
  "level": "INFO",
  "message": "Round 1 started with 3 clients"
}
```

**Client Example (JavaScript)**:
```javascript
const client = new Client({
  brokerURL: 'ws://localhost:8080/ws',
  onConnect: () => {
    client.subscribe('/topic/logs/project-123', (message) => {
      console.log(message.body);
    });
  }
});
client.activate();
```

---

## Key Components Explained

### 1. FlowerServerManager

Manages FL server lifecycle by spawning Python processes.

**Responsibilities**:
- Start FL server process for a project
- Monitor server status
- Stop server process
- Stream server output to WebSocket

**Key Methods** (refer to actual code for implementation):
- `startServer(projectId, port)` - Spawns `python fl_server.py --project-id {id} --port {port}`
- `stopServer(projectId)` - Kills the server process
- `isServerRunning(projectId)` - Checks if server is active

---

### 2. ProjectService

Business logic for project management.

**Responsibilities**:
- Create/Read/Update/Delete projects
- Validate project configuration
- Coordinate with FlowerServerManager
- Update project status

**Key Operations**:
- Create project → Save to DB → Return DTO
- Start server → Call FlowerServerManager → Update status to RUNNING
- Stop server → Call FlowerServerManager → Update status to STOPPED

---

### 3. WebSocketService

Handles WebSocket message broadcasting.

**Responsibilities**:
- Send log messages to subscribed clients
- Broadcast project status updates
- Handle connection management

**Message Topics**:
- `/topic/logs/{projectId}` - Server logs
- `/topic/status/{projectId}` - Status updates

---

### 4. JwtTokenProvider & JwtAuthenticationFilter

JWT-based authentication.

**JwtTokenProvider**:
- Generate JWT tokens on login
- Validate tokens
- Extract user info from tokens

**JwtAuthenticationFilter**:
- Intercept requests
- Extract JWT from Authorization header
- Validate token
- Set authentication in SecurityContext

---

## Database Schema

### User Table

| Column | Type | Constraints |
|--------|------|-------------|
| id | BIGINT | Primary Key, Auto-increment |
| username | VARCHAR(50) | Unique, Not Null |
| email | VARCHAR(100) | Unique, Not Null |
| password | VARCHAR(255) | Not Null (BCrypt hashed) |
| created_at | TIMESTAMP | Not Null |

---

### Project Table

| Column | Type | Constraints |
|--------|------|-------------|
| id | VARCHAR(36) | Primary Key (UUID) |
| user_id | BIGINT | Foreign Key → User |
| name | VARCHAR(100) | Not Null |
| type | VARCHAR(50) | Not Null |
| model | VARCHAR(50) | Not Null |
| optimizer | VARCHAR(50) | Not Null |
| strategy | VARCHAR(50) | Not Null |
| rounds | INT | Not Null |
| min_clients | INT | Not Null |
| status | VARCHAR(20) | RUNNING, STOPPED, COMPLETED |
| port | INT | Server port number |
| created_at | TIMESTAMP | Not Null |
| updated_at | TIMESTAMP | Not Null |

---

### RoundResult Table

| Column | Type | Constraints |
|--------|------|-------------|
| id | BIGINT | Primary Key, Auto-increment |
| project_id | VARCHAR(36) | Foreign Key → Project |
| round_number | INT | Not Null |
| accuracy | DOUBLE | Nullable |
| loss | DOUBLE | Nullable |
| metrics | JSON/TEXT | Additional metrics |
| timestamp | TIMESTAMP | Not Null |

---

## Security

### JWT Configuration

- **Algorithm**: HS256
- **Expiration**: Configurable (default: 24 hours)
- **Secret**: Stored in application.properties (use env var in production)

### Password Security

- **Hashing**: BCrypt with strength 10
- **Storage**: Never store plaintext passwords

### CORS Configuration

Configured in `SecurityConfig.java` to dynamically allow frontend origins based on deployment architecture.

**Allowed Origins**:
Origins are resolved dynamically via environment variables (e.g. `FRONTEND_URL`) to seamlessly support LAN deployments and AWS integration, strictly avoiding hardcoded bindings to `http://localhost:5173` in production.

---

## Python FL Server Integration

### How It Works

1. **User clicks "Start Server"** in frontend
2. **Backend receives** POST `/api/projects/{id}/start`
3. **FlowerServerManager spawns** Python process:
   ```bash
   python src/main/resources/scripts/fl_server.py \
     --project-id {projectId} \
     --port {port} \
     --rounds {rounds} \
     --min-clients {minClients} \
     --strategy {strategy}
   ```
4. **Process output** is captured and streamed via WebSocket
5. **Server runs** until training completes or user stops it

### FL Server Script Parameters

The `fl_server.py` script accepts:
- `--project-id`: Unique project identifier
- `--port`: gRPC server port (e.g., 50051)
- `--rounds`: Number of training rounds
- `--min-clients`: Minimum clients required
- `--strategy`: Aggregation strategy (FedAvg, DeComFL)

---

## Deployment

### Local Development

```bash
# Start PostgreSQL
docker run -d \
  --name fedlearn-postgres \
  -e POSTGRES_DB=fedlearn_db \
  -e POSTGRES_USER=fedlearn \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15

# Run Spring Boot
mvn spring-boot:run
```

### Production (AWS/Cloud)

1. **Database**: Use managed PostgreSQL (AWS RDS, etc.)
2. **Application**: Deploy JAR to EC2/Elastic Beanstalk
3. **Environment Variables**: Set via platform
4. **Python Environment**: Ensure Python 3.10+ installed on server

### Docker Deployment

```bash
# Build JAR
mvn clean package

# Build Docker image
docker build -t fedlearn-backend .

# Run container
docker run -p 8080:8080 \
  -e DATABASE_URL=jdbc:postgresql://host:5432/db \
  -e DB_USERNAME=user \
  -e DB_PASSWORD=pass \
  fedlearn-backend
```

---

## Troubleshooting

### Issue: Database connection fails

**Check**:
1. PostgreSQL is running
2. Database credentials correct
3. Database exists

### Issue: FL server won't start

**Check**:
1. Python 3.10+ installed
2. `fl_server.py` exists in scripts directory
3. Required Python packages installed
4. Port not already in use

### Issue: JWT authentication fails

**Check**:
1. Token included in Authorization header
2. Token not expired
3. JWT secret matches between requests

### Issue: WebSocket not connecting

**Check**:
1. CORS configuration allows frontend origin
2. WebSocket endpoint enabled
3. Firewall allows WebSocket connections

---

## Development Guide

See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Adding new endpoints
- Creating new entities
- Extending services
- Testing guidelines

## API Documentation

For interactive API docs, see [API.md](API.md) for complete endpoint reference.

## Contributing

When modifying the backend:
1. Follow Spring Boot best practices
2. Keep controllers thin, services thick
3. Use DTOs for API requests/responses
4. Write JUnit tests for services
5. Document new endpoints

---

**Repository**: [GitHub URL]
**API Base URL**: `http://localhost:8080/api`
**WebSocket URL**: `ws://localhost:8080/ws`