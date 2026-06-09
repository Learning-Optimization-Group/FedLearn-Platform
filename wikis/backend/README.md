# Backend Architecture Wiki

Welcome to the internal documentation for the **FedLearn-Platform Backend**.

This section of the wiki covers the Spring Boot 3 API, the PostgreSQL database interactions, security, WebSockets, and how the federated learning python processes are orchestrated natively or scaled on AWS ECS Fargate.

## Documentation Index

1. **[Architecture & Core Concepts](01_architecture_overview.md)**
   Provides a high-level overview of the backend's directory structure, the technology stack, and the primary domain models (Projects, Results, Logs).

2. **[Security & Authentication](02_security_and_auth.md)**
   Explains how stateless JWT validation works via filters, how WebSocket connections are secured during the HTTP upgrade handshake, and the internal API key mechanism used by FL servers.

3. **[Project Management Lifecycle](03_project_management.md)**
   Details the `ProjectService` and `ProjectController` logic. Explains how training rounds are configured, how projects are persisted to PostgreSQL, and how models are initialized.

4. **[Federated Orchestration (FlowerServerManager)](04_federated_orchestration.md)**
   The most complex component. Documents how the Java API dynamically provisions the Python FL aggregation servers, differentiating between local machine `ProcessBuilder` execution and cloud-native AWS ECS Fargate orchestration.

5. **[WebSocket Log Streaming](05_websocket_logs_streaming.md)**
   Explains the real-time observability pipeline. Shows how the backend captures standard output from the Python FL servers, routes it via STOMP topics to the React frontend, and persists it for export.

6. **[Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)**
   Documents the identity subsystem: the three-layer role model (platform / organization / project), organization-scoped multi-tenant isolation (`OrgScope`), the `@Auditable` audit trail, the email + first-run bootstrap plumbing, the V4–V6 migrations, and the new membership/admin/access-request REST endpoints. (Backend-first; the client RBAC UI is deferred.)
