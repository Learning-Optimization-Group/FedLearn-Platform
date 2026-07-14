# Backend Architecture Wiki

Welcome to the internal documentation for the **FedLearn-Platform Backend**.

This section of the wiki covers the Spring Boot 3 API, the database interactions, security, WebSockets, and how the federated learning python processes are orchestrated natively or scaled on AWS ECS Fargate.

> ⚠️ **Branch reality.** The backend runs on **PostgreSQL** for every profile (H2 has been retired) — `dev`/`ec2demo` against a local Postgres (`backend/fl-platform-api/docker-compose.yml` → `docker compose up -d`) and `test` against Testcontainers Postgres. The highest committed Flyway migration is **`V19`**. Authorization on this branch is the single coarse `users.role IN (USER, ADMIN)` column (migration `V2`). The identity / multi-tenancy / audit subsystem documented in page 06 is **designed on a separate identity-foundations branch and is not present here.**

## Documentation Index

1. **[Architecture & Core Concepts](01_architecture_overview.md)**
   Provides a high-level overview of the backend's directory structure, the technology stack, and the primary domain models (Projects, Results, Logs).

2. **[Security & Authentication](02_security_and_auth.md)**
   Explains how stateless JWT validation works via filters, how WebSocket connections are secured during the HTTP upgrade handshake, and the internal API key mechanism used by FL servers.

3. **[Project Management Lifecycle](03_project_management.md)**
   Details the `ProjectService` and `ProjectController` logic. Explains how training rounds are configured, how projects are persisted to the database, and how models are initialized.

4. **[Federated Orchestration (FlowerServerManager)](04_federated_orchestration.md)**
   The most complex component. Documents how the Java API dynamically provisions the Python FL aggregation servers, differentiating between local machine `ProcessBuilder` execution and cloud-native AWS ECS Fargate orchestration.

5. **[WebSocket Log Streaming](05_websocket_logs_streaming.md)**
   Explains the real-time observability pipeline. Shows how the backend captures standard output from the Python FL servers, routes it via STOMP topics to the React frontend, and persists it for export.

6. **[Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)** — ⚠️ **designed on a separate identity-foundations branch; not present on this branch.**
   Documents the identity subsystem: the three-layer role model (platform / organization / project), organization-scoped multi-tenant isolation (`OrgScope`), the `@Auditable` audit trail, the email + first-run bootstrap plumbing, the V4–V6 migrations, and the membership/admin/access-request REST endpoints. None of this is committed on the current branch — it is included for reference.

7. **[Content-Addressed Model Artifact Registry](07_artifact_registry.md)**
   Documents the registry that replaced the single overwritable `.npz` at `projects.model_path`: the `artifact_blobs` / `model_artifacts` / `artifact_lineage` data model, the `ArtifactBlobStore` write-once content store, `RegistryModelResolver`'s registry-first read path for inference and FL-server warm-start, the artifact/lineage/marketplace REST surface, and the `V12`/`V18` migrations.
