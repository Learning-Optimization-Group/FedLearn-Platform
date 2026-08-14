# Backend Architecture Wiki

Welcome to the internal documentation for the **FedLearn-Platform Backend**.

This section of the wiki covers the Spring Boot 3 API, the database interactions, security, WebSockets, and how the federated learning python processes are orchestrated as local OS processes on the host VM.

> ⚠️ **Orchestration reality (OP-14).** The **only** supported deployed architecture is the **hardened single-VM**: the backend spawns each FL aggregation server as a local Python process (via the `FlServerProcessRunner` seam, DA-8) on a port in the `50000-50010` range. Managed cloud task orchestration on AWS ECS/Fargate *was* implemented once (`1239dda`) and later **removed** along with the AWS SDK (`9124b62`) — no ECS code and no AWS SDK dependency remain in the repo. The one surviving knob, `ecs.cluster-name` (blank default), exists **solely to be rejected**: `FlOrchestrationModeValidator` throws at boot in **every** profile if it is set to a non-blank value. The `production` profile is the single-VM profile, not an ECS profile. Managed-task orchestration is deferred to **OP-12**.

> ✅ **Branch reality.** The backend runs on **PostgreSQL** for every profile (H2 has been retired) — `dev`/`ec2demo` against a local Postgres (`backend/fl-platform-api/docker-compose.yml` → `docker compose up -d`) and `test` against Testcontainers Postgres. The highest committed Flyway migration is **`V23`** (`V23__training_arm_ova_lp.sql`). The full **identity / multi-tenancy / audit subsystem documented in page 06 IS present on this branch**: the three-layer role model (`PlatformRole` platform role, `OrgRole` org role, `MembershipRole` project membership), organization-scoped isolation (`OrgScopeFilter`), `AuthorizationService`, the `@Auditable`/`AuditEvent` trail, and the `email/` + `bootstrap/` plumbing — backed by the `V4`–`V7` identity migrations. (The single coarse `users.role IN (USER, ADMIN)` column from `V2` was the *original* model; it has since been superseded by the layered `PlatformRole`.)
>
> The four migrations past `V19`, none of which older wiki text mentions:
>
> | Migration | Adds |
> |---|---|
> | **`V20__project_derivation.sql`** | `projects.init_from_pretrained` (NOT NULL, default `FALSE`), `base_ref_sha256`, `derivation_spec` — the opt-in, nullable record of a project deriving from a frozen/pretrained base instead of training an architecture from scratch. `chk_projects_base_ref_sha256_hex` pins the ref to lowercase-hex sha256 (or NULL). |
> | **`V21__base_ref_unique_index.sql`** | The partial unique index `uq_base_ref_org_model ON model_artifacts (org_id, base_model_ref) WHERE kind = 'BASE_REF'` — makes "one `BASE_REF` per (org, base model)" a database invariant and backs the `ON CONFLICT DO NOTHING` insert-if-absent that replaced a racy read-then-insert. See [07](07_artifact_registry.md). |
> | **`V22__project_training_arm.sql`** | `projects.training_arm VARCHAR(32) NOT NULL DEFAULT 'FULL'` plus `chk_projects_training_arm`. The training arm becomes a stored, queryable project property instead of something the Python client inferred from the recipe key. |
> | **`V23__training_arm_ova_lp.sql`** | Widens `chk_projects_training_arm` to `('FULL','FROZEN_HEAD','OVA_LP')` for the OvA-LP arm. |
>
> Note that `ls` sorts these lexicographically, so `V5`–`V9` list *after* `V21` — the last line of an `ls` is not the highest version.

## Documentation Index

1. **[Architecture & Core Concepts](01_architecture_overview.md)**
   Provides a high-level overview of the backend's directory structure, the technology stack, and the primary domain models (Projects, Results, Logs).

2. **[Security & Authentication](02_security_and_auth.md)**
   Explains how stateless JWT validation works via filters, how WebSocket connections are secured during the HTTP upgrade handshake, and the internal API key mechanism used by FL servers.

3. **[Project Management Lifecycle](03_project_management.md)**
   Details the `ProjectService` and `ProjectController` logic: the full `ProjectController` route surface, how a project is persisted and its model asynchronously initialized (BA-1), how the **training arm** (`FULL` / `FROZEN_HEAD` / `OVA_LP`) is chosen at creation and enforced from the DTO down to the CHECK constraint, how a run is started, and why direct deletion is admin-only.

4. **[Federated Orchestration (FlServerManager)](04_federated_orchestration.md)**
   The most complex component. Documents how the Java API dynamically provisions the Python FL aggregation servers as local OS processes — port reservation in the `50000-50010` range, the `FlServerProcessRunner` seam that shells out to the `fl-runtime/` scripts, and the `ProcessHandle` tracking that makes `/stop` work. Also covers why the removed ECS/Fargate path is now fail-closed at boot (OP-14) and deferred to OP-12.

5. **[WebSocket Log Streaming](05_websocket_logs_streaming.md)**
   Explains the real-time observability pipeline. Shows how the backend captures standard output from the Python FL servers, routes it via STOMP topics to the React frontend, and persists it for export.

6. **[Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)**
   Documents the identity subsystem — **present on this branch**: the three-layer role model (platform / organization / project), organization-scoped multi-tenant isolation (`OrgScopeFilter`), the `@Auditable` audit trail, the email + first-run bootstrap plumbing, the `V4`–`V7` identity migrations, and the membership/admin/access-request REST endpoints.

7. **[Content-Addressed Model Artifact Registry](07_artifact_registry.md)**
   Documents the registry that replaced the single overwritable `.npz` at `projects.model_path`: the `artifact_blobs` / `model_artifacts` / `artifact_lineage` data model, the `ArtifactBlobStore` write-once content store, `RegistryModelResolver`'s registry-first read path for inference and FL-server warm-start, the artifact/lineage/marketplace REST surface, and the `V12`/`V18` migrations.
