# 03 - Project Management Lifecycle

This document explains how a Federated Learning Project is created, initialized, and managed within the Spring Boot backend.

> ⚠️ **Branch reality.** The project CRUD, model initialization, and FL-server start/stop flows on this page are **current**. The **org-scoping and audit layers** described here — `projects.org_id` (the V5 migration), `authz.requireOrgScope(...)`, `OrgScope`, `isPlatformAdmin()`, and the `@Auditable` annotations — are **designed on a separate identity-foundations branch and are _not present_ here.** On this branch a project is owned by a `User` with no org pinning, and authorization is the coarse `users.role IN (USER, ADMIN)` model. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).

## 1. Project Creation Flow

The lifecycle begins when a user submits a configuration via the React dashboard.

### `ProjectController`
The `POST /api/projects` endpoint accepts a `CreateProjectRequest` payload containing:
- `name` (e.g., "Pneumonia Detection")
- `modelType` (e.g., "CNN")
- `modelName` (e.g., "net")
- `optimizer` (e.g., "Adam")
- `pretrainEpochs` (int)

### `ProjectService.createProject()`
This service method is annotated with `@Transactional`. If any step fails, the entire database transaction rolls back, preventing "orphaned" projects.

1. **Persist the Shell:** An empty `Project` entity is saved to the database to generate a UUID.
2. **Pin to an Organization:** Since the V5 migration made `projects.org_id` **NOT NULL**, the new project is pinned to the owner's first org membership, falling back to the single `DEFAULT_ORG_ID` for membership-less users. Its `visibility` defaults to `PRIVATE`.
3. **Determine File Path:** A target file path is generated (e.g., `models/<uuid>.npz`).
4. **Model Initialization:** `ModelInitializer.initializeModelFile()` is invoked. It executes a local Python script (`run_init_model.sh`) that constructs the initial model architecture (PyTorch) based on the `modelType` and saves it to the `.npz` file.
5. **Finalize DB Entry:** The `Project` entity is updated with the absolute path to the `.npz` file and its status is set to `CREATED`.

`createProject` is also annotated `@Auditable(action = PROJECT_CREATED)`, so a successful creation writes an `audit_events` row in the same transaction (see [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)).

## 2. Project Ownership, Membership, and Org Isolation

A project is no longer just "owned by one user." It belongs to an
**Organization** (`org_id` NOT NULL), has a **visibility** (`PRIVATE`/`PUBLIC`),
and can have **project memberships** (`OWNER`/`MEMBER`/`CLIENT`) and **access
requests**. Authorization is centralised in `AuthorizationService` and applies two
layers in sequence.

### Layer 1 — Org-scope (multi-tenant isolation)

Before any ownership check, mutating paths call `authz.requireOrgScope(project.getOrgId())`,
which throws **403** if the project's org is outside the caller's request-scoped
`OrgScope`. Pure-read paths use the boolean `authz.isInOrgScope(...)` (or
`orgScope.allows(...)`) and instead return **404**, so a caller can't even learn
that a cross-tenant project exists.

### Layer 2 — Ownership / membership

```java
public void requireOwnerOrAdmin(Project project) {
    if (isPlatformAdmin() || isOwner(project)) return;
    throw new AccessDeniedException("You do not have access to this project");
}
```

`isPlatformAdmin()` checks the `ROLE_PLATFORM_ADMIN` authority (platform admins
bypass both layers). The service also offers `requireOwnerOrMemberOrAdmin` and
`requireParticipant` for read endpoints that members/clients may see. An
unauthorized `start`/`stop`/`delete` throws `AccessDeniedException` → 403.

### Visibility, memberships & access requests

- **Visibility** (`PATCH /api/projects/{id}`) flips a project between `PRIVATE`
  and `PUBLIC`; participants are notified on change.
- **Memberships** (`/api/projects/{id}/memberships`) — owners/admins add/remove
  any role; a `MEMBER` may add/remove only `CLIENT`. Member add/remove are
  `@Auditable`.
- **Access requests** (`/api/projects/{id}/access-requests`) — joining a PUBLIC
  project creates a `CLIENT` membership immediately; requesting a PRIVATE project
  files a `PENDING` request the owner decides.

See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)
for the full membership/access-request workflow and the org-scope mechanism.

### List views are org-scoped

`getProjectsForCurrentUser()` no longer returns a flat owned-or-joined list:
unless the caller is an unrestricted platform admin, it runs the org-scoped
`ProjectRepository.findOwnedOrMemberOfInOrgs(userId, orgScope.visibleOrgIds())`,
so the "My Projects" dashboard only shows projects from the caller's visible
orgs. The discover feed (`GET /api/projects/discover`) is likewise constrained via
`findDiscoverableInOrgs`.

## 3. Starting the Training Server

The `POST /api/projects/{projectId}/start` endpoint kicks off the machine learning phase.

1. The user specifies the training `strategy` (e.g., `FedAvg`), the `minClients` required, and the `numRounds`.
2. The `ProjectService` enforces org-scope then ownership (`requireOrgScope` → `requireOwnerOrAdmin`) and ensures the server isn't already running. `startServerForProject` is `@Auditable(action = RUN_STARTED)`; `stopServerForProject` is `@Auditable(action = RUN_STOPPED)`.
3. It calls `FlowerServerManager.startServerForProject(...)`. (See [04 - Federated Orchestration](04_federated_orchestration.md) for full details on this component).
4. The backend updates the project's status to `RUNNING` and saves the network `serverPort` where the FL Server is listening.
5. A real-time `ProjectStatusUpdateDto` is fired over WebSockets to instantly update the React dashboard UI.

## 4. Completion and Cleanup

### Marking as Completed
When the FL Server successfully finishes all its federated rounds, it sends a final REST call to the internal API endpoint.
`ProjectService.markProjectAsCompleted()` clears the active `serverPort` and sets the status to `COMPLETED`.

### Deletion
When a user deletes a project (`DELETE /api/projects/{projectId}`):
1. The service enforces org-scope (`requireOrgScope`) then ownership (`requireOwnerOrAdmin`). `deleteProject` is `@Auditable(action = PROJECT_DELETED)`.
2. It makes a best-effort attempt to terminate any actively running FL Server processes via `FlowerServerManager.stopServerForProject()`. This prevents ghost processes or orphaned AWS Fargate tasks from lingering and consuming resources.
3. The database row is deleted. Cascade rules automatically delete the associated `ServerLog`, `RoundResult`, `ProjectMembership`, and `ProjectAccessRequest` entries.
