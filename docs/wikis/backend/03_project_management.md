# 03 - Project Management Lifecycle

This document explains how a Federated Learning Project is created, initialized, and managed within the Spring Boot backend.

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
2. **Determine File Path:** A target file path is generated (e.g., `models/<uuid>.npz`).
3. **Model Initialization:** `ModelInitializer.initializeModelFile()` is invoked. It executes a local Python script (`run_init_model.sh`) that constructs the initial model architecture (PyTorch) based on the `modelType` and saves it to the `.npz` file.
4. **Finalize DB Entry:** The `Project` entity is updated with the absolute path to the `.npz` file and its status is set to `CREATED`.

## 2. Project Ownership and Authorization

Users are only permitted to interact with projects they have created. 

The `ProjectService` enforces this via the `requireOwnerOrAdmin` helper:

```java
private void requireOwnerOrAdmin(Project project) {
    Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
    User caller = currentUser();
    User owner = project.getUser();
    
    if (!isAdmin(authentication) && (owner == null || !owner.getId().equals(caller.getId()))) {
        throw new AccessDeniedException("You do not have access to this project");
    }
}
```

If an unauthorized user attempts to start, stop, or delete a project, this check immediately throws an `AccessDeniedException` which is mapped to a 403 Forbidden response.

## 3. Starting the Training Server

The `POST /api/projects/{projectId}/start` endpoint kicks off the machine learning phase.

1. The user specifies the training `strategy` (e.g., `FedAvg`), the `minClients` required, and the `numRounds`.
2. The `ProjectService` validates ownership and ensures the server isn't already running.
3. It calls `FlowerServerManager.startServerForProject(...)`. (See [04 - Federated Orchestration](04_federated_orchestration.md) for full details on this component).
4. The backend updates the project's status to `RUNNING` and saves the network `serverPort` where the FL Server is listening.
5. A real-time `ProjectStatusUpdateDto` is fired over WebSockets to instantly update the React dashboard UI.

## 4. Completion and Cleanup

### Marking as Completed
When the FL Server successfully finishes all its federated rounds, it sends a final REST call to the internal API endpoint.
`ProjectService.markProjectAsCompleted()` clears the active `serverPort` and sets the status to `COMPLETED`.

### Deletion
When a user deletes a project (`DELETE /api/projects/{projectId}`):
1. The service checks ownership.
2. It makes a best-effort attempt to terminate any actively running FL Server processes via `FlowerServerManager.stopServerForProject()`. This prevents ghost processes or orphaned AWS Fargate tasks from lingering and consuming resources.
3. The database row is deleted. cascade rules automatically delete the associated `ServerLog` and `RoundResult` entries.
