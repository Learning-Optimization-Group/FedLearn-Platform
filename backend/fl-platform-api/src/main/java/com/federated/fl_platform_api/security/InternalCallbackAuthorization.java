package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import jakarta.persistence.EntityManager;
import jakarta.persistence.LockModeType;
import jakarta.persistence.PersistenceContext;
import java.util.UUID;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

/** Hold callback authorization stable for the caller's entire write transaction. */
@Service
public class InternalCallbackAuthorization {
    public record AuthorizedRun(Project project, Run run) {}

    private final ProjectRepository projects;
    private final RunRepository runs;
    @PersistenceContext private EntityManager entityManager;

    public InternalCallbackAuthorization(ProjectRepository projects, RunRepository runs) {
        this.projects = projects;
        this.runs = runs;
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public AuthorizedRun requireActiveRun(UUID projectId, InternalCallbackContext context) {
        if (context == null || !projectId.equals(context.projectId()) || context.runId() == null) {
            throw new AccessDeniedException("Callback identity does not match the project");
        }
        // Always project first, then run. Keep both locks until the caller commits its mutation.
        var project = projects.lockById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        entityManager.refresh(project, LockModeType.PESSIMISTIC_WRITE);
        var run = runs.lockById(context.runId())
                .orElseThrow(() -> new IllegalStateException("Callback run no longer exists"));
        entityManager.refresh(run, LockModeType.PESSIMISTIC_WRITE);
        if (!projectId.equals(run.getProjectId()) || context.tokenHash() == null
                || !context.tokenHash().equals(run.getInternalTokenHash())) {
            throw new AccessDeniedException("Callback authorization has been retired");
        }
        if (!run.getId().equals(project.getActiveRunId())
                || (run.getStatus() != RunStatus.RUNNING && run.getStatus() != RunStatus.STARTING)) {
            throw new IllegalStateException("Callback run is not active");
        }
        return new AuthorizedRun(project, run);
    }
}
