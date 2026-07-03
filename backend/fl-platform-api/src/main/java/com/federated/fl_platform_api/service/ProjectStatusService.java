package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectInitStatus;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.repository.RunRepository;
import org.springframework.stereotype.Service;

import java.util.UUID;

/**
 * Derives a project's {@link ProjectStatus} from its active {@link com.federated.fl_platform_api.model.Run}
 * (BA-4). This is the single read-time source of truth for project status: DTO-building callers ask
 * here instead of trusting the denormalized, drift-prone {@code projects.status} string column (which,
 * for example, stayed {@code RUNNING} after a run failed because nothing wrote the failure back).
 */
@Service
public class ProjectStatusService {

    private final RunRepository runRepository;

    public ProjectStatusService(RunRepository runRepository) {
        this.runRepository = runRepository;
    }

    /**
     * The project's current status. The one-time model-init phase (BA-1) takes precedence: a project
     * that is still initialising — or whose init failed — has no run yet, so run-derivation would
     * misread it as the idle {@code CREATED}. Only once init is {@code DONE} do we defer to the
     * active run (no active run -> {@code CREATED}).
     */
    public ProjectStatus currentStatus(Project project) {
        ProjectInitStatus init = project.getInitStatus();
        if (init == ProjectInitStatus.INITIALIZING) {
            return ProjectStatus.INITIALIZING;
        }
        if (init == ProjectInitStatus.FAILED) {
            return ProjectStatus.FAILED;
        }
        UUID activeRunId = project.getActiveRunId();
        if (activeRunId == null) {
            return ProjectStatus.CREATED;
        }
        return runRepository.findById(activeRunId)
                .map(run -> ProjectStatus.fromActiveRunStatus(run.getStatus()))
                .orElse(ProjectStatus.CREATED);   // active_run_id points at a since-deleted run
    }
}
