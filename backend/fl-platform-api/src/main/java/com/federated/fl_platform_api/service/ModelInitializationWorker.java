package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ProjectStatusUpdateDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectInitStatus;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.util.UUID;

/**
 * BA-1: runs a project's one-time model initialisation OFF the request thread.
 *
 * <p>{@code createProject} persists the project shell as {@link ProjectInitStatus#INITIALIZING} and
 * returns 201 immediately; it then dispatches {@link #initialize} on the bounded
 * {@code modelInitExecutor} (see {@link com.federated.fl_platform_api.config.AsyncConfig}). Because
 * the actual init spawns an unbounded Python process, doing it inside the request's transaction pinned
 * a DB connection and a Tomcat thread for as long as the process ran. Here it runs on a worker thread
 * — the subprocess is itself timeout-bounded and force-killed by {@link ModelInitializer} — and the
 * outcome is written back as a short, independent unit of work.</p>
 *
 * <p>Runs after the creating transaction has committed, so {@link #initialize} sees the persisted row.
 * On success the project transitions to {@link ProjectInitStatus#DONE} (status then defers to the
 * active run); on any failure it transitions to {@link ProjectInitStatus#FAILED} — the row persists in
 * a failed state the owner can see and delete/retry, rather than being rolled back. Either way the
 * derived project status is broadcast so a polling client learns the outcome promptly.</p>
 */
@Component
public class ModelInitializationWorker {

    private static final Logger log = LoggerFactory.getLogger(ModelInitializationWorker.class);

    private final ModelInitializer modelInitializer;
    private final ProjectRepository projectRepository;
    private final WebSocketService webSocketService;
    private final ProjectStatusService projectStatusService;

    public ModelInitializationWorker(ModelInitializer modelInitializer,
                                     ProjectRepository projectRepository,
                                     WebSocketService webSocketService,
                                     ProjectStatusService projectStatusService) {
        this.modelInitializer = modelInitializer;
        this.projectRepository = projectRepository;
        this.webSocketService = webSocketService;
        this.projectStatusService = projectStatusService;
    }

    /**
     * Materialise the initial weights file, then record the terminal init phase. Never propagates:
     * a failure is captured as {@link ProjectInitStatus#FAILED} on the project, not thrown (there is
     * no request left to receive it).
     */
    @Async("modelInitExecutor")
    public void initialize(UUID projectId, String modelType, String modelName, String optimizer,
                           String outputPath, int pretrainEpochs, String taskType) {
        try {
            modelInitializer.initializeModelFile(
                    modelType, modelName, optimizer, outputPath, pretrainEpochs, taskType);
            complete(projectId, ProjectInitStatus.DONE);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("Model init interrupted for project {} — marking FAILED", projectId, e);
            complete(projectId, ProjectInitStatus.FAILED);
        } catch (Exception e) {
            log.error("Model init failed for project {} — marking FAILED", projectId, e);
            complete(projectId, ProjectInitStatus.FAILED);
        }
    }

    /**
     * Persist the terminal init phase and broadcast the resulting derived status. Each repository call
     * is its own transaction (the creating transaction is long committed by now); no ambient
     * transaction is required. A project deleted mid-init is a no-op.
     */
    void complete(UUID projectId, ProjectInitStatus phase) {
        Project project = projectRepository.findById(projectId).orElse(null);
        if (project == null) {
            log.warn("Project {} no longer exists; skipping init transition to {}", projectId, phase);
            return;
        }
        project.setInitStatus(phase);
        projectRepository.save(project);

        ProjectStatus derived = projectStatusService.currentStatus(project);
        webSocketService.sendStatusUpdate(
                new ProjectStatusUpdateDto(projectId, derived.name(), project.getServerPort()));
        log.info("Project {} init {} -> project status {}", projectId, phase, derived);
    }
}
