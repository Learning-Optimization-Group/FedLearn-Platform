package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import java.util.Objects;
import java.util.UUID;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class RoundResultIngestService {
    public record Recorded(RoundResult result, boolean created) {}

    private final ProjectRepository projects;
    private final RunRepository runs;
    private final RoundResultRepository results;

    public RoundResultIngestService(ProjectRepository projects, RunRepository runs, RoundResultRepository results) {
        this.projects = projects;
        this.runs = runs;
        this.results = results;
    }

    @Transactional
    public Recorded record(UUID projectId, InternalCallbackContext context, RoundResultDto dto) {
        if (context == null || !projectId.equals(context.projectId()) || context.runId() == null) {
            throw new AccessDeniedException("Callback identity does not match the project");
        }
        // Project first, then run: active-run changes and token/status writes cannot pass this
        // transaction's authorization check and commit before the result is recorded.
        var project = projects.lockById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        var run = runs.lockById(context.runId())
                .orElseThrow(() -> new IllegalStateException("Callback run no longer exists"));
        if (!projectId.equals(run.getProjectId()) || context.tokenHash() == null
                || !context.tokenHash().equals(run.getInternalTokenHash())) {
            throw new AccessDeniedException("Callback authorization has been retired");
        }
        if (!run.getId().equals(project.getActiveRunId())
                || (run.getStatus() != RunStatus.RUNNING && run.getStatus() != RunStatus.STARTING)) {
            throw new IllegalStateException("Callback run is not active");
        }
        if (dto.getServerRound() == null || dto.getServerRound() < 1 || dto.getServerRound() > run.getNumRounds()) {
            throw new IllegalArgumentException("Round must be within the run's round budget");
        }
        requireFinite(dto.getLoss());
        requireFinite(dto.getAccuracy());
        requireFinite(dto.getGpuUtilization());
        var existing = results.findByRunIdAndServerRound(run.getId(), dto.getServerRound());
        if (existing.isPresent()) {
            var previous = existing.get();
            if (!Objects.equals(previous.getLoss(), dto.getLoss())
                    || !Objects.equals(previous.getAccuracy(), dto.getAccuracy())
                    || !Objects.equals(previous.getGpuUtilization(), dto.getGpuUtilization())) {
                throw new IllegalStateException("Round result conflicts with the recorded metrics");
            }
            return new Recorded(previous, false);
        }
        var result = new RoundResult();
        result.setProject(project);
        result.setRunId(run.getId());
        result.setServerRound(dto.getServerRound());
        result.setLoss(dto.getLoss());
        result.setAccuracy(dto.getAccuracy());
        result.setGpuUtilization(dto.getGpuUtilization());
        return new Recorded(results.saveAndFlush(result), true);
    }

    private static void requireFinite(Double value) {
        if (value != null && !Double.isFinite(value)) {
            throw new IllegalArgumentException("Round metrics must be finite or null");
        }
    }
}
