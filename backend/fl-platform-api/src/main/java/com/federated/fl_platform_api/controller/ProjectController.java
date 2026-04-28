package com.federated.fl_platform_api.controller;


import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.dto.ServerLogDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    private static final Logger log = LoggerFactory.getLogger(ProjectController.class);

    @Autowired
    private ProjectService projectService;

    @PostMapping
    public ResponseEntity<ProjectResponseDto> createProject(@Valid @RequestBody CreateProjectRequest request)
            throws Exception {
        ProjectResponseDto newProject = projectService.createProject(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(newProject);
    }

    @GetMapping
    public ResponseEntity<List<ProjectResponseDto>> getAllProjects() {
        return ResponseEntity.ok(projectService.getProjectsForCurrentUser());
    }

    @PostMapping("/{projectId}/start")
    public ResponseEntity<ProjectResponseDto> startProjectServer(@PathVariable @NonNull UUID projectId,
                                                                 @Valid @RequestBody StartProject request)
            throws Exception {
        log.debug("Start request for project {} (strategy={}, rounds={}, minClients={})",
                projectId, request.getStrategy(), request.getNumRounds(), request.getMinClients());
        return ResponseEntity.ok(projectService.startServerForProject(projectId, request));
    }

    @PostMapping("/{projectId}/stop")
    public ResponseEntity<ProjectResponseDto> stopProjectServer(@PathVariable @NonNull UUID projectId) {
        return ResponseEntity.ok(projectService.stopServerForProject(projectId));
    }

    @GetMapping("/{projectId}/results")
    public ResponseEntity<List<RoundResultDto>> getProjectResults(@PathVariable @NonNull UUID projectId) {
        return ResponseEntity.ok(projectService.getResultsForProject(projectId));
    }

    /**
     * Returns one page of training logs for a project. Caller controls
     * {@code ?page=N&size=M} (Spring binds {@link Pageable} automatically).
     * The service clamps {@code size} to a hard upper bound — long-running
     * projects can produce millions of rows and we won't ship them all in
     * a single response.
     */
    @GetMapping("/{projectId}/logs")
    public ResponseEntity<List<ServerLogDto>> getProjectLogs(
            @PathVariable @NonNull UUID projectId,
            @PageableDefault(size = 200) Pageable pageable) {
        return ResponseEntity.ok(projectService.getLogsForProject(projectId, pageable));
    }

    /**
     * Export logs as a downloadable plain-text file for external review.
     * Capped server-side at {@code ProjectService.MAX_LOGS_EXPORT_SIZE}; for
     * larger projects we'll later route to S3-archived dumps.
     */
    @GetMapping("/{projectId}/logs/export")
    public ResponseEntity<String> exportProjectLogs(@PathVariable @NonNull UUID projectId) {
        List<ServerLogDto> logs = projectService.getLogsForExport(projectId);
        StringBuilder sb = new StringBuilder();
        sb.append("=== FedLearn Training Logs ===\n");
        sb.append("Project ID: ").append(projectId).append("\n");
        sb.append("Exported at: ").append(java.time.Instant.now()).append("\n");
        sb.append("Total log entries: ").append(logs.size()).append("\n");
        sb.append("=".repeat(60)).append("\n\n");

        for (ServerLogDto entry : logs) {
            sb.append(String.format("[%s] [%s] %s%n",
                    entry.getTimestamp(), entry.getLevel(), entry.getMessage()));
            if (entry.getStackTrace() != null && !entry.getStackTrace().isEmpty()) {
                sb.append("  STACKTRACE: ").append(entry.getStackTrace()).append("\n");
            }
        }

        String filename = "fedlearn-logs-" + projectId + ".txt";
        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=\"" + filename + "\"")
                .header("Content-Type", "text/plain; charset=UTF-8")
                .body(sb.toString());
    }

    @DeleteMapping("/{projectId}")
    public ResponseEntity<Map<String, Object>> deleteProject(@PathVariable @NonNull UUID projectId) {
        projectService.deleteProject(projectId);
        return ResponseEntity.ok(Map.of(
                "projectId", projectId,
                "message", "Project deleted successfully"
        ));
    }

    /**
     * Legacy POST endpoint kept temporarily for clients that have not migrated
     * to the canonical DELETE /{projectId} route. Remove once the desktop and
     * web clients no longer call it.
     */
    @PostMapping("/{projectId}/delete")
    @Deprecated
    public ResponseEntity<Map<String, Object>> deleteProjectLegacy(@PathVariable @NonNull UUID projectId) {
        return deleteProject(projectId);
    }
}
