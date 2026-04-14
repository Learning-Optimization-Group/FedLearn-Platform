package com.federated.fl_platform_api.controller;


import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.service.ProjectService;
import org.springframework.lang.NonNull;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    @Autowired
    private ProjectService projectService;

    @PostMapping
    public ResponseEntity<ProjectResponseDto> createProject(@Valid @RequestBody CreateProjectRequest request) {
        try {

            ProjectResponseDto newProject = projectService.createProject(request);
            return ResponseEntity.ok(newProject);
        } catch (Exception e) {
            System.err.println("=== ERROR IN CREATE PROJECT ===");
            e.printStackTrace();
            return ResponseEntity.badRequest().build();
        }
    }

    @GetMapping
    public ResponseEntity<List<ProjectResponseDto>> getAllProjects() {
        try {
            List<ProjectResponseDto> projects = projectService.getProjectsForCurrentUser();
            return ResponseEntity.ok(projects);
        } catch (Exception e) {
            // Handle cases where the user might not be found or other errors
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @PostMapping("/{projectId}/start")
    public ResponseEntity<ProjectResponseDto> startProjectServer(@PathVariable @NonNull UUID projectId,
                                                                 @RequestBody StartProject request,
                                                                 HttpServletRequest httpRequest) {

        System.out.println("=== REQUEST DEBUG ===");
        System.out.println("Content-Type: " + httpRequest.getContentType());
        System.out.println("Request object: " + request);
        System.out.println("Strategy: " + request.getStrategy());
        System.out.println("NumRounds: " + request.getNumRounds());
        System.out.println("MinClients:"+request.getMinClients());
        System.out.println("NumRounds is null: " + (request.getNumRounds() == null));
        System.out.println("===================");
        try{

            ProjectResponseDto startedProject = projectService.startServerForProject(projectId, request);
            return ResponseEntity.ok(startedProject);
        }catch (RuntimeException e){
            return ResponseEntity.badRequest().build();
        }catch (Exception e){
            return ResponseEntity.internalServerError().build();
        }

    }

    @PostMapping("/{projectId}/stop")
    public ResponseEntity<ProjectResponseDto> stopProjectServer(@PathVariable @NonNull UUID projectId) {
        try {
            ProjectResponseDto stoppedProject = projectService.stopServerForProject(projectId);
            return ResponseEntity.ok(stoppedProject);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }



    @GetMapping("/{projectId}/results")
    public ResponseEntity<List<RoundResultDto>> getProjectResults(@PathVariable @NonNull UUID projectId) {
        List<RoundResultDto> resultDtos = projectService.getResultsForProject(projectId);
        return ResponseEntity.ok(resultDtos);
    }

    @GetMapping("/{projectId}/logs")
    public ResponseEntity<List<com.federated.fl_platform_api.dto.ServerLogDto>> getProjectLogs(@PathVariable @NonNull UUID projectId) {
        List<com.federated.fl_platform_api.dto.ServerLogDto> logs = projectService.getLogsForProject(projectId);
        return ResponseEntity.ok(logs);
    }

    /**
     * Export logs as a downloadable plain-text file for external review.
     */
    @GetMapping("/{projectId}/logs/export")
    public ResponseEntity<String> exportProjectLogs(@PathVariable @NonNull UUID projectId) {
        List<com.federated.fl_platform_api.dto.ServerLogDto> logs = projectService.getLogsForProject(projectId);
        StringBuilder sb = new StringBuilder();
        sb.append("=== FedLearn Training Logs ===\n");
        sb.append("Project ID: ").append(projectId).append("\n");
        sb.append("Exported at: ").append(java.time.Instant.now()).append("\n");
        sb.append("Total log entries: ").append(logs.size()).append("\n");
        sb.append("=".repeat(60)).append("\n\n");

        for (com.federated.fl_platform_api.dto.ServerLogDto log : logs) {
            sb.append(String.format("[%s] [%s] %s\n", log.getTimestamp(), log.getLevel(), log.getMessage()));
            if (log.getStackTrace() != null && !log.getStackTrace().isEmpty()) {
                sb.append("  STACKTRACE: ").append(log.getStackTrace()).append("\n");
            }
        }

        String filename = "fedlearn-logs-" + projectId + ".txt";
        return ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=\"" + filename + "\"")
                .header("Content-Type", "text/plain; charset=UTF-8")
                .body(sb.toString());
    }

    @PostMapping("/{projectId}/delete")
    public ResponseEntity<String> deleteProject(@PathVariable @NonNull UUID projectId) {
        projectService.deleteProject(projectId);
        String msg = projectId+"Project deleted successfully";
        return ResponseEntity.ok(msg);
    }
}
