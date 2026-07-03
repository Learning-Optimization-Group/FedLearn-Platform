package com.federated.fl_platform_api.controller;


import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/internal/results")
public class ResultsController {

    @Autowired
    private RoundResultRepository roundResultRepository;

    @Autowired
    private ProjectRepository projectRepository;

    @Autowired
    private ProjectService projectService;

    @Autowired
    private com.federated.fl_platform_api.service.WebSocketService webSocketService;

    @org.springframework.beans.factory.annotation.Value("${feature.round-result-reporting.enabled:true}")
    private boolean roundResultsEnabled;

    @PostMapping("/{projectId}")
    public ResponseEntity<Void> reportRoundResult(@PathVariable @NonNull UUID projectId,
                                                  @Valid @RequestBody RoundResultDto resultDto) {
        if (!roundResultsEnabled) {
            return ResponseEntity.ok().build();
        }
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));

        RoundResult result = new RoundResult();
        result.setProject(project);
        result.setServerRound(resultDto.getServerRound());
        result.setLoss(resultDto.getLoss());
        result.setAccuracy(resultDto.getAccuracy());
        result.setGpuUtilization(resultDto.getGpuUtilization());

        RoundResult saved = roundResultRepository.save(result);

        // Broadcast a DTO, not the JPA entity. The entity's lazy Hibernate proxies (project -> user)
        // are not JSON-serializable by the STOMP message converter (no Hibernate Jackson module is
        // registered), which previously threw MessageConversionException and 500'd this callback
        // AFTER the row had already been saved — the client never got the live update.
        RoundResultDto out = new RoundResultDto();
        out.setId(saved.getId());
        out.setServerRound(saved.getServerRound());
        out.setLoss(saved.getLoss());
        out.setAccuracy(saved.getAccuracy());
        out.setGpuUtilization(saved.getGpuUtilization());
        webSocketService.sendResultUpdate(projectId, out);

        return ResponseEntity.ok().build();
    }

    @PostMapping("/{projectId}/finished")
    public ResponseEntity<Void> markProjectAsFinished(@PathVariable @NonNull UUID projectId) {
        projectService.markProjectAsCompleted(projectId);
        return ResponseEntity.ok().build();
    }
}
