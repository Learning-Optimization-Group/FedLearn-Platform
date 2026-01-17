package com.federated.fl_platform_api.controller;


import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.service.ProjectService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
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

    @PostMapping("/{projectId}")
    public ResponseEntity<Void> reportRoundResult(@PathVariable UUID projectId, @RequestBody RoundResultDto resultDto) {
        Project project = projectRepository.findById(projectId).orElse(null);
        if (project == null) {
            return ResponseEntity.notFound().build();

        }
        RoundResult result = new RoundResult();
        result.setProject(project);
        result.setServerRound(resultDto.getServerRound());
        result.setLoss(resultDto.getLoss());
        result.setAccuracy(resultDto.getAccuracy());

        roundResultRepository.save(result);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{projectId}/finished")
    public ResponseEntity<Void> markProjectAsFinished(@PathVariable UUID projectId) {
        projectService.markProjectAsCompleted(projectId);
        return ResponseEntity.ok().build();
    }



}
