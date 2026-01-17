package com.federated.fl_platform_api.controller;


import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    @Autowired
    private ProjectService projectService;

    @Autowired
    private RoundResultRepository roundResultRepository;

    @PostMapping
    public ResponseEntity<ProjectResponseDto> createProject(@Valid @RequestBody CreateProjectRequest request) {
        try {

            ProjectResponseDto newProject = projectService.createProject(request);
            return ResponseEntity.ok(newProject);
        } catch (Exception e) {
            // It's better to have a proper exception handler, but for now this is ok
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
    public ResponseEntity<ProjectResponseDto> startProjectServer(@PathVariable UUID projectId,
                                                                 @RequestBody StartProject request,
                                                                 HttpServletRequest httpRequest) {

        System.out.println("=== REQUEST DEBUG ===");
        System.out.println("Content-Type: " + httpRequest.getContentType());
        System.out.println("Request object: " + request);
        System.out.println("Strategy: " + request.getStrategy());
        System.out.println("NumRounds: " + request.getNumRounds());
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
    public ResponseEntity<ProjectResponseDto> stopProjectServer(@PathVariable UUID projectId) {
        try {
            ProjectResponseDto stoppedProject = projectService.stopServerForProject(projectId);
            return ResponseEntity.ok(stoppedProject);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }



    @GetMapping("/{projectId}/results")
    public ResponseEntity<List<RoundResultDto>> getProjectResults(@PathVariable UUID projectId) {
        List<RoundResultDto> resultDtos = projectService.getResultsForProject(projectId);
        return ResponseEntity.ok(resultDtos);
    }

    @PostMapping("/{projectId}/delete")
    public ResponseEntity<String> deleteProject(@PathVariable UUID projectId) {
        projectService.deleteProject(projectId);
        String msg = projectId+"Project deleted successfully";
        return ResponseEntity.ok(msg);
    }
}
