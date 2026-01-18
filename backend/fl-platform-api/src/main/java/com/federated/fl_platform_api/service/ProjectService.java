package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.*;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.flower.FlowerServerManager;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ProjectService {

    @Autowired
    private ProjectRepository projectRepository;
    @Autowired
    private FlowerServerManager flowerServerManager;
    @Autowired
    private ModelInitializer modelInitializer;
    @Autowired
    private UserRepository userRepository;
    @Autowired
    private RoundResultRepository roundResultRepository;
    @Autowired
    private WebSocketService webSocketService;


    private RoundResultDto convertToDto(RoundResult result) {
        RoundResultDto dto = new RoundResultDto();
        dto.setId(result.getId());
        dto.setServerRound(result.getServerRound());
        dto.setLoss(result.getLoss());
        dto.setAccuracy(result.getAccuracy());
        dto.setGpuUtilization(result.getGpuUtilization());
        return dto;
    }

    private ProjectResponseDto convertToDto(Project project) {
        ProjectResponseDto dto = new ProjectResponseDto();
        dto.setId(project.getId());
        dto.setName(project.getName());
        dto.setModelType(project.getModelType());
        dto.setModelName(project.getModelName());
        dto.setServerPort(project.getServerPort());
        dto.setOptimizer(project.getOptimizer());
        dto.setStatus(project.getStatus());

        return dto;
    }

    public ProjectResponseDto createProject(CreateProjectRequest request) throws IOException, InterruptedException {
        System.out.println("\n\n==================== NEW PROJECT REQUEST RECEIVED ====================");
        System.out.println("=> Project Name: " + request.getName());
        System.out.println("=> Model Type: " + request.getModelType());

        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();
        User currentUser = userRepository.findByUsername(currentUsername)
                .orElseThrow(() -> new UsernameNotFoundException("Authenticated user not found in database"));

        // --- Step 1: Initial Database Entry ---
        System.out.println("\n[1/3] Persisting initial project state to database...");
        Project project = new Project();
        project.setName(request.getName());
        project.setModelType(request.getModelType());
        project.setModelName(request.getModelName());
        project.setOptimizer(request.getOptimizer());
        project.setUser(currentUser);
        project.setStatus("CREATED");
        Project savedProject = projectRepository.save(project);
        System.out.println("...Success! Project ID: " + savedProject.getId());

        // --- Step 2: Model Initialization (The "Loader" Part) ---
        File modelFile = new File("models/" + savedProject.getId().toString() + ".npz");
        String absoluteModelPath = modelFile.getAbsolutePath();
        savedProject.setModelPath(absoluteModelPath);
        System.out.println("Saving model at - "+absoluteModelPath);

        System.out.println("\n[2/3] Initializing model file... (This may take a moment)");
        System.out.println("------------------------- LOADER START -------------------------");

        // This is the long-running, blocking call.
        modelInitializer.initializeModelFile(request.getModelType(), request.getModelName(), request.getOptimizer(), absoluteModelPath, request.getPretrainEpochs());

        System.out.println("-------------------------- LOADER END --------------------------");
        System.out.println("...Success! Model file created at: " + absoluteModelPath);

//        // --- Step 3: Start Federated Learning Server ---
//        System.out.println("\n[3/4] Starting dedicated Flower server process...");
//        int port = flowerServerManager.startServerForProject(savedProject,false);
//        savedProject.setServerPort(port);
//        System.out.println("...Success! Flower server started on port: " + port);

        // --- Step 3: Final Database Update ---
        System.out.println("\n[3/3] Updating project with server details...");
        Project finalProject = projectRepository.save(savedProject);
        System.out.println("...Success! Project is fully configured and ready.");
        System.out.println("==================== PROJECT CREATION COMPLETE ====================\n");

        return convertToDto(finalProject);
    }

    public ProjectResponseDto startServerForProject(UUID projectId, StartProject request) throws IOException, InterruptedException {

        Optional<Project> savedProject = projectRepository.findById(projectId);
        System.out.println("\n[1/4] Finding project with ID: " + projectId);

        Project project = projectRepository.findById(projectId).orElseThrow(() -> new RuntimeException("Project not found with ID: " + projectId));

        String strategyToUse = (request != null && request.getStrategy() != null && !request.getStrategy().isEmpty())
                ? request.getStrategy()
                : "FedAvg";

        // --- THIS IS THE NEW LOGIC FOR NUMBER OF ROUNDS ---
        System.out.println("request.getNumRounds() - "+request.getNumRounds());
        Integer numRoundsToUse;
        if (request != null && request.getNumRounds() != null && request.getNumRounds() > 0) {
            // 1. Use the user's value if provided and valid
            numRoundsToUse = request.getNumRounds();
            System.out.println("...User specified number of rounds: " + numRoundsToUse);
        } else {
            // 2. Otherwise, set a default based on the model type
            if ("Transformer".equalsIgnoreCase(project.getModelType())) {
                numRoundsToUse = 5; // Default for LLMs
            } else {
                numRoundsToUse = 5; // Default for CNNs and others
            }
            System.out.println("...Using default number of rounds for " + project.getModelType() + ": " + numRoundsToUse);
        }
        // Check if server is already running for this project

        if(flowerServerManager.isServerRunning(projectId)){
            System.out.println("...Server is already running for this project on port: " + project.getServerPort());
            convertToDto(project);
        }

        System.out.println("\n[2/4] Starting dedicated Flower server process...");

        int port = flowerServerManager.startServerForProject(project,true, strategyToUse, numRoundsToUse);
        project.setServerPort(port);
        project.setStatus("RUNNING");


        System.out.println("...Success! Flower server started on port: " + port);

        System.out.println("\n[3/4] Updating project with server details...");
//        Project finalProject = projectRepository.save(project);
        Project updatedProject = projectRepository.save(project);

        ProjectStatusUpdateDto update = new ProjectStatusUpdateDto(updatedProject.getId(), "RUNNING", updatedProject.getServerPort());
        webSocketService.sendStatusUpdate(update);
        System.out.println("...Success! Project is fully configured and ready.");
        System.out.println("\n[4/4] Starting dedicated Flower server process...");
        return convertToDto(updatedProject);

    }

    @Transactional
    public ProjectResponseDto stopServerForProject(UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> new RuntimeException("Project not found with ID: " + projectId));

        boolean stopped = flowerServerManager.stopServerForProject(projectId);
        Project finalProjectState = project;
        if (stopped || project.getStatus().equals("RUNNING")) {
            System.out.println("!!! SERVER PROCESS WAS STOPPED. SETTING PORT TO NULL. !!!");
            // If the server was successfully stopped, update the project state
            project.setServerPort(null);
            project.setStatus("STOPPED");
            finalProjectState = projectRepository.save(project);
            System.out.println("...Server process stopped and project state updated.");
        }

        return convertToDto(finalProjectState);
    }

    // in ProjectService.java
    public List<ProjectResponseDto> getProjectsForCurrentUser() {
        // Get the currently authenticated user's details
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String currentUsername = authentication.getName();

        // You'll need your UserRepository for this
        // (Assuming you have a UserRepository that can find by username)
        User currentUser = userRepository.findByUsername(currentUsername)
                .orElseThrow(() -> new UsernameNotFoundException("User not found"));

        List<Project> projects = projectRepository.findByUserId(currentUser.getId());

        // Convert the list of entities to a list of DTOs
        return projects.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    public List<RoundResultDto> getResultsForProject(UUID projectId) {
        List<RoundResult> results = roundResultRepository.findByProjectIdOrderByServerRoundAsc(projectId);

        // Convert the list of entities to a list of DTOs
        return results.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    public void markProjectAsCompleted(UUID projectId){
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> new RuntimeException("Project not found with ID: " + projectId));

        project.setStatus("COMPLETED");
        project.setServerPort(null);
        projectRepository.save(project);

        ProjectStatusUpdateDto update = new ProjectStatusUpdateDto(project.getId(), "COMPLETED", null);
        webSocketService.sendStatusUpdate(update);
        System.out.println("...Success! Project is marked as completed.");

    }

    public void deleteProject(UUID projectId){
        projectRepository.deleteById(projectId);
        System.out.println("...Success! Project deleted.");
    }
}