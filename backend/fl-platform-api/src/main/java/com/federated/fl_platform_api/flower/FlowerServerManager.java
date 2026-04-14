package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.WebSocketService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.time.Instant;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.model.ServerLog;
import com.federated.fl_platform_api.repository.ServerLogRepository;


@Component
public class FlowerServerManager {

    // This property points to the run_fl_server wrapper script
    @Value("${python.script.fl-server.path}")
    private String flServerWrapperPath;

    // A map to keep track of the running server processes for cleanup
    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    @Autowired
    private WebSocketService logBroadcaster;

    @Autowired
    private ServerLogRepository serverLogRepository;

    @Autowired
    private ObjectMapper objectMapper;

    /**
     * Starts a dedicated Flower server process via AWS Fargate orchestration.
     */
    public int startServerForProject(Project project, boolean isPretrained, String strategy, Integer numRounds, Integer minClients) {

        System.out.println("--- Preparing to Start AWS Fargate Flower Server ---");

        // Port mapping and IP resolution are handled internally by ECS Service Connect / awsvpc mode
        software.amazon.awssdk.services.ecs.EcsClient ecsClient = software.amazon.awssdk.services.ecs.EcsClient.builder().build();
        
        java.util.List<software.amazon.awssdk.services.ecs.model.KeyValuePair> envVars = new java.util.ArrayList<>();
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("PROJECT_ID").value(project.getId().toString()).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("MODEL_PATH").value(project.getModelPath()).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("STRATEGY").value(strategy).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("NUM_ROUNDS").value(String.valueOf(numRounds)).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("MIN_CLIENTS").value(String.valueOf(minClients)).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("MODEL_TYPE").value(project.getModelType()).build());
        envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("MODEL_NAME").value(project.getModelName()).build());
        
        if (!isPretrained) {
            envVars.add(software.amazon.awssdk.services.ecs.model.KeyValuePair.builder().name("PRETRAIN").value("true").build());
        }

        software.amazon.awssdk.services.ecs.model.RunTaskRequest runTaskRequest = software.amazon.awssdk.services.ecs.model.RunTaskRequest.builder()
            .cluster("fedlearn-production-cluster")
            .taskDefinition("fl-server-task")
            .launchType(software.amazon.awssdk.services.ecs.model.LaunchType.FARGATE)
            .overrides(software.amazon.awssdk.services.ecs.model.TaskOverride.builder()
                .containerOverrides(software.amazon.awssdk.services.ecs.model.ContainerOverride.builder()
                    .name("fl-server-container")
                    .environment(envVars)
                    .build())
                .build())
            .build();
            
        ecsClient.runTask(runTaskRequest);
        System.out.println("Dispatched AWS Fargate task for project " + project.getName());
        return 0; // Return dummy port 0 as routing is managed natively via AWS Cloud Map
    }

    public boolean stopServerForProject(UUID projectId) {
        System.out.println("AWS Task termination is managed independently via API.");
        return true;
    }

    private int findFreePort() {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            if (serverSocket != null) {
                return serverSocket.getLocalPort();
            }
        } catch (IOException e) {
            throw new IllegalStateException("Could not find a free TCP/IP port", e);
        }
        throw new IllegalStateException("Could not find a free TCP/IP port");
    }

    public boolean isServerRunning(UUID projectId) {
        Process p = runningServers.get(projectId);
        return (p != null && p.isAlive());
    }
}