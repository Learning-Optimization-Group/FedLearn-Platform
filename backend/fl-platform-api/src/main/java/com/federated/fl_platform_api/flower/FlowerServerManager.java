package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.WebSocketService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.services.ecs.EcsClient;
import software.amazon.awssdk.services.ecs.model.AssignPublicIp;
import software.amazon.awssdk.services.ecs.model.AwsVpcConfiguration;
import software.amazon.awssdk.services.ecs.model.ContainerOverride;
import software.amazon.awssdk.services.ecs.model.Failure;
import software.amazon.awssdk.services.ecs.model.KeyValuePair;
import software.amazon.awssdk.services.ecs.model.LaunchType;
import software.amazon.awssdk.services.ecs.model.NetworkConfiguration;
import software.amazon.awssdk.services.ecs.model.RunTaskRequest;
import software.amazon.awssdk.services.ecs.model.RunTaskResponse;
import software.amazon.awssdk.services.ecs.model.TaskOverride;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Component
public class FlowerServerManager {

    private static final Logger log = LoggerFactory.getLogger(FlowerServerManager.class);

    @Value("${ecs.cluster-name}")
    private String ecsClusterName;

    @Value("${ecs.task-definition}")
    private String ecsTaskDefinition;

    @Value("${ecs.subnets}")
    private String ecsSubnetsCsv;

    @Value("${ecs.security-groups}")
    private String ecsSecurityGroupsCsv;

    @Value("${ecs.assign-public-ip:DISABLED}")
    private String ecsAssignPublicIp;

    @Value("${ecs.container-name:fl-server-container}")
    private String ecsContainerName;

    @Value("${app.internal.api-key:}")
    private String internalApiKey;

    @Value("${app.backend.internal-url:}")
    private String backendInternalUrl;

    @Value("${python.script.fl-server.path:src/main/resources/scripts/run_fl_server.sh}")
    private String flServerWrapperPath;

    @Autowired
    private WebSocketService logBroadcaster;

    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    public int startServerForProject(Project project, boolean isPretrained, String strategy,
                                     Integer numRounds, Integer minClients) {
        if (!isBlank(ecsClusterName)) {
            return startEcsFargateServer(project, isPretrained, strategy, numRounds, minClients);
        } else {
            return startLocalServer(project, isPretrained, strategy, numRounds);
        }
    }

    private int startEcsFargateServer(Project project, boolean isPretrained, String strategy,
                                      Integer numRounds, Integer minClients) {
        validateEcsConfig();
        if (isBlank(internalApiKey)) {
            throw new IllegalStateException(
                    "APP_INTERNAL_API_KEY is not configured; FL-server tasks cannot report results back.");
        }

        List<KeyValuePair> envVars = buildEnvOverrides(project, isPretrained, strategy, numRounds, minClients);

        RunTaskRequest request = RunTaskRequest.builder()
                .cluster(ecsClusterName)
                .taskDefinition(ecsTaskDefinition)
                .launchType(LaunchType.FARGATE)
                .networkConfiguration(NetworkConfiguration.builder()
                        .awsvpcConfiguration(AwsVpcConfiguration.builder()
                                .subnets(splitCsv(ecsSubnetsCsv))
                                .securityGroups(splitCsv(ecsSecurityGroupsCsv))
                                .assignPublicIp(AssignPublicIp.fromValue(ecsAssignPublicIp))
                                .build())
                        .build())
                .overrides(TaskOverride.builder()
                        .containerOverrides(ContainerOverride.builder()
                                .name(ecsContainerName)
                                .environment(envVars)
                                .build())
                        .build())
                .build();

        try (EcsClient ecsClient = EcsClient.builder().build()) {
            RunTaskResponse response = ecsClient.runTask(request);
            if (response.hasFailures() && !response.failures().isEmpty()) {
                String joined = response.failures().stream()
                        .map(Failure::toString)
                        .reduce((a, b) -> a + "; " + b)
                        .orElse("unknown failure");
                throw new IllegalStateException("ECS runTask reported failures: " + joined);
            }
            String taskArn = response.tasks().isEmpty() ? "<none>" : response.tasks().get(0).taskArn();
            log.info("Dispatched Fargate task {} for project {}", taskArn, project.getId());
            return 0;
        } catch (RuntimeException e) {
            log.error("Failed to dispatch Fargate task for project {}: {}", project.getId(), e.getMessage());
            throw e;
        }
    }

    private int startLocalServer(Project project, boolean isPretrained, String strategy, Integer numRounds) {
        try {
            stopServerForProject(project.getId());
            Thread.sleep(2000);

            int freePort = findFreePort();
            File scriptFile = new File(flServerWrapperPath);
            String absoluteScriptPath = scriptFile.getAbsolutePath();
            ProcessBuilder pb;
            System.out.println("absoluteScriptPath - " + absoluteScriptPath);
            List<String> command = new ArrayList<>();
            String os = System.getProperty("os.name").toLowerCase();

            if (!os.contains("win")) {
                command.add("bash");
            }

            command.add(absoluteScriptPath);
            command.add("--project-id");
            command.add(project.getId().toString());
            command.add("--model-path");
            command.add(project.getModelPath());
            command.add("--port");
            command.add(String.valueOf(freePort));
            command.add("--strategy");
            command.add(strategy);
            command.add("--num-rounds");
            command.add(String.valueOf(numRounds));
            command.add("--model-type");
            command.add(project.getModelType());
            command.add("--model-name");
            command.add(project.getModelName());

            pb = new ProcessBuilder(command);

            if (!isPretrained) {
                pb.command().add("--pretrain");
            }

            System.out.println("--- Preparing to Start Flower Server ---");
            System.out.println("Executing command: " + String.join(" ", pb.command()));

            pb.redirectErrorStream(true);
            pb.directory(new File("."));

            Process process = pb.start();
            runningServers.put(project.getId(), process);

            final StringBuilder startupOutput = new StringBuilder();
            final boolean[] errorOccurred = {false};

            Thread outputReaderThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("[FL_SERVER_LOG " + project.getId() + "] " + line);
                        if (logBroadcaster != null) {
                            logBroadcaster.sendLogs(project.getId(), line);
                        }
                        startupOutput.append(line).append("\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error reading output from Flower server for project " + project.getId());
                    errorOccurred[0] = true;
                    if (logBroadcaster != null) {
                        logBroadcaster.sendLogs(project.getId(), "ERROR: " + e);
                    }
                    e.printStackTrace();
                }
            });
            outputReaderThread.setDaemon(true);
            outputReaderThread.start();

            boolean exited = process.waitFor(3, TimeUnit.SECONDS);

            if (exited || errorOccurred[0]) {
                outputReaderThread.join(1000);
                throw new RuntimeException("Flower server process failed to start. Exit code: " + process.exitValue() +
                        "\nFull Output:\n" + startupOutput);
            }

            System.out.println("Started Flower server for project " + project.getName() + " on port " + freePort);
            return freePort;
        } catch (Exception e) {
            throw new RuntimeException("Failed to start local server process", e);
        }
    }

    public boolean stopServerForProject(UUID projectId) {
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            System.out.println("Stopping Flower server for project: " + projectId);
            process.destroyForcibly();
            runningServers.remove(projectId);
            return true;
        }
        System.out.println("No running server found for project: " + projectId);
        return false;
    }

    public boolean isServerRunning(UUID projectId) {
        Process p = runningServers.get(projectId);
        return (p != null && p.isAlive());
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

    private void validateEcsConfig() {
        if (isBlank(ecsClusterName) || isBlank(ecsTaskDefinition)
                || isBlank(ecsSubnetsCsv) || isBlank(ecsSecurityGroupsCsv)) {
            throw new IllegalStateException(
                    "ECS configuration incomplete. Required: ECS_CLUSTER_NAME, ECS_TASK_DEFINITION, "
                            + "ECS_SUBNETS, ECS_SECURITY_GROUPS.");
        }
    }

    private List<KeyValuePair> buildEnvOverrides(Project project, boolean isPretrained, String strategy,
                                                 Integer numRounds, Integer minClients) {
        List<KeyValuePair> envVars = new ArrayList<>();
        envVars.add(kv("PROJECT_ID", project.getId().toString()));
        envVars.add(kv("MODEL_PATH", project.getModelPath()));
        envVars.add(kv("STRATEGY", strategy));
        envVars.add(kv("NUM_ROUNDS", String.valueOf(numRounds)));
        envVars.add(kv("MIN_CLIENTS", String.valueOf(minClients)));
        envVars.add(kv("MODEL_TYPE", project.getModelType()));
        envVars.add(kv("MODEL_NAME", project.getModelName()));
        if (!isPretrained) {
            envVars.add(kv("PRETRAIN", "true"));
        }

        envVars.add(kv("FEDLEARN_INTERNAL_API_KEY", internalApiKey));
        if (!isBlank(backendInternalUrl)) {
            envVars.add(kv("FEDLEARN_BACKEND_URL", backendInternalUrl));
        }
        return envVars;
    }

    private static KeyValuePair kv(String name, String value) {
        return KeyValuePair.builder().name(name).value(value).build();
    }

    private static List<String> splitCsv(String csv) {
        return Arrays.stream(csv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
    }

    private static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }
}
