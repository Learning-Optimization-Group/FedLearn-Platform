package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

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

    /** Shared secret the FL-server task must send as X-Internal-Key on /api/internal/** calls. */
    @Value("${app.internal.api-key:}")
    private String internalApiKey;

    /**
     * URL the FL-server task uses to call back into this backend (round results,
     * finished notifications). Use an internal, VPC-reachable URL in production
     * (ALB, Cloud Map DNS) — not the public domain. Optional in dev.
     */
    @Value("${app.backend.internal-url:}")
    private String backendInternalUrl;

    /**
     * Dispatches a Flower FL server task on AWS ECS Fargate for the given project.
     * Returns 0 on success; callers resolve the task endpoint via Cloud Map / Service Connect.
     * Throws IllegalStateException if ECS config is missing or if task launch fails.
     */
    public int startServerForProject(Project project, boolean isPretrained, String strategy,
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

    public boolean stopServerForProject(UUID projectId) {
        log.info("ECS task termination for project {} is managed out-of-band via the ECS StopTask API.", projectId);
        return true;
    }

    public boolean isServerRunning(UUID projectId) {
        return false;
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

        // Service-to-service auth + callback URL for /api/internal/** endpoints.
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
