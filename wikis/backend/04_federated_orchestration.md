# 04 - Federated Orchestration (FlServerManager)

The `FlServerManager` is the most operationally complex class in the Spring Boot backend. It is responsible for bridging the gap between the stateless Java REST API and the stateful, heavily computational Python Federated Learning servers.

The application supports two distinct execution paths, determined entirely by the presence of AWS configuration properties (`ecs.cluster-name`, `ecs.task-definition`, etc.) in the `application.properties`.

## Path A: Local Execution (ProcessBuilder)

When AWS configurations are not present, the `FlServerManager` falls back to local execution. This is primarily used for development, testing on a Macbook, or bare-metal deployments (like the RIT lab environment).

### Process Lifecycle
1. **Port Allocation:** The manager asks the host OS kernel for an ephemeral port via `new ServerSocket(0)`.
2. **Process Construction:** It builds a shell command array using `bash` and the path to the `run_fl_server.sh` script.
3. **Environment Injection:** It injects internal configuration strings directly into the child process's environment variables.
   ```java
   Map<String, String> env = pb.environment();
   env.put("FEDLEARN_INTERNAL_API_KEY", internalApiKey);
   env.put("FEDLEARN_BACKEND_URL", backendInternalUrl);
   ```
4. **Execution and Tracking:** The process is started via `ProcessBuilder.start()`. The returned `java.lang.Process` object is stored in a `ConcurrentHashMap` keyed by the `UUID projectId`.

### Output Redirection
Because a spawned local process does not automatically print to the parent's console, the manager creates a dedicated daemon thread `fl-server-stdout-{id}`. This thread continuously reads from the child's `InputStream`.

Every line read is passed to the `WebSocketService` to be broadcast to the UI. If the process crashes during startup (exits within 3 seconds), an exception is thrown containing the captured `stdout` to help developers debug.

## Path B: Cloud-Native Execution (AWS ECS Fargate)

When the backend is deployed to a production environment (like AWS), local process spawning is disabled. A Spring Boot container running behind an Application Load Balancer cannot spawn heavily computational Python tasks inside itself—it would cause OOM kills, and the dynamically allocated ports would be inaccessible from the public internet.

Instead, the `FlServerManager` utilizes the AWS SDK (`EcsClient`) to orchestrate infrastructure-level task provisioning.

### ECS `RunTaskRequest`
The manager dynamically builds a `RunTaskRequest`. It targets a serverless AWS Fargate cluster.

```java
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
        ...
```

### Dynamic Overrides
Instead of hardcoding environment variables in the task definition in AWS, the Java application uses `TaskOverride` and `ContainerOverride` to dynamically inject the specific parameters for the requested federated training round:

```java
.overrides(TaskOverride.builder()
        .containerOverrides(ContainerOverride.builder()
                .name(ecsContainerName)
                .environment(
                        kv("PROJECT_ID", project.getId().toString()),
                        kv("MODEL_PATH", project.getModelPath()),
                        kv("STRATEGY", strategy),
                        kv("NUM_ROUNDS", String.valueOf(numRounds)),
                        kv("FEDLEARN_INTERNAL_API_KEY", internalApiKey)
                )
                .build())
        .build())
```

### Network Resolution (awsvpc)
When launched via ECS, the task runs in `awsvpc` mode. It receives its own Elastic Network Interface (ENI) and private IP. The Java app does *not* try to allocate a port. Instead, AWS ECS Service Connect (Envoy Proxies) registers the container in a private Cloud Map namespace, allowing edge clients to connect to it using a logical hostname rather than a hardcoded port.

## Graceful Shutdown
To prevent orphaned processes, the manager hooks into the Spring Application Context lifecycle via the `@PreDestroy` annotation. When the Java backend is shutting down, it iterates through the `ConcurrentHashMap` of running servers and forcibly destroys all active child processes.
