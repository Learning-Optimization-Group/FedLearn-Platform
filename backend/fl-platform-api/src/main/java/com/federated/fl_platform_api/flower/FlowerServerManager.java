package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.exception.ServerProcessException;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.WebSocketService;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

@Component
public class FlowerServerManager {

    private static final Logger log = LoggerFactory.getLogger(FlowerServerManager.class);

    @Value("${ecs.cluster-name:}")
    private String ecsClusterName;

    @Value("${app.internal.api-key:}")
    private String internalApiKey;

    @Value("${app.backend.internal-url:}")
    private String backendInternalUrl;

    @Value("${python.script.fl-server.path:src/main/resources/scripts/run_fl_server.sh}")
    private String flServerWrapperPath;

    @Value("${python.script.fot-server.path:src/main/resources/scripts/run_fot_server.sh}")
    private String fotServerWrapperPath;

    @Value("${fl.server.port-range.start:50000}")
    private int portRangeStart;

    @Value("${fl.server.port-range.end:50010}")
    private int portRangeEnd;

    @Autowired
    private WebSocketService logBroadcaster;

    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    public int startServerForProject(Project project, String strategy,
                                     Integer numRounds, Integer minClients) {
        if (!isBlank(ecsClusterName)) {
            // The ECS/Fargate production path is not implemented: runTask returned no reachable
            // host:port (it handed back 0), the task was never tracked in runningServers, and
            // stop/delete could not terminate it — so it would leak a running, billing task while
            // the project was marked RUNNING on an unreachable port. Fail closed rather than record
            // that bogus state. Unset ecs.cluster-name to use the local-process path.
            // See docs/guides/AWS_AUDIT.md before implementing the managed-task path.
            throw new UnsupportedOperationException(
                    "ECS/Fargate FL-server orchestration is not implemented yet "
                            + "(tasks cannot be tracked or stopped). "
                            + "Unset ecs.cluster-name to run FL servers as local processes.");
        }
        return startLocalServer(project, strategy, numRounds, minClients);
    }

    private int startLocalServer(Project project, String strategy, Integer numRounds, Integer minClients) {
        Process process = null;
        try {
            stopServerForProject(project.getId());

            int freePort = findFreePort();
            // Federation over Text (FoT) is a SEPARATE text-federation server spawned through the
            // same seam as the gradient FL server. This is purely ADDITIVE: the FoT branch selects
            // its own wrapper + flag contract, and the gradient (FedAvg/DeComFL) spawn is the
            // else-branch below (unaffected by adding this branch).
            boolean isFoT = "FoT".equalsIgnoreCase(strategy);
            String wrapperPath = isFoT ? fotServerWrapperPath : flServerWrapperPath;
            String absoluteScriptPath = new File(wrapperPath).getAbsolutePath();

            List<String> command = new ArrayList<>();
            String os = System.getProperty("os.name").toLowerCase();
            if (!os.contains("win")) {
                command.add("bash");
            }
            command.add(absoluteScriptPath);

            if (isFoT) {
                // FoT has no global model: no model/strategy/min-clients args. round-seconds,
                // quorum and backend use the entrypoint defaults until exposed in StartProject.
                command.add("--project-id");
                command.add(project.getId().toString());
                command.add("--port");
                command.add(String.valueOf(freePort));
                command.add("--num-rounds");
                command.add(String.valueOf(numRounds));
            } else {
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
                command.add("--min-clients");
                command.add(String.valueOf(minClients));
            }

            ProcessBuilder pb = new ProcessBuilder(command);

            Map<String, String> env = pb.environment();
            env.put("FEDLEARN_INTERNAL_API_KEY", internalApiKey == null ? "" : internalApiKey);
            if (!isBlank(backendInternalUrl)) {
                env.put("FEDLEARN_BACKEND_URL", backendInternalUrl);
            }

            log.debug("Starting FL server for project {} via script {}", project.getId(), absoluteScriptPath);

            pb.redirectErrorStream(true);
            pb.directory(new File("."));

            process = pb.start();
            runningServers.put(project.getId(), process);

            final StringBuilder startupOutput = new StringBuilder();
            final boolean[] errorOccurred = {false};
            final Process readerProcess = process;

            Thread outputReaderThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(readerProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        log.debug("[FL_SERVER {}] {}", project.getId(), line);
                        if (logBroadcaster != null) {
                            logBroadcaster.sendLogs(project.getId(), line);
                        }
                        startupOutput.append(line).append('\n');
                    }
                } catch (IOException e) {
                    errorOccurred[0] = true;
                    log.warn("Failed reading FL server output for project {}: {}",
                            project.getId(), e.getClass().getSimpleName());
                    if (logBroadcaster != null) {
                        logBroadcaster.sendLogs(project.getId(),
                                "ERROR: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                    }
                }
            }, "fl-server-stdout-" + project.getId());
            outputReaderThread.setDaemon(true);
            outputReaderThread.start();

            boolean exited = process.waitFor(3, TimeUnit.SECONDS);

            if (exited) {
                outputReaderThread.join(1000);
                throw new ServerProcessException(
                        "FL server exited during startup for project " + project.getId()
                                + " (exit code " + process.exitValue() + ")\nOutput:\n" + startupOutput);
            }
            if (errorOccurred[0]) {
                outputReaderThread.join(1000);
                throw new ServerProcessException(
                        "FL server stdout reader failed for project " + project.getId());
            }

            log.info("Started FL server for project {} on port {}", project.getId(), freePort);
            return freePort;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            if (process != null) {
                process.destroyForcibly();
                runningServers.remove(project.getId());
            }
            throw new ServerProcessException(
                    "Interrupted while starting FL server for project " + project.getId(), e);
        } catch (IOException e) {
            throw new ServerProcessException(
                    "Failed to spawn FL server process for project " + project.getId(), e);
        }
    }

    public boolean stopServerForProject(UUID projectId) {
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            log.info("Stopping FL server for project {}", projectId);
            process.destroyForcibly();
            runningServers.remove(projectId);
            return true;
        }
        log.debug("No running FL server found for project {}", projectId);
        return false;
    }

    /**
     * Stop every spawned FL server when the application context shuts down.
     * Without this, child Python processes survive backend restarts and
     * become orphans on the host (no longer reachable, but still bound to
     * their gRPC ports).
     */
    @PreDestroy
    public void stopAllOnShutdown() {
        if (runningServers.isEmpty()) {
            return;
        }
        log.info("Shutdown: terminating {} running FL server process(es)", runningServers.size());
        runningServers.forEach((id, p) -> {
            try {
                if (p.isAlive()) {
                    p.destroyForcibly();
                }
            } catch (RuntimeException e) {
                log.warn("Failed to terminate FL server for project {}: {}",
                        id, e.getClass().getSimpleName());
            }
        });
        runningServers.clear();
    }

    public boolean isServerRunning(UUID projectId) {
        Process p = runningServers.get(projectId);
        return (p != null && p.isAlive());
    }

    private int findFreePort() {
        for (int port = portRangeStart; port <= portRangeEnd; port++) {
            try (ServerSocket s = new ServerSocket(port)) {
                return port;
            } catch (IOException ignored) {
                // port in use, try next
            }
        }
        throw new IllegalStateException(
            "No free port in range " + portRangeStart + "–" + portRangeEnd);
    }

    private static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }
}
