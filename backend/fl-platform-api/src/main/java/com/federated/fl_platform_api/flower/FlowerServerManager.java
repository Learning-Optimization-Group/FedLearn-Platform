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
import java.util.Optional;
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

    // SE-1/SE-7: the FL connection-token verify secret handed to the spawned FL server, and whether
    // it enforces client auth. Kept OFF by default — activating locks out clients whose launcher
    // does not yet pass FEDLEARN_CONNECTION_TOKEN.
    @Value("${app.fl.token-secret:}")
    private String flTokenSecret;

    @Value("${app.fl.require-client-auth:false}")
    private boolean requireClientAuth;

    // SE-2: require the FL server to serve TLS (fail closed on plaintext). Default OFF — activating
    // needs server certs provisioned; flipping it without them would refuse to start.
    @Value("${app.fl.require-tls:false}")
    private boolean requireTls;

    @Value("${python.script.fl-server.path:src/main/resources/scripts/run_fl_server.sh}")
    private String flServerWrapperPath;

    @Value("${python.script.fot-server.path:src/main/resources/scripts/run_fot_server.sh}")
    private String fotServerWrapperPath;

    @Value("${fl.server.port-range.start:50000}")
    private int portRangeStart;

    @Value("${fl.server.port-range.end:50010}")
    private int portRangeEnd;

    @Value("${fl.server.startup-probe-seconds:3}")
    private long startupProbeSeconds;

    @Value("${fl.server.stdout-drain-millis:5000}")
    private long stdoutDrainMillis;

    @Autowired
    private WebSocketService logBroadcaster;

    @Autowired
    private com.federated.fl_platform_api.security.RunTokenRegistry runTokenRegistry;

    // BA-3: used to record the spawned child's PID + start-instant on the active Run so orphans can be
    // reconciled after a backend crash.
    @Autowired
    private com.federated.fl_platform_api.repository.RunRepository runRepository;

    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    // Ports that have been picked by findFreePort() but whose Python child
    // has not yet bound — see findFreePort/releasePort. Without this,
    // concurrent project starts can race: both probes find the same port
    // free, both close their probe socket, and both spawn Python on it.
    private final java.util.Set<Integer> reservedPorts = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private final Object portReservationLock = new Object();

    /**
     * Start the FL server for a project and return the reserved local port.
     *
     * <p>Returns {@link Optional#empty()} only on the managed/ECS path, which is
     * deliberately unimplemented here (see the fail-closed block below). On the
     * local-process path a port is always reserved, so the result is present.</p>
     */
    public Optional<Integer> startServerForProject(Project project, String strategy,
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

    private Optional<Integer> startLocalServer(Project project, String strategy,
                                               Integer numRounds, Integer minClients) {
        Process process = null;
        int freePort = -1;
        try {
            stopServerForProject(project.getId());

            freePort = findFreePort();
            // Federation over Text (FoT) is a SEPARATE text-federation server spawned through the
            // same seam as the gradient FL server. This is purely ADDITIVE: the FoT branch selects
            // its own wrapper + flag contract, and the gradient (FedAvg/DeComFL) spawn is the
            // else-branch below (unaffected by adding this branch).
            boolean isFoT = "FoT".equalsIgnoreCase(strategy);
            String wrapperPath = isFoT ? fotServerWrapperPath : flServerWrapperPath;
            String absoluteScriptPath = new File(wrapperPath).getAbsolutePath();

            boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
            List<String> command = buildServerCommand(
                    project, strategy, numRounds, minClients, freePort, absoluteScriptPath, isWindows);

            ProcessBuilder pb = new ProcessBuilder(command);

            // SE-7: mint a random per-run internal token scoped to (projectId, runId) and hand ONLY
            // it to the child — never a secret it could use to forge another project's token. It is
            // evicted when the server stops (stopServerForProject).
            String internalRunToken = runTokenRegistry.mint(project.getId(), project.getActiveRunId());
            configureChildEnv(pb.environment(), internalApiKey, backendInternalUrl,
                    flTokenSecret, requireClientAuth,
                    project.getActiveRunId() != null ? project.getActiveRunId().toString() : null,
                    requireTls, internalRunToken);

            log.debug("Starting FL server for project {} via script {}", project.getId(), absoluteScriptPath);

            pb.redirectErrorStream(true);
            pb.directory(new File("."));

            process = pb.start();
            runningServers.put(project.getId(), process);
            recordProcessIdentity(project.getActiveRunId(), process, freePort);

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

            boolean exited = process.waitFor(startupProbeSeconds, TimeUnit.SECONDS);

            if (exited) {
                // Stdout is buffered: give the reader a generous window to
                // drain remaining output before we surface the failure.
                // Truncating here is the difference between "Python crashed"
                // and a usable stack trace.
                outputReaderThread.join(stdoutDrainMillis);
                runningServers.remove(project.getId());
                throw new ServerProcessException(
                        "FL server exited during startup for project " + project.getId()
                                + " (exit code " + process.exitValue() + ")\nOutput:\n" + startupOutput);
            }
            if (errorOccurred[0]) {
                outputReaderThread.join(stdoutDrainMillis);
                process.destroyForcibly();
                runningServers.remove(project.getId());
                throw new ServerProcessException(
                        "FL server stdout reader failed for project " + project.getId()
                                + "\nOutput:\n" + startupOutput);
            }

            log.info("Started FL server for project {} on port {}", project.getId(), freePort);
            return Optional.of(freePort);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            if (process != null) {
                process.destroyForcibly();
                runningServers.remove(project.getId());
            }
            throw new ServerProcessException(
                    "Interrupted while starting FL server for project " + project.getId(), e);
        } catch (IOException e) {
            // pb.start() is the only IOException source in this try; it
            // throws before `process` is assigned, so there is nothing to
            // tear down here.
            throw new ServerProcessException(
                    "Failed to spawn FL server process for project " + project.getId(), e);
        } finally {
            // Release the reservation regardless of outcome. On success the
            // Python child is now bound, so the next findFreePort() probe
            // will naturally skip this port via the ServerSocket check; on
            // failure no one holds the port and it's free for reuse.
            if (freePort != -1) {
                releasePort(freePort);
            }
        }
    }

    // SE-10: identifier-like fields (strategy, model-type, task-type) — a bare token, no separators.
    private static final java.util.regex.Pattern SAFE_TOKEN =
            java.util.regex.Pattern.compile("[A-Za-z0-9_]+");
    // SE-10: a model reference (local filename stem or a HuggingFace repo id such as
    // "Qwen/Qwen2.5-0.5B") — alphanumeric start, then the limited set '/._-'. The leading-alnum
    // anchor blocks option injection ('-x'); ".." is rejected separately to block path traversal.
    private static final java.util.regex.Pattern SAFE_MODEL_REF =
            java.util.regex.Pattern.compile("[A-Za-z0-9][A-Za-z0-9._/-]*");

    /**
     * Reject any attacker-influenceable field before it reaches the {@code fl_server} argv (SE-10).
     * {@link ProcessBuilder} with a {@code List} never invokes a shell, so the concrete risks are
     * option injection (a value beginning with {@code -} being read as an argparse flag) and path
     * traversal via {@code --model-path}/{@code --model-name}. We fail closed — the message names the
     * field but never echoes the rejected value, so a poisoned input can't inject into logs either.
     */
    private static void requireSafeToken(String field, String value) {
        if (value == null || !SAFE_TOKEN.matcher(value).matches()) {
            throw new IllegalArgumentException("Illegal " + field + " for FL-server spawn");
        }
    }

    private static void requireSafeModelRef(String field, String value) {
        if (value == null || value.isBlank() || value.contains("..")
                || !SAFE_MODEL_REF.matcher(value).matches()) {
            throw new IllegalArgumentException("Illegal " + field + " for FL-server spawn");
        }
    }

    private static void requireSafePath(String field, String value) {
        if (value == null || value.isBlank() || value.startsWith("-") || value.contains("..")
                || value.chars().anyMatch(c -> c == '\0' || c == '\n' || c == '\r')) {
            throw new IllegalArgumentException("Illegal " + field + " for FL-server spawn");
        }
    }

    /** Build the fl_server (or FoT) launch command. LLM_LORA carries --aggregation FFA_LORA. */
    static List<String> buildServerCommand(Project project, String strategy, Integer numRounds,
                                           Integer minClients, int freePort, String absoluteScriptPath,
                                           boolean isWindows) {
        boolean isFoT = "FoT".equalsIgnoreCase(strategy);
        // SE-10: allowlist every project-derived string this branch will place on the argv. Ints
        // (rounds/clients/port) and the server-generated UUID are type-safe and need no check.
        requireSafeToken("strategy", strategy);
        if (!isFoT) {
            requireSafePath("model-path", project.getModelPath());
            requireSafeModelRef("model-name", project.getModelName());
            requireSafeToken("model-type", project.getModelType());
            String taskType = project.getTaskType();
            if (taskType != null && !taskType.isBlank()) {
                requireSafeToken("task-type", taskType);
            }
        }
        List<String> command = new ArrayList<>();
        if (!isWindows) {
            command.add("bash");
        }
        command.add(absoluteScriptPath);
        if (isFoT) {
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
            if ("LLM_LORA".equalsIgnoreCase(project.getModelType())) {
                command.add("--aggregation");
                command.add("FFA_LORA");
                String tt = project.getTaskType();
                command.add("--task-type");
                command.add(tt == null || tt.isBlank() ? "SEQ_CLASSIFICATION" : tt);
            }
        }
        return command;
    }

    /**
     * Populate the spawned FL server's child environment (SE-1/SE-7). Sets the internal-API key and
     * backend URL (as before), hands the FL server the connection-token VERIFY secret and the
     * enforcement toggle, and — crucially — SCRUBS the web-auth JWT secret from the child so a
     * compromise of the network-facing FL server cannot forge web/admin sessions (trust-domain
     * isolation). Package-private + static so it is unit-testable without spawning a process.
     */
    static void configureChildEnv(Map<String, String> env, String internalApiKey, String backendUrl,
                                  String flTokenSecret, boolean requireClientAuth, String runId,
                                  boolean requireTls, String internalRunToken) {
        env.put("FEDLEARN_INTERNAL_API_KEY", internalApiKey == null ? "" : internalApiKey);
        // SE-7: scoped per-run token the child presents on /api/internal/** callbacks, so the backend
        // can bind each call to this run's project (a leaked run token can mutate only its project).
        if (!isBlank(internalRunToken)) {
            env.put("FEDLEARN_INTERNAL_RUN_TOKEN", internalRunToken);
        }
        if (!isBlank(backendUrl)) {
            env.put("FEDLEARN_BACKEND_URL", backendUrl);
        }
        if (!isBlank(flTokenSecret)) {
            env.put("FEDLEARN_FL_TOKEN_SECRET", flTokenSecret);
        }
        env.put("FEDLEARN_REQUIRE_CLIENT_AUTH", requireClientAuth ? "1" : "0");
        // FR-7: bind the server to the run it serves, so a token minted for another run is rejected.
        if (!isBlank(runId)) {
            env.put("FEDLEARN_RUN_ID", runId);
        }
        // SE-2: when the deploy requires TLS, enable it AND fail closed on plaintext. The cert paths
        // (FEDLEARN_GRPC_SERVER_KEY/CERT) are inherited from the backend process env.
        if (requireTls) {
            env.put("FEDLEARN_GRPC_USE_TLS", "1");
            env.put("FEDLEARN_REQUIRE_TLS", "1");
        }
        // The FL server verifies with FEDLEARN_FL_TOKEN_SECRET and never needs the web-auth secret.
        env.remove("APP_JWT_SECRET");
    }

    /**
     * BA-3: record the spawned child's OS identity (PID + start instant) and reserved port on the
     * active {@link com.federated.fl_platform_api.model.Run}, so a startup reconciler can later tell a
     * still-live FL server from a dead — or PID-reused — one after a backend restart. The start
     * instant is the anti-PID-reuse guard: a recycled PID belonging to an unrelated process will not
     * share it.
     *
     * <p>Best-effort and self-contained: a {@code null} run (e.g. a bare-manager test with no bound
     * run) or a since-deleted run is a no-op, and a persistence hiccup is logged rather than allowed
     * to fail an otherwise-healthy spawn.
     */
    void recordProcessIdentity(UUID runId, Process process, int port) {
        if (runId == null) {
            return;
        }
        try {
            runRepository.findById(runId).ifPresent(run -> {
                run.setServerPid(process.pid());
                run.setProcessStartedAt(process.info().startInstant().orElse(null));
                run.setServerPort(port);
                runRepository.save(run);
            });
        } catch (RuntimeException e) {
            log.warn("Could not persist FL-server process identity (pid={}, port={}) to run {}: {}",
                    process.pid(), port, runId, e.toString());
        }
    }

    public boolean stopServerForProject(UUID projectId) {
        runTokenRegistry.evictForProject(projectId);   // SE-7: invalidate this run's internal token
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            log.info("Stopping FL server for project {}", projectId);
            process.destroyForcibly();
            try {
                process.waitFor(stopWaitSeconds(), TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                log.warn("Interrupted while waiting for FL server {} to terminate", projectId);
                Thread.currentThread().interrupt();
            }
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
                    p.waitFor(stopWaitSeconds(), TimeUnit.SECONDS);
                }
            } catch (InterruptedException e) {
                log.warn("Interrupted while waiting for FL server {} to terminate during shutdown", id);
                Thread.currentThread().interrupt();
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

    /**
     * Reserve a free port in [portRangeStart, portRangeEnd]. The port is
     * tracked in {@link #reservedPorts} so concurrent callers cannot pick
     * the same port between probe-close and Python bind. Callers MUST call
     * {@link #releasePort(int)} once the spawned process has bound or has
     * failed to start.
     */
    private int findFreePort() {
        synchronized (portReservationLock) {
            for (int port = portRangeStart; port <= portRangeEnd; port++) {
                if (reservedPorts.contains(port)) {
                    continue;
                }
                try (ServerSocket s = new ServerSocket(port)) {
                    reservedPorts.add(port);
                    return port;
                } catch (IOException ignored) {
                    // port in use, try next
                }
            }
            throw new IllegalStateException(
                "No free port in range " + portRangeStart + "–" + portRangeEnd);
        }
    }

    private void releasePort(int port) {
        reservedPorts.remove(port);
    }

    private long stopWaitSeconds() {
        return Math.max(1L, TimeUnit.MILLISECONDS.toSeconds(stdoutDrainMillis));
    }

    private static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }
}
