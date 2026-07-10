package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.exception.ProjectStateException;
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

    @Autowired
    private com.federated.fl_platform_api.service.ModelRecipeService modelRecipeService;

    // DA-8: the FL-server orchestration seam. Raw ProcessBuilder mechanics live behind this runner
    // instead of inline in this JVM orchestration class, so the spawn path is unit-testable with a fake
    // runner and a future managed-task (ECS) runner can be swapped in without changing the orchestration.
    // Defaults to the local-process runner; a Spring bean of type FlServerProcessRunner, if one is
    // defined, overrides it (setProcessRunner). Tests inject a fake via ReflectionTestUtils.
    private FlServerProcessRunner processRunner = new LocalProcessFlServerRunner();

    @Autowired(required = false)
    public void setProcessRunner(FlServerProcessRunner runner) {
        if (runner != null) {
            this.processRunner = runner;
        }
    }

    // BA-3: ProcessHandle (not Process) so a StartupReconciler can re-adopt a child that outlived a
    // backend crash — a restarted JVM can only recover a handle to an orphan, never the original
    // Process object. Freshly-spawned servers are stored via process.toHandle(); the stdout reader
    // still uses the live Process captured at spawn.
    private final Map<UUID, ProcessHandle> runningServers = new ConcurrentHashMap<>();

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
        requireDpPolicySatisfied(project);   // SE-11: gate every start path, before any spawn
        requireModelTypeInCatalog(project, strategy);   // SE-10: unknown modelType -> 400 before spawn
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
        SpawnedFlProcess process = null;
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

            // SE-7: mint a random per-run internal token scoped to (projectId, runId) and hand ONLY
            // it to the child — never a secret it could use to forge another project's token. It is
            // evicted when the server stops (stopServerForProject).
            String internalRunToken = runTokenRegistry.mint(project.getId(), project.getActiveRunId());
            // BA-3: persist the token's hash on the run so a re-adopted server's token can be
            // rehydrated after a restart (the plaintext goes only to the child, below).
            String internalTokenHash = runTokenRegistry.hash(internalRunToken);
            String runIdArg = project.getActiveRunId() != null
                    ? project.getActiveRunId().toString() : null;

            log.debug("Starting FL server for project {} via script {}", project.getId(), absoluteScriptPath);

            // DA-8: delegate the raw process launch to the runner seam. The env customization (SE-1/SE-7
            // secret scrub + per-run token) runs inside the runner as it configures the child env, so
            // the security contract is unchanged — only the ProcessBuilder mechanics moved.
            process = processRunner.start(command,
                    env -> configureChildEnv(env, internalApiKey, backendInternalUrl,
                            flTokenSecret, requireClientAuth, runIdArg, requireTls, internalRunToken),
                    new File("."));
            runningServers.put(project.getId(), process.toHandle());
            try {
                recordProcessIdentity(project.getActiveRunId(), process.pid(),
                        process.startInstant().orElse(null), freePort, internalTokenHash);
            } catch (RuntimeException e) {
                // BA-3: we spawned a child but could not persist its identity — after a crash it would
                // be an unreconcilable orphan holding its port with no PID on record to reap it. Fail
                // closed: kill the child and surface the failure rather than leak silently.
                log.error("Could not record FL-server identity for project {}; terminating the child to "
                        + "avoid an unrecoverable orphan", project.getId(), e);
                process.destroyForcibly();
                runningServers.remove(project.getId());
                // the outer finally releases the reserved port
                throw new ServerProcessException(
                        "Failed to record FL-server process identity for project " + project.getId(), e);
            }

            final StringBuilder startupOutput = new StringBuilder();
            final boolean[] errorOccurred = {false};
            final SpawnedFlProcess readerProcess = process;

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
            // processRunner.start() is the only IOException source in this try; the local runner throws
            // it from ProcessBuilder.start() before returning a handle, so `process` is still null and
            // there is nothing to tear down here (the finally still releases the reserved port).
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

    /**
     * SE-11 run-start DP policy gate: a {@code regulated} project may not start unless DP is
     * enabled AND its (epsilon, delta, clip-norm) config is complete. Enforced here — the single
     * spawn seam — so no start path can bypass it. Throws {@link ProjectStateException} (→ 409 with
     * the message intact via {@code GlobalExceptionHandler}) because the project's stored config,
     * not this request, is what blocks the start.
     */
    private static void requireDpPolicySatisfied(Project project) {
        if (!project.isRegulated()) {
            return;
        }
        if (!project.isDpEnabled()) {
            throw new ProjectStateException(
                    "Cannot start regulated project " + project.getId()
                            + ": differential privacy must be enabled (dpEnabled=true) before a "
                            + "regulated project may train.");
        }
        if (!project.hasCompleteDpConfig()) {
            throw new ProjectStateException(
                    "Cannot start regulated project " + project.getId()
                            + ": incomplete DP config — requires dpTargetEpsilon > 0 (guidance: "
                            + "4-8 for medical/regulated data), dpDelta in (0,1) exclusive, and "
                            + "dpClipNorm > 0 (the per-user contribution bound).");
        }
    }

    /**
     * SE-10: refuse to spawn a gradient FL server whose modelType is not a key in the recipe
     * catalog (recipes.py -- the same source of truth GET /api/model-recipes and the inference
     * paths use). buildServerCommand's char-class allowlist blocks argv option-injection but still
     * lets an unknown-but-clean token (e.g. "FOOBAR") reach a spawn that can only crash at recipe
     * load. Fail fast here.
     *
     * <p>Requires an EXACT-CASE catalog key, not merely a case-insensitive hit. This is a
     * canonical-key CONSISTENCY policy, not crash-prevention: {@code fl_server.py}'s {@code --model-type}
     * argparse uses {@code type=str.upper}, so a lowercase variant is in fact normalized there and would
     * NOT mistrain (an earlier rationale here claiming a case-sensitive downstream mistrain was
     * incorrect -- corrected per an adversarial audit). Requiring the canonical key keeps the persisted
     * modelType and its registry recipeKey identical to the catalog; legit clients always send the exact
     * key from GET /api/model-recipes, so this rejects only hand-crafted non-canonical requests.
     *
     * <p>Throws {@link IllegalArgumentException} (mapped to 400 by GlobalExceptionHandler) because an
     * unknown/non-canonical modelType is invalid input -- distinct from the DP gate's 409, which is a
     * fixable regulated-config conflict. The message names the field + projectId but never echoes the
     * raw value (SE-10 no-reflect convention: the stored modelType is not yet char-class validated
     * here). FoT text-federation carries no model-type on its argv (see buildServerCommand's !isFoT
     * branch), so it is exempt. A catalog LOAD failure surfaces as IllegalStateException from
     * getRecipes() and is handled by GlobalExceptionHandler as 409 (an infra failure is arguably a
     * 5xx, but the key property is only that it is never masked as this 400).
     */
    private void requireModelTypeInCatalog(Project project, String strategy) {
        if ("FoT".equalsIgnoreCase(strategy)) {
            return;
        }
        String modelType = project.getModelType();
        boolean canonicalMatch = modelRecipeService.findByKey(modelType)
                .map(recipe -> recipe.key().equals(modelType))   // exact-case, not just a lenient hit
                .orElse(false);
        if (!canonicalMatch) {
            throw new IllegalArgumentException(
                    "Unknown or non-canonical model type for project " + project.getId()
                            + " -- must be an exact recipe-catalog key; cannot start an FL server.");
        }
    }

    /** Build the fl_server (or FoT) launch command. LLM_LORA carries --aggregation FFA_LORA. */
    static List<String> buildServerCommand(Project project, String strategy, Integer numRounds,
                                           Integer minClients, int freePort, String absoluteScriptPath,
                                           boolean isWindows) {
        boolean isFoT = "FoT".equalsIgnoreCase(strategy);
        // SE-11: the FoT text-federation server has no DP flag contract; spawning it for a
        // DP-enabled project would silently train without DP. Fail closed.
        if (isFoT && project.isDpEnabled()) {
            throw new IllegalArgumentException(
                    "DP is not supported for FoT text-federation runs; disable dpEnabled or use a "
                            + "gradient strategy.");
        }
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
            if (project.isDpEnabled()) {
                // SE-11: the --dp-* flag names are a pinned contract with fl_server.py's argparse —
                // do not rename. Creation validates completeness, but the spawn seam re-checks so a
                // null knob can never reach the argv as the string "null" (SE-10 fail-closed). All
                // values are typed numbers formatted via String.valueOf, never raw strings.
                Double epsilon = project.getDpTargetEpsilon();
                Double delta = project.getDpDelta();
                Double clipNorm = project.getDpClipNorm();
                if (!Project.isCompleteDpConfig(epsilon, delta, clipNorm)) {
                    throw new IllegalArgumentException(
                            "Incomplete DP config for FL-server spawn: requires dpTargetEpsilon > 0 "
                                    + "(guidance: 4-8), dpDelta in (0,1), and dpClipNorm > 0.");
                }
                command.add("--dp-enabled");
                command.add("--dp-clip-norm");
                command.add(String.valueOf(clipNorm.doubleValue()));
                command.add("--dp-target-epsilon");
                command.add(String.valueOf(epsilon.doubleValue()));
                command.add("--dp-delta");
                command.add(String.valueOf(delta.doubleValue()));
                command.add("--dp-rounds");
                command.add(String.valueOf(numRounds));
                command.add("--dp-num-clients");
                command.add(String.valueOf(minClients));
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
     * <p>A {@code null} run (e.g. a bare-manager test with no bound run) or a since-deleted run is a
     * no-op. A persistence failure, however, PROPAGATES: the caller must fail the spawn closed (kill
     * the child), because a live server whose identity was never recorded is an orphan that can never
     * be reconciled or reaped — exactly the leak this feature exists to prevent.
     *
     * <p>DA-8: takes the identity values ({@code pid}, {@code startInstant}) directly rather than a
     * {@link Process}, so it is decoupled from the process seam — the manager reads them off the
     * {@link SpawnedFlProcess} at the call site.</p>
     */
    void recordProcessIdentity(UUID runId, long pid, java.time.Instant startInstant, int port,
                               String internalTokenHash) {
        if (runId == null) {
            return;
        }
        runRepository.findById(runId).ifPresent(run -> {
            run.setServerPid(pid);
            run.setProcessStartedAt(startInstant);
            run.setServerPort(port);
            run.setInternalTokenHash(internalTokenHash);
            runRepository.save(run);
        });
    }

    public boolean stopServerForProject(UUID projectId) {
        runTokenRegistry.evictForProject(projectId);   // SE-7: invalidate this run's internal token
        ProcessHandle handle = runningServers.get(projectId);
        if (handle != null && handle.isAlive()) {
            log.info("Stopping FL server for project {}", projectId);
            handle.destroyForcibly();
            try {
                handle.onExit().get(stopWaitSeconds(), TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                log.warn("Interrupted while waiting for FL server {} to terminate", projectId);
                Thread.currentThread().interrupt();
            } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException e) {
                log.warn("FL server {} did not terminate within {}s of destroyForcibly: {}",
                        projectId, stopWaitSeconds(), e.getClass().getSimpleName());
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
                    p.onExit().get(stopWaitSeconds(), TimeUnit.SECONDS);
                }
            } catch (InterruptedException e) {
                log.warn("Interrupted while waiting for FL server {} to terminate during shutdown", id);
                Thread.currentThread().interrupt();
            } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException e) {
                log.warn("FL server {} did not terminate within {}s during shutdown: {}",
                        id, stopWaitSeconds(), e.getClass().getSimpleName());
            } catch (RuntimeException e) {
                log.warn("Failed to terminate FL server for project {}: {}",
                        id, e.getClass().getSimpleName());
            }
        });
        runningServers.clear();
    }

    public boolean isServerRunning(UUID projectId) {
        ProcessHandle p = runningServers.get(projectId);
        return (p != null && p.isAlive());
    }

    /**
     * BA-3: re-adopt an FL-server child that outlived a backend restart, so it is tracked again and a
     * later {@link #stopServerForProject} can terminate it. The port needs no reservation — a live
     * server holds it at the OS level, so {@link #findFreePort} (which probes an actual bind) won't
     * hand it out. Only the startup reconciler, which has already verified PID + start-instant
     * identity, calls this.
     */
    public void adopt(UUID projectId, ProcessHandle handle) {
        ProcessHandle existing = runningServers.putIfAbsent(projectId, handle);
        if (existing != null) {
            // A process is already tracked for this project. Reconciliation runs before the HTTP layer
            // opens, so this should not happen — but never clobber a live tracked handle (that would
            // orphan it and leak its port). Keep the existing one.
            log.warn("Skipped re-adopting FL server for project {} (pid {}): a process is already tracked",
                    projectId, handle.pid());
            return;
        }
        log.info("Re-adopted orphaned FL server for project {} (pid {})", projectId, handle.pid());
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
