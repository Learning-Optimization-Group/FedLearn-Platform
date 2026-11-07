package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.RunService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * BA-3: reconciles FL-server processes against persisted run state on backend startup.
 *
 * <p>FL servers are spawned as child OS processes and tracked only in {@link FlServerManager}'s
 * in-memory map, so a backend crash orphans them: children keep running (holding gRPC ports) while
 * their runs sit forever in a non-terminal state with no handle to stop them. On boot this loads every
 * still-in-flight run and, using the PID + OS start-instant recorded at spawn (see
 * {@link FlServerManager#recordProcessIdentity}):
 * <ul>
 *   <li><b>re-adopts</b> a run whose recorded PID is still live and whose start-instant matches — the
 *       server survived the restart, so it is tracked again and a later stop can terminate it;</li>
 *   <li><b>reaps</b> a run whose process is dead, whose PID was never recorded, or whose live PID
 *       belongs to a <em>different</em> process (PID reuse) — the run is reset to FAILED and its
 *       project returned to idle. A PID-reused process is never killed (it is not ours; our own server
 *       is already dead and its port already free).</li>
 * </ul>
 *
 * <p>Reconciliation never throws out of {@link #run}: a failure here must not stop the app booting.
 * The last summary is exposed as an actuator health-indicator detail.
 */
@Component
public class StartupReconciler implements HealthIndicator {

    private static final Logger log = LoggerFactory.getLogger(StartupReconciler.class);

    /** Non-terminal run states that could have been left orphaned by a crash. */
    private static final List<RunStatus> IN_FLIGHT =
            List.of(RunStatus.PENDING, RunStatus.STARTING, RunStatus.RUNNING);

    /**
     * Tolerance when matching the recorded start-instant against the live process's. The recorded
     * value is truncated to the DB column's resolution (microseconds) while the live handle carries
     * full nanoseconds, so an exact {@code equals} would spuriously fail. A couple of seconds absorbs
     * that yet stays far tighter than any realistic PID-reuse gap — a reused PID's process can only
     * have started after ours died.
     */
    private static final long IDENTITY_TOLERANCE_MS = 2_000;

    private final RunRepository runRepository;
    private final ProjectRepository projectRepository;
    private final RunService runService;
    private final FlServerManager serverManager;
    private final ProcessProbe processProbe;
    private final RunTokenRegistry runTokenRegistry;

    // Auto-reconcile on boot in real deploys; disabled under test so the ApplicationRunner doesn't
    // reap runs across the shared Testcontainers DB (the reconcile() logic is exercised directly).
    @Value("${app.fl.reconcile-on-startup:true}")
    private boolean reconcileOnStartup;

    // Finding-4 follow-up: enable the periodic stuck-run sweep. On (default) in real deploys; OFF under
    // test so the scheduler doesn't reap runs across the shared Testcontainers DB (sweepStuckRuns() is
    // exercised directly by the unit test instead).
    @Value("${app.fl.stuck-run-sweep.enabled:true}")
    private boolean stuckRunSweepEnabled;

    // Runs seen RUNNING-but-untracked on the PREVIOUS sweep. A run is only reaped once it appears here
    // AND is still stuck this pass (two-consecutive-pass debounce) — see sweepStuckRuns().
    private final Set<UUID> suspectedStuck = ConcurrentHashMap.newKeySet();

    private volatile ReconciliationResult lastResult;

    public StartupReconciler(RunRepository runRepository, ProjectRepository projectRepository,
                             RunService runService, FlServerManager serverManager,
                             ProcessProbe processProbe, RunTokenRegistry runTokenRegistry) {
        this.runRepository = runRepository;
        this.projectRepository = projectRepository;
        this.runService = runService;
        this.serverManager = serverManager;
        this.processProbe = processProbe;
        this.runTokenRegistry = runTokenRegistry;
    }

    /**
     * Reconcile during bean initialisation — i.e. BEFORE the embedded web server starts accepting
     * requests. An ApplicationRunner would run after Tomcat opens, letting a concurrent {@code /start}
     * for a just-orphaned project race the re-adoption into a double-live server + leaked port. Running
     * here closes that window: reconciliation completes while the HTTP layer is still shut.
     */
    @PostConstruct
    void reconcileOnStartup() {
        if (reconcileOnStartup) {
            ReconciliationResult result = reconcile();
            if (result.error() != null) {
                throw new IllegalStateException("Startup reconciliation failed: " + result.error());
            }
        }
    }

    /**
     * Reconcile every in-flight run against its recorded process. Package-private and returns the
     * summary so tests can drive it directly. Never throws out of this method.
     */
    ReconciliationResult reconcile() {
        List<Run> inFlight;
        try {
            inFlight = runRepository.findByStatusIn(IN_FLIGHT);
        } catch (RuntimeException e) {
            log.error("Startup reconciliation could not load in-flight runs; skipping", e);
            this.lastResult = new ReconciliationResult(0, 0, e.getClass().getSimpleName() + ": " + e.getMessage());
            return this.lastResult;
        }

        List<LiveCandidate> liveCandidates = new ArrayList<>();
        List<Run> deadCandidates = new ArrayList<>();

        for (Run run : inFlight) {
            Optional<ProcessHandle> verified = verifyLiveProcess(run);
            if (verified.isPresent()) {
                liveCandidates.add(new LiveCandidate(run, verified.get()));
            } else {
                deadCandidates.add(run);
            }
        }

        String preflightError = preflightLiveCohort(liveCandidates);
        if (preflightError != null) {
            log.error("Startup reconciliation preflight failed; refusing adoption and reaping to prevent unsafe state: {}",
                    preflightError);
            this.lastResult = new ReconciliationResult(0, 0, preflightError);
            return this.lastResult;
        }

        int adopted = 0;
        for (LiveCandidate candidate : liveCandidates) {
            Run run = candidate.run();
            if (run.getCurrentAttemptId() != null) {
                serverManager.adopt(run.getProjectId(), run.getId(), run.getCurrentAttemptId(), candidate.handle(), run.getServerPort());
            } else {
                serverManager.adopt(run.getProjectId(), run.getId(), candidate.handle(), run.getServerPort());
            }
            if (run.getInternalTokenHash() != null) {
                runTokenRegistry.rehydrate(run.getInternalTokenHash(),
                        new RunTokenRegistry.Scope(run.getProjectId(), run.getId()));
            }
            adopted++;
        }

        int reaped = 0;
        for (Run run : deadCandidates) {
            try {
                reap(run);
                reaped++;
            } catch (RuntimeException e) {
                log.warn("Reconciliation reaping failed for run {} (project {}); leaving it as-is: {}",
                        run.getId(), run.getProjectId(), e.toString());
            }
        }

        this.lastResult = new ReconciliationResult(adopted, reaped, null);
        if (adopted + reaped > 0) {
            log.info("Startup reconciliation: re-adopted {} live FL server(s), reaped {} orphaned run(s)",
                    adopted, reaped);
        } else {
            log.debug("Startup reconciliation: no in-flight runs to reconcile");
        }
        return this.lastResult;
    }

    private String preflightLiveCohort(List<LiveCandidate> candidates) {
        Set<UUID> seenProjects = new HashSet<>();
        Set<Integer> seenPorts = new HashSet<>();

        for (LiveCandidate candidate : candidates) {
            Run run = candidate.run();
            Integer port = run.getServerPort();
            if (port == null || port < 1 || port > 65535) {
                return "Run " + run.getId() + " has invalid persisted server port: " + port;
            }

            UUID projectId = run.getProjectId();
            if (projectId == null) {
                return "Run " + run.getId() + " has null projectId";
            }

            if (!seenProjects.add(projectId)) {
                return "Multiple live runs claim project " + projectId;
            }
            if (!seenPorts.add(port)) {
                return "Multiple live runs claim port " + port;
            }

            Optional<Project> projectOpt = projectRepository.findById(projectId);
            if (projectOpt.isEmpty()) {
                return "Run " + run.getId() + " references missing project " + projectId;
            }
            Project project = projectOpt.get();
            if (!run.getId().equals(project.getActiveRunId())) {
                return "Run " + run.getId() + " is live but not the active run for project " + projectId
                        + " (activeRunId=" + project.getActiveRunId() + ")";
            }

            if (serverManager.isServerRunning(projectId)) {
                return "Server is already running in manager for project " + projectId;
            }
            if (serverManager.hasReservedPort(projectId)) {
                return "Port is already reserved in manager for project " + projectId;
            }
            if (serverManager.isPortReserved(port)) {
                return "Port " + port + " is already reserved in manager";
            }
        }
        return null;
    }

    /**
     * @return an engaged {@link ProcessHandle} if the run's recorded process is live and identity-matched;
     *         {@link Optional#empty()} if the run's process is gone, unrecorded, or PID-reused.
     */
    private Optional<ProcessHandle> verifyLiveProcess(Run run) {
        Long pid = run.getServerPid();
        if (pid == null) {
            log.warn("Run {} is in-flight but recorded no server PID; candidate for reaping", run.getId());
            return Optional.empty();
        }
        Optional<ProcessHandle> live = processProbe.of(pid);
        if (live.isEmpty()) {
            return Optional.empty();   // our server is gone; its OS port is already free
        }
        ProcessHandle handle = live.get();
        if (!handle.isAlive()) {
            return Optional.empty();
        }

        Instant recorded = run.getProcessStartedAt();
        Optional<Instant> liveStart = handle.info().startInstant();
        if (recorded != null && liveStart.isPresent()) {
            long deltaMs = Math.abs(Duration.between(recorded, liveStart.get()).toMillis());
            if (deltaMs > IDENTITY_TOLERANCE_MS) {
                log.warn("Run {} PID {} is live but its start-instant differs by {}ms from the recorded "
                                + "one (PID reuse); candidate for reaping without touching that process",
                        run.getId(), pid, deltaMs);
                return Optional.empty();
            }
        } else {
            log.warn("Run {} PID {} is live but a start-instant is unavailable; matching on PID alone",
                    run.getId(), pid);
        }

        return Optional.of(handle);
    }

    private record LiveCandidate(Run run, ProcessHandle handle) {}

    /** Reset an orphaned run to FAILED and, if it is still the project's active run, return it to idle. */
    private void reap(Run run) {
        if (run.getCurrentAttemptId() != null) {
            runService.markFailed(run.getId(), run.getCurrentAttemptId(), "Reaped by startup reconciler");
        } else {
            runService.markFailed(run.getId());
        }
        projectRepository.findById(run.getProjectId()).ifPresent(project -> {
            // Only clear the project pointer if THIS run is still its active one. A newer run may have
            // become active (and been re-adopted) in the meantime — don't null out its pointer.
            if (run.getId().equals(project.getActiveRunId())) {
                project.setActiveRunId(null);
                project.setServerPort(null);
                projectRepository.save(project);
            }
        });
    }

    /**
     * Finding 4: a mid-run crash while the backend stays UP leaves the run stuck in RUNNING forever —
     * the boot {@link #reconcile()} runs once, and normal completion relies on the FL server's terminal
     * callback, which a crash skips. This periodic sweep reaps such runs. Gated + fail-soft; the sweep
     * logic lives in {@link #sweepStuckRuns()} so tests drive it without the scheduler.
     */
    @Scheduled(fixedDelayString = "${app.fl.stuck-run-sweep.interval-ms:60000}",
            initialDelayString = "${app.fl.stuck-run-sweep.interval-ms:60000}")
    void scheduledStuckRunSweep() {
        if (!stuckRunSweepEnabled) {
            return;
        }
        try {
            sweepStuckRuns();
        } catch (RuntimeException e) {
            log.warn("Stuck-run sweep failed; will retry next interval: {}", e.toString());
        }
    }

    /**
     * Reap runs stuck in RUNNING because their FL server died without a terminal callback. Returns the
     * number reaped (for tests). Two-pass debounce: a RUNNING run whose FL server is no longer tracked
     * (post-BA-13, {@code onChildExit} evicts a crashed child) is reaped only if it was ALSO untracked
     * on the previous pass — so a run that is merely mid-completion or mid-stop (server gone, status not
     * yet COMPLETED/STOPPED) settles to a terminal status (and drops out of the RUNNING query) before
     * pass 2, which avoids a FAILED-clobbers-COMPLETED race (endRun does not guard terminal
     * transitions). Only RUNNING is swept: PENDING/STARTING runs are legitimately mid-launch (a slow LLM
     * start can be STARTING for minutes) and must not be reaped.
     */
    int sweepStuckRuns() {
        List<Run> running;
        try {
            running = runRepository.findByStatusIn(List.of(RunStatus.RUNNING));
        } catch (RuntimeException e) {
            log.warn("Stuck-run sweep could not load RUNNING runs; skipping: {}", e.toString());
            return 0;
        }
        Set<UUID> deadThisPass = new HashSet<>();
        int reaped = 0;
        for (Run run : running) {
            try {
                if (serverManager.isServerRunning(run.getProjectId())) {
                    continue;   // a live tracked FL server holds this run -> not stuck
                }
                if (suspectedStuck.contains(run.getId())) {
                    log.warn("Run {} (project {}) is RUNNING but its FL server has been untracked across "
                            + "two sweeps; reaping it as a crashed run", run.getId(), run.getProjectId());
                    reap(run);
                    reaped++;
                } else {
                    deadThisPass.add(run.getId());   // first sighting -> confirm (or clear) on the next pass
                }
            } catch (RuntimeException e) {
                log.warn("Stuck-run sweep failed for run {} (project {}); leaving it: {}",
                        run.getId(), run.getProjectId(), e.toString());
            }
        }
        // A run that recovered, completed, stopped, or was just reaped is absent from deadThisPass, so
        // replacing the set (not merging) clears stale suspicions.
        suspectedStuck.clear();
        suspectedStuck.addAll(deadThisPass);
        return reaped;
    }

    @Override
    public Health health() {
        ReconciliationResult r = this.lastResult;
        if (r == null) {
            return Health.unknown().withDetail("state", "not-run-yet").build();
        }
        Health.Builder builder = (r.error() == null) ? Health.up() : Health.down();
        builder.withDetail("adopted", r.adopted())
                .withDetail("reaped", r.reaped());
        if (r.error() != null) {
            builder = builder.withDetail("error", r.error());
        }
        return builder.build();
    }

    /** Summary of a reconciliation pass (exposed via the health indicator; returned to tests). */
    record ReconciliationResult(int adopted, int reaped, String error) {
    }
}
