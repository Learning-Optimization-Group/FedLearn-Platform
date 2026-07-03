package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.flower.FlowerServerManager;
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
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * BA-3: reconciles FL-server processes against persisted run state on backend startup.
 *
 * <p>FL servers are spawned as child OS processes and tracked only in {@link FlowerServerManager}'s
 * in-memory map, so a backend crash orphans them: children keep running (holding gRPC ports) while
 * their runs sit forever in a non-terminal state with no handle to stop them. On boot this loads every
 * still-in-flight run and, using the PID + OS start-instant recorded at spawn (see
 * {@link FlowerServerManager#recordProcessIdentity}):
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
    private final FlowerServerManager serverManager;
    private final ProcessProbe processProbe;
    private final RunTokenRegistry runTokenRegistry;

    // Auto-reconcile on boot in real deploys; disabled under test so the ApplicationRunner doesn't
    // reap runs across the shared Testcontainers DB (the reconcile() logic is exercised directly).
    @Value("${app.fl.reconcile-on-startup:true}")
    private boolean reconcileOnStartup;

    private volatile ReconciliationResult lastResult;

    public StartupReconciler(RunRepository runRepository, ProjectRepository projectRepository,
                             RunService runService, FlowerServerManager serverManager,
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
            reconcile();
        }
    }

    /**
     * Reconcile every in-flight run against its recorded process. Package-private and returns the
     * summary so tests can drive it directly. Never throws.
     */
    ReconciliationResult reconcile() {
        List<Run> inFlight;
        try {
            inFlight = runRepository.findByStatusIn(IN_FLIGHT);
        } catch (RuntimeException e) {
            log.error("Startup reconciliation could not load in-flight runs; skipping", e);
            this.lastResult = new ReconciliationResult(0, 0, e.getClass().getSimpleName());
            return this.lastResult;
        }

        int adopted = 0;
        int reaped = 0;
        for (Run run : inFlight) {
            try {
                if (adoptIfLive(run)) {
                    adopted++;
                } else {
                    reap(run);
                    reaped++;
                }
            } catch (RuntimeException e) {
                log.warn("Reconciliation failed for run {} (project {}); leaving it as-is: {}",
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

    /**
     * @return {@code true} if the run's recorded process is live and identity-matched (and was
     *         re-adopted); {@code false} if the run should be reaped. Never kills the inspected
     *         process — a reaped run's own server is already dead.
     */
    private boolean adoptIfLive(Run run) {
        Long pid = run.getServerPid();
        if (pid == null) {
            // No PID recorded. With fail-closed spawn (a spawn that can't persist its identity kills
            // the child), the only way here is a JVM crash in the sub-millisecond window between
            // pb.start() and the identity write — extremely rare. We cannot identify or kill a process
            // we have no PID for, so reap the run; if a child did survive that window its port may need
            // manual reclamation.
            log.warn("Run {} is in-flight but recorded no server PID; reaping it (any surviving child "
                    + "process from the crash window cannot be identified)", run.getId());
            return false;
        }
        Optional<ProcessHandle> live = processProbe.of(pid);
        if (live.isEmpty()) {
            return false;   // our server is gone; its OS port is already free
        }
        ProcessHandle handle = live.get();

        Instant recorded = run.getProcessStartedAt();
        Optional<Instant> liveStart = handle.info().startInstant();
        if (recorded != null && liveStart.isPresent()) {
            // Both known: a start-instant that differs by more than the tolerance is a CONFIRMED
            // different process (the PID was recycled). Reap the run, but never touch that process.
            long deltaMs = Math.abs(Duration.between(recorded, liveStart.get()).toMillis());
            if (deltaMs > IDENTITY_TOLERANCE_MS) {
                log.warn("Run {} PID {} is live but its start-instant differs by {}ms from the recorded "
                                + "one (PID reuse); reaping the run without touching that process",
                        run.getId(), pid, deltaMs);
                return false;
            }
        } else {
            // Start-instant unavailable (rare on Linux). We recorded exactly this PID for this run, so
            // a live process on it is almost certainly still ours — adopt to avoid leaking its port,
            // rather than reap-and-leak.
            log.warn("Run {} PID {} is live but a start-instant is unavailable; adopting on PID match "
                    + "alone", run.getId(), pid);
        }

        serverManager.adopt(run.getProjectId(), handle);
        // Restore the surviving server's token so its result/benchmark callbacks keep authorizing —
        // the plaintext lives only in the still-running child; we rehydrate from the persisted hash.
        runTokenRegistry.rehydrate(run.getInternalTokenHash(),
                new RunTokenRegistry.Scope(run.getProjectId(), run.getId()));
        return true;
    }

    /** Reset an orphaned run to FAILED and, if it is still the project's active run, return it to idle. */
    private void reap(Run run) {
        runService.markFailed(run.getId());   // status=FAILED, server_port=null, ended_at=now
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

    @Override
    public Health health() {
        ReconciliationResult r = this.lastResult;
        if (r == null) {
            return Health.unknown().withDetail("state", "not-run-yet").build();
        }
        Health.Builder builder = Health.up()
                .withDetail("adopted", r.adopted())
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
