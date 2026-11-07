package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * BA-3 reconciliation policy in isolation (ProcessProbe mocked, so no real OS processes):
 * live+identity-matched -> re-adopt; dead / never-recorded / PID-reused -> reset to FAILED, and a
 * PID-reused (identity-mismatched) process is reset rather than killed.
 */
@ExtendWith(MockitoExtension.class)
class StartupReconcilerTest {

    private final RunRepository runRepo = mock(RunRepository.class);
    private final ProjectRepository projectRepo = mock(ProjectRepository.class);
    private final RunService runService = mock(RunService.class);
    private final FlServerManager manager = mock(FlServerManager.class);
    private final ProcessProbe probe = mock(ProcessProbe.class);
    private final RunTokenRegistry tokenRegistry = mock(RunTokenRegistry.class);

    private final StartupReconciler reconciler =
            new StartupReconciler(runRepo, projectRepo, runService, manager, probe, tokenRegistry);

    private static final Instant STARTED = Instant.parse("2026-07-03T12:00:00Z");

    private Run runWith(UUID projectId, Long pid, Instant startedAt) {
        Run r = new Run();
        ReflectionTestUtils.setField(r, "id", UUID.randomUUID());
        r.setProjectId(projectId);
        r.setServerPid(pid);
        r.setProcessStartedAt(startedAt);
        r.setStatus(RunStatus.RUNNING);
        return r;
    }

    private ProcessHandle handleStartedAt(Instant startInstant) {
        ProcessHandle handle = mock(ProcessHandle.class);
        when(handle.isAlive()).thenReturn(true);
        ProcessHandle.Info info = mock(ProcessHandle.Info.class);
        when(handle.info()).thenReturn(info);
        when(info.startInstant()).thenReturn(Optional.ofNullable(startInstant));
        return handle;
    }

    @Test
    void liveIdentityMatchedRun_isReadopted() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 100L, STARTED);
        run.setServerPort(50001);
        run.setInternalTokenHash("hash-100");
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        ProcessHandle handle = handleStartedAt(STARTED);   // exact match, within tolerance
        when(probe.of(100L)).thenReturn(Optional.of(handle));

        Project project = new Project();
        project.setId(projectId);
        project.setActiveRunId(run.getId());
        project.setServerPort(50001);
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        verify(manager).adopt(eq(projectId), eq(run.getId()), eq(handle), eq(50001));
        // BA-3: the survivor's token is rehydrated so its callbacks keep authorizing after restart.
        verify(tokenRegistry).rehydrate("hash-100", new RunTokenRegistry.Scope(projectId, run.getId()));
        verify(runService, never()).markFailed(any());
        assertEquals(1, result.adopted());
        assertEquals(0, result.reaped());
        assertNull(result.error());
    }

    @Test
    void deadPid_isReaped_andProjectReturnedToIdle() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 200L, STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        when(probe.of(200L)).thenReturn(Optional.empty());   // process is gone

        Project project = new Project();
        project.setActiveRunId(run.getId());   // this run IS the project's active one
        project.setServerPort(50001);
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        verify(runService).markFailed(run.getId());
        assertNull(project.getActiveRunId(), "project active run cleared");
        assertNull(project.getServerPort(), "project server port cleared");
        verify(projectRepo).save(project);
        verify(manager, never()).adopt(any(), any());
        assertEquals(1, result.reaped());
    }

    @Test
    void reapingAnOrphan_doesNotClobberADifferentActiveRun() {
        UUID projectId = UUID.randomUUID();
        Run orphan = runWith(projectId, 200L, STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(orphan));
        when(probe.of(200L)).thenReturn(Optional.empty());   // orphan's process is gone -> reap

        // The project's active run is a DIFFERENT, current run (e.g. re-adopted or freshly started).
        UUID currentRunId = UUID.randomUUID();
        Project project = new Project();
        project.setActiveRunId(currentRunId);
        project.setServerPort(50002);
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        reconciler.reconcile();

        verify(runService).markFailed(orphan.getId());          // the orphan run is still failed
        assertEquals(currentRunId, project.getActiveRunId(), "the current run's pointer must survive");
        assertEquals(50002, project.getServerPort());
        verify(projectRepo, never()).save(project);             // project untouched
    }

    @Test
    void reusedPid_isReaped_butTheUnrelatedProcessIsNotKilled() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 300L, STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        // Same PID is alive, but its start-instant is 10 minutes off — a different, unrelated process.
        ProcessHandle handle = handleStartedAt(STARTED.plusSeconds(600));
        when(probe.of(300L)).thenReturn(Optional.of(handle));
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(new Project()));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        verify(runService).markFailed(run.getId());
        verify(manager, never()).adopt(any(), any());
        verify(handle, never()).destroy();
        verify(handle, never()).destroyForcibly();
        assertEquals(1, result.reaped());
    }

    @Test
    void runWithNoRecordedPid_isReaped_withoutProbing() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, null, null);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(new Project()));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        verify(runService).markFailed(run.getId());
        verifyNoInteractions(probe);
        assertEquals(1, result.reaped());
    }

    @Test
    void health_reportsAdoptedAndReapedCounts() {
        UUID p1 = UUID.randomUUID();
        UUID p2 = UUID.randomUUID();
        Run live = runWith(p1, 100L, STARTED);
        live.setServerPort(50001);
        Project project = new Project();
        project.setId(p1);
        project.setActiveRunId(live.getId());
        project.setServerPort(50001);
        when(projectRepo.findById(p1)).thenReturn(Optional.of(project));

        Run dead = runWith(p2, 200L, STARTED);
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live, dead));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));
        when(probe.of(200L)).thenReturn(Optional.empty());
        when(projectRepo.findById(p2)).thenReturn(Optional.of(new Project()));

        reconciler.reconcile();

        var health = reconciler.health();
        assertEquals(1, health.getDetails().get("adopted"));
        assertEquals(1, health.getDetails().get("reaped"));
        assertEquals("UP", health.getStatus().getCode());
    }

    @Test
    void preflightRefusesAdoption_whenServerPortIsNull() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(null); // missing port
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted(), "no adoption on preflight failure");
        assertEquals(0, result.reaped(), "live run must not be reaped on preflight failure");
        assertTrue(result.error().contains("invalid persisted server port"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
        verify(runService, never()).markFailed(any());
        assertEquals("DOWN", reconciler.health().getStatus().getCode());
    }

    @Test
    void preflightRefusesAdoption_whenServerPortIsOutOfRange() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(70000); // out of valid TCP port range
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("invalid persisted server port"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
    }

    @Test
    void preflightRefusesAdoption_whenProjectIsMissing() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(50001);
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));
        when(projectRepo.findById(projectId)).thenReturn(Optional.empty());

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("references missing project"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
    }

    @Test
    void preflightRefusesAdoption_whenProjectActiveRunMismatched() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(50001);
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        Project project = new Project();
        project.setId(projectId);
        project.setActiveRunId(UUID.randomUUID()); // different active run
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("not the active run for project"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
        verify(runService, never()).markFailed(any());
    }

    @Test
    void preflightRefusesAdoption_whenTwoLiveRunsClaimSameProject() {
        UUID projectId = UUID.randomUUID();
        Run live1 = runWith(projectId, 100L, STARTED);
        live1.setServerPort(50001);
        Run live2 = runWith(projectId, 101L, STARTED);
        live2.setServerPort(50002);
        ProcessHandle handle1 = handleStartedAt(STARTED);
        ProcessHandle handle2 = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live1, live2));
        when(probe.of(100L)).thenReturn(Optional.of(handle1));
        when(probe.of(101L)).thenReturn(Optional.of(handle2));

        Project project = new Project();
        project.setId(projectId);
        project.setActiveRunId(live1.getId());
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("Multiple live runs claim project"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
    }

    @Test
    void preflightRefusesAdoption_whenTwoLiveRunsClaimSamePort() {
        UUID p1 = UUID.randomUUID();
        UUID p2 = UUID.randomUUID();
        Run live1 = runWith(p1, 100L, STARTED);
        live1.setServerPort(50001);
        Run live2 = runWith(p2, 101L, STARTED);
        live2.setServerPort(50001); // duplicate port
        ProcessHandle handle1 = handleStartedAt(STARTED);
        ProcessHandle handle2 = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live1, live2));
        when(probe.of(100L)).thenReturn(Optional.of(handle1));
        when(probe.of(101L)).thenReturn(Optional.of(handle2));

        Project proj1 = new Project();
        proj1.setId(p1);
        proj1.setActiveRunId(live1.getId());
        when(projectRepo.findById(p1)).thenReturn(Optional.of(proj1));

        Project proj2 = new Project();
        proj2.setId(p2);
        proj2.setActiveRunId(live2.getId());
        when(projectRepo.findById(p2)).thenReturn(Optional.of(proj2));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("Multiple live runs claim port"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
    }

    @Test
    void preflightRefusesAdoption_whenManagerAlreadyHasServerOrPortReserved() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(50001);
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        Project project = new Project();
        project.setId(projectId);
        project.setActiveRunId(live.getId());
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));
        when(manager.isPortReserved(50001)).thenReturn(true);

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(0, result.adopted());
        assertEquals(0, result.reaped());
        assertTrue(result.error().contains("already reserved in manager"));
        verify(manager, never()).adopt(any(), any(), any(), anyInt());
    }

    @Test
    void preflightAcceptsValidPortOutsideConfiguredPortRange() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(62345); // outside normal 50000..50010 range
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        Project project = new Project();
        project.setId(projectId);
        project.setActiveRunId(live.getId());
        project.setServerPort(62345);
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(project));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        assertEquals(1, result.adopted());
        assertEquals(0, result.reaped());
        assertNull(result.error());
        verify(manager).adopt(eq(projectId), eq(live.getId()), eq(liveHandle), eq(62345));
    }

    @Test
    void reconcileOnStartup_throwsIllegalStateException_whenPreflightFails() {
        UUID projectId = UUID.randomUUID();
        Run live = runWith(projectId, 100L, STARTED);
        live.setServerPort(null); // preflight failure
        ProcessHandle liveHandle = handleStartedAt(STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(live));
        when(probe.of(100L)).thenReturn(Optional.of(liveHandle));

        ReflectionTestUtils.setField(reconciler, "reconcileOnStartup", true);
        assertThrows(IllegalStateException.class, reconciler::reconcileOnStartup,
                "startup must abort boot admission if preflight fails");
    }

    // ---- Finding 4: periodic stuck-run sweep (two-consecutive-pass debounce) --------------------

    @Test
    void stuckRunningRun_isReapedOnlyAfterTwoConsecutivePasses() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 300L, STARTED);   // status RUNNING
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        when(manager.isServerRunning(projectId)).thenReturn(false);   // FL server not tracked (crashed)
        when(projectRepo.findById(projectId)).thenReturn(Optional.of(new Project()));

        // Pass 1: first time seen stuck -> suspected, NOT yet reaped (a mid-completion/stop run gets to settle).
        int reapedPass1 = reconciler.sweepStuckRuns();
        assertEquals(0, reapedPass1);
        verify(runService, never()).markFailed(any());

        // Pass 2: still RUNNING + still untracked -> genuinely stuck -> reaped exactly once.
        int reapedPass2 = reconciler.sweepStuckRuns();
        assertEquals(1, reapedPass2);
        verify(runService, times(1)).markFailed(run.getId());
    }

    @Test
    void trackedRunningRun_isNeverReaped() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 301L, STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        when(manager.isServerRunning(projectId)).thenReturn(true);   // a live tracked FL server

        reconciler.sweepStuckRuns();
        reconciler.sweepStuckRuns();

        verify(runService, never()).markFailed(any());
    }

    @Test
    void runThatRecoversBetweenPasses_isNotReaped() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 302L, STARTED);
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        // Pass 1 sees it untracked (suspected); pass 2 sees it tracked again (transient blip).
        when(manager.isServerRunning(projectId)).thenReturn(false).thenReturn(true);

        reconciler.sweepStuckRuns();
        reconciler.sweepStuckRuns();

        verify(runService, never()).markFailed(any());   // recovery clears the suspicion
    }

    @Test
    void runThatLeavesRunningBetweenPasses_isNotReaped() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 303L, STARTED);
        // Pass 1: run is RUNNING (in the query) but untracked -> suspected. Pass 2: it has completed/stopped,
        // so it is no longer RUNNING and drops out of the query -> never reaped (no FAILED-clobbers-COMPLETED).
        when(runRepo.findByStatusIn(anyCollection()))
                .thenReturn(List.of(run))
                .thenReturn(List.of());
        when(manager.isServerRunning(projectId)).thenReturn(false);

        reconciler.sweepStuckRuns();
        reconciler.sweepStuckRuns();

        verify(runService, never()).markFailed(any());
    }
}
