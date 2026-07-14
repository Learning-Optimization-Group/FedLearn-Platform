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
        ProcessHandle.Info info = mock(ProcessHandle.Info.class);
        when(handle.info()).thenReturn(info);
        when(info.startInstant()).thenReturn(Optional.ofNullable(startInstant));
        return handle;
    }

    @Test
    void liveIdentityMatchedRun_isReadopted() {
        UUID projectId = UUID.randomUUID();
        Run run = runWith(projectId, 100L, STARTED);
        run.setInternalTokenHash("hash-100");
        when(runRepo.findByStatusIn(anyCollection())).thenReturn(List.of(run));
        ProcessHandle handle = handleStartedAt(STARTED);   // exact match, within tolerance
        when(probe.of(100L)).thenReturn(Optional.of(handle));

        StartupReconciler.ReconciliationResult result = reconciler.reconcile();

        verify(manager).adopt(eq(projectId), eq(handle));
        // BA-3: the survivor's token is rehydrated so its callbacks keep authorizing after restart.
        verify(tokenRegistry).rehydrate("hash-100", new RunTokenRegistry.Scope(projectId, run.getId()));
        verify(runService, never()).markFailed(any());
        assertEquals(1, result.adopted());
        assertEquals(0, result.reaped());
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
    }
}
