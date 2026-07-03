package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.repository.RunRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.*;

/**
 * BA-3: at spawn the manager records the child's OS identity (PID + start instant) and reserved port
 * on the active Run so a startup reconciler can later reap orphans. Unit-level (no spawn, no Spring):
 * the Process is mocked, so this covers the write logic and its guards in isolation.
 */
@ExtendWith(MockitoExtension.class)
class FlowerServerManagerProcessIdentityTest {

    private FlowerServerManager managerWith(RunRepository repo) {
        FlowerServerManager m = new FlowerServerManager();
        ReflectionTestUtils.setField(m, "runRepository", repo);
        return m;
    }

    private Process processWith(long pid, Instant startInstant) {
        Process p = mock(Process.class);
        when(p.pid()).thenReturn(pid);
        ProcessHandle.Info info = mock(ProcessHandle.Info.class);
        when(p.info()).thenReturn(info);
        when(info.startInstant()).thenReturn(Optional.of(startInstant));
        return p;
    }

    @Test
    void recordsPidStartInstantAndPortOnTheActiveRun() {
        RunRepository repo = mock(RunRepository.class);
        UUID runId = UUID.randomUUID();
        Run run = new Run();
        when(repo.findById(runId)).thenReturn(Optional.of(run));

        Instant started = Instant.parse("2026-07-03T12:00:00Z");
        managerWith(repo).recordProcessIdentity(runId, processWith(4242L, started), 50005);

        assertEquals(4242L, run.getServerPid());
        assertEquals(started, run.getProcessStartedAt());
        assertEquals(50005, run.getServerPort());
        verify(repo).save(run);
    }

    @Test
    void nullRunId_isANoOp() {
        RunRepository repo = mock(RunRepository.class);
        managerWith(repo).recordProcessIdentity(null, mock(Process.class), 50005);
        verifyNoInteractions(repo);
    }

    @Test
    void missingRun_doesNotSave() {
        RunRepository repo = mock(RunRepository.class);
        UUID runId = UUID.randomUUID();
        when(repo.findById(runId)).thenReturn(Optional.empty());

        // Process is never touched when the run is gone — so leave it unstubbed.
        managerWith(repo).recordProcessIdentity(runId, mock(Process.class), 50005);

        verify(repo, never()).save(any());
    }

    @Test
    void persistenceFailure_propagates_soTheSpawnCanFailClosed() {
        // A live server whose identity can't be persisted would be an unreconcilable orphan, so the
        // failure must NOT be swallowed — it propagates and the spawn path kills the child + fails.
        RunRepository repo = mock(RunRepository.class);
        UUID runId = UUID.randomUUID();
        Run run = new Run();
        when(repo.findById(runId)).thenReturn(Optional.of(run));
        doThrow(new RuntimeException("db down")).when(repo).save(run);

        assertThrows(RuntimeException.class, () -> managerWith(repo)
                .recordProcessIdentity(runId, processWith(7L, Instant.parse("2026-07-03T12:00:00Z")), 50007));
    }
}
