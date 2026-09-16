package com.federated.fl_platform_api.orchestration;

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
 * on the active Run so a startup reconciler can later reap orphans. Unit-level (no spawn, no Spring).
 *
 * <p>DA-8: recordProcessIdentity now takes the identity values directly (pid + start instant) rather
 * than a {@link Process} — it is decoupled from the process seam, so these tests pass primitives
 * instead of mocking a Process/ProcessHandle.Info.</p>
 */
@ExtendWith(MockitoExtension.class)
class FlServerManagerProcessIdentityTest {

    private static final Instant STARTED = Instant.parse("2026-07-03T12:00:00Z");

    private FlServerManager managerWith(RunRepository repo) {
        FlServerManager m = new FlServerManager();
        ReflectionTestUtils.setField(m, "runRepository", repo);
        return m;
    }

    @Test
    void recordsPidStartInstantAndPortOnTheActiveRun() {
        RunRepository repo = mock(RunRepository.class);
        UUID runId = UUID.randomUUID();
        Run run = new Run();
        when(repo.findById(runId)).thenReturn(Optional.of(run));

        managerWith(repo).recordProcessIdentity(runId, 4242L, STARTED, 50005, "deadbeefhash");

        assertEquals(4242L, run.getServerPid());
        assertEquals(STARTED, run.getProcessStartedAt());
        assertEquals(50005, run.getServerPort());
        assertEquals("deadbeefhash", run.getInternalTokenHash());   // BA-3: persisted for token rehydration
        verify(repo).save(run);
    }

    @Test
    void nullRunId_isANoOp() {
        RunRepository repo = mock(RunRepository.class);
        managerWith(repo).recordProcessIdentity(null, 1L, STARTED, 50005, "hash");
        verifyNoInteractions(repo);
    }

    @Test
    void missingRun_doesNotSave() {
        RunRepository repo = mock(RunRepository.class);
        UUID runId = UUID.randomUUID();
        when(repo.findById(runId)).thenReturn(Optional.empty());

        managerWith(repo).recordProcessIdentity(runId, 1L, STARTED, 50005, "hash");

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
                .recordProcessIdentity(runId, 7L, STARTED, 50007, "hash"));
    }
}
