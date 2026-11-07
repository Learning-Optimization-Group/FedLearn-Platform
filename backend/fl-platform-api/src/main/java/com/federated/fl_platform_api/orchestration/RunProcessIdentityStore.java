package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.RunExecutionAttemptRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import jakarta.persistence.EntityManager;
import jakarta.persistence.LockModeType;
import jakarta.persistence.PersistenceContext;
import java.time.Instant;
import java.util.Objects;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

/** Commit launch authorization and OS identity independently of a startup request's cached entities. */
@Service
public class RunProcessIdentityStore {
    private final RunRepository runs;
    private final RunExecutionAttemptRepository attemptRepository;
    @PersistenceContext private EntityManager entityManager;

    @Autowired
    public RunProcessIdentityStore(RunRepository runs, @Autowired(required = false) RunExecutionAttemptRepository attemptRepository) {
        this.runs = runs;
        this.attemptRepository = attemptRepository;
    }

    public RunProcessIdentityStore(RunRepository runs) {
        this(runs, null);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void authorizeLaunch(UUID runId, UUID attemptId, String tokenHash) {
        Run run = lockCurrent(runId);
        if (run == null) return;
        if (run.getStatus() != RunStatus.STARTING) {
            throw new IllegalStateException("Only a starting run may authorize a process launch");
        }
        if (attemptId != null && run.getCurrentAttemptId() != null
                && !attemptId.equals(run.getCurrentAttemptId())) {
            throw new IllegalStateException("Attempt " + attemptId + " is not the current attempt for run " + runId);
        }
        if (tokenHash == null || tokenHash.isBlank()) {
            throw new IllegalArgumentException("Process authorization hash is required");
        }
        run.setInternalTokenHash(tokenHash);
        runs.saveAndFlush(run);

        UUID activeAttemptId = attemptId != null ? attemptId : run.getCurrentAttemptId();
        if (activeAttemptId != null && attemptRepository != null) {
            attemptRepository.findById(activeAttemptId).ifPresent(attempt -> {
                attempt.setInternalTokenHash(tokenHash);
                attemptRepository.saveAndFlush(attempt);
            });
        }
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void authorizeLaunch(UUID runId, String tokenHash) {
        authorizeLaunch(runId, null, tokenHash);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void recordProcessIdentity(UUID runId, UUID attemptId, long pid, Instant startedAt, int port, String tokenHash) {
        Run run = lockCurrent(runId);
        if (run == null || run.getStatus() == RunStatus.COMPLETED) return;
        if (run.getStatus() != RunStatus.STARTING && run.getStatus() != RunStatus.RUNNING) {
            throw new IllegalStateException("Run no longer accepts a process identity");
        }
        if (attemptId != null && run.getCurrentAttemptId() != null
                && !attemptId.equals(run.getCurrentAttemptId())) {
            throw new IllegalStateException("Attempt " + attemptId + " is not the current attempt for run " + runId);
        }
        if (tokenHash == null || !Objects.equals(tokenHash, run.getInternalTokenHash())) {
            throw new IllegalStateException("Process authorization has been retired");
        }
        run.setServerPid(pid);
        run.setProcessStartedAt(startedAt);
        run.setServerPort(port);
        runs.save(run);

        UUID activeAttemptId = attemptId != null ? attemptId : run.getCurrentAttemptId();
        if (activeAttemptId != null && attemptRepository != null) {
            attemptRepository.findById(activeAttemptId).ifPresent(attempt -> {
                attempt.setServerPid(pid);
                attempt.setProcessStartedAt(startedAt);
                attempt.setServerPort(port);
                attemptRepository.save(attempt);
            });
        }
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void recordProcessIdentity(UUID runId, long pid, Instant startedAt, int port, String tokenHash) {
        recordProcessIdentity(runId, null, pid, startedAt, port, tokenHash);
    }

    private Run lockCurrent(UUID runId) {
        if (runId == null) return null;
        Run run = runs.lockById(runId).orElse(null);
        if (run != null) entityManager.refresh(run, LockModeType.PESSIMISTIC_WRITE);
        return run;
    }
}
