package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "run_execution_attempts",
       uniqueConstraints = @UniqueConstraint(name = "uq_run_attempt_number", columnNames = {"run_id", "attempt_number"}))
public class RunExecutionAttempt {

    @Id
    private UUID id;

    @Column(name = "run_id", nullable = false)
    private UUID runId;

    @Column(name = "attempt_number", nullable = false)
    private int attemptNumber;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private RunStatus status;

    @Column(name = "server_pid")
    private Long serverPid;

    @Column(name = "process_started_at")
    private Instant processStartedAt;

    @Column(name = "server_port")
    private Integer serverPort;

    @Column(name = "internal_token_hash", length = 64)
    private String internalTokenHash;

    @Column(name = "started_at")
    private Instant startedAt;

    @Column(name = "ended_at")
    private Instant endedAt;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "failure_reason")
    private String failureReason;

    public RunExecutionAttempt() {}

    public RunExecutionAttempt(UUID id, UUID runId, int attemptNumber, RunStatus status) {
        this.id = id;
        this.runId = runId;
        this.attemptNumber = attemptNumber;
        this.status = status;
        this.createdAt = Instant.now();
    }

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }

    public int getAttemptNumber() { return attemptNumber; }
    public void setAttemptNumber(int attemptNumber) { this.attemptNumber = attemptNumber; }

    public RunStatus getStatus() { return status; }
    public void setStatus(RunStatus status) { this.status = status; }

    public Long getServerPid() { return serverPid; }
    public void setServerPid(Long serverPid) { this.serverPid = serverPid; }

    public Instant getProcessStartedAt() { return processStartedAt; }
    public void setProcessStartedAt(Instant processStartedAt) { this.processStartedAt = processStartedAt; }

    public Integer getServerPort() { return serverPort; }
    public void setServerPort(Integer serverPort) { this.serverPort = serverPort; }

    public String getInternalTokenHash() { return internalTokenHash; }
    public void setInternalTokenHash(String internalTokenHash) { this.internalTokenHash = internalTokenHash; }

    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }

    public Instant getEndedAt() { return endedAt; }
    public void setEndedAt(Instant endedAt) { this.endedAt = endedAt; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }

    public String getFailureReason() { return failureReason; }
    public void setFailureReason(String failureReason) { this.failureReason = failureReason; }
}
