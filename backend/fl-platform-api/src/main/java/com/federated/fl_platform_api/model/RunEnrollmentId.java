package com.federated.fl_platform_api.model;

import jakarta.persistence.Column;
import jakarta.persistence.Embeddable;
import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

@Embeddable
public class RunEnrollmentId implements Serializable {

    @Column(name = "run_id")
    private UUID runId;

    @Column(name = "user_id")
    private Long userId;

    public RunEnrollmentId() {}

    public RunEnrollmentId(UUID runId, Long userId) {
        this.runId = runId;
        this.userId = userId;
    }

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RunEnrollmentId that)) return false;
        return Objects.equals(runId, that.runId) && Objects.equals(userId, that.userId);
    }
    @Override public int hashCode() { return Objects.hash(runId, userId); }
}
