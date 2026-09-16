package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "run_enrollments")
public class RunEnrollment {

    @EmbeddedId
    private RunEnrollmentId id;

    @Column(name = "partition_id", nullable = false)
    private int partitionId;

    @Enumerated(EnumType.STRING)
    @Column(name = "client_kind", nullable = false, length = 16)
    private ClientKind clientKind;

    @Column(name = "enrolled_at", nullable = false)
    private Instant enrolledAt;

    @Column(name = "token_issued_at")
    private Instant tokenIssuedAt;

    public RunEnrollment() {}

    public RunEnrollment(RunEnrollmentId id, int partitionId, ClientKind clientKind, Instant enrolledAt) {
        this.id = id;
        this.partitionId = partitionId;
        this.clientKind = clientKind;
        this.enrolledAt = enrolledAt;
    }

    public RunEnrollmentId getId() { return id; }
    public void setId(RunEnrollmentId id) { this.id = id; }
    public int getPartitionId() { return partitionId; }
    public void setPartitionId(int partitionId) { this.partitionId = partitionId; }
    public ClientKind getClientKind() { return clientKind; }
    public void setClientKind(ClientKind clientKind) { this.clientKind = clientKind; }
    public Instant getEnrolledAt() { return enrolledAt; }
    public void setEnrolledAt(Instant enrolledAt) { this.enrolledAt = enrolledAt; }
    public Instant getTokenIssuedAt() { return tokenIssuedAt; }
    public void setTokenIssuedAt(Instant tokenIssuedAt) { this.tokenIssuedAt = tokenIssuedAt; }
}
