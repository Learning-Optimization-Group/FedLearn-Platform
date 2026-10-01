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

    /** The device's capability report as JSON (V31); informational, never an approval to train. */
    @org.hibernate.annotations.JdbcTypeCode(org.hibernate.type.SqlTypes.JSON)
    @Column(name = "capability_report", columnDefinition = "jsonb")
    private String capabilityReport;

    @Column(name = "capability_reported_at")
    private Instant capabilityReportedAt;

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

    public String getCapabilityReport() {
        return capabilityReport;
    }

    public Instant getCapabilityReportedAt() {
        return capabilityReportedAt;
    }

    /** Records the report and when it arrived; a null report clears both. */
    public void setCapabilityReport(String report, Instant reportedAt) {
        this.capabilityReport = report;
        this.capabilityReportedAt = report == null ? null : reportedAt;
    }
}
