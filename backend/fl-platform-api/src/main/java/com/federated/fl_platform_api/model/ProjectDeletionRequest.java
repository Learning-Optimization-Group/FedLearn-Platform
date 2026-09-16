package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

/**
 * An owner's request to delete a project. The project is parked in
 * {@code PENDING_DELETION} status (and its FL server stopped) when the request
 * is filed; a platform admin then approves (the project is hard-deleted) or
 * denies (the prior status is restored). One row per project.
 */
@Entity
@Table(
    name = "project_deletion_requests",
    uniqueConstraints = @UniqueConstraint(columnNames = {"project_id"})
)
public class ProjectDeletionRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "project_id", nullable = false)
    private Project project;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "requested_by", nullable = false)
    private User requestedBy;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private AccessRequestStatus status;

    @Column(columnDefinition = "TEXT")
    private String reason;

    @Column(name = "requested_at", nullable = false)
    private Instant requestedAt;

    @Column(name = "decided_at")
    private Instant decidedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "decided_by")
    private User decidedBy;

    public ProjectDeletionRequest() {}

    public ProjectDeletionRequest(Project project, User requestedBy, String reason) {
        this.project = project;
        this.requestedBy = requestedBy;
        this.reason = reason;
        this.status = AccessRequestStatus.PENDING;
        this.requestedAt = Instant.now();
    }

    @PrePersist
    void prePersist() {
        if (requestedAt == null) requestedAt = Instant.now();
        if (status == null) status = AccessRequestStatus.PENDING;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Project getProject() { return project; }
    public void setProject(Project project) { this.project = project; }
    public User getRequestedBy() { return requestedBy; }
    public void setRequestedBy(User requestedBy) { this.requestedBy = requestedBy; }
    public AccessRequestStatus getStatus() { return status; }
    public void setStatus(AccessRequestStatus status) { this.status = status; }
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    public Instant getRequestedAt() { return requestedAt; }
    public void setRequestedAt(Instant requestedAt) { this.requestedAt = requestedAt; }
    public Instant getDecidedAt() { return decidedAt; }
    public void setDecidedAt(Instant decidedAt) { this.decidedAt = decidedAt; }
    public User getDecidedBy() { return decidedBy; }
    public void setDecidedBy(User decidedBy) { this.decidedBy = decidedBy; }
}
