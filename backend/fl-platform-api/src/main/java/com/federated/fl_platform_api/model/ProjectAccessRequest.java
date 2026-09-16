package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(
    name = "project_access_requests",
    uniqueConstraints = @UniqueConstraint(columnNames = {"project_id", "user_id"})
)
public class ProjectAccessRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "project_id", nullable = false)
    private Project project;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(name = "requested_role", nullable = false, length = 32)
    private MembershipRole requestedRole = MembershipRole.CLIENT;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private AccessRequestStatus status;

    @Column(columnDefinition = "TEXT")
    private String message;

    @Column(name = "requested_at", nullable = false)
    private Instant requestedAt;

    @Column(name = "decided_at")
    private Instant decidedAt;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "decided_by")
    private User decidedBy;

    public ProjectAccessRequest() {}

    public ProjectAccessRequest(Project project, User user, String message) {
        this.project = project;
        this.user = user;
        this.message = message;
        this.status = AccessRequestStatus.PENDING;
        this.requestedAt = Instant.now();
    }

    /**
     * Defensive default for callers that build the entity via the no-arg
     * constructor + setters and forget to set requestedAt. The column is
     * NOT NULL, so a missing value would otherwise produce a constraint
     * violation at flush — surfaced far from the bug site.
     */
    @PrePersist
    void prePersist() {
        if (requestedAt == null) requestedAt = Instant.now();
        if (status == null) status = AccessRequestStatus.PENDING;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Project getProject() { return project; }
    public void setProject(Project project) { this.project = project; }
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }
    public MembershipRole getRequestedRole() { return requestedRole; }
    public void setRequestedRole(MembershipRole requestedRole) { this.requestedRole = requestedRole; }
    public AccessRequestStatus getStatus() { return status; }
    public void setStatus(AccessRequestStatus status) { this.status = status; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public Instant getRequestedAt() { return requestedAt; }
    public void setRequestedAt(Instant requestedAt) { this.requestedAt = requestedAt; }
    public Instant getDecidedAt() { return decidedAt; }
    public void setDecidedAt(Instant decidedAt) { this.decidedAt = decidedAt; }
    public User getDecidedBy() { return decidedBy; }
    public void setDecidedBy(User decidedBy) { this.decidedBy = decidedBy; }
}
