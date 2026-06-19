package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

/**
 * A user's request to be promoted from {@link PlatformRole#USER} to
 * {@link PlatformRole#PROJECT_OWNER}. Decided by a platform admin. One row per
 * user (a re-request after a DENY updates the same row), mirroring
 * {@link ProjectAccessRequest}.
 */
@Entity
@Table(
    name = "owner_promotion_requests",
    uniqueConstraints = @UniqueConstraint(columnNames = {"user_id"})
)
public class OwnerPromotionRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

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

    public OwnerPromotionRequest() {}

    public OwnerPromotionRequest(User user, String message) {
        this.user = user;
        this.message = message;
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
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }
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
