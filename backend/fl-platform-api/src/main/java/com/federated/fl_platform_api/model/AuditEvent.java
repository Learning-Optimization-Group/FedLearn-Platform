package com.federated.fl_platform_api.model;

import jakarta.persistence.*;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "audit_events")
public class AuditEvent {

    @Id
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Column(name = "occurred_at", nullable = false, updatable = false)
    private Instant occurredAt;

    @Column(name = "actor_user_id")
    private Long actorUserId;

    @Column(name = "org_id")
    private UUID orgId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 64)
    private AuditAction action;

    @Column(name = "target_type", length = 32)
    private String targetType;

    @Column(name = "target_id", length = 64)
    private String targetId;

    @Lob
    @Column
    private String metadata;        // JSON string; not indexable until Postgres + JSONB

    @Column(name = "request_ip", length = 45)
    private String requestIp;

    @Column(name = "user_agent", length = 256)
    private String userAgent;

    protected AuditEvent() { /* JPA */ }

    public static Builder builder() { return new Builder(); }

    public UUID getId() { return id; }
    public Instant getOccurredAt() { return occurredAt; }
    public Long getActorUserId() { return actorUserId; }
    public UUID getOrgId() { return orgId; }
    public AuditAction getAction() { return action; }
    public String getTargetType() { return targetType; }
    public String getTargetId() { return targetId; }
    public String getMetadata() { return metadata; }
    public String getRequestIp() { return requestIp; }
    public String getUserAgent() { return userAgent; }

    public static class Builder {
        private final AuditEvent e = new AuditEvent();
        public Builder() { e.id = UUID.randomUUID(); e.occurredAt = Instant.now(); }
        public Builder action(AuditAction a)     { e.action = a; return this; }
        public Builder actorUserId(Long u)       { e.actorUserId = u; return this; }
        public Builder orgId(UUID o)             { e.orgId = o; return this; }
        public Builder targetType(String t)      { e.targetType = t; return this; }
        public Builder targetId(String t)        { e.targetId = t; return this; }
        public Builder metadata(String json)     { e.metadata = json; return this; }
        public Builder requestIp(String ip)      { e.requestIp = ip; return this; }
        public Builder userAgent(String ua)      { e.userAgent = ua; return this; }
        public AuditEvent build() {
            if (e.action == null) throw new IllegalStateException("action required");
            return e;
        }
    }
}
