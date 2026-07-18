package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

/**
 * Read model for the admin audit-event explorer. Mirrors
 * {@link com.federated.fl_platform_api.model.AuditEvent} with the actor's
 * username resolved server-side (batched per page — no N+1). {@code metadata}
 * is the raw JSON string stored on the row.
 */
public class AuditEventDto {

    private UUID id;
    private Instant occurredAt;
    private Long actorUserId;
    private String actorUsername;
    private String action;
    private String targetType;
    private String targetId;
    private String requestIp;
    private String metadata;

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public Instant getOccurredAt() { return occurredAt; }
    public void setOccurredAt(Instant occurredAt) { this.occurredAt = occurredAt; }
    public Long getActorUserId() { return actorUserId; }
    public void setActorUserId(Long actorUserId) { this.actorUserId = actorUserId; }
    public String getActorUsername() { return actorUsername; }
    public void setActorUsername(String actorUsername) { this.actorUsername = actorUsername; }
    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }
    public String getTargetType() { return targetType; }
    public void setTargetType(String targetType) { this.targetType = targetType; }
    public String getTargetId() { return targetId; }
    public void setTargetId(String targetId) { this.targetId = targetId; }
    public String getRequestIp() { return requestIp; }
    public void setRequestIp(String requestIp) { this.requestIp = requestIp; }
    public String getMetadata() { return metadata; }
    public void setMetadata(String metadata) { this.metadata = metadata; }
}
