package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

public class NotificationDto {

    public enum Type {
        ACCESS_REQUEST_CREATED,
        ACCESS_REQUEST_DECIDED,
        MEMBERSHIP_ADDED,
        MEMBERSHIP_REMOVED,
        PROJECT_VISIBILITY_CHANGED,
        OWNER_PROMOTION_REQUESTED,
        OWNER_PROMOTION_DECIDED,
        PROJECT_DELETION_REQUESTED,
        PROJECT_DELETION_DECIDED
    }

    private UUID id = UUID.randomUUID();
    private Type type;
    private UUID projectId;
    private String projectName;
    private Long actorId;
    private String actorUsername;
    private Long subjectId;
    private String subjectUsername;
    private String decision;   // 'APPROVED' / 'DENIED' for ACCESS_REQUEST_DECIDED only
    private String role;       // 'MEMBER' / 'CLIENT' for MEMBERSHIP_* only
    private Instant timestamp = Instant.now();

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public Type getType() { return type; }
    public void setType(Type type) { this.type = type; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getProjectName() { return projectName; }
    public void setProjectName(String projectName) { this.projectName = projectName; }
    public Long getActorId() { return actorId; }
    public void setActorId(Long actorId) { this.actorId = actorId; }
    public String getActorUsername() { return actorUsername; }
    public void setActorUsername(String actorUsername) { this.actorUsername = actorUsername; }
    public Long getSubjectId() { return subjectId; }
    public void setSubjectId(Long subjectId) { this.subjectId = subjectId; }
    public String getSubjectUsername() { return subjectUsername; }
    public void setSubjectUsername(String subjectUsername) { this.subjectUsername = subjectUsername; }
    public String getDecision() { return decision; }
    public void setDecision(String decision) { this.decision = decision; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }
}
