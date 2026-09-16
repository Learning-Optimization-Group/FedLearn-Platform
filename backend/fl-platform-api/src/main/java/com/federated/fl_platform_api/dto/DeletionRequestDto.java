package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

/** Read model for a project-deletion request (owner-facing + admin queue). */
public class DeletionRequestDto {
    private Long id;
    private UUID projectId;
    private String projectName;
    private Long requestedById;
    private String requestedByUsername;
    private String status;            // PENDING | APPROVED | DENIED
    private String reason;
    private Instant requestedAt;
    private Instant decidedAt;
    private String decidedByUsername;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getProjectName() { return projectName; }
    public void setProjectName(String projectName) { this.projectName = projectName; }
    public Long getRequestedById() { return requestedById; }
    public void setRequestedById(Long requestedById) { this.requestedById = requestedById; }
    public String getRequestedByUsername() { return requestedByUsername; }
    public void setRequestedByUsername(String requestedByUsername) { this.requestedByUsername = requestedByUsername; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    public Instant getRequestedAt() { return requestedAt; }
    public void setRequestedAt(Instant requestedAt) { this.requestedAt = requestedAt; }
    public Instant getDecidedAt() { return decidedAt; }
    public void setDecidedAt(Instant decidedAt) { this.decidedAt = decidedAt; }
    public String getDecidedByUsername() { return decidedByUsername; }
    public void setDecidedByUsername(String decidedByUsername) { this.decidedByUsername = decidedByUsername; }
}
