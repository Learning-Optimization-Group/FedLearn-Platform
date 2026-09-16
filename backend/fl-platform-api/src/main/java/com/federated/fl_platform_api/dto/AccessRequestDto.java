package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

public class AccessRequestDto {
    private Long id;
    private UUID projectId;
    private String projectName;
    private Long userId;
    private String username;
    private String status;
    private String message;
    private Instant requestedAt;
    private Instant decidedAt;
    private String decidedByUsername;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getProjectName() { return projectName; }
    public void setProjectName(String projectName) { this.projectName = projectName; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public Instant getRequestedAt() { return requestedAt; }
    public void setRequestedAt(Instant requestedAt) { this.requestedAt = requestedAt; }
    public Instant getDecidedAt() { return decidedAt; }
    public void setDecidedAt(Instant decidedAt) { this.decidedAt = decidedAt; }
    public String getDecidedByUsername() { return decidedByUsername; }
    public void setDecidedByUsername(String decidedByUsername) { this.decidedByUsername = decidedByUsername; }
}
