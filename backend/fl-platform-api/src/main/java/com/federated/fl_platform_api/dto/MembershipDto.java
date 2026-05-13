package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

public class MembershipDto {
    private UUID projectId;
    private Long userId;
    private String username;
    private String role;        // MEMBER | CLIENT | OWNER
    private Integer partitionId;
    private String joinedVia;
    private Instant addedAt;

    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public Integer getPartitionId() { return partitionId; }
    public void setPartitionId(Integer partitionId) { this.partitionId = partitionId; }
    public String getJoinedVia() { return joinedVia; }
    public void setJoinedVia(String joinedVia) { this.joinedVia = joinedVia; }
    public Instant getAddedAt() { return addedAt; }
    public void setAddedAt(Instant addedAt) { this.addedAt = addedAt; }
}
