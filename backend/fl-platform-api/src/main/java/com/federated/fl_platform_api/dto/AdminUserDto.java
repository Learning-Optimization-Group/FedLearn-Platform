package com.federated.fl_platform_api.dto;

import java.time.Instant;

public class AdminUserDto {
    private Long id;
    private String username;
    private String email;
    private String role;
    private long projectsOwned;
    private long memberships;
    private Instant createdAt;
    // Additive fields for the paginated admin directory (all nullable-safe).
    private String status;
    private String displayName;
    private Instant lastLoginAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public long getProjectsOwned() { return projectsOwned; }
    public void setProjectsOwned(long projectsOwned) { this.projectsOwned = projectsOwned; }
    public long getMemberships() { return memberships; }
    public void setMemberships(long memberships) { this.memberships = memberships; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getDisplayName() { return displayName; }
    public void setDisplayName(String displayName) { this.displayName = displayName; }
    public Instant getLastLoginAt() { return lastLoginAt; }
    public void setLastLoginAt(Instant lastLoginAt) { this.lastLoginAt = lastLoginAt; }
}
