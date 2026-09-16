package com.federated.fl_platform_api.dto;

import java.time.Instant;

/** Read model for an owner-promotion request (user-facing + admin queue). */
public class OwnerRequestDto {
    private Long id;
    private Long userId;
    private String username;
    private String email;
    private String status;            // PENDING | APPROVED | DENIED
    private String message;
    private Instant requestedAt;
    private Instant decidedAt;
    private String decidedByUsername;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
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
