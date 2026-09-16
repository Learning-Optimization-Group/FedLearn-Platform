package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Size;

/** Body for POST /api/projects/{id}/deletion-request — owner's reason for deletion. */
public class CreateDeletionRequestRequest {
    @Size(max = 1000)
    private String reason;

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
}
