package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Size;

/** Body for POST /api/owner-requests — a user's request to become a project owner. */
public class CreateOwnerRequestRequest {
    @Size(max = 1000)
    private String message;

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}
