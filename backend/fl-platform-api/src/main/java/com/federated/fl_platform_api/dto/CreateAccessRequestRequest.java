package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Size;

public class CreateAccessRequestRequest {
    @Size(max = 1000)
    private String message;

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}
