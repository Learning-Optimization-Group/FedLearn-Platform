package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public class CreateMembershipRequest {
    @NotBlank
    private String username;

    @NotNull
    @Pattern(regexp = "MEMBER|CLIENT")
    private String role;

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
}
