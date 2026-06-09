package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public class UpdateUserRoleRequest {
    @NotNull
    @Pattern(regexp = "USER|PLATFORM_ADMIN")
    private String role;
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
}
