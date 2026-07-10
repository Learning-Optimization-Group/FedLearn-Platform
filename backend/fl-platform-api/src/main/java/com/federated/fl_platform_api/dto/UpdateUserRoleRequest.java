package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.validation.ValueOfEnum;
import jakarta.validation.constraints.NotNull;

public class UpdateUserRoleRequest {
    // Validated against the PlatformRole enum so the accepted set can never drift from the
    // source of truth (BA-15). @NotNull keeps the field required.
    @NotNull
    @ValueOfEnum(enumClass = PlatformRole.class)
    private String role;
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
}
