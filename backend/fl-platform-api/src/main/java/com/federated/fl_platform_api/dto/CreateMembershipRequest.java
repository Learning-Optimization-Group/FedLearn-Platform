package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.validation.ValueOfEnum;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class CreateMembershipRequest {
    @NotBlank
    private String username;

    // Validated against MembershipRole minus OWNER (BA-15): a member may be added as
    // MEMBER or CLIENT via this endpoint, but project ownership (OWNER) is never grantable
    // here. Enum-derived so a newly added role is auto-accepted; OWNER stays excluded.
    @NotNull
    @ValueOfEnum(enumClass = MembershipRole.class, exclude = {"OWNER"})
    private String role;

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
}
