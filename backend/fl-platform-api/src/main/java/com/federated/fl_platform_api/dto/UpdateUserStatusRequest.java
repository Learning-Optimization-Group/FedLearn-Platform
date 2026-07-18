package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.validation.ValueOfEnum;
import jakarta.validation.constraints.NotNull;

public class UpdateUserStatusRequest {
    // Enum-derived accepted set (BA-15). PENDING is the initial registration
    // state and can never be (re)entered through the admin status endpoint,
    // so it is excluded the same way an access decision excludes PENDING.
    @NotNull
    @ValueOfEnum(enumClass = UserStatus.class, exclude = "PENDING")
    private String status;

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
