package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.validation.ValueOfEnum;
import jakarta.validation.constraints.NotNull;

public class DecideAccessRequestRequest {
    // Validated against AccessRequestStatus minus PENDING (BA-15): a decision is a terminal
    // APPROVED or DENIED, never the initial not-yet-decided PENDING state. Enum-derived so it
    // can't drift from the status enum; PENDING stays excluded from the accepted set.
    @NotNull
    @ValueOfEnum(enumClass = AccessRequestStatus.class, exclude = {"PENDING"})
    private String decision;

    public String getDecision() { return decision; }
    public void setDecision(String decision) { this.decision = decision; }
}
