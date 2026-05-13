package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public class DecideAccessRequestRequest {
    @NotNull
    @Pattern(regexp = "APPROVED|DENIED")
    private String decision;

    public String getDecision() { return decision; }
    public void setDecision(String decision) { this.decision = decision; }
}
