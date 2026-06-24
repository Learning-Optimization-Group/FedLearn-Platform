package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class ActiveRunDto {
    private UUID runId;
    private String status;

    public ActiveRunDto() {}

    public ActiveRunDto(UUID runId, String status) {
        this.runId = runId;
        this.status = status;
    }

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
