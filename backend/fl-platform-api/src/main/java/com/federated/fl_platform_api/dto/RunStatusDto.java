package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class RunStatusDto {
    private UUID runId;
    private String status;
    private String grpcEndpoint;   // null unless RUNNING
    private String caFingerprint;  // null in Phase 1

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getGrpcEndpoint() { return grpcEndpoint; }
    public void setGrpcEndpoint(String grpcEndpoint) { this.grpcEndpoint = grpcEndpoint; }
    public String getCaFingerprint() { return caFingerprint; }
    public void setCaFingerprint(String caFingerprint) { this.caFingerprint = caFingerprint; }
}
