package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

public class EnrollmentDto {
    private UUID runId;
    private UUID projectId;
    private String grpcEndpoint;
    private int partitionId;
    private String clientKind;
    private String caFingerprint;   // null in Phase 1
    private String connectionToken;
    private Instant expiresAt;
    private RunManifestDto manifest;

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getGrpcEndpoint() { return grpcEndpoint; }
    public void setGrpcEndpoint(String grpcEndpoint) { this.grpcEndpoint = grpcEndpoint; }
    public int getPartitionId() { return partitionId; }
    public void setPartitionId(int partitionId) { this.partitionId = partitionId; }
    public String getClientKind() { return clientKind; }
    public void setClientKind(String clientKind) { this.clientKind = clientKind; }
    public String getCaFingerprint() { return caFingerprint; }
    public void setCaFingerprint(String caFingerprint) { this.caFingerprint = caFingerprint; }
    public String getConnectionToken() { return connectionToken; }
    public void setConnectionToken(String connectionToken) { this.connectionToken = connectionToken; }
    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
    public RunManifestDto getManifest() { return manifest; }
    public void setManifest(RunManifestDto manifest) { this.manifest = manifest; }
}
