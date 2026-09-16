package com.federated.fl_platform_api.dto;

import java.time.Instant;
import java.util.UUID;

public class EnrollmentDto {
    private UUID runId;
    private UUID projectId;
    private String grpcEndpoint;
    private int partitionId;
    private String clientKind;
    private String caFingerprint;   // sha256 of the FL server's certificate when TLS is required; null otherwise
    private String connectionToken;
    // SE-12: per-client mTLS cert issued at enrollment (null unless feature.fl-client-cert.enabled). The
    // client presents these to the FL gRPC server when require_client_auth is on; the key never leaves it.
    private String clientCertPem;
    private String clientKeyPem;
    // The SE-12 client-cert bundle's issuing CA, kept apart from caFingerprint, which describes the FL server.
    private String clientCaFingerprint;
    // Server trust: whether the FL server requires TLS, and the certificate to verify it with (null on a plaintext
    // deployment, or when none is configured and the client verifies against its system roots).
    private boolean grpcTls;
    private String grpcServerCertPem;
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
    public String getClientCertPem() { return clientCertPem; }
    public void setClientCertPem(String clientCertPem) { this.clientCertPem = clientCertPem; }
    public String getClientKeyPem() { return clientKeyPem; }
    public void setClientKeyPem(String clientKeyPem) { this.clientKeyPem = clientKeyPem; }
    public String getClientCaFingerprint() { return clientCaFingerprint; }
    public void setClientCaFingerprint(String clientCaFingerprint) { this.clientCaFingerprint = clientCaFingerprint; }
    public boolean isGrpcTls() { return grpcTls; }
    public void setGrpcTls(boolean grpcTls) { this.grpcTls = grpcTls; }
    public String getGrpcServerCertPem() { return grpcServerCertPem; }
    public void setGrpcServerCertPem(String grpcServerCertPem) { this.grpcServerCertPem = grpcServerCertPem; }
    public Instant getExpiresAt() { return expiresAt; }
    public void setExpiresAt(Instant expiresAt) { this.expiresAt = expiresAt; }
    public RunManifestDto getManifest() { return manifest; }
    public void setManifest(RunManifestDto manifest) { this.manifest = manifest; }
}
