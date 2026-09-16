package com.federated.fl_platform_api.dto;

import java.util.UUID;

/**
 * The resolved, enrolled identity a valid connection token carries — returned on a successful
 * {@code POST /api/internal/runs/{projectId}/{runId}/verify-connection-token}. Lets the FL
 * server bind an accepted gRPC client to exactly one enrollment (user + partition + client kind)
 * without trusting anything the client asserts on the wire.
 */
public class ConnectionTokenVerificationDto {

    private Long userId;
    private UUID runId;
    private UUID projectId;
    private int partitionId;
    private String clientKind;
    private String grpcEndpoint;

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }

    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }

    public int getPartitionId() { return partitionId; }
    public void setPartitionId(int partitionId) { this.partitionId = partitionId; }

    public String getClientKind() { return clientKind; }
    public void setClientKind(String clientKind) { this.clientKind = clientKind; }

    public String getGrpcEndpoint() { return grpcEndpoint; }
    public void setGrpcEndpoint(String grpcEndpoint) { this.grpcEndpoint = grpcEndpoint; }
}
