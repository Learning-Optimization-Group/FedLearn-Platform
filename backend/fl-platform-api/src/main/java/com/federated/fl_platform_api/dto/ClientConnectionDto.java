package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class ClientConnectionDto {
    private UUID projectId;
    private String name;
    private String modelType;
    private String serverAddress;
    private Integer partitionId;
    private String status;
    private String connectionToken;
    // The running aggregation strategy (from the active Run). The desktop client threads this into
    // fl-runtime/client.py's --strategy so the client picks the matching path (e.g. DeComFL) instead
    // of always defaulting to the FedAvg path — otherwise a DeComFL project silently mismatches for
    // any non-MLP model type.
    private String strategy;
    // The project's training arm (FULL / FROZEN_HEAD). Same rationale as `strategy` above: the
    // server filters its parameters to the arm's trainable subset, so a client that does not know
    // the arm uploads the FULL state dict against a server expecting the head only. Carried here
    // rather than inferred, because the arm is a project property the client cannot derive.
    private String trainingArm;

    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public String getServerAddress() { return serverAddress; }
    public void setServerAddress(String serverAddress) { this.serverAddress = serverAddress; }
    public Integer getPartitionId() { return partitionId; }
    public void setPartitionId(Integer partitionId) { this.partitionId = partitionId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getConnectionToken() { return connectionToken; }
    public void setConnectionToken(String connectionToken) { this.connectionToken = connectionToken; }
    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }
    public String getTrainingArm() { return trainingArm; }
    public void setTrainingArm(String trainingArm) { this.trainingArm = trainingArm; }
}
