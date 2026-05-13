package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class ClientConnectionDto {
    private UUID projectId;
    private String name;
    private String modelType;
    private String serverAddress;
    private Integer partitionId;
    private String status;

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
}
