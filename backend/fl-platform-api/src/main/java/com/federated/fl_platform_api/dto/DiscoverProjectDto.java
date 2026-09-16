package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class DiscoverProjectDto {
    private UUID id;
    private String name;
    private String visibility;
    private String ownerUsername;
    private String modelType;
    private String myRequestStatus;   // NONE | PENDING | APPROVED | DENIED
    private Double lastAccuracy;
    private String description;

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
    public String getOwnerUsername() { return ownerUsername; }
    public void setOwnerUsername(String ownerUsername) { this.ownerUsername = ownerUsername; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public String getMyRequestStatus() { return myRequestStatus; }
    public void setMyRequestStatus(String myRequestStatus) { this.myRequestStatus = myRequestStatus; }
    public Double getLastAccuracy() { return lastAccuracy; }
    public void setLastAccuracy(Double lastAccuracy) { this.lastAccuracy = lastAccuracy; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}
