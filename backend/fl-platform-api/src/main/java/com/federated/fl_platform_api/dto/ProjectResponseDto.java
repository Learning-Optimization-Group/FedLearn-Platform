package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class ProjectResponseDto {

    private UUID id;
    private String name;
    private String modelType;
    private String modelName;
    private Integer serverPort;
    private String optimizer;
    private String status;
    private String myRelationship;  // "OWNER" | "MEMBER" | "CLIENT" | null
    private String visibility;       // "PUBLIC" | "RESTRICTED" | "PRIVATE"
    // Populated only by the admin all-projects view (null elsewhere): lets the
    // admin see who owns each project and how many participants it has.
    private String ownerUsername;
    private Integer participantCount;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Integer getServerPort() {
        return serverPort;
    }

    public void setServerPort(Integer serverPort) {
        this.serverPort = serverPort;
    }

    public String getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(String optimizer) {
        this.optimizer = optimizer;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getMyRelationship() {
        return myRelationship;
    }

    public void setMyRelationship(String myRelationship) {
        this.myRelationship = myRelationship;
    }

    public String getVisibility() {
        return visibility;
    }

    public void setVisibility(String visibility) {
        this.visibility = visibility;
    }

    public String getOwnerUsername() {
        return ownerUsername;
    }

    public void setOwnerUsername(String ownerUsername) {
        this.ownerUsername = ownerUsername;
    }

    public Integer getParticipantCount() {
        return participantCount;
    }

    public void setParticipantCount(Integer participantCount) {
        this.participantCount = participantCount;
    }
}
