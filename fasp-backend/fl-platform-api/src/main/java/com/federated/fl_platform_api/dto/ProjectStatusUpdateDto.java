package com.federated.fl_platform_api.dto;

import java.util.UUID;

public class ProjectStatusUpdateDto {

    private UUID projectId;
    private String newStatus;
    private Integer serverPort;

    public UUID getProjectId() {
        return projectId;
    }

    public void setProjectId(UUID projectId) {
        this.projectId = projectId;
    }

    public String getNewStatus() {
        return newStatus;
    }

    public void setNewStatus(String newStatus) {
        this.newStatus = newStatus;
    }

    public Integer getServerPort() {
        return serverPort;
    }

    public void setServerPort(Integer serverPort) {
        this.serverPort = serverPort;
    }

    public ProjectStatusUpdateDto(UUID projectId, String newStatus, Integer serverPort) {
        this.projectId = projectId;
        this.newStatus = newStatus;
        this.serverPort = serverPort;
    }


}
