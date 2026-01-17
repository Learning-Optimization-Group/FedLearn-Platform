package com.federated.fl_platform_api.dto;

import lombok.Data;

import java.util.UUID;


public class RoundResultDto {
    private UUID id;
    private Integer serverRound;
    private Double loss;
    private Double accuracy;
    private Double gpuUtilization;

    public Integer getServerRound() {
        return serverRound;
    }

    public void setServerRound(Integer serverRound) {
        this.serverRound = serverRound;
    }

    public Double getLoss() {
        return loss;
    }

    public void setLoss(Double loss) {
        this.loss = loss;
    }

    public Double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }

    public Double getGpuUtilization() {
        return gpuUtilization;
    }

    public void setGpuUtilization(Double gpuUtilization) {
        this.gpuUtilization = gpuUtilization;
    }

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }
}
