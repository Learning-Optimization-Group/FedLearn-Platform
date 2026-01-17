package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import lombok.Data;

import java.util.UUID;

@Entity

public class RoundResult {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne
    @JoinColumn(name = "project_id",nullable = false)
    private Project project;

    @Column(nullable = false)
    private Integer serverRound;

    private Double loss;
    private Double accuracy;
    private Double gpuUtilization;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public Project getProject() {
        return project;
    }

    public void setProject(Project project) {
        this.project = project;
    }

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
}
