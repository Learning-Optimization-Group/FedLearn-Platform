package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.util.UUID;

@Entity
@Table(uniqueConstraints = @UniqueConstraint(name = "uq_round_result_run_round", columnNames = {"run_id", "server_round"}))
public class RoundResult {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne
    @JoinColumn(name = "project_id",nullable = false)
    private Project project;

    // Historical project-only rows cannot be assigned to a run reliably.
    @Column(name = "run_id")
    private UUID runId;

    @Column(nullable = false)
    private Integer serverRound;

    private Double loss;
    private Double accuracy;
    private Double gpuUtilization;

    @Column(name = "time_data_loading_s")
    private Double timeDataLoadingS;

    @Column(name = "time_forward_backward_s")
    private Double timeForwardBackwardS;

    @Column(name = "time_optimizer_step_s")
    private Double timeOptimizerStepS;

    @Column(name = "time_wire_encode_s")
    private Double timeWireEncodeS;

    @Column(name = "time_aggregation_s")
    private Double timeAggregationS;

    @Column(name = "time_wire_decode_s")
    private Double timeWireDecodeS;

    @Column(name = "time_eval_s")
    private Double timeEvalS;

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

    public UUID getRunId() { return runId; }

    public void setRunId(UUID runId) { this.runId = runId; }

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

    public Double getTimeDataLoadingS() {
        return timeDataLoadingS;
    }

    public void setTimeDataLoadingS(Double timeDataLoadingS) {
        this.timeDataLoadingS = timeDataLoadingS;
    }

    public Double getTimeForwardBackwardS() {
        return timeForwardBackwardS;
    }

    public void setTimeForwardBackwardS(Double timeForwardBackwardS) {
        this.timeForwardBackwardS = timeForwardBackwardS;
    }

    public Double getTimeOptimizerStepS() {
        return timeOptimizerStepS;
    }

    public void setTimeOptimizerStepS(Double timeOptimizerStepS) {
        this.timeOptimizerStepS = timeOptimizerStepS;
    }

    public Double getTimeWireEncodeS() {
        return timeWireEncodeS;
    }

    public void setTimeWireEncodeS(Double timeWireEncodeS) {
        this.timeWireEncodeS = timeWireEncodeS;
    }

    public Double getTimeAggregationS() {
        return timeAggregationS;
    }

    public void setTimeAggregationS(Double timeAggregationS) {
        this.timeAggregationS = timeAggregationS;
    }

    public Double getTimeWireDecodeS() {
        return timeWireDecodeS;
    }

    public void setTimeWireDecodeS(Double timeWireDecodeS) {
        this.timeWireDecodeS = timeWireDecodeS;
    }

    public Double getTimeEvalS() {
        return timeEvalS;
    }

    public void setTimeEvalS(Double timeEvalS) {
        this.timeEvalS = timeEvalS;
    }
}
