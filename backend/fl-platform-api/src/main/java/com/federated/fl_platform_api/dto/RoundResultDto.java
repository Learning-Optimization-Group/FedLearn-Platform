package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import java.util.UUID;


public class RoundResultDto {
    private UUID id;
    @JsonDeserialize(using = IntegralNumberDeserializers.IntegerValue.class)
    private Integer serverRound;
    private Double loss;
    private Double accuracy;
    private Double gpuUtilization;

    @JsonProperty("timeDataLoadingS")
    @JsonAlias("time_data_loading_s")
    private Double timeDataLoadingS;

    @JsonProperty("timeForwardBackwardS")
    @JsonAlias("time_forward_backward_s")
    private Double timeForwardBackwardS;

    @JsonProperty("timeOptimizerStepS")
    @JsonAlias("time_optimizer_step_s")
    private Double timeOptimizerStepS;

    @JsonProperty("timeWireEncodeS")
    @JsonAlias("time_wire_encode_s")
    private Double timeWireEncodeS;

    @JsonProperty("timeAggregationS")
    @JsonAlias("time_aggregation_s")
    private Double timeAggregationS;

    @JsonProperty("timeWireDecodeS")
    @JsonAlias("time_wire_decode_s")
    private Double timeWireDecodeS;

    @JsonProperty("timeEvalS")
    @JsonAlias("time_eval_s")
    private Double timeEvalS;

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
