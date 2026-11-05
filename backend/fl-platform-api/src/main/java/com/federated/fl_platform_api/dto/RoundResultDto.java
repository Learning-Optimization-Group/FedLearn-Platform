package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import java.io.IOException;
import java.util.UUID;


public class RoundResultDto {
    private UUID id;
    @JsonDeserialize(using = RoundNumberDeserializer.class)
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

    /** Round identity must not be truncated from a float or coerced from a string. */
    public static class RoundNumberDeserializer extends JsonDeserializer<Integer> {
        @Override
        public Integer deserialize(JsonParser parser, DeserializationContext context) throws IOException {
            if (!parser.hasToken(JsonToken.VALUE_NUMBER_INT)) {
                return context.reportInputMismatch(Integer.class, "serverRound must be a JSON integer");
            }
            return parser.getIntValue();
        }
    }
}
