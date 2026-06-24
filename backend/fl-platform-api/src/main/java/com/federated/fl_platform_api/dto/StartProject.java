package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Pattern;

/**
 * Body for POST /api/projects/{id}/start. All fields are optional — the
 * service supplies sensible defaults — but if provided they must satisfy
 * the bounds below. Validation rejects pathological inputs (e.g.
 * numRounds=Integer.MAX_VALUE) before they reach the FL spawn path,
 * where they could exhaust resources or get interpolated into a shell
 * command.
 */
public class StartProject {

    /**
     * Gradient strategies (FedAvg, DeComFL) mirror those registered in
     * {@code framework/src/fedlearn/server/strategy.py} + {@code decomfl_strategy.py} and spawn
     * the gradient FL server. {@code FoT} (Federation over Text) is a SEPARATE, additive
     * text-federation mode that spawns the standalone {@code fl_fot_server.py} instead. Keep this
     * regex in sync when a strategy/mode is added on the Python side.
     * <p>
     * NOTE: FedLoRA is NOT listed here — it is derived server-side for LLM_LORA runs (see ProjectService.resolveStrategy), never user-submitted.
     */
    @Pattern(
            regexp = "FedAvg|DeComFL|FoT",
            message = "strategy must be one of: FedAvg, DeComFL, FoT"
    )
    private String strategy;

    @Min(value = 1, message = "numRounds must be at least 1")
    @Max(value = 100, message = "numRounds must be at most 100")
    private Integer numRounds;

    @Min(value = 1, message = "minClients must be at least 1")
    @Max(value = 100, message = "minClients must be at most 100")
    private Integer minClients;

    @Min(value = 1, message = "clientsPerRound must be at least 1")
    @Max(value = 100, message = "clientsPerRound must be at most 100")
    private Integer clientsPerRound;

    public String getStrategy() {
        return strategy;
    }

    public void setStrategy(String strategy) {
        this.strategy = strategy;
    }

    public Integer getNumRounds() {
        return numRounds;
    }

    public void setNumRounds(Integer numRounds) {
        this.numRounds = numRounds;
    }

    public Integer getMinClients() {
        return minClients;
    }

    public void setMinClients(Integer minClients) {
        this.minClients = minClients;
    }

    public Integer getClientsPerRound() {
        return clientsPerRound;
    }

    public void setClientsPerRound(Integer clientsPerRound) {
        this.clientsPerRound = clientsPerRound;
    }

    @Override
    public String toString() {
        return "StartProject{" +
                "strategy='" + strategy + '\'' +
                ", numRounds=" + numRounds +
                ", minClients=" + minClients +
                ", clientsPerRound=" + clientsPerRound +
                '}';
    }
}
