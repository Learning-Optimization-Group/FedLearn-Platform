package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.UUID;

public class RunManifestDto {
    private UUID runId;
    private UUID projectId;
    private String recipeKey;
    private String strategy;
    private int numRounds;
    private int clientsPerRound;
    private String partitioningMode;
    private Long seed;
    private String torchVersion;
    // MO-4: true when a trainable .pte was staged for this run (real on-device first-order FedAvg is
    // possible). The mobile client fail-closes on FedAvg when this is false/absent, running the DeComFL
    // zeroth-order path instead. Tied to the staged bundle's actual trainablePtePath, not the recipe alone.
    private boolean firstOrderSupported;
    // V24: the Byzantine-robust rule a Robust run used and its settings; all null for other strategies.
    private String robustMethod;
    private Double robustByzantineFraction;
    private Double robustTrimRatio;
    private Double centeredClipTau;
    // V25: true when the run uses secure aggregation; the threshold is null when it does not. A client that
    // cannot mask its update must refuse such a run, and the phone does.
    private boolean secureAggregation;
    private Integer secureAggThreshold;
    // Execution contract v1, emitted beside the legacy fields during the compatibility window: always its state
    // (PENDING, READY, UNAVAILABLE or LEGACY_ONLY); the ProtoJSON contract and its opaque ID only when READY; the
    // machine-readable reason only when UNAVAILABLE. The server-side detail behind a reason is never sent.
    private String contractState;
    private String contractId;
    private JsonNode executionContract;
    private String contractUnavailableReason;

    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getRecipeKey() { return recipeKey; }
    public void setRecipeKey(String recipeKey) { this.recipeKey = recipeKey; }
    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }
    public int getNumRounds() { return numRounds; }
    public void setNumRounds(int numRounds) { this.numRounds = numRounds; }
    public int getClientsPerRound() { return clientsPerRound; }
    public void setClientsPerRound(int clientsPerRound) { this.clientsPerRound = clientsPerRound; }
    public String getPartitioningMode() { return partitioningMode; }
    public void setPartitioningMode(String partitioningMode) { this.partitioningMode = partitioningMode; }
    public Long getSeed() { return seed; }
    public void setSeed(Long seed) { this.seed = seed; }
    public String getTorchVersion() { return torchVersion; }
    public void setTorchVersion(String torchVersion) { this.torchVersion = torchVersion; }
    public boolean isFirstOrderSupported() { return firstOrderSupported; }
    public void setFirstOrderSupported(boolean firstOrderSupported) { this.firstOrderSupported = firstOrderSupported; }
    public String getRobustMethod() { return robustMethod; }
    public void setRobustMethod(String robustMethod) { this.robustMethod = robustMethod; }
    public Double getRobustByzantineFraction() { return robustByzantineFraction; }
    public void setRobustByzantineFraction(Double robustByzantineFraction) { this.robustByzantineFraction = robustByzantineFraction; }
    public Double getRobustTrimRatio() { return robustTrimRatio; }
    public void setRobustTrimRatio(Double robustTrimRatio) { this.robustTrimRatio = robustTrimRatio; }
    public Double getCenteredClipTau() { return centeredClipTau; }
    public void setCenteredClipTau(Double centeredClipTau) { this.centeredClipTau = centeredClipTau; }
    public boolean isSecureAggregation() { return secureAggregation; }
    public void setSecureAggregation(boolean secureAggregation) { this.secureAggregation = secureAggregation; }
    public Integer getSecureAggThreshold() { return secureAggThreshold; }
    public void setSecureAggThreshold(Integer secureAggThreshold) { this.secureAggThreshold = secureAggThreshold; }
    public String getContractState() { return contractState; }
    public void setContractState(String contractState) { this.contractState = contractState; }
    public String getContractId() { return contractId; }
    public void setContractId(String contractId) { this.contractId = contractId; }
    public JsonNode getExecutionContract() { return executionContract; }
    public void setExecutionContract(JsonNode executionContract) { this.executionContract = executionContract; }
    public String getContractUnavailableReason() { return contractUnavailableReason; }
    public void setContractUnavailableReason(String contractUnavailableReason) {
        this.contractUnavailableReason = contractUnavailableReason;
    }
}
