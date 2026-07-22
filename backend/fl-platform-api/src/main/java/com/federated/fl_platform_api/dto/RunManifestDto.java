package com.federated.fl_platform_api.dto;

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
}
