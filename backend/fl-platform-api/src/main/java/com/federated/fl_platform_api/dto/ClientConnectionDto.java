package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.UUID;

public class ClientConnectionDto {
    private UUID projectId;
    // The active run and its execution contract, as the run manifest reports them: the state always; the ProtoJSON
    // contract and its ID only when READY; the reason only when UNAVAILABLE. A launcher hands a READY contract to
    // fl-runtime/client.py (--execution-contract, --run-id), which refuses training it would not execute exactly.
    private UUID runId;
    private String contractState;
    private String contractId;
    private JsonNode executionContract;
    private String contractUnavailableReason;
    private String name;
    private String modelType;
    private String serverAddress;
    private Integer partitionId;
    private String status;
    private String connectionToken;
    // The running aggregation strategy (from the active Run). The desktop client threads this into
    // fl-runtime/client.py's --strategy so the client picks the matching path (e.g. DeComFL) instead
    // of always defaulting to the FedAvg path — otherwise a DeComFL project silently mismatches for
    // any non-MLP model type.
    private String strategy;
    // The project's training arm (FULL / FROZEN_HEAD). Same rationale as `strategy` above: the
    // server filters its parameters to the arm's trainable subset, so a client that does not know
    // the arm uploads the FULL state dict against a server expecting the head only. Carried here
    // rather than inferred, because the arm is a project property the client cannot derive.
    private String trainingArm;
    // Server trust, as enrollment resolved it: whether the FL server serves TLS, the certificate to verify it with,
    // and that certificate's sha256. The desktop writes the certificate to a file its client process trusts.
    private boolean grpcTls;
    private String grpcServerCertPem;
    private String grpcServerCertFingerprint;

    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public String getServerAddress() { return serverAddress; }
    public void setServerAddress(String serverAddress) { this.serverAddress = serverAddress; }
    public Integer getPartitionId() { return partitionId; }
    public void setPartitionId(Integer partitionId) { this.partitionId = partitionId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getConnectionToken() { return connectionToken; }
    public void setConnectionToken(String connectionToken) { this.connectionToken = connectionToken; }
    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }
    public String getTrainingArm() { return trainingArm; }
    public void setTrainingArm(String trainingArm) { this.trainingArm = trainingArm; }

    public boolean isGrpcTls() { return grpcTls; }
    public void setGrpcTls(boolean grpcTls) { this.grpcTls = grpcTls; }
    public String getGrpcServerCertPem() { return grpcServerCertPem; }
    public void setGrpcServerCertPem(String grpcServerCertPem) { this.grpcServerCertPem = grpcServerCertPem; }
    public String getGrpcServerCertFingerprint() { return grpcServerCertFingerprint; }
    public void setGrpcServerCertFingerprint(String grpcServerCertFingerprint) {
        this.grpcServerCertFingerprint = grpcServerCertFingerprint;
    }
    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
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
