package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "runs")
public class Run {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(name = "project_id", nullable = false)
    private UUID projectId;

    @Column(nullable = false, length = 32)
    private String strategy;

    @Column(name = "num_rounds", nullable = false)
    private int numRounds;

    @Column(name = "min_clients", nullable = false)
    private int minClients;

    @Column(name = "clients_per_round", nullable = false)
    private int clientsPerRound;

    @Enumerated(EnumType.STRING)
    @Column(name = "partitioning_mode", nullable = false, length = 16)
    private PartitioningMode partitioningMode = PartitioningMode.SHARDED;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private RunStatus status;

    @Column(name = "server_host")
    private String serverHost;

    @Column(name = "server_port")
    private Integer serverPort;

    // BA-3: OS identity of the spawned FL-server child, recorded at spawn so a StartupReconciler can
    // reap orphans after a backend crash. process_started_at guards against PID reuse.
    @Column(name = "server_pid")
    private Long serverPid;

    @Column(name = "process_started_at")
    private Instant processStartedAt;

    @Column(name = "grpc_ca_fingerprint", length = 128)
    private String grpcCaFingerprint;

    @Column
    private Long seed;

    @Column(name = "torch_version", length = 32)
    private String torchVersion;

    @Column(name = "recipe_key", nullable = false, length = 64)
    private String recipeKey;

    @Column(name = "created_by")
    private Long createdBy;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "started_at")
    private Instant startedAt;

    @Column(name = "ended_at")
    private Instant endedAt;

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public String getStrategy() { return strategy; }
    public void setStrategy(String strategy) { this.strategy = strategy; }
    public int getNumRounds() { return numRounds; }
    public void setNumRounds(int numRounds) { this.numRounds = numRounds; }
    public int getMinClients() { return minClients; }
    public void setMinClients(int minClients) { this.minClients = minClients; }
    public int getClientsPerRound() { return clientsPerRound; }
    public void setClientsPerRound(int clientsPerRound) { this.clientsPerRound = clientsPerRound; }
    public PartitioningMode getPartitioningMode() { return partitioningMode; }
    public void setPartitioningMode(PartitioningMode partitioningMode) { this.partitioningMode = partitioningMode; }
    public RunStatus getStatus() { return status; }
    public void setStatus(RunStatus status) { this.status = status; }
    public String getServerHost() { return serverHost; }
    public void setServerHost(String serverHost) { this.serverHost = serverHost; }
    public Integer getServerPort() { return serverPort; }
    public void setServerPort(Integer serverPort) { this.serverPort = serverPort; }

    public Long getServerPid() { return serverPid; }
    public void setServerPid(Long serverPid) { this.serverPid = serverPid; }

    public Instant getProcessStartedAt() { return processStartedAt; }
    public void setProcessStartedAt(Instant processStartedAt) { this.processStartedAt = processStartedAt; }
    public String getGrpcCaFingerprint() { return grpcCaFingerprint; }
    public void setGrpcCaFingerprint(String grpcCaFingerprint) { this.grpcCaFingerprint = grpcCaFingerprint; }
    public Long getSeed() { return seed; }
    public void setSeed(Long seed) { this.seed = seed; }
    public String getTorchVersion() { return torchVersion; }
    public void setTorchVersion(String torchVersion) { this.torchVersion = torchVersion; }
    public String getRecipeKey() { return recipeKey; }
    public void setRecipeKey(String recipeKey) { this.recipeKey = recipeKey; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }
    public Instant getEndedAt() { return endedAt; }
    public void setEndedAt(Instant endedAt) { this.endedAt = endedAt; }
}
