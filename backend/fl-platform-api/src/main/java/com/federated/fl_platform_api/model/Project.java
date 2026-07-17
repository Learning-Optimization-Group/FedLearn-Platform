package com.federated.fl_platform_api.model;


import jakarta.persistence.*;


import java.util.UUID;

@Entity
@Table(name = "projects")
public class Project {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;


    @Column(nullable = false, unique = true)
    private String name;


    @Column(nullable = false)
    private String modelType;

    @Column(nullable = false)
    private String modelName;


    @Column
    private Integer serverPort;


    @Column
    private String modelPath;

    @Column
    private String optimizer;

    @Column
    private String taskType;

    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @Column(name = "org_id", nullable = false)
    private UUID orgId;

    @Column(nullable = false)
    private String status;

    // BA-1: the one-time model-init phase, independent of the run-derived status (BA-4). Defaults to
    // DONE so every legacy row (created synchronously) reads as already-initialised; createProject
    // sets INITIALIZING and the async worker flips it to DONE/FAILED.
    @Enumerated(EnumType.STRING)
    @Column(name = "init_status", nullable = false, length = 16)
    private ProjectInitStatus initStatus = ProjectInitStatus.DONE;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private ProjectVisibility visibility = ProjectVisibility.PRIVATE;

    @Column(name = "model_published", nullable = false)
    private boolean modelPublished = false;

    @Column(name = "model_description", columnDefinition = "TEXT")
    private String modelDescription;

    @Column(name = "model_tags", length = 512)
    private String modelTags;

    @Column(name = "model_published_at")
    private java.time.Instant modelPublishedAt;

    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public Integer getServerPort() {
        return serverPort;
    }

    public void setServerPort(Integer serverPort) {
        this.serverPort = serverPort;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getOptimizer() { return optimizer; }
    public void setOptimizer(String optimizer) { this.optimizer = optimizer; }

    public User getUser() { return user; }

    public void setUser(User user) { this.user = user; }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getStatus() { return status; }

    public ProjectInitStatus getInitStatus() { return initStatus; }

    public void setInitStatus(ProjectInitStatus initStatus) { this.initStatus = initStatus; }

    public ProjectVisibility getVisibility() {
        return visibility;
    }

    public void setVisibility(ProjectVisibility visibility) {
        this.visibility = visibility;
    }

    public boolean isModelPublished() {
        return modelPublished;
    }

    public void setModelPublished(boolean modelPublished) {
        this.modelPublished = modelPublished;
    }

    public String getModelDescription() {
        return modelDescription;
    }

    public void setModelDescription(String modelDescription) {
        this.modelDescription = modelDescription;
    }

    public String getModelTags() {
        return modelTags;
    }

    public void setModelTags(String modelTags) {
        this.modelTags = modelTags;
    }

    public java.time.Instant getModelPublishedAt() {
        return modelPublishedAt;
    }

    public void setModelPublishedAt(java.time.Instant modelPublishedAt) {
        this.modelPublishedAt = modelPublishedAt;
    }

    public UUID getOrgId() { return orgId; }
    public void setOrgId(UUID orgId) { this.orgId = orgId; }

    @Column(name = "active_run_id")
    private java.util.UUID activeRunId;

    public java.util.UUID getActiveRunId() { return activeRunId; }
    public void setActiveRunId(java.util.UUID activeRunId) { this.activeRunId = activeRunId; }

    @Convert(converter = com.federated.fl_platform_api.model.DeviceRequirementsConverter.class)
    @Column(name = "requirements_override", columnDefinition = "TEXT")
    private com.federated.fl_platform_api.dto.DeviceRequirements requirementsOverride;

    public com.federated.fl_platform_api.dto.DeviceRequirements getRequirementsOverride() { return requirementsOverride; }
    public void setRequirementsOverride(com.federated.fl_platform_api.dto.DeviceRequirements requirementsOverride) { this.requirementsOverride = requirementsOverride; }

    // SE-11: run-level DP policy (V17). A regulated project may not start a run unless DP is
    // enabled with a complete config (enforced at the FlServerManager start gate); when
    // dp_enabled, the three knobs flow to the FL server as --dp-* flags. Knobs are nullable by
    // design: a non-DP project carries no config.

    @Column(name = "regulated", nullable = false)
    private boolean regulated = false;

    @Column(name = "dp_enabled", nullable = false)
    private boolean dpEnabled = false;

    /** Target privacy budget epsilon (> 0; guidance ~4-8 for medical/regulated data). */
    @Column(name = "dp_target_epsilon")
    private Double dpTargetEpsilon;

    /** DP failure probability delta, in (0,1) exclusive. */
    @Column(name = "dp_delta")
    private Double dpDelta;

    /** Per-user (per-client) L2 contribution bound S (> 0). */
    @Column(name = "dp_clip_norm")
    private Double dpClipNorm;

    public boolean isRegulated() { return regulated; }
    public void setRegulated(boolean regulated) { this.regulated = regulated; }

    public boolean isDpEnabled() { return dpEnabled; }
    public void setDpEnabled(boolean dpEnabled) { this.dpEnabled = dpEnabled; }

    public Double getDpTargetEpsilon() { return dpTargetEpsilon; }
    public void setDpTargetEpsilon(Double dpTargetEpsilon) { this.dpTargetEpsilon = dpTargetEpsilon; }

    public Double getDpDelta() { return dpDelta; }
    public void setDpDelta(Double dpDelta) { this.dpDelta = dpDelta; }

    public Double getDpClipNorm() { return dpClipNorm; }
    public void setDpClipNorm(Double dpClipNorm) { this.dpClipNorm = dpClipNorm; }

    // DA-14 Ph3.2: per-project derivation record (V20). A NULL/false derivation == a normal
    // from-scratch recipe project (today's behavior); nothing on the training path reads these yet.
    @Column(name = "init_from_pretrained", nullable = false)
    private boolean initFromPretrained = false;

    /** Content address (sha256) of the frozen BASE_REF backbone this project derives from; null = from-scratch. */
    @Column(name = "base_ref_sha256")
    private String baseRefSha256;

    /** JSON derivation spec (dataset / head / freeze / lora); null when absent. */
    @Column(name = "derivation_spec")
    private String derivationSpec;

    public boolean isInitFromPretrained() { return initFromPretrained; }
    public void setInitFromPretrained(boolean initFromPretrained) { this.initFromPretrained = initFromPretrained; }

    public String getBaseRefSha256() { return baseRefSha256; }
    public void setBaseRefSha256(String baseRefSha256) { this.baseRefSha256 = baseRefSha256; }

    public String getDerivationSpec() { return derivationSpec; }
    public void setDerivationSpec(String derivationSpec) { this.derivationSpec = derivationSpec; }

    /**
     * SE-11: true iff (epsilon, delta, S) form a complete, sane DP config — epsilon > 0, delta in
     * (0,1) exclusive, clip norm S > 0. Single source of truth for the creation validation, the
     * run-start gate and the spawn-time argv check.
     */
    public static boolean isCompleteDpConfig(Double epsilon, Double delta, Double clipNorm) {
        return epsilon != null && epsilon > 0
                && delta != null && delta > 0 && delta < 1
                && clipNorm != null && clipNorm > 0;
    }

    /** SE-11: this project's stored DP config is complete and sane. */
    public boolean hasCompleteDpConfig() {
        return isCompleteDpConfig(dpTargetEpsilon, dpDelta, dpClipNorm);
    }
}
