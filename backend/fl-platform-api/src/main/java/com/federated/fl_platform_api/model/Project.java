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

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @Column(name = "org_id", nullable = false)
    private UUID orgId;

    @Column(nullable = false)
    private String status;

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
}
