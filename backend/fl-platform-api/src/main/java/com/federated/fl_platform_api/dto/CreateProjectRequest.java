package com.federated.fl_platform_api.dto;


import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

public class CreateProjectRequest {

    @NotEmpty(message = "Project name cannot be empty")
    private String name;

    @NotEmpty(message = "Model type cannot be empty")
    private String modelType;

    private String modelName;

    private String optimizer;

    @jakarta.validation.constraints.Pattern(regexp = "SEQ_CLASSIFICATION|CAUSAL_LM",
            message = "taskType must be SEQ_CLASSIFICATION or CAUSAL_LM")
    private String taskType;

    public String getTaskType() { return taskType; }
    public void setTaskType(String taskType) { this.taskType = taskType; }

    @NotNull(message = "pretrainEpochs must be provided")
    @Min(value = 0, message = "pretrainEpochs cannot be negative")
    private Integer pretrainEpochs;


    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getModelType() {
        return modelType;
    }

    public String getModelName() {
        return modelName;
    }

    public String getOptimizer() {
        return optimizer;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public Integer getPretrainEpochs() {
        return pretrainEpochs;
    }

    public void setPretrainEpochs(Integer pretrainEpochs) {
        this.pretrainEpochs = pretrainEpochs;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public void setOptimizer(String optimizer) { this.optimizer = optimizer; }

    private DeviceRequirements requirementsOverride;
    public DeviceRequirements getRequirementsOverride() { return requirementsOverride; }
    public void setRequirementsOverride(DeviceRequirements requirementsOverride) { this.requirementsOverride = requirementsOverride; }

    // SE-11: run-level DP policy. If regulated or dpEnabled is true, the three knobs must form a
    // complete config — dpTargetEpsilon > 0 (guidance ~4-8 for medical/regulated data), dpDelta in
    // (0,1) exclusive, dpClipNorm > 0 — enforced cross-field in ProjectService.createProject
    // (single-field bean validation can't express the conditional completeness rule).

    private Boolean regulated;

    private Boolean dpEnabled;

    private Double dpTargetEpsilon;

    private Double dpDelta;

    private Double dpClipNorm;

    public Boolean getRegulated() { return regulated; }
    public void setRegulated(Boolean regulated) { this.regulated = regulated; }

    public Boolean getDpEnabled() { return dpEnabled; }
    public void setDpEnabled(Boolean dpEnabled) { this.dpEnabled = dpEnabled; }

    public Double getDpTargetEpsilon() { return dpTargetEpsilon; }
    public void setDpTargetEpsilon(Double dpTargetEpsilon) { this.dpTargetEpsilon = dpTargetEpsilon; }

    public Double getDpDelta() { return dpDelta; }
    public void setDpDelta(Double dpDelta) { this.dpDelta = dpDelta; }

    public Double getDpClipNorm() { return dpClipNorm; }
    public void setDpClipNorm(Double dpClipNorm) { this.dpClipNorm = dpClipNorm; }
}
