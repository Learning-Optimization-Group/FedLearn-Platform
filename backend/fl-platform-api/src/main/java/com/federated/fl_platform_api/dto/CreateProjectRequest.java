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

    @NotNull(message = "pretrainEpochs must be provided")
    @Min(value = 0, message = "pretrainEpochs cannot be negative")
    private Integer preTrainEpocs;


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
        return preTrainEpocs;
    }

    public void setPretrainEpochs(Integer preTrainEpocs) {
        this.preTrainEpocs = preTrainEpocs;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public void setOptimizer(String optimizer) { this.optimizer = optimizer; }
}
