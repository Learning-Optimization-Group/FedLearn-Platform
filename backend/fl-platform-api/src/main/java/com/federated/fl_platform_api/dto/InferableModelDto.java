package com.federated.fl_platform_api.dto;

import java.util.List;
import java.util.UUID;

/**
 * A trained model the current user may run inference against. Surfaced by
 * {@code GET /api/inference/models} to power the "Use a model" page on the web
 * and desktop clients.
 */
public class InferableModelDto {

    private UUID projectId;
    private String name;
    private String modelType;
    private String modelName;
    private String status;
    /** "image" | "vector" | null (null ⇒ not runnable interactively, e.g. Transformer). */
    private String inputKind;
    /** Human-readable class labels in output order. */
    private List<String> classes;
    /** False for model types we cannot run interactively yet (Transformer). */
    private boolean supported;

    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getInputKind() { return inputKind; }
    public void setInputKind(String inputKind) { this.inputKind = inputKind; }

    public List<String> getClasses() { return classes; }
    public void setClasses(List<String> classes) { this.classes = classes; }

    public boolean isSupported() { return supported; }
    public void setSupported(boolean supported) { this.supported = supported; }
}
