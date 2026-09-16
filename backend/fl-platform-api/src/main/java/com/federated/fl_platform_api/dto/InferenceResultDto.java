package com.federated.fl_platform_api.dto;

import java.util.List;

/**
 * Result of one inference run. {@code probabilities} is the softmax over the
 * model's outputs (same order as {@code classes}); {@code predictedIndex} is the
 * argmax and {@code predictedLabel} its human-readable class.
 */
public class InferenceResultDto {

    private String modelType;
    private int predictedIndex;
    private String predictedLabel;
    private List<String> classes;
    private List<Double> probabilities;
    private List<Double> logits;

    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }

    public int getPredictedIndex() { return predictedIndex; }
    public void setPredictedIndex(int predictedIndex) { this.predictedIndex = predictedIndex; }

    public String getPredictedLabel() { return predictedLabel; }
    public void setPredictedLabel(String predictedLabel) { this.predictedLabel = predictedLabel; }

    public List<String> getClasses() { return classes; }
    public void setClasses(List<String> classes) { this.classes = classes; }

    public List<Double> getProbabilities() { return probabilities; }
    public void setProbabilities(List<Double> probabilities) { this.probabilities = probabilities; }

    public List<Double> getLogits() { return logits; }
    public void setLogits(List<Double> logits) { this.logits = logits; }
}
