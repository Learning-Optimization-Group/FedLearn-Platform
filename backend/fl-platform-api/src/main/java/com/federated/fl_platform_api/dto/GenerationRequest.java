package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/** Request for streaming text generation on a CAUSAL_LM LLM_LORA project. */
public class GenerationRequest {

    @NotBlank
    @Size(max = 10_000)
    private String prompt;

    @Min(1)
    @Max(2048)
    private int maxNewTokens = 256;

    @DecimalMin("0.0")
    @DecimalMax("2.0")
    private double temperature = 0.7;

    public String getPrompt() { return prompt; }
    public void setPrompt(String prompt) { this.prompt = prompt; }
    public int getMaxNewTokens() { return maxNewTokens; }
    public void setMaxNewTokens(int maxNewTokens) { this.maxNewTokens = maxNewTokens; }
    public double getTemperature() { return temperature; }
    public void setTemperature(double temperature) { this.temperature = temperature; }

    @jakarta.validation.Valid
    @Size(max = 100)
    private java.util.List<ChatTurn> history;
    public java.util.List<ChatTurn> getHistory() { return history; }
    public void setHistory(java.util.List<ChatTurn> history) { this.history = history; }
}
