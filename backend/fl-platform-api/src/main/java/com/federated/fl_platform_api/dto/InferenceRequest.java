package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Size;

import java.util.List;

/**
 * Inference input for {@code POST /api/inference/{projectId}}. Exactly one field
 * is expected, depending on the model's input kind:
 * <ul>
 *   <li>{@code imageBase64} — a base64-encoded image (any common format); used by
 *       image models (CNN). May include a {@code data:} URL prefix, which the
 *       service strips.</li>
 *   <li>{@code values} — a numeric feature vector; used by tabular models (MLP/ECG).</li>
 *   <li>{@code text} — raw text input; used by text-classification models
 *       (LLM_LORA, TRANSFORMER). The backend tokenizes on the Python side.</li>
 * </ul>
 */
public class InferenceRequest {

    /**
     * Bounded so a malicious caller can't force an unbounded heap allocation: the
     * base64 string is base64-decoded (allocating ~0.75x more) before the 9 MB
     * decoded-byte cap is even checked. 12 MB of base64 (~9 MB decoded) is the
     * field ceiling; the service still enforces the precise decoded-size limit.
     */
    @Size(max = 12_000_000, message = "imageBase64 exceeds the maximum allowed size")
    private String imageBase64;

    /** Sanity bound mirrored by InferenceService#MAX_VECTOR_LENGTH (defense in depth). */
    @Size(max = 100_000, message = "values vector exceeds the maximum allowed length")
    private List<Double> values;

    /** Raw text for text-classification models (LLM_LORA, TRANSFORMER). Backend tokenizes. */
    @Size(max = 10_000, message = "text exceeds the maximum allowed length")
    private String text;

    public String getImageBase64() { return imageBase64; }
    public void setImageBase64(String imageBase64) { this.imageBase64 = imageBase64; }

    public List<Double> getValues() { return values; }
    public void setValues(List<Double> values) { this.values = values; }

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
}
