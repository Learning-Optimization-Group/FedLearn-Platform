package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ProjectServiceGenerationKindTest {
    @Test
    void generationOnlyForLlmLoraCausal() {
        assertTrue(ProjectService.isGenerationProject("LLM_LORA", "CAUSAL_LM"));
        assertTrue(ProjectService.isGenerationProject("llm_lora", "causal_lm"));
        assertFalse(ProjectService.isGenerationProject("LLM_LORA", "SEQ_CLASSIFICATION"));
        assertFalse(ProjectService.isGenerationProject("CNN", "CAUSAL_LM"));
        assertFalse(ProjectService.isGenerationProject("LLM_LORA", null));
    }
}
