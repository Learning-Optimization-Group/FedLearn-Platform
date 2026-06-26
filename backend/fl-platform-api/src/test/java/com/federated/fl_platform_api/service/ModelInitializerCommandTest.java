package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class ModelInitializerCommandTest {

    @Test
    void llmLoraInitCommandCarriesFfaAggregation() {
        List<String> cmd = ModelInitializer.buildInitCommand(
                "LLM_LORA", "qwen2.5-0.5b", "AdamW", "/tmp/out.npz", 0, "SEQ_CLASSIFICATION", "/x/run_init_model.sh", false);
        assertTrue(cmd.contains("--aggregation"));
        assertEquals("FFA_LORA", cmd.get(cmd.indexOf("--aggregation") + 1));
        assertEquals("LLM_LORA", cmd.get(cmd.indexOf("--model-type") + 1));
    }

    @Test
    void nonLlmLoraInitCommandHasNoAggregationFlag() {
        List<String> cmd = ModelInitializer.buildInitCommand(
                "CNN", "net", "Adam", "/tmp/out.npz", 0, "SEQ_CLASSIFICATION", "/x/run_init_model.sh", false);
        assertFalse(cmd.contains("--aggregation"));
    }

    @Test
    void llmLoraInitCommandCarriesTaskType() {
        List<String> cmd = ModelInitializer.buildInitCommand(
                "LLM_LORA", "qwen2.5-0.5b", "AdamW", "/tmp/o.npz", 0, "CAUSAL_LM", "/x/run_init_model.sh", false);
        assertTrue(cmd.contains("--task-type"));
        assertEquals("CAUSAL_LM", cmd.get(cmd.indexOf("--task-type") + 1));
    }
}
