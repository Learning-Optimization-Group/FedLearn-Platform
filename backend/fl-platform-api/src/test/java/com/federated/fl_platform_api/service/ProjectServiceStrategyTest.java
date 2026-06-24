package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ProjectServiceStrategyTest {

    @Test
    void llmLoraAlwaysForcesFedLoRA() {
        assertEquals("FedLoRA", ProjectService.resolveStrategy("LLM_LORA", "FedAvg"));
        assertEquals("FedLoRA", ProjectService.resolveStrategy("llm_lora", "DeComFL"));
        assertEquals("FedLoRA", ProjectService.resolveStrategy("LLM_LORA", null));
        assertEquals("FedLoRA", ProjectService.resolveStrategy("LLM_LORA", ""));
    }

    @Test
    void nonLlmLoraKeepsRequestedOrDefaults() {
        assertEquals("DeComFL", ProjectService.resolveStrategy("CNN", "DeComFL"));
        assertEquals("FoT", ProjectService.resolveStrategy("TRANSFORMER", "FoT"));
        assertEquals("FedAvg", ProjectService.resolveStrategy("CNN", null));
        assertEquals("FedAvg", ProjectService.resolveStrategy("MLP", ""));
    }
}
