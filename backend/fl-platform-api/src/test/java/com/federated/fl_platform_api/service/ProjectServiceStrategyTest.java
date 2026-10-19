package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ProjectServiceStrategyTest {

    @Test void nihApprovesOnlyFirstOrderFedAvgAndRobust() {
        assertEquals("FedAvg", ProjectService.resolveStrategy("NIH_PNEUMONIA_CNN", null));
        assertEquals("Robust", ProjectService.resolveStrategy("NIH_PNEUMONIA_CNN", "Robust"));
        for (String strategy : new String[]{"FedOpt", "FedProx", "DeComFL", "FoT", "XgbBagging"}) {
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> ProjectService.resolveStrategy("NIH_PNEUMONIA_CNN", strategy));
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> ProjectService.resolveStrategy("nih_pneumonia_cnn", strategy));
        }
    }

    @Test void treesRequireTreeAggregationAndNeuralModelsCannotRequestIt() {
        assertEquals("XgbBagging", ProjectService.resolveStrategy("XGBOOST", null));
        assertEquals("XgbBagging", ProjectService.resolveStrategy("XGBOOST", "XgbBagging"));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
            () -> ProjectService.resolveStrategy("XGBOOST", "FedAvg"));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
            () -> ProjectService.resolveStrategy("MLP", "XgbBagging"));
    }

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

    @Test
    void newlyExposedStrategiesPassThrough() {
        // FedOpt (server-side adaptive) and Robust (Byzantine-robust aggregation, median by default) now run end-to-end and
        // are user-selectable; resolveStrategy forwards them unchanged for a non-LLM_LORA model.
        assertEquals("FedOpt", ProjectService.resolveStrategy("CNN", "FedOpt"));
        assertEquals("Robust", ProjectService.resolveStrategy("MLP", "Robust"));
    }
}
