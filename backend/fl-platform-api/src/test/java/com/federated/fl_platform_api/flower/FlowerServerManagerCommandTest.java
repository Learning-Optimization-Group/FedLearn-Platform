package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.UUID;
import static org.junit.jupiter.api.Assertions.*;

class FlowerServerManagerCommandTest {

    private Project project(String modelType) {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType(modelType);
        p.setModelName("qwen2.5-0.5b");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    @Test
    void llmLoraCommandCarriesFedLoRAStrategyAndFfaAggregation() {
        List<String> cmd = FlowerServerManager.buildServerCommand(
                project("LLM_LORA"), "FedLoRA", 5, 1, 50000, "/x/run_fl_server.sh", false);
        assertTrue(cmd.contains("--strategy"));
        assertEquals("FedLoRA", cmd.get(cmd.indexOf("--strategy") + 1));
        assertTrue(cmd.contains("--aggregation"), "LLM_LORA must pass --aggregation");
        assertEquals("FFA_LORA", cmd.get(cmd.indexOf("--aggregation") + 1));
        assertEquals("LLM_LORA", cmd.get(cmd.indexOf("--model-type") + 1));
    }

    @Test
    void nonLlmLoraCommandHasNoAggregationFlag() {
        List<String> cmd = FlowerServerManager.buildServerCommand(
                project("CNN"), "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false);
        assertFalse(cmd.contains("--aggregation"), "non-LLM_LORA must NOT pass --aggregation");
        assertEquals("FedAvg", cmd.get(cmd.indexOf("--strategy") + 1));
    }

    @Test
    void fotCommandIsUnaffected() {
        List<String> cmd = FlowerServerManager.buildServerCommand(
                project("TRANSFORMER"), "FoT", 5, 1, 50000, "/x/run_fot.sh", false);
        assertFalse(cmd.contains("--strategy"), "FoT branch has no --strategy");
        assertFalse(cmd.contains("--aggregation"));
    }
}
