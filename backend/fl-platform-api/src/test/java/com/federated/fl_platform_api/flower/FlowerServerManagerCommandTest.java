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

    @Test
    void llmLoraServerCommandCarriesTaskType() {
        Project p = project("LLM_LORA");
        p.setTaskType("CAUSAL_LM");
        List<String> cmd = FlowerServerManager.buildServerCommand(
                p, "FedLoRA", 5, 1, 50000, "/x/run_fl_server.sh", false);
        assertTrue(cmd.contains("--task-type"));
        assertEquals("CAUSAL_LM", cmd.get(cmd.indexOf("--task-type") + 1));
    }

    // --- SE-10: allowlist attacker-influenceable project fields before they reach the fl_server
    // argv. ProcessBuilder(List) never invokes a shell, so the real risks are option injection (a
    // value starting with '-' misread as a flag by fl_server.py's argparse) and path traversal via
    // --model-path / --model-name feeding the server-side model load. Fail closed: refuse to build
    // the command rather than spawn with a poisoned argument.

    @Test
    void modelNameStartingWithDash_isRejected() {
        Project p = project("LLM_LORA");
        p.setModelName("--num-rounds");   // option-injection attempt
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false));
    }

    @Test
    void modelPathWithTraversal_isRejected() {
        Project p = project("CNN");
        p.setModelPath("/tmp/../../etc/passwd");
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false));
    }

    @Test
    void strategyWithMetacharacter_isRejected() {
        Project p = project("CNN");
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FedAvg;whoami", 5, 1, 50000, "/x/run_fl_server.sh", false));
    }

    @Test
    void modelNameWithShellMetacharacter_isRejected() {
        Project p = project("LLM_LORA");
        p.setModelName("qwen$(whoami)");
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false));
    }

    @Test
    void legitimateHuggingFaceModelName_isAccepted() {
        // Guard against over-blocking: a real HF repo id carries '/', '.' and '-'.
        Project p = project("LLM_LORA");
        p.setModelName("Qwen/Qwen2.5-0.5B");
        List<String> cmd = FlowerServerManager.buildServerCommand(
                p, "FedLoRA", 5, 1, 50000, "/x/run_fl_server.sh", false);
        assertEquals("Qwen/Qwen2.5-0.5B", cmd.get(cmd.indexOf("--model-name") + 1));
    }
}
