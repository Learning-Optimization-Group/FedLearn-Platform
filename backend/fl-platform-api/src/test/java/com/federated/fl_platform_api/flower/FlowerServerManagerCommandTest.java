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
    void initModelPath_isPassedSeparately_whileModelPathStaysTheWriteTarget() {
        // BA-11: a continued run reads init weights from the registry-resolved path, but WRITES its
        // output to the project's .npz — the two must not be conflated.
        List<String> cmd = FlowerServerManager.buildServerCommand(
                project("CNN"), "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false,
                "/var/models/blob-cache/deadbeef.npz");
        assertTrue(cmd.contains("--init-model-path"), "registry-resolved init weights must be passed");
        assertEquals("/var/models/blob-cache/deadbeef.npz", cmd.get(cmd.indexOf("--init-model-path") + 1));
        assertEquals("/tmp/model.npz", cmd.get(cmd.indexOf("--model-path") + 1)); // write target unchanged
    }

    @Test
    void noInitModelPath_whenNull_orViaTheDefaultArity() {
        assertFalse(FlowerServerManager.buildServerCommand(
                        project("CNN"), "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false, null)
                .contains("--init-model-path"));
        assertFalse(FlowerServerManager.buildServerCommand(
                        project("CNN"), "FedAvg", 5, 1, 50000, "/x/run_fl_server.sh", false)
                .contains("--init-model-path"));
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

    // --- SE-1/SE-7: the spawned FL server's child environment ---
    @Test
    void configureChildEnv_setsFlSecretAndAuthFlagAndScrubsWebSecret() {
        java.util.Map<String, String> env = new java.util.HashMap<>();
        env.put("APP_JWT_SECRET", "web-auth-secret");   // inherited from the backend process
        env.put("PATH", "/usr/bin");                     // unrelated inherited var must survive
        FlowerServerManager.configureChildEnv(env, "internal-key", "http://backend", "fl-secret", true,
                "run-abc", true, "scoped-run-token-xyz");
        assertEquals("internal-key", env.get("FEDLEARN_INTERNAL_API_KEY"));
        assertEquals("scoped-run-token-xyz", env.get("FEDLEARN_INTERNAL_RUN_TOKEN"));  // SE-7: scoped per-run token
        assertEquals("http://backend", env.get("FEDLEARN_BACKEND_URL"));
        assertEquals("fl-secret", env.get("FEDLEARN_FL_TOKEN_SECRET"));  // FL server can verify tokens
        assertEquals("1", env.get("FEDLEARN_REQUIRE_CLIENT_AUTH"));       // enforcement activated
        assertEquals("run-abc", env.get("FEDLEARN_RUN_ID"));             // FR-7: server bound to its run
        assertEquals("1", env.get("FEDLEARN_GRPC_USE_TLS"));            // SE-2: TLS enabled
        assertEquals("1", env.get("FEDLEARN_REQUIRE_TLS"));             // SE-2: fail closed on plaintext
        assertNull(env.get("APP_JWT_SECRET"),
                "the web-auth secret must be scrubbed from the network-facing FL child (SE-7)");
        assertEquals("/usr/bin", env.get("PATH"));
    }

    @Test
    void configureChildEnv_authAndTlsDisabledByDefaultNoBackendUrlNoRun() {
        java.util.Map<String, String> env = new java.util.HashMap<>();
        FlowerServerManager.configureChildEnv(env, "k", null, "fl-secret", false, null, false, null);
        assertEquals("0", env.get("FEDLEARN_REQUIRE_CLIENT_AUTH"));  // off unless explicitly required
        assertNull(env.get("FEDLEARN_BACKEND_URL"));                 // blank backend url -> unset
        assertNull(env.get("FEDLEARN_RUN_ID"));                      // null run -> unset
        assertNull(env.get("FEDLEARN_INTERNAL_RUN_TOKEN"));          // SE-7: null token -> unset
        assertNull(env.get("FEDLEARN_REQUIRE_TLS"));                 // SE-2: TLS not forced -> plaintext (dev/demo)
        assertNull(env.get("FEDLEARN_GRPC_USE_TLS"));
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

    // --- SE-11: DP flag passthrough. The --dp-* flag names are a pinned contract with
    // fl_server.py's argparse — do not rename. All values are typed numbers formatted via
    // String.valueOf (SE-10: nothing project-derived reaches the argv as a raw string).

    private Project dpProject() {
        Project p = project("CNN");
        p.setDpEnabled(true);
        p.setDpTargetEpsilon(6.0);
        p.setDpDelta(1e-5);
        p.setDpClipNorm(1.5);
        return p;
    }

    @Test
    void dpEnabledProject_carriesTheExactPinnedDpArgv() {
        Project p = dpProject();
        List<String> cmd = FlowerServerManager.buildServerCommand(
                p, "FedAvg", 5, 2, 50000, "/x/run_fl_server.sh", false);
        assertEquals(List.of(
                "bash", "/x/run_fl_server.sh",
                "--project-id", p.getId().toString(),
                "--model-path", "/tmp/model.npz",
                "--port", "50000",
                "--strategy", "FedAvg",
                "--num-rounds", "5",
                "--model-type", "CNN",
                "--model-name", "qwen2.5-0.5b",
                "--min-clients", "2",
                "--dp-enabled",
                "--dp-clip-norm", "1.5",
                "--dp-target-epsilon", "6.0",
                "--dp-delta", "1.0E-5",
                "--dp-rounds", "5",
                "--dp-num-clients", "2"), cmd);
    }

    @Test
    void nonDpProject_argvIsByteForByteUnchanged() {
        Project p = project("CNN");
        List<String> cmd = FlowerServerManager.buildServerCommand(
                p, "FedAvg", 5, 2, 50000, "/x/run_fl_server.sh", false);
        assertEquals(List.of(
                "bash", "/x/run_fl_server.sh",
                "--project-id", p.getId().toString(),
                "--model-path", "/tmp/model.npz",
                "--port", "50000",
                "--strategy", "FedAvg",
                "--num-rounds", "5",
                "--model-type", "CNN",
                "--model-name", "qwen2.5-0.5b",
                "--min-clients", "2"), cmd);
        assertTrue(cmd.stream().noneMatch(a -> a.startsWith("--dp")),
                "a non-DP project must emit no --dp-* flags");
    }

    @Test
    void dpEnabledWithIncompleteConfig_failsClosedAtSpawn() {
        // Creation validates completeness, but the spawn seam re-checks: a null knob must never
        // reach the argv as the string "null".
        Project p = dpProject();
        p.setDpClipNorm(null);
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FedAvg", 5, 2, 50000, "/x/run_fl_server.sh", false));
    }

    @Test
    void dpEnabledFoTRun_isRejected() {
        // The FoT text-federation server has no DP contract; spawning it for a DP-enabled project
        // would silently train without DP. Fail closed instead.
        Project p = dpProject();
        assertThrows(IllegalArgumentException.class, () ->
                FlowerServerManager.buildServerCommand(p, "FoT", 5, 2, 50000, "/x/run_fot.sh", false));
    }
}
