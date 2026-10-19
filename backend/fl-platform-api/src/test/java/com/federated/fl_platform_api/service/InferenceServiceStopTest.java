package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;
import java.util.UUID;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class InferenceServiceStopTest {

    private InferenceService newService() {
        return new InferenceService(mock(ProjectService.class), mock(WebSocketService.class), 2, 300);
    }

    @Test
    void stopTrackedReturnsFalseWhenNothingRunning() {
        assertFalse(newService().stopTrackedGeneration(UUID.randomUUID()));
    }

    @Test
    void stopTrackedKillsAndFlagsWhenRunning() {
        InferenceService svc = newService();
        UUID pid = UUID.randomUUID();
        Process p = mock(Process.class);
        svc.runningGenerations.put(pid, p);

        assertTrue(svc.stopTrackedGeneration(pid));
        verify(p).destroyForcibly();
        assertTrue(svc.stoppedGenerations.contains(pid));
    }

    // The generation script is a bash wrapper that forks python; stopping must end the python, not only bash, or the
    // "stopped" generation keeps running and streaming.
    @Test
    @org.junit.jupiter.api.condition.DisabledOnOs(org.junit.jupiter.api.condition.OS.WINDOWS)
    void stopTrackedKillsWhatTheGenerationScriptForked() throws Exception {
        InferenceService svc = newService();
        UUID pid = UUID.randomUUID();
        java.nio.file.Path pidFile = java.nio.file.Files.createTempFile("gen", ".pid");
        java.nio.file.Files.delete(pidFile);
        Process wrapper = com.federated.fl_platform_api.orchestration.ProcessTreesTest.startForkingWrapper(pidFile);
        ProcessHandle child = com.federated.fl_platform_api.orchestration.ProcessTreesTest.forkedChild(pidFile);
        svc.runningGenerations.put(pid, wrapper);

        assertTrue(svc.stopTrackedGeneration(pid));

        org.awaitility.Awaitility.await().atMost(10, java.util.concurrent.TimeUnit.SECONDS)
                .untilAsserted(() -> assertFalse(child.isAlive(), "the forked generation process survived stop"));
        java.nio.file.Files.deleteIfExists(pidFile);
    }

    @Test
    void stoppedResultHasStoppedFinishReason() {
        JsonNode n = newService().stoppedResult("LLM_LORA");
        assertTrue(n.path("ok").asBoolean());
        assertEquals("LLM_LORA", n.path("modelType").asText());
        assertEquals("stopped", n.path("finishReason").asText());
        assertEquals("", n.path("generatedText").asText());
        assertEquals(0, n.path("tokenCount").asInt());
    }
}
