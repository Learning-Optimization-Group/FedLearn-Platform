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
