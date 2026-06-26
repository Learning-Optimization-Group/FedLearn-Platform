package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import org.mockito.InOrder;
import java.util.UUID;
import static org.mockito.Mockito.*;

class InferenceServiceBroadcastTest {
    @Test
    void broadcastsOnlyTokenLinesInOrder() {
        ProjectService ps = mock(ProjectService.class);
        WebSocketService ws = mock(WebSocketService.class);
        InferenceService svc = new InferenceService(ps, ws, 2, 300);
        UUID pid = UUID.randomUUID();

        assertBroadcast(svc, pid, "{\"token\":\"He\"}", true);
        assertBroadcast(svc, pid, "{\"token\":\"llo\"}", true);
        assertBroadcast(svc, pid, "[infer] loading model", false);   // diag line, not JSON
        assertBroadcast(svc, pid, "{\"ok\":true}", false);            // JSON but no "token"

        InOrder ord = inOrder(ws);
        ord.verify(ws).sendInferenceToken(pid, "{\"token\":\"He\"}");
        ord.verify(ws).sendInferenceToken(pid, "{\"token\":\"llo\"}");
        verify(ws, times(2)).sendInferenceToken(any(), any());
    }

    private void assertBroadcast(InferenceService svc, UUID pid, String line, boolean expected) {
        org.junit.jupiter.api.Assertions.assertEquals(expected, svc.broadcastIfToken(pid, line));
    }
}
