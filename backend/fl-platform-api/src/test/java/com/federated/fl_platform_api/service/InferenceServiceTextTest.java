package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.InferenceRequest;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InferenceServiceTextTest {

    @Test
    void textIsNowAnInteractiveInputKind() {
        assertTrue(InferenceService.isInteractiveInputKind("text"));
        assertTrue(InferenceService.isInteractiveInputKind("image"));
        assertTrue(InferenceService.isInteractiveInputKind("vector"));
        assertFalse(InferenceService.isInteractiveInputKind(null));
        assertFalse(InferenceService.isInteractiveInputKind("audio"));
    }

    @Test
    void textFieldRoundTrips() {
        InferenceRequest r = new InferenceRequest();
        r.setText("a great movie");
        assertEquals("a great movie", r.getText());
    }
}
