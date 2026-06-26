package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.Project;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ProjectServiceTaskTypeTest {
    @Test
    void taskTypeRoundTripsOnEntity() {
        Project p = new Project();
        p.setTaskType("CAUSAL_LM");
        assertEquals("CAUSAL_LM", p.getTaskType());
    }

    @Test
    void defaultTaskTypeWhenAbsent() {
        // resolveTaskType returns SEQ_CLASSIFICATION for null/blank, CAUSAL_LM only when asked.
        assertEquals("SEQ_CLASSIFICATION", ProjectService.resolveTaskType(null));
        assertEquals("SEQ_CLASSIFICATION", ProjectService.resolveTaskType(""));
        assertEquals("CAUSAL_LM", ProjectService.resolveTaskType("CAUSAL_LM"));
    }
}
