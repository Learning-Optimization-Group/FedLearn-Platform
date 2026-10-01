package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.CapabilityReport;
import com.federated.fl_platform_api.dto.EnrollmentDto;
import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Stage 3 D1, HTTP side: enroll takes an optional capability report. Clients that send no body (the desktop app, the
 * laptop CLI) enroll exactly as before; a report outside its bounds is refused with a 400 rather than stored.
 */
class RunControllerEnrollTest {

    private MockMvc mvc;
    private RunService runService;
    private final UUID runId = UUID.randomUUID();

    @BeforeEach
    void setUp() {
        runService = mock(RunService.class);
        when(runService.enroll(eq(runId), any())).thenReturn(new EnrollmentDto());
        RunController controller = new RunController();
        ReflectionTestUtils.setField(controller, "runService", runService);
        mvc = MockMvcBuilders.standaloneSetup(controller).build();
    }

    @Test
    void anEnrollmentWithoutABodyIsUnchanged() throws Exception {
        mvc.perform(post("/api/runs/" + runId + "/enroll")).andExpect(status().isOk());
        ArgumentCaptor<CapabilityReport> report = ArgumentCaptor.forClass(CapabilityReport.class);
        verify(runService).enroll(eq(runId), report.capture());
        assertNull(report.getValue());
    }

    @Test
    void aReportIsPassedOnWithTheEnrollment() throws Exception {
        mvc.perform(post("/api/runs/" + runId + "/enroll").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capabilityReport\":{\"platform\":\"android\",\"apiLevel\":27,\"batteryPct\":80}}"))
                .andExpect(status().isOk());
        ArgumentCaptor<CapabilityReport> report = ArgumentCaptor.forClass(CapabilityReport.class);
        verify(runService).enroll(eq(runId), report.capture());
        assertEquals("android", report.getValue().platform());
        assertEquals(27, report.getValue().apiLevel());
    }

    @Test
    void aReportOutsideItsBoundsIsRefused() throws Exception {
        mvc.perform(post("/api/runs/" + runId + "/enroll").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"capabilityReport\":{\"platform\":\"android\",\"batteryPct\":150}}"))
                .andExpect(status().isBadRequest());
        verify(runService, never()).enroll(any(), any());
    }
}
