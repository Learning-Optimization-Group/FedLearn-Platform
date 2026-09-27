package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Stage 3 slice A2, HTTP side: an artifact is served with a strong ETag equal to its SHA-256 and honours byte ranges,
 * so a phone can resume a large download. A Range whose If-Range validator is stale gets the whole file, never a
 * slice of different bytes.
 */
class RunControllerArtifactTest {

    @TempDir Path dir;
    private MockMvc mvc;
    private final UUID runId = UUID.randomUUID();
    private final String sha = "ab".repeat(32);
    private final byte[] bytes = new byte[1000];

    @BeforeEach
    void setUp() throws Exception {
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) (i % 251);
        Path file = Files.write(dir.resolve("loss.pte"), bytes);
        RunService runService = mock(RunService.class);
        when(runService.getContractArtifact(runId, sha)).thenReturn(new RunService.ContractArtifact(file, sha, bytes.length));
        RunController controller = new RunController();
        ReflectionTestUtils.setField(controller, "runService", runService);
        mvc = MockMvcBuilders.standaloneSetup(controller).build();
    }

    private String url() {
        return "/api/runs/" + runId + "/artifacts/" + sha;
    }

    private MvcResult perform(org.springframework.test.web.servlet.RequestBuilder request) throws Exception {
        MvcResult started = mvc.perform(request).andExpect(request().asyncStarted()).andReturn();
        return mvc.perform(asyncDispatch(started)).andReturn();
    }

    @Test
    void theWholeFileIsServedWithItsHashAsAStrongETag() throws Exception {
        MvcResult r = mvc.perform(asyncDispatch(mvc.perform(get(url())).andReturn()))
                .andExpect(status().isOk())
                .andExpect(header().string("ETag", "\"" + sha + "\""))
                .andExpect(header().string("Accept-Ranges", "bytes"))
                .andReturn();
        assertArrayEquals(bytes, r.getResponse().getContentAsByteArray());
    }

    @Test
    void aRangeWithAMatchingValidatorGetsTheRestOfTheFile() throws Exception {
        MvcResult r = mvc.perform(asyncDispatch(
                        mvc.perform(get(url()).header("Range", "bytes=600-").header("If-Range", "\"" + sha + "\"")).andReturn()))
                .andExpect(status().isPartialContent())
                .andExpect(header().string("Content-Range", "bytes 600-999/1000"))
                .andReturn();
        assertArrayEquals(Arrays.copyOfRange(bytes, 600, 1000), r.getResponse().getContentAsByteArray());
    }

    @Test
    void aRangeWithAStaleValidatorGetsTheWholeFile() throws Exception {
        MvcResult r = perform(get(url()).header("Range", "bytes=600-").header("If-Range", "\"stale\""));
        assertEquals(200, r.getResponse().getStatus());
        assertArrayEquals(bytes, r.getResponse().getContentAsByteArray());
    }

    @Test
    void aBoundedRangeIsServedExactly() throws Exception {
        MvcResult r = perform(get(url()).header("Range", "bytes=10-19"));
        assertEquals(206, r.getResponse().getStatus());
        assertEquals("bytes 10-19/1000", r.getResponse().getHeader("Content-Range"));
        assertArrayEquals(Arrays.copyOfRange(bytes, 10, 20), r.getResponse().getContentAsByteArray());
    }

    @Test
    void aRangeStartingPastTheEndIsUnsatisfiable() throws Exception {
        MvcResult r = mvc.perform(get(url()).header("Range", "bytes=1000-")).andReturn();
        assertEquals(416, r.getResponse().getStatus());
        assertEquals("bytes */1000", r.getResponse().getHeader("Content-Range"));
    }
}
