package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.GenerationRequest;
import com.federated.fl_platform_api.dto.InferenceRequest;
import com.federated.fl_platform_api.exception.ServerProcessException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Inference and generation read the script's output to EOF before waiting with their timeout, and EOF only arrives
 * once every process holding the pipe exits. So a hung script that printed nothing blocked the request for good and
 * the timeout never fired. A watchdog must kill the script's process tree at the timeout.
 */
@DisabledOnOs(OS.WINDOWS)
class InferenceServiceTimeoutTest {

    private static final UUID PROJECT = UUID.randomUUID();

    private static Path hungScript(Path dir, Path pidFile) throws Exception {
        Path script = dir.resolve("hang.sh");
        Files.writeString(script, "#!/bin/bash\nsleep 300 &\necho $! > " + pidFile + "\nwait\n");
        return script;
    }

    private static InferenceService service(Path script, String inputKind) {
        ProjectService projects = mock(ProjectService.class);
        when(projects.resolveInferenceTarget(PROJECT)).thenReturn(
                new ProjectService.InferenceTarget("/tmp/m.npz", "TINYNET_GOLDEN", "m", "COMPLETED", null));
        when(projects.inputKindFor(any())).thenReturn(inputKind);
        when(projects.inputKindFor(any(), any())).thenReturn(inputKind);
        InferenceService svc = new InferenceService(projects, mock(WebSocketService.class), 2, 1);
        ReflectionTestUtils.setField(svc, "inferWrapperPath", script.toString());
        ReflectionTestUtils.setField(svc, "inferenceTimeoutSeconds", 1L);
        return svc;
    }

    private static void assertForkedChildIsDead(Path pidFile) throws Exception {
        ProcessHandle child = ProcessHandle.of(Long.parseLong(Files.readString(pidFile).trim())).orElse(null);
        await().atMost(10, TimeUnit.SECONDS)
                .untilAsserted(() -> assertTrue(child == null || !child.isAlive(), "the script's child survived"));
    }

    @Test
    void aHungInferenceTimesOutAndIsKilled(@TempDir Path dir) throws Exception {
        Path pidFile = dir.resolve("child.pid");
        InferenceService svc = service(hungScript(dir, pidFile), "vector");
        InferenceRequest request = new InferenceRequest();
        request.setValues(List.of(0.1, 0.2, 0.3, 0.4));

        assertTimeoutPreemptively(Duration.ofSeconds(15), () -> {
            ServerProcessException e = assertThrows(ServerProcessException.class,
                    () -> svc.runInference(PROJECT, request));
            assertTrue(e.getMessage().contains("timed out"), e.getMessage());
        });
        assertForkedChildIsDead(pidFile);
    }

    @Test
    void aHungGenerationTimesOutAndIsKilled(@TempDir Path dir) throws Exception {
        Path pidFile = dir.resolve("child.pid");
        InferenceService svc = service(hungScript(dir, pidFile), "generation");
        GenerationRequest request = new GenerationRequest();
        request.setPrompt("hello");

        assertTimeoutPreemptively(Duration.ofSeconds(15), () -> {
            ServerProcessException e = assertThrows(ServerProcessException.class, () -> svc.generate(PROJECT, request));
            assertTrue(e.getMessage().contains("timed out"), e.getMessage());
        });
        assertForkedChildIsDead(pidFile);
    }
}
