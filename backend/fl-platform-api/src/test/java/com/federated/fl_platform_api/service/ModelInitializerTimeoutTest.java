package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.exception.ServerProcessException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * BA-1: model-init must not block the request (and hold a DB connection) forever on a hung python
 * process — the waitFor is bounded and a timed-out process is killed. Spawns a real hanging script,
 * so it's Unix-only (bash/sleep); CI runs on Linux.
 */
@DisabledOnOs(OS.WINDOWS)
class ModelInitializerTimeoutTest {

    @Test
    void initializeModelFile_timesOutAndKillsAHungProcess() throws Exception {
        Path script = Files.createTempFile("hang-init", ".sh");
        Files.writeString(script, "#!/bin/bash\nsleep 20\n");   // ignores args, hangs well past the timeout
        try {
            ModelInitializer init = new ModelInitializer();
            ReflectionTestUtils.setField(init, "initModelWrapperPath", script.toString());
            ReflectionTestUtils.setField(init, "initTimeoutSeconds", 1L);

            long start = System.currentTimeMillis();
            assertThrows(ServerProcessException.class, () ->
                    init.initializeModelFile("CNN", "m", "Adam", "/tmp/ba1-out.npz", 0, null));
            long elapsedMs = System.currentTimeMillis() - start;

            assertTrue(elapsedMs < 10_000,
                    "should fail on the 1s timeout, not wait out the 20s sleep (elapsed=" + elapsedMs + "ms)");
        } finally {
            Files.deleteIfExists(script);
        }
    }
}
