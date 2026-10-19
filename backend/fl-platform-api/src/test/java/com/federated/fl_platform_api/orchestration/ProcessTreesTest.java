package com.federated.fl_platform_api.orchestration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.file.Files;
import java.nio.file.Path;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * The backend's scripts are bash wrappers that fork Python rather than exec it, so killing only the wrapper orphans
 * the real work. ProcessTrees kills a process together with everything it forked.
 */
@DisabledOnOs(OS.WINDOWS)
public class ProcessTreesTest {

    /** A wrapper that forks a long-lived child, records its PID, and waits on it. */
    public static Process startForkingWrapper(Path pidFile) throws Exception {
        Process p = new ProcessBuilder("bash", "-c", "sleep 300 & echo $! > " + pidFile + "; wait").start();
        await().atMost(10, SECONDS).until(() -> Files.exists(pidFile) && !Files.readString(pidFile).isBlank());
        return p;
    }

    public static ProcessHandle forkedChild(Path pidFile) throws Exception {
        long pid = Long.parseLong(Files.readString(pidFile).trim());
        return ProcessHandle.of(pid).orElseThrow();
    }

    @Test
    void killingAProcessAlsoKillsWhatItForked() throws Exception {
        Path pidFile = Files.createTempFile("forked", ".pid");
        Files.delete(pidFile);
        Process wrapper = startForkingWrapper(pidFile);
        ProcessHandle child = forkedChild(pidFile);

        ProcessTrees.destroyForcibly(wrapper);

        await().atMost(10, SECONDS).untilAsserted(() -> {
            assertFalse(wrapper.isAlive(), "the wrapper must be dead");
            assertFalse(child.isAlive(), "the process the wrapper forked must be dead too");
        });
        Files.deleteIfExists(pidFile);
    }

    @Test
    void aWatchdogKillsTheTreeWhenItExpires() throws Exception {
        Path pidFile = Files.createTempFile("wd", ".pid");
        Files.delete(pidFile);
        Process wrapper = startForkingWrapper(pidFile);
        ProcessHandle child = forkedChild(pidFile);

        try (ProcessTrees.Watchdog watchdog = ProcessTrees.killAfter(wrapper, 1)) {
            await().atMost(10, SECONDS).untilAsserted(() -> {
                assertFalse(wrapper.isAlive());
                assertFalse(child.isAlive());
            });
            org.junit.jupiter.api.Assertions.assertTrue(watchdog.fired());
        }
        Files.deleteIfExists(pidFile);
    }

    @Test
    void closingAWatchdogCancelsTheKill() throws Exception {
        Path pidFile = Files.createTempFile("wd-closed", ".pid");
        Files.delete(pidFile);
        Process wrapper = startForkingWrapper(pidFile);
        ProcessHandle child = forkedChild(pidFile);
        try {
            ProcessTrees.Watchdog watchdog = ProcessTrees.killAfter(wrapper, 1);
            watchdog.close();
            Thread.sleep(2500);
            org.junit.jupiter.api.Assertions.assertTrue(wrapper.isAlive(), "a closed watchdog must not kill");
            assertFalse(watchdog.fired());
        } finally {
            ProcessTrees.destroyForcibly(wrapper);
            await().atMost(10, SECONDS).until(() -> !child.isAlive());
            Files.deleteIfExists(pidFile);
        }
    }
}
