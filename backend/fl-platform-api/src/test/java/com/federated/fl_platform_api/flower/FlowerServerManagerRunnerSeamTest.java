package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.exception.ServerProcessException;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.ModelRecipeService;
import com.federated.fl_platform_api.service.WebSocketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.contains;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * DA-8: the FL-server orchestration seam ({@link FlServerProcessRunner}) makes the spawn path
 * unit-testable WITHOUT launching a real process — the whole orchestration (build command → apply the
 * SE-1/SE-7 env contract → track handle → broadcast stdout → startup probe → surface an early exit)
 * runs against a fake runner. This complements {@link FlowerServerManagerIntegrationTest} (which still
 * exercises the real {@link LocalProcessFlServerRunner} end to end for behaviour preservation).
 */
class FlowerServerManagerRunnerSeamTest {

    private FlowerServerManager manager;
    private WebSocketService ws;

    @BeforeEach
    void setUp() {
        ws = mock(WebSocketService.class);

        RunTokenRegistry runTokenRegistry = mock(RunTokenRegistry.class);
        when(runTokenRegistry.mint(any(), any())).thenReturn("per-run-token");
        when(runTokenRegistry.hash(any())).thenReturn("tokenhash");

        ModelRecipeService recipes = mock(ModelRecipeService.class);
        when(recipes.findByKey(any())).thenReturn(Optional.of(
                new ModelRecipeDto("CNN", "CNN", "image",
                        List.of(), List.of(), List.of(), null)));

        manager = new FlowerServerManager();
        ReflectionTestUtils.setField(manager, "logBroadcaster", ws);
        ReflectionTestUtils.setField(manager, "runTokenRegistry", runTokenRegistry);
        ReflectionTestUtils.setField(manager, "runRepository", mock(RunRepository.class));
        ReflectionTestUtils.setField(manager, "modelRecipeService", recipes);
        // BA-11: a bare-mock resolver returns Optional.empty() (no registry head) → no --init-model-path,
        // i.e. the pre-BA-11 spawn behavior these tests assert.
        ReflectionTestUtils.setField(manager, "registryModelResolver",
                new com.federated.fl_platform_api.service.RegistryModelResolver(
                        mock(com.federated.fl_platform_api.repository.ModelArtifactRepository.class),
                        mock(com.federated.fl_platform_api.service.ArtifactBlobStore.class), "unused"));
        ReflectionTestUtils.setField(manager, "internalApiKey", "the-api-key");
        // The fake runner ignores the command's script path, but the manager still builds an absolute
        // File() from it — a null would NPE before the runner is reached (@Value default isn't applied
        // under `new FlowerServerManager()`).
        ReflectionTestUtils.setField(manager, "flServerWrapperPath", "run_fl_server.sh");
        ReflectionTestUtils.setField(manager, "portRangeStart", 50000);
        ReflectionTestUtils.setField(manager, "portRangeEnd", 50010);
        ReflectionTestUtils.setField(manager, "startupProbeSeconds", 1L);
        ReflectionTestUtils.setField(manager, "stdoutDrainMillis", 2000L);
    }

    private Project project() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("qwen2.5-0.5b");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    @Test
    void start_delegatesTheBuiltCommandToTheRunner_tracksHandle_andReturnsReservedPort() {
        FakeRunner runner = new FakeRunner(FakeProcess.alive("STUB started\n"));
        ReflectionTestUtils.setField(manager, "processRunner", runner);
        Project p = project();

        Optional<Integer> port = manager.startServerForProject(p, "FedAvg", 5, 1);

        assertTrue(port.isPresent(), "the seam path must still reserve and return a port");
        assertTrue(port.get() >= 50000 && port.get() <= 50010);
        assertTrue(manager.isServerRunning(p.getId()), "the (fake) process handle must be tracked");

        // The manager built the argv and handed it to the runner — the seam only executes, never builds.
        assertEquals(1, runner.startCount);
        assertTrue(runner.lastCommand.contains("--model-type"));
        assertTrue(runner.lastCommand.contains("CNN"));
        assertTrue(runner.lastCommand.contains(String.valueOf(port.get())),
                "the reserved port must be on the argv handed to the runner");

        manager.stopServerForProject(p.getId());
    }

    @Test
    void start_appliesTheSecurityEnvContract_throughTheRunnerCustomizer() {
        // Seed the child env with a web-auth secret to prove the customizer scrubs it (trust-domain
        // isolation, SE-1) and injects the per-run internal token (SE-7) — the invariant survives the seam.
        FakeRunner runner = new FakeRunner(FakeProcess.alive("ok\n"));
        runner.seedEnv.put("APP_JWT_SECRET", "web-secret-must-not-reach-child");
        ReflectionTestUtils.setField(manager, "processRunner", runner);
        Project p = project();

        manager.startServerForProject(p, "FedAvg", 5, 1);

        assertNull(runner.lastEnv.get("APP_JWT_SECRET"), "SE-1: the web-auth secret must be scrubbed");
        assertEquals("per-run-token", runner.lastEnv.get("FEDLEARN_INTERNAL_RUN_TOKEN"), "SE-7 token");
        assertEquals("the-api-key", runner.lastEnv.get("FEDLEARN_INTERNAL_API_KEY"));

        manager.stopServerForProject(p.getId());
    }

    @Test
    void start_immediateChildExit_surfacesCapturedOutput_withoutARealProcess() {
        FakeRunner runner = new FakeRunner(FakeProcess.exitsWith(1, "STUB_CRASH boom\n"));
        ReflectionTestUtils.setField(manager, "processRunner", runner);
        Project p = project();

        ServerProcessException ex = assertThrows(ServerProcessException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 1));

        assertTrue(ex.getMessage().contains("STUB_CRASH boom"),
                "captured child stdout must surface in the exception, was: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("exit code 1"));
        assertFalse(manager.isServerRunning(p.getId()), "nothing tracked after an early-exit failure");

        // And the child's stdout was broadcast to /topic/logs/{projectId} before the failure surfaced.
        await().atMost(5, TimeUnit.SECONDS).untilAsserted(() ->
                verify(ws, atLeastOnce()).sendLogs(eq(p.getId()), contains("STUB_CRASH boom")));
    }

    // --- fakes --------------------------------------------------------------

    /** A {@link FlServerProcessRunner} that records the command/env and hands back a canned process. */
    private static final class FakeRunner implements FlServerProcessRunner {
        private final FakeProcess process;
        final Map<String, String> seedEnv = new HashMap<>();
        List<String> lastCommand;
        Map<String, String> lastEnv;
        int startCount;

        FakeRunner(FakeProcess process) {
            this.process = process;
        }

        @Override
        public SpawnedFlProcess start(List<String> command, Consumer<Map<String, String>> envCustomizer,
                                      File workingDir) {
            startCount++;
            lastCommand = command;
            Map<String, String> env = new HashMap<>(seedEnv);
            envCustomizer.accept(env); // run the real configureChildEnv contract against a controllable map
            lastEnv = env;
            return process;
        }
    }

    /** A {@link SpawnedFlProcess} with no OS process behind it — canned stdout + exit behaviour. */
    private static final class FakeProcess implements SpawnedFlProcess {
        private final byte[] stdout;
        private final boolean exits;
        private final int exitCode;

        private FakeProcess(String stdout, boolean exits, int exitCode) {
            this.stdout = stdout.getBytes(StandardCharsets.UTF_8);
            this.exits = exits;
            this.exitCode = exitCode;
        }

        static FakeProcess alive(String stdout) {
            return new FakeProcess(stdout, false, -1);
        }

        static FakeProcess exitsWith(int code, String stdout) {
            return new FakeProcess(stdout, true, code);
        }

        @Override
        public long pid() {
            return 4242L;
        }

        @Override
        public Optional<Instant> startInstant() {
            return Optional.of(Instant.parse("2026-07-09T00:00:00Z"));
        }

        @Override
        public ProcessHandle toHandle() {
            ProcessHandle handle = mock(ProcessHandle.class);
            when(handle.isAlive()).thenReturn(!exits);
            // stopServerForProject calls handle.onExit().get(...) — a bare mock returns null → NPE.
            when(handle.onExit()).thenReturn(java.util.concurrent.CompletableFuture.completedFuture(handle));
            return handle;
        }

        @Override
        public InputStream getInputStream() {
            return new ByteArrayInputStream(stdout);
        }

        @Override
        public boolean waitFor(long timeout, TimeUnit unit) {
            // "alive" never exits within the probe window; a crash reports it exited immediately.
            return exits;
        }

        @Override
        public int exitValue() {
            return exitCode;
        }

        @Override
        public boolean isAlive() {
            return !exits;
        }

        @Override
        public void destroyForcibly() {
            // no-op — nothing to kill
        }
    }
}
