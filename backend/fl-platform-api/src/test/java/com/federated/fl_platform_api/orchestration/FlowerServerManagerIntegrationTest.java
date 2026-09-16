package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.exception.ServerProcessException;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.WebSocketService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * TE-7: end-to-end integration test for the FL-server orchestration path — the cross-component seam
 * that the rest of the suite only covers with static string assertions (see
 * {@link FlServerManagerCommandTest}). Here we actually drive
 * {@link FlServerManager#startServerForProject} against a hermetic stub script and assert the
 * real behaviour: a port in {@code 50000-50010} is reserved, a child process is spawned, its stdout
 * is streamed to {@link WebSocketService#sendLogs} (i.e. broadcast to {@code /topic/logs/{projectId}}),
 * {@code stop} force-terminates the child and releases the port for reuse, and an early child exit is
 * surfaced as a {@link ServerProcessException} carrying the captured output within the startup window.
 *
 * <p><b>Test seam (zero production change):</b> the wrapper-script path, the port range and the timing
 * knobs are all {@code @Value}-injected fields with production defaults. We construct the manager with
 * {@code new} and override those fields via {@link ReflectionTestUtils} — the same pattern used by
 * {@link com.federated.fl_platform_api.service.ModelInitializerTimeoutTest} — so the manager spawns a
 * tiny bash/python stub instead of the real {@code run_fl_server.sh}. No production code is modified.</p>
 *
 * <p>The stubs use bash + a trivial python socket bind (no torch, no framework), so the test is
 * Unix-only; CI runs on Linux.</p>
 */
@DisabledOnOs(OS.WINDOWS)
class FlServerManagerIntegrationTest {

    private static final int RANGE_START = 50000;
    private static final int RANGE_END = 50010;

    private FlServerManager manager;
    private WebSocketService ws;
    private RunTokenRegistry runTokenRegistry;

    private Path aliveStub;
    private Path crashStub;
    private final List<UUID> startedProjects = new ArrayList<>();

    @BeforeEach
    void setUp() throws IOException {
        ws = mock(WebSocketService.class);
        runTokenRegistry = mock(RunTokenRegistry.class);
        when(runTokenRegistry.mint(any(), any())).thenReturn("test-run-token");

        manager = new FlServerManager();
        ReflectionTestUtils.setField(manager, "logBroadcaster", ws);
        ReflectionTestUtils.setField(manager, "runTokenRegistry", runTokenRegistry);
        // BA-3: the stub project carries no active run, so recordProcessIdentity short-circuits — but
        // wire a mock repo so the field is never null if that changes.
        ReflectionTestUtils.setField(manager, "runRepository",
                mock(com.federated.fl_platform_api.repository.RunRepository.class));

        com.federated.fl_platform_api.service.ModelRecipeService modelRecipeService =
                mock(com.federated.fl_platform_api.service.ModelRecipeService.class);
        when(modelRecipeService.findByKey(any())).thenReturn(Optional.of(
                new com.federated.fl_platform_api.dto.ModelRecipeDto(
                        "CNN", "CNN", "image",
                        java.util.List.of(), java.util.List.of(), java.util.List.of(), null)));
        ReflectionTestUtils.setField(manager, "modelRecipeService", modelRecipeService);
        // BA-11: a bare-mock resolver returns Optional.empty() (no registry head) → no --init-model-path.
        ReflectionTestUtils.setField(manager, "registryModelResolver",
                new com.federated.fl_platform_api.service.RegistryModelResolver(
                        mock(com.federated.fl_platform_api.repository.ModelArtifactRepository.class),
                        mock(com.federated.fl_platform_api.service.ArtifactBlobStore.class), "unused"));

        ReflectionTestUtils.setField(manager, "portRangeStart", RANGE_START);
        ReflectionTestUtils.setField(manager, "portRangeEnd", RANGE_END);
        // Short probe window so a healthy start returns quickly; still long enough for the stub to
        // print its markers and stay alive past the window.
        ReflectionTestUtils.setField(manager, "startupProbeSeconds", 2L);
        ReflectionTestUtils.setField(manager, "stdoutDrainMillis", 2000L);

        // Alive stub: prints two known marker lines, then binds the --port it was handed (on BOTH the
        // IPv4 and IPv6 wildcard) and blocks. Binding both families is deliberate: the manager detects a
        // taken port with a plain Java `new ServerSocket(port)`, which binds the IPv6 wildcard by default
        // on macOS but the IPv4 wildcard on Linux — an IPv4-only child would be invisible to the probe on
        // macOS. `exec` replaces bash with python so the tracked PID becomes the port holder, and so
        // destroyForcibly actually frees the port (a forked child would survive the kill and keep it bound).
        aliveStub = Files.createTempFile("stub-fl-alive", ".sh");
        Files.writeString(aliveStub,
                "#!/bin/bash\n"
                        + "PORT=\"\"\n"
                        + "while [ \"$#\" -gt 0 ]; do\n"
                        + "  case \"$1\" in\n"
                        + "    --port) PORT=\"$2\"; shift 2 ;;\n"
                        + "    *) shift ;;\n"
                        + "  esac\n"
                        + "done\n"
                        + "echo \"STUB_FL_SERVER_MARKER_1 started\"\n"
                        + "echo \"STUB_FL_SERVER_MARKER_2 port=${PORT}\"\n"
                        + "exec python3 -c \"import socket, time\n"
                        + "p = int('${PORT}')\n"
                        + "s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                        + "s4.bind(('0.0.0.0', p)); s4.listen(1)\n"
                        + "s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
                        + "s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)\n"
                        + "s6.bind(('::', p)); s6.listen(1)\n"
                        + "time.sleep(300)\"\n");

        // Crash stub: prints one error line to stdout, then exits non-zero immediately.
        crashStub = Files.createTempFile("stub-fl-crash", ".sh");
        Files.writeString(crashStub, "#!/bin/bash\necho \"STUB_FL_SERVER_CRASH boom\"\nexit 1\n");
    }

    @AfterEach
    void tearDown() throws IOException {
        for (UUID id : startedProjects) {
            try {
                manager.stopServerForProject(id);
            } catch (RuntimeException ignored) {
                // best-effort cleanup
            }
        }
        Files.deleteIfExists(aliveStub);
        Files.deleteIfExists(crashStub);
    }

    @Test
    void startSpawnsChildReservesPortInRangeAndBroadcastsStdout() {
        useWrapper(aliveStub);
        Project p = project("CNN");

        Optional<Integer> port = manager.startServerForProject(p, "FedAvg", 5, 1);
        startedProjects.add(p.getId());

        assertTrue(port.isPresent(), "the local-process path must reserve and return a port");
        assertTrue(port.get() >= RANGE_START && port.get() <= RANGE_END,
                "reserved port must fall inside the configured 50000-50010 range, was " + port.get());
        assertTrue(manager.isServerRunning(p.getId()), "the child FL-server process must be alive");

        // The stub's stdout marker lines must be streamed to WebSocketService.sendLogs(projectId, line),
        // i.e. broadcast to /topic/logs/{projectId}. Poll to stay robust against the reader-thread timing.
        await().atMost(10, SECONDS).untilAsserted(() -> {
            verify(ws, atLeastOnce()).sendLogs(eq(p.getId()), contains("STUB_FL_SERVER_MARKER_1"));
            verify(ws, atLeastOnce()).sendLogs(eq(p.getId()), contains("STUB_FL_SERVER_MARKER_2"));
        });
    }

    @Test
    void stopForceTerminatesChildAndReleasesPortForReuse() throws IOException {
        int port = pickFreePortInRange();
        useWrapper(aliveStub);
        // Constrain the range to a single port so "held while running / free after stop" is directly
        // observable through a second reservation attempt.
        ReflectionTestUtils.setField(manager, "portRangeStart", port);
        ReflectionTestUtils.setField(manager, "portRangeEnd", port);

        Project p1 = project("CNN");
        Optional<Integer> first = manager.startServerForProject(p1, "FedAvg", 5, 1);
        startedProjects.add(p1.getId());
        assertEquals(Optional.of(port), first, "the only free port in range must be the one reserved");
        assertTrue(manager.isServerRunning(p1.getId()));

        // Once the child has actually bound the port, a different project cannot reserve it.
        await().atMost(10, SECONDS).until(() -> isPortBound(port));
        Project p2 = project("CNN");
        assertThrows(IllegalStateException.class,
                () -> manager.startServerForProject(p2, "FedAvg", 5, 1),
                "the single port is held by the running child, so a new reservation must fail");

        // Stopping p1 force-terminates the child and frees the port.
        assertTrue(manager.stopServerForProject(p1.getId()), "stop must report it terminated a live server");
        assertFalse(manager.isServerRunning(p1.getId()), "the child must be gone after stop");

        // The freed port can now be reserved again (poll: the OS may need a moment to release it).
        Project p3 = project("CNN");
        await().atMost(10, SECONDS).ignoreExceptions().untilAsserted(() -> {
            Optional<Integer> reuse = manager.startServerForProject(p3, "FedAvg", 5, 1);
            assertEquals(Optional.of(port), reuse, "the released port must be reservable again");
        });
        startedProjects.add(p3.getId());
        assertTrue(manager.isServerRunning(p3.getId()));
    }

    @Test
    void earlyChildExitSurfacesCapturedOutputWithinStartupWindow() {
        useWrapper(crashStub);
        Project p = project("CNN");

        long start = System.currentTimeMillis();
        ServerProcessException ex = assertThrows(ServerProcessException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 1));
        long elapsedMs = System.currentTimeMillis() - start;

        assertTrue(ex.getMessage().contains("STUB_FL_SERVER_CRASH boom"),
                "the captured child stdout must be surfaced in the thrown exception, was: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("exit code 1"),
                "the exception must report the child's non-zero exit code, was: " + ex.getMessage());
        assertTrue(elapsedMs < 8_000,
                "an early exit must fail fast within the startup window, not hang; elapsed=" + elapsedMs + "ms");
        assertFalse(manager.isServerRunning(p.getId()), "no server should be tracked after an early-exit failure");

        // The port reserved for the failed start must have been released — an alive stub can reserve
        // within the same range straight after.
        useWrapper(aliveStub);
        Optional<Integer> port = manager.startServerForProject(p, "FedAvg", 5, 1);
        startedProjects.add(p.getId());
        assertTrue(port.isPresent() && port.get() >= RANGE_START && port.get() <= RANGE_END,
                "the port from the failed start must be free for a subsequent successful start");
    }

    // --- helpers ------------------------------------------------------------

    private void useWrapper(Path stub) {
        ReflectionTestUtils.setField(manager, "flServerWrapperPath", stub.toString());
    }

    private Project project(String modelType) {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType(modelType);
        p.setModelName("qwen2.5-0.5b");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    private int pickFreePortInRange() throws IOException {
        for (int candidate = RANGE_START; candidate <= RANGE_END; candidate++) {
            try (ServerSocket s = new ServerSocket(candidate)) {
                return s.getLocalPort();
            } catch (IOException ignored) {
                // in use on this host — try the next
            }
        }
        throw new IllegalStateException("no free port in test range " + RANGE_START + "-" + RANGE_END);
    }

    private boolean isPortBound(int port) {
        try (ServerSocket s = new ServerSocket(port)) {
            return false;
        } catch (IOException e) {
            return true;
        }
    }
}
