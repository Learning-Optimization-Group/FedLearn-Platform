package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.ModelRecipeService;
import com.federated.fl_platform_api.service.RegistryModelResolver;
import com.federated.fl_platform_api.service.WebSocketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.ServerSocket;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * BA-13: the FL-server orchestrator holds a reserved port for the spawned child's WHOLE LIFE, not just
 * the fixed startup-probe window. The old {@code startLocalServer} finally released the port after the
 * probe on the unverified assumption the child had already bound; but the Python child binds late
 * (torch import + model build routinely > the probe), so the port was freed while still unbound and a
 * concurrent CROSS-project start could grab it — silently cross-wiring two servers onto one port.
 *
 * <p>This is a PURE Mockito unit test (no {@code @SpringBootTest} / Testcontainers): it drives the real
 * orchestration through the {@link FlServerProcessRunner} seam with a fake runner + a controllable
 * {@link ProcessHandle}, so it needs no Docker and runs fast. The port pool is squeezed to a SINGLE
 * free port, so "is the port still reserved?" reduces to "does the private {@code findFreePort()} throw
 * (held) or hand the port straight back (released)?" — the crisp RED/GREEN signal for the fix.</p>
 *
 * <p>Contract pinned here:</p>
 * <ol>
 *   <li>after a SUCCESSFUL start the reserved port stays held (findFreePort throws — nothing free);</li>
 *   <li>{@code stopServerForProject} releases the held port back to the pool;</li>
 *   <li>the child exiting (its {@code onExit} future completing) releases the port AND evicts the
 *       tracking entry via the {@code onChildExit} watcher.</li>
 * </ol>
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)   // manual mocks + not-every-stub-used-in-every-test
class FlServerManagerPortLifecycleTest {

    private FlServerManager manager;

    /** The single controllable child handle the fake process hands back on every toHandle() call. */
    private ProcessHandle childHandle;
    /** The child's onExit future — completing it drives the BA-13 onChildExit watcher. */
    private CompletableFuture<ProcessHandle> childExitFuture;
    /** The one and only port in the manager's range, so the range is exhausted once it is reserved. */
    private int reservedPort;

    @BeforeEach
    void setUp() {
        RunTokenRegistry runTokenRegistry = mock(RunTokenRegistry.class);
        when(runTokenRegistry.mint(any(), any())).thenReturn("tok");
        when(runTokenRegistry.hash(any())).thenReturn("h");
        // evictForProject(any) is void on a mock -> no-op by default.

        ModelRecipeService modelRecipeService = mock(ModelRecipeService.class);
        // SE-10 catalog gate: requireModelTypeInCatalog needs an EXACT-CASE recipe whose key()=="CNN".
        when(modelRecipeService.findByKey("CNN")).thenReturn(Optional.of(
                new ModelRecipeDto("CNN", "CNN", "image", List.of(), List.of(), List.of(), null)));

        RegistryModelResolver registryModelResolver = mock(RegistryModelResolver.class);
        // BA-11: no registry head -> no --init-model-path (the pre-BA-11 spawn shape).
        when(registryModelResolver.resolveModelPath(any())).thenReturn(Optional.empty());

        RunRepository runRepository = mock(RunRepository.class);
        // BA-3 recordProcessIdentity: no bound Run -> the persistence step is a no-op.
        when(runRepository.findById(any())).thenReturn(Optional.empty());

        // One handle instance, returned by every toHandle() call, so the keyed
        // runningServers.remove(projectId, handle) in onChildExit matches what start stored.
        childExitFuture = new CompletableFuture<>();
        childHandle = mock(ProcessHandle.class);
        when(childHandle.isAlive()).thenReturn(true);
        when(childHandle.pid()).thenReturn(1234L);
        // onExit() must return the SAME future on every call (start registers the watcher on it; stop
        // later awaits it) so completing it in the test fires the registered onChildExit callback.
        when(childHandle.onExit()).thenReturn(childExitFuture);

        reservedPort = pickFreePort();

        manager = new FlServerManager();
        ReflectionTestUtils.setField(manager, "logBroadcaster", mock(WebSocketService.class));
        ReflectionTestUtils.setField(manager, "runTokenRegistry", runTokenRegistry);
        ReflectionTestUtils.setField(manager, "runRepository", runRepository);
        ReflectionTestUtils.setField(manager, "modelRecipeService", modelRecipeService);
        ReflectionTestUtils.setField(manager, "registryModelResolver", registryModelResolver);
        ReflectionTestUtils.setField(manager, "processRunner",
                new FakeRunner(new FakeChildProcess(childHandle)));

        // @Value defaults are NOT applied under `new FlServerManager()`, so set every field the start
        // path reads. A null wrapper path would NPE at new File(path) before the runner is reached.
        ReflectionTestUtils.setField(manager, "flServerWrapperPath", "run_fl_server.sh");
        ReflectionTestUtils.setField(manager, "ecsClusterName", "");   // local-process path, not ECS
        ReflectionTestUtils.setField(manager, "portRangeStart", reservedPort);
        ReflectionTestUtils.setField(manager, "portRangeEnd", reservedPort);   // single-port pool
        ReflectionTestUtils.setField(manager, "startupProbeSeconds", 1L);
        ReflectionTestUtils.setField(manager, "stdoutDrainMillis", 100L);
    }

    /**
     * BA-13 CORE: after a successful start the port is held for the child's LIFE. With the pool squeezed
     * to one port, findFreePort() must throw ("No free port") — proving the port was NOT released on the
     * startup-probe timer. Pre-BA-13 the finally freed it after the probe, so findFreePort() would hand
     * the very same port straight back instead of throwing (the RED failure this test catches).
     */
    @Test
    void startedServer_holdsItsPortForTheChildsLife_notJustTheStartupProbe() {
        Project project = newProject();

        Optional<Integer> port = manager.startServerForProject(project, "CNN", 1, 1);

        assertTrue(port.isPresent(), "local-process start must reserve and return a port");
        assertEquals(reservedPort, port.get().intValue());

        assertThrows(IllegalStateException.class,
                () -> ReflectionTestUtils.invokeMethod(manager, "findFreePort"),
                "the reserved port must stay HELD for the child's life, not be freed by the probe timer");
    }

    /** BA-13: stopServerForProject removes + releases the held port, returning it to the pool. */
    @Test
    void stop_releasesTheHeldPort() {
        Project project = newProject();
        manager.startServerForProject(project, "CNN", 1, 1);

        boolean stopped = manager.stopServerForProject(project.getId());

        assertTrue(stopped, "stop must report it terminated the tracked child");
        Integer freed = ReflectionTestUtils.invokeMethod(manager, "findFreePort");
        assertNotNull(freed, "the port must be free again after stop");
        assertEquals(reservedPort, freed.intValue(), "stop must release the held port back to the pool");
    }

    /**
     * BA-13: when the child exits, the onExit watcher (onChildExit) releases the port AND evicts the
     * runningServers entry — so a mid-run crash frees the port and clears tracking without a stop call.
     */
    @Test
    void childExit_releasesThePortAndEvictsTheTrackingEntry() {
        Project project = newProject();
        manager.startServerForProject(project, "CNN", 1, 1);
        assertTrue(manager.isServerRunning(project.getId()), "child must be tracked while alive");

        // Complete the child's onExit future: the watcher registered at start runs synchronously here.
        childExitFuture.complete(childHandle);

        Integer freed = ReflectionTestUtils.invokeMethod(manager, "findFreePort");
        assertNotNull(freed, "child exit must free the port");
        assertEquals(reservedPort, freed.intValue(), "child exit must release the held port");
        assertFalse(manager.isServerRunning(project.getId()),
                "child exit must evict the tracking entry");
    }

    // --- helpers ------------------------------------------------------------

    /** A regulated=false CNN project whose argv fields all pass buildServerCommand's SAFE_* validators. */
    private static Project newProject() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setActiveRunId(UUID.randomUUID());
        p.setRegulated(false);          // SE-11 DP gate skipped (dpEnabled defaults false too)
        p.setModelType("CNN");
        p.setModelPath("model.npz");    // requireSafePath
        p.setModelName("cnn");          // requireSafeModelRef
        p.setTaskType("classification");// requireSafeToken
        return p;
    }

    /** Grab an OS-free port to use as the manager's single-port range (nothing stays bound to it). */
    private static int pickFreePort() {
        try (ServerSocket s = new ServerSocket(0)) {
            return s.getLocalPort();
        } catch (IOException e) {
            throw new IllegalStateException("could not pick a free port for the test", e);
        }
    }

    // --- fakes --------------------------------------------------------------

    /** A {@link FlServerProcessRunner} that hands back a canned process without launching anything. */
    private static final class FakeRunner implements FlServerProcessRunner {
        private final SpawnedFlProcess process;

        FakeRunner(SpawnedFlProcess process) {
            this.process = process;
        }

        @Override
        public SpawnedFlProcess start(List<String> command, Consumer<Map<String, String>> envCustomizer,
                                      File workingDir) {
            return process;   // env customization is irrelevant to the port-lifetime behaviour under test
        }
    }

    /**
     * A {@link SpawnedFlProcess} with no OS process behind it. waitFor() returns false (NOT exited =
     * startup success) and toHandle() always returns the one controllable handle, so the manager's
     * long-lived tracking + onExit watcher operate on a handle the test can drive.
     */
    private static final class FakeChildProcess implements SpawnedFlProcess {
        private final ProcessHandle handle;

        FakeChildProcess(ProcessHandle handle) {
            this.handle = handle;
        }

        @Override
        public long pid() {
            return 1234L;
        }

        @Override
        public Optional<Instant> startInstant() {
            return Optional.of(Instant.now());
        }

        @Override
        public ProcessHandle toHandle() {
            return handle;   // SAME instance every call (put + tracked handle must match for keyed remove)
        }

        @Override
        public InputStream getInputStream() {
            return new ByteArrayInputStream(new byte[0]);   // reader thread sees EOF immediately
        }

        @Override
        public boolean waitFor(long timeout, TimeUnit unit) {
            return false;   // did not exit within the probe window -> startup succeeded
        }

        @Override
        public int exitValue() {
            return 0;
        }

        @Override
        public boolean isAlive() {
            return true;
        }

        @Override
        public void destroyForcibly() {
            // no-op — nothing to kill
        }
    }
}
