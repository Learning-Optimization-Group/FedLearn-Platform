package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.exception.ProjectStateException;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * SE-11: the run-start DP policy gate — a {@code regulated} project must not reach the spawn path
 * unless DP is enabled with a complete (epsilon, delta, clip-norm) config. Same hermetic-stub
 * pattern as {@link FlowerServerManagerIntegrationTest}: the manager is constructed with {@code new},
 * its {@code @Value} knobs are overridden via {@link ReflectionTestUtils}, and the "spawn" is a tiny
 * bash/python stub — so the refusal is asserted at the real
 * {@link FlowerServerManager#startServerForProject} seam, not against a mock of it.
 */
@DisabledOnOs(OS.WINDOWS)
class FlowerServerManagerDpPolicyTest {

    private static final int RANGE_START = 50000;
    private static final int RANGE_END = 50010;

    private FlowerServerManager manager;
    private WebSocketService ws;

    private Path aliveStub;
    private final List<UUID> startedProjects = new ArrayList<>();

    @BeforeEach
    void setUp() throws IOException {
        ws = mock(WebSocketService.class);
        RunTokenRegistry runTokenRegistry = mock(RunTokenRegistry.class);
        when(runTokenRegistry.mint(any(), any())).thenReturn("test-run-token");

        manager = new FlowerServerManager();
        ReflectionTestUtils.setField(manager, "logBroadcaster", ws);
        ReflectionTestUtils.setField(manager, "runTokenRegistry", runTokenRegistry);
        ReflectionTestUtils.setField(manager, "runRepository",
                mock(com.federated.fl_platform_api.repository.RunRepository.class));
        ReflectionTestUtils.setField(manager, "portRangeStart", RANGE_START);
        ReflectionTestUtils.setField(manager, "portRangeEnd", RANGE_END);
        ReflectionTestUtils.setField(manager, "startupProbeSeconds", 2L);
        ReflectionTestUtils.setField(manager, "stdoutDrainMillis", 2000L);

        // Same alive stub as FlowerServerManagerIntegrationTest: unknown flags (including the
        // --dp-* passthrough) are shifted away, then the stub binds its --port and blocks.
        aliveStub = Files.createTempFile("stub-fl-dp", ".sh");
        Files.writeString(aliveStub,
                "#!/bin/bash\n"
                        + "PORT=\"\"\n"
                        + "while [ \"$#\" -gt 0 ]; do\n"
                        + "  case \"$1\" in\n"
                        + "    --port) PORT=\"$2\"; shift 2 ;;\n"
                        + "    *) shift ;;\n"
                        + "  esac\n"
                        + "done\n"
                        + "echo \"STUB_FL_SERVER started port=${PORT}\"\n"
                        + "exec python3 -c \"import socket, time\n"
                        + "p = int('${PORT}')\n"
                        + "s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                        + "s4.bind(('0.0.0.0', p)); s4.listen(1)\n"
                        + "s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
                        + "s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)\n"
                        + "s6.bind(('::', p)); s6.listen(1)\n"
                        + "time.sleep(300)\"\n");
        ReflectionTestUtils.setField(manager, "flServerWrapperPath", aliveStub.toString());
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
    }

    @Test
    void regulatedProjectWithDpDisabled_isRefusedBeforeAnySpawn() {
        Project p = regulatedProject();
        p.setDpEnabled(false);

        ProjectStateException ex = assertThrows(ProjectStateException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 2));

        assertTrue(ex.getMessage().contains("regulated"), ex.getMessage());
        assertTrue(ex.getMessage().contains("differential privacy"), ex.getMessage());
        assertFalse(manager.isServerRunning(p.getId()), "no child may be spawned for a refused start");
        verifyNoInteractions(ws);   // refused before the spawn — nothing was streamed
    }

    @Test
    void regulatedProjectWithIncompleteDpConfig_isRefusedWithAnActionableError() {
        Project p = regulatedProject();
        p.setDpEnabled(true);
        p.setDpTargetEpsilon(6.0);
        p.setDpDelta(null);          // incomplete: no delta
        p.setDpClipNorm(1.5);

        ProjectStateException ex = assertThrows(ProjectStateException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 2));

        assertTrue(ex.getMessage().contains("incomplete DP config"), ex.getMessage());
        assertTrue(ex.getMessage().contains("4-8"),
                "the refusal should carry the epsilon guidance range: " + ex.getMessage());
        assertFalse(manager.isServerRunning(p.getId()));
        verifyNoInteractions(ws);
    }

    @Test
    void regulatedProjectWithCompleteDpConfig_proceedsPastThePolicyGate() {
        Project p = regulatedProject();
        p.setDpEnabled(true);
        p.setDpTargetEpsilon(6.0);
        p.setDpDelta(1e-5);
        p.setDpClipNorm(1.5);

        // The gate lets a fully-configured regulated project through to the (stubbed) spawn: a port
        // is reserved and the child comes up. Anything after the gate is the generic spawn path
        // already covered by FlowerServerManagerIntegrationTest.
        Optional<Integer> port = manager.startServerForProject(p, "FedAvg", 5, 2);
        startedProjects.add(p.getId());

        assertTrue(port.isPresent(), "a compliant regulated project must start");
        assertTrue(port.get() >= RANGE_START && port.get() <= RANGE_END);
        assertTrue(manager.isServerRunning(p.getId()));
    }

    private Project regulatedProject() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("qwen2.5-0.5b");
        p.setModelPath("/tmp/model.npz");
        p.setRegulated(true);
        return p;
    }
}
