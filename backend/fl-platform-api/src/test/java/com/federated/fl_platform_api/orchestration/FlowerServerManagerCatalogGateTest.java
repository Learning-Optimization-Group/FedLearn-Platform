package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.ModelRecipeService;
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
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * SE-10: the model-type catalog gate at the {@link FlServerManager#startServerForProject} spawn
 * seam — a gradient-strategy project whose {@code modelType} is not a key in the recipe catalog
 * ({@link ModelRecipeService}) must be refused before any process is spawned. Same hermetic-stub
 * pattern as {@link FlServerManagerDpPolicyTest}: the manager is constructed with {@code new}, its
 * {@code @Value} knobs are overridden via {@link ReflectionTestUtils}, and the "spawn" is a tiny
 * bash/python stub — so the refusal is asserted at the real seam, not against a mock of it.
 */
@DisabledOnOs(OS.WINDOWS)
class FlServerManagerCatalogGateTest {

    private static final int RANGE_START = 50000;
    private static final int RANGE_END = 50010;

    private FlServerManager manager;
    private WebSocketService ws;
    private ModelRecipeService modelRecipeService;

    private Path aliveStub;
    private final List<UUID> startedProjects = new ArrayList<>();

    @BeforeEach
    void setUp() throws IOException {
        ws = mock(WebSocketService.class);
        RunTokenRegistry runTokenRegistry = mock(RunTokenRegistry.class);
        when(runTokenRegistry.mint(any(), any())).thenReturn("test-run-token");
        modelRecipeService = mock(ModelRecipeService.class);

        manager = new FlServerManager();
        ReflectionTestUtils.setField(manager, "logBroadcaster", ws);
        ReflectionTestUtils.setField(manager, "runTokenRegistry", runTokenRegistry);
        ReflectionTestUtils.setField(manager, "runRepository",
                mock(com.federated.fl_platform_api.repository.RunRepository.class));
        ReflectionTestUtils.setField(manager, "modelRecipeService", modelRecipeService);
        // BA-11: a bare-mock resolver returns Optional.empty() (no registry head) → no --init-model-path.
        ReflectionTestUtils.setField(manager, "registryModelResolver",
                new com.federated.fl_platform_api.service.RegistryModelResolver(
                        mock(com.federated.fl_platform_api.repository.ModelArtifactRepository.class),
                        mock(com.federated.fl_platform_api.service.ArtifactBlobStore.class), "unused"));
        ReflectionTestUtils.setField(manager, "portRangeStart", RANGE_START);
        ReflectionTestUtils.setField(manager, "portRangeEnd", RANGE_END);
        ReflectionTestUtils.setField(manager, "startupProbeSeconds", 2L);
        ReflectionTestUtils.setField(manager, "stdoutDrainMillis", 2000L);

        // Same alive stub as FlServerManagerDpPolicyTest: unknown flags are shifted away, then
        // the stub binds its --port and blocks. Wired as BOTH the gradient and FoT wrapper so a
        // successful spawn is available on either path if the gate lets it through.
        aliveStub = Files.createTempFile("stub-fl-catalog", ".sh");
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
        ReflectionTestUtils.setField(manager, "fotServerWrapperPath", aliveStub.toString());
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
    void unknownModelType_isRefusedWith400BoundIllegalArgumentException() {
        Project p = project("FOOBAR");
        when(modelRecipeService.findByKey("FOOBAR")).thenReturn(Optional.empty());

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 1));

        assertTrue(ex.getMessage().contains("model type"), ex.getMessage());
        assertTrue(ex.getMessage().contains(p.getId().toString()), ex.getMessage());
        assertFalse(ex.getMessage().contains("FOOBAR"), ex.getMessage());
        assertFalse(manager.isServerRunning(p.getId()), "no child may be spawned for a refused start");
        verifyNoInteractions(ws);
    }

    @Test
    void knownCatalogModelType_proceedsPastTheGate() {
        Project p = project("CNN");
        ModelRecipeDto cnnDto = new ModelRecipeDto(
                "CNN", "CNN", "image", List.of(), List.of(), List.of(), null);
        when(modelRecipeService.findByKey("CNN")).thenReturn(Optional.of(cnnDto));

        Optional<Integer> port = manager.startServerForProject(p, "FedAvg", 5, 1);
        startedProjects.add(p.getId());

        assertTrue(port.isPresent(), "a known catalog model type must be allowed to start");
        assertTrue(port.get() >= RANGE_START && port.get() <= RANGE_END);
        assertTrue(manager.isServerRunning(p.getId()));
    }

    @Test
    void fotStart_isExemptFromTheCatalogGate() {
        Project p = project("ANYTHING_NOT_IN_CATALOG");

        try {
            manager.startServerForProject(p, "FoT", 5, 1);
            startedProjects.add(p.getId());
        } catch (RuntimeException ignored) {
            // this test asserts only that the catalog gate is skipped for FoT -- a successful stub
            // spawn is a bonus, not the assertion under test.
        }

        verify(modelRecipeService, never()).findByKey(any());
    }

    @Test
    void caseVariantModelType_isRejected_becauseFlServerComparesCaseSensitively() {
        // "mlp" is a case-INSENSITIVE hit on the catalog key "MLP", but fl_server.py compares
        // model_type case-SENSITIVELY (== 'MLP' / == 'LLM_LORA' / ...), so a lowercase value would
        // clear a lenient gate and then silently mis-train (wrong dataset / artifact kind). The gate
        // must require the exact canonical key.
        Project p = project("mlp");
        ModelRecipeDto canonical = new ModelRecipeDto(
                "MLP", "MLP", "tabular", List.of(), List.of(), List.of(), null);
        when(modelRecipeService.findByKey("mlp")).thenReturn(Optional.of(canonical));

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> manager.startServerForProject(p, "FedAvg", 5, 1));

        assertTrue(ex.getMessage().contains("model type"), ex.getMessage());
        assertFalse(ex.getMessage().contains("mlp"), ex.getMessage());   // no-reflect
        assertFalse(manager.isServerRunning(p.getId()), "a case-variant modelType must not spawn");
        verifyNoInteractions(ws);
    }

    private Project project(String modelType) {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType(modelType);
        p.setModelName("qwen2.5-0.5b");
        p.setModelPath("/tmp/model.npz");
        return p;
    }
}
