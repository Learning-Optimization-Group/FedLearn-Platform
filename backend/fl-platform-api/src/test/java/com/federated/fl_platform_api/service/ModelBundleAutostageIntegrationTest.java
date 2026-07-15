package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ModelBundleDto;
import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;

/**
 * BA-16 acceptance: starting a fixture-backed {@code TINYNET_GOLDEN} DeComFL run auto-stages the mobile
 * on-device bundle so a phone can join with ZERO manual terminal steps — i.e. without an operator running
 * {@code scripts/stage_model_bundle.py} by hand. Drives the real HTTP path end-to-end (login cookie →
 * {@code POST /api/projects/{id}/start} → {@code GET /api/runs/{runId}/model-bundle}) and asserts a 200
 * with the staged manifest, plus all five bundle files on disk.
 *
 * <p>The FL-server spawn is the only thing stubbed ({@link FlServerManager} is a {@code @MockBean}, so
 * no real {@code python fl_server.py} runs); the auto-stage itself uses the real
 * {@link ScriptModelBundleStager} bean and the real, stdlib-only {@code scripts/stage_model_bundle.py} —
 * exactly the code path a live host takes. Staging is made synchronous via the package-private executor
 * seam so the served bundle is deterministic (no polling race with the background worker).</p>
 *
 * <p>Unix-only (the stager shells out to {@code python3}); CI runs on Linux.</p>
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DisabledOnOs(OS.WINDOWS)
class ModelBundleAutostageIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired PasswordEncoder passwordEncoder;
    @Autowired ScriptModelBundleStager stager;

    @MockitoBean FlServerManager flServerManager;   // never spawn a real python FL server under test

    private static Path bundleDir;

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry registry) throws IOException {
        bundleDir = Files.createTempDirectory("ba16-bundles");
        // JVM cwd under gradle is the module dir (backend/fl-platform-api); the staging scripts live at
        // the repo root. Point the script @Value paths at their real absolute locations so ProcessBuilder
        // finds them regardless of cwd.
        Path repoRoot = Path.of("").toAbsolutePath().getParent().getParent();
        registry.add("app.model-bundle.dir", () -> bundleDir.toString());
        registry.add("feature.model-bundle-autostage.enabled", () -> "true");
        registry.add("python.script.stage-model-bundle.path",
                () -> repoRoot.resolve("scripts/stage_model_bundle.py").toString());
        registry.add("python.script.export-model.path",
                () -> repoRoot.resolve("scripts/export_model.py").toString());
    }

    @BeforeEach
    void setUp() {
        when(flServerManager.isServerRunning(any())).thenReturn(false);
        when(flServerManager.startServerForProject(any(), any(), anyInt(), anyInt()))
                .thenReturn(Optional.of(50000));
        // Stage on the calling thread so the served bundle is ready the moment /start returns (the same
        // executor seam the ScriptModelBundleStager unit test uses).
        stager.setExecutor(Runnable::run);
    }

    @Test
    void startingTinynetGoldenRun_autostagesTheMobileBundle_servedAt200() {
        User owner = new User("ba16-owner", "ba16-owner@example.com", passwordEncoder.encode("Password1!"));
        owner.setPlatformRole(PlatformRole.PROJECT_OWNER);
        userRepository.save(owner);
        String cookie = login("ba16-owner");

        Project p = new Project();
        p.setName("tinynet-" + System.nanoTime());
        p.setModelType("TINYNET_GOLDEN");
        p.setModelName("tinynet");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(ProjectVisibility.PRIVATE);
        p = projectRepository.save(p);

        // Start a DeComFL run — this is the moment the bundle must auto-stage.
        ResponseEntity<String> start = restTemplate.exchange(
                "/api/projects/" + p.getId() + "/start", HttpMethod.POST,
                new HttpEntity<>(Map.of("strategy", "DeComFL", "minClients", 1, "numRounds", 1), authJson(cookie)),
                String.class);
        assertEquals(HttpStatus.OK, start.getStatusCode(), "start must succeed: " + start.getBody());

        UUID runId = projectRepository.findById(p.getId()).orElseThrow().getActiveRunId();
        assertNotNull(runId, "start must have created an active run");

        // The bundle must be served 200 with the manifest — no manual stage step required (BA-16).
        ResponseEntity<ModelBundleDto> bundle = restTemplate.exchange(
                "/api/runs/" + runId + "/model-bundle", HttpMethod.GET,
                new HttpEntity<>(auth(cookie)), ModelBundleDto.class);
        assertEquals(HttpStatus.OK, bundle.getStatusCode(), "the auto-staged bundle must be served 200");
        assertNotNull(bundle.getBody());
        assertFalse(bundle.getBody().paramLayout().isEmpty(), "staged manifest must carry the paramLayout");
        assertTrue(bundle.getBody().totalParamCount() > 0, "staged manifest must carry a param count");

        // ...and all five staged files landed on disk in the served bundle dir.
        Path runDir = bundleDir.resolve(runId.toString());
        for (String f : List.of("manifest.json", "loss.pte", "infer.pte", "inputs.f32", "targets.i64")) {
            assertTrue(Files.isRegularFile(runDir.resolve(f)), "auto-stage must produce " + f);
        }
    }

    // --- helpers ------------------------------------------------------------

    private String login(String username) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
                "/api/auth/login", HttpMethod.POST,
                new HttpEntity<>(Map.of("username", username, "password", "Password1!"), h),
                Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        return resp.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];
    }

    private HttpHeaders auth(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    private HttpHeaders authJson(String cookie) {
        HttpHeaders h = auth(cookie);
        h.setContentType(MediaType.APPLICATION_JSON);
        return h;
    }
}
