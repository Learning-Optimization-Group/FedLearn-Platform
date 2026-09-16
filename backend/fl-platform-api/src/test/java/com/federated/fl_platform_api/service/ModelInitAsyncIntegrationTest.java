package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectInitStatus;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * BA-1 acceptance: model init runs off the request thread. Points {@link ModelInitializer} at a stub
 * wrapper that hangs (sleep) well past the subprocess timeout, then asserts the "done when" contract:
 *
 * <ol>
 *   <li>{@code createProject} returns fast (project INITIALIZING) even though init hangs — the request
 *       thread and its DB connection are not held for the duration of init;</li>
 *   <li>a second creation succeeds just as fast while the first init is still hanging — the first init
 *       did not starve the request path;</li>
 *   <li>the hung init is force-killed on the (short) subprocess timeout and the project transitions to
 *       FAILED, persisted (not rolled back).</li>
 * </ol>
 *
 * Unix-only (the stub is a bash script), matching {@link ModelInitializerTimeoutTest}; CI is Linux.
 */
@SpringBootTest
@ActiveProfiles("test")
@DisabledOnOs(OS.WINDOWS)
class ModelInitAsyncIntegrationTest {

    @Autowired ProjectService projectService;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private static Path hangScript;

    @DynamicPropertySource
    static void initProps(DynamicPropertyRegistry registry) throws IOException {
        hangScript = Files.createTempFile("ba1-hang-init", ".sh");
        Files.writeString(hangScript, "#!/bin/bash\nsleep 30\n");   // ignores args, hangs past the timeout
        registry.add("python.executable.path", () -> hangScript.toString());
        registry.add("python.script.init-model.timeout-seconds", () -> "2");  // fail fast, don't wait out the sleep
        registry.add("app.model-init.pool-size", () -> "2");                  // both creations can init concurrently
    }

    @AfterAll
    static void removeStub() throws IOException {
        if (hangScript != null) {
            Files.deleteIfExists(hangScript);
        }
    }

    @BeforeEach
    void seedOwner() {
        if (userRepository.findByUsername("ba1-owner").isEmpty()) {
            User u = new User("ba1-owner", "ba1-owner@example.com", passwordEncoder.encode("Password1!"));
            u.setPlatformRole(PlatformRole.PROJECT_OWNER);
            userRepository.save(u);
        }
    }

    private CreateProjectRequest request(String name) {
        CreateProjectRequest req = new CreateProjectRequest();
        req.setName(name);
        req.setModelType("CNN");
        req.setModelName("simple-cnn");
        req.setOptimizer("Adam");
        req.setPretrainEpochs(0);
        return req;
    }

    @Test
    @WithMockUser(username = "ba1-owner", roles = "PROJECT_OWNER")
    void createReturnsFastWhileInitHangs_thenTransitionsToFailedOnTimeout() throws Exception {
        // (1) createProject returns immediately with the project INITIALIZING, though the init hangs 30s.
        long start = System.nanoTime();
        ProjectResponseDto created = projectService.createProject(request("ba1-hang-1"));
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;

        assertTrue(elapsedMs < 5_000,
                "createProject must return without waiting on the hung init (elapsed=" + elapsedMs + "ms)");
        assertEquals(ProjectStatus.INITIALIZING.name(), created.getStatus());
        UUID firstId = created.getId();

        // (2) A second creation is not blocked by the first still-hanging init — no thread/connection
        // was held by createProject #1.
        long start2 = System.nanoTime();
        ProjectResponseDto created2 = projectService.createProject(request("ba1-hang-2"));
        long elapsedMs2 = (System.nanoTime() - start2) / 1_000_000;

        assertTrue(elapsedMs2 < 5_000,
                "second createProject must not block behind the first hanging init (elapsed=" + elapsedMs2 + "ms)");
        assertEquals(ProjectStatus.INITIALIZING.name(), created2.getStatus());

        // (3) The hung init is killed on the 2s subprocess timeout; the project persists as FAILED.
        ProjectInitStatus terminal = pollInitStatusUntilTerminal(firstId, 30_000);
        assertEquals(ProjectInitStatus.FAILED, terminal,
                "a hung init must transition the project to FAILED after the timeout");
    }

    /** Poll the persisted init status (each read its own transaction) until it leaves INITIALIZING. */
    private ProjectInitStatus pollInitStatusUntilTerminal(UUID projectId, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        ProjectInitStatus status = ProjectInitStatus.INITIALIZING;
        while (System.currentTimeMillis() < deadline) {
            status = projectRepository.findById(projectId)
                    .map(Project::getInitStatus)
                    .orElse(ProjectInitStatus.INITIALIZING);
            if (status != ProjectInitStatus.INITIALIZING) {
                return status;
            }
            Thread.sleep(200);
        }
        return status;
    }
}
