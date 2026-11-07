package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.RunStatusDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.orchestration.RunProcessIdentityStore;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunExecutionAttemptRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.security.OrgScopeFilter;
import com.federated.fl_platform_api.service.ProjectService;
import com.federated.fl_platform_api.service.RunService;
import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.context.SecurityContextImpl;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

@SpringBootTest
@ActiveProfiles("test")
class ExecutionAttemptContinuationIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired private ProjectRepository projectRepository;
    @Autowired private RunRepository runRepository;
    @Autowired private RunExecutionAttemptRepository attemptRepository;
    @Autowired private UserRepository userRepository;
    @Autowired private PasswordEncoder passwordEncoder;
    @Autowired private ProjectService projectService;
    @Autowired private RunService runService;
    @Autowired private FlServerManager serverManager;
    @Autowired private RunProcessIdentityStore processIdentities;
    @Autowired private OrgScope orgScope;
    @Autowired private OrgScopeFilter orgScopeFilter;

    private User testUser;
    private Project testProject;
    private Path createdCheckpointDir;
    private Path aliveStub;
    private String originalWrapperPath;
    private Long originalProbeSeconds;
    private Long originalDrainMillis;

    private void authAndScope(User u) throws Exception {
        MockHttpServletRequest req = new MockHttpServletRequest();
        RequestContextHolder.setRequestAttributes(new ServletRequestAttributes(req));
        SecurityContextImpl ctx = new SecurityContextImpl();
        ctx.setAuthentication(new UsernamePasswordAuthenticationToken(
                u.getUsername(), "x", List.of(new SimpleGrantedAuthority("ROLE_USER"))));
        SecurityContextHolder.setContext(ctx);
        orgScopeFilter.doFilter(req, new MockHttpServletResponse(), mock(FilterChain.class));
    }

    @BeforeEach
    void setUp() throws Exception {
        originalWrapperPath = (String) ReflectionTestUtils.getField(serverManager, "flServerWrapperPath");
        originalProbeSeconds = (Long) ReflectionTestUtils.getField(serverManager, "startupProbeSeconds");
        originalDrainMillis = (Long) ReflectionTestUtils.getField(serverManager, "stdoutDrainMillis");

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
                        + "echo \"STUB_FL_SERVER started port=${PORT}\"\n"
                        + "exec python3 -c \"import socket, time\n"
                        + "p = int('${PORT}')\n"
                        + "s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                        + "s4.bind(('0.0.0.0', p)); s4.listen(1)\n"
                        + "s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
                        + "s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)\n"
                        + "s6.bind(('::', p)); s6.listen(1)\n"
                        + "time.sleep(300)\"\n");
        ReflectionTestUtils.setField(serverManager, "flServerWrapperPath", aliveStub.toString());
        ReflectionTestUtils.setField(serverManager, "startupProbeSeconds", 1L);
        ReflectionTestUtils.setField(serverManager, "stdoutDrainMillis", 500L);

        String suffix = UUID.randomUUID().toString().substring(0, 8);
        testUser = userRepository.save(new User("owner-" + suffix, "owner-" + suffix + "@example.com",
                passwordEncoder.encode("Password1!")));
        authAndScope(testUser);

        testProject = new Project();
        testProject.setName("proj-" + suffix);
        testProject.setModelType("CNN");
        testProject.setModelName("net");
        testProject.setModelPath("/tmp/model.npz");
        testProject.setStatus(ProjectStatus.CREATED.name());
        testProject.setUser(testUser);
        testProject.setOrgId(DEFAULT_ORG_ID);
        testProject.setVisibility(ProjectVisibility.PRIVATE);
        testProject = projectRepository.save(testProject);
    }

    @AfterEach
    void tearDown() throws IOException {
        SecurityContextHolder.clearContext();
        RequestContextHolder.resetRequestAttributes();
        if (testProject != null) {
            try {
                serverManager.stopServerForProject(testProject.getId());
            } catch (RuntimeException ignored) {
            }
        }
        if (aliveStub != null) {
            Files.deleteIfExists(aliveStub);
        }
        ReflectionTestUtils.setField(serverManager, "flServerWrapperPath", originalWrapperPath);
        ReflectionTestUtils.setField(serverManager, "startupProbeSeconds", originalProbeSeconds);
        ReflectionTestUtils.setField(serverManager, "stdoutDrainMillis", originalDrainMillis);

        if (createdCheckpointDir != null && Files.exists(createdCheckpointDir)) {
            Files.walk(createdCheckpointDir)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
    }

    private void createSessionFile(UUID runId) throws IOException {
        createdCheckpointDir = serverManager.getCheckpointDirectory(runId);
        Files.createDirectories(createdCheckpointDir);
        Path sessionFile = createdCheckpointDir.resolve("session.json");
        Files.writeString(sessionFile, "{\"run_id\":\"" + runId + "\",\"seed\":42}");
    }

    @Test
    void runCreation_initializesAttemptOne() {
        Run run = runService.createForStart(testProject, "FedAvg", 3, 1, 1);

        assertNotNull(run.getCurrentAttemptId());
        assertEquals(1, run.getAttemptCount());
        assertEquals(RunStatus.STARTING, run.getStatus());

        List<RunExecutionAttempt> attempts = attemptRepository.findByRunIdOrderByAttemptNumberAsc(run.getId());
        assertEquals(1, attempts.size());
        RunExecutionAttempt first = attempts.get(0);
        assertEquals(run.getCurrentAttemptId(), first.getId());
        assertEquals(1, first.getAttemptNumber());
        assertEquals(RunStatus.STARTING, first.getStatus());
    }

    @Test
    void staleAttemptLauncher_isRejectedWhenRunReopens() {
        Run run = runService.createForStart(testProject, "FedAvg", 3, 1, 1);
        final UUID runId = run.getId();
        UUID firstAttemptId = run.getCurrentAttemptId();

        // Mark attempt 1 running, then failed
        runService.markRunning(runId, firstAttemptId, 50001);
        runService.markFailed(runId, firstAttemptId, "Crashed");

        // Simulate reopening into attempt 2
        UUID secondAttemptId = UUID.randomUUID();
        run.setCurrentAttemptId(secondAttemptId);
        run.setAttemptCount(2);
        run.setStatus(RunStatus.STARTING);
        run = runRepository.save(run);

        RunExecutionAttempt attempt2 = new RunExecutionAttempt(secondAttemptId, runId, 2, RunStatus.STARTING);
        attemptRepository.save(attempt2);

        // A delayed old launcher from attempt 1 tries to authorize: must be rejected!
        assertThrows(IllegalStateException.class, () ->
                processIdentities.authorizeLaunch(runId, firstAttemptId, "stale-token-hash"));

        // Current attempt 2 succeeds
        assertDoesNotThrow(() ->
                processIdentities.authorizeLaunch(runId, secondAttemptId, "current-token-hash"));
    }

    @Test
    void delayedOldAttemptExit_cannotFailReopenedAttempt() {
        Run run = runService.createForStart(testProject, "FedAvg", 3, 1, 1);
        final UUID runId = run.getId();
        UUID attempt1 = run.getCurrentAttemptId();

        runService.markRunning(runId, attempt1, 50001);
        runService.markFailed(runId, attempt1, "Simulated initial crash");

        // Reopen into attempt 2
        UUID attempt2 = UUID.randomUUID();
        run.setCurrentAttemptId(attempt2);
        run.setAttemptCount(2);
        run.setStatus(RunStatus.STARTING);
        run = runRepository.save(run);

        RunExecutionAttempt att2Entity = new RunExecutionAttempt(attempt2, runId, 2, RunStatus.STARTING);
        attemptRepository.save(att2Entity);

        // Attempt 2 marks RUNNING
        runService.markRunning(runId, attempt2, 50002);
        assertEquals(RunStatus.RUNNING, runRepository.findById(runId).orElseThrow().getStatus());

        // A delayed exit from attempt 1 arrives late: must NOT clobber attempt 2
        runService.markFailed(runId, attempt1, "Delayed old exit");

        Run preserved = runRepository.findById(runId).orElseThrow();
        assertEquals(RunStatus.RUNNING, preserved.getStatus(), "Delayed attempt 1 exit must not fail attempt 2");
        assertEquals(attempt2, preserved.getCurrentAttemptId());
        assertEquals(Integer.valueOf(50002), preserved.getServerPort());

        RunExecutionAttempt att2Checked = attemptRepository.findById(attempt2).orElseThrow();
        assertEquals(RunStatus.RUNNING, att2Checked.getStatus());
    }

    @Test
    void resumeRun_rejectsIneligibleConditions() throws IOException {
        Run run = runService.createForStart(testProject, "FedAvg", 3, 1, 1);
        final UUID runId = run.getId();
        testProject.setActiveRunId(runId);
        projectRepository.save(testProject);

        // 1. Rejects while still STARTING
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));

        // Mark completed
        runService.markCompleted(runId);
        // 2. Rejects COMPLETED runs
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));

        // Set to FAILED
        run.setStatus(RunStatus.FAILED);
        run = runRepository.save(run);

        // 3. Rejects when checkpoint directory is missing
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));

        // Create checkpoint state
        createSessionFile(runId);

        // 4. Rejects if not the active run for project
        testProject.setActiveRunId(UUID.randomUUID());
        projectRepository.save(testProject);
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));
        testProject.setActiveRunId(runId);
        projectRepository.save(testProject);

        // 5. Rejects unsupported strategies (e.g. DeComFL)
        run.setStrategy("DeComFL");
        runRepository.save(run);
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));
        run.setStrategy("FedAvg");
        runRepository.save(run);

        // 6. Rejects DP-enabled runs
        RunIntent dpIntent = new RunIntent(TrainingArm.FULL, "net", "classification", true, 8.0, 1e-5, 1.0, false, false, 120000L, TrainingDataSource.FIXTURE);
        run.setIntent(dpIntent);
        runRepository.save(run);
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));

        // Reset to non-DP
        run.setIntent(new RunIntent(TrainingArm.FULL, "net", "classification", false, null, null, null, false, false, 120000L, TrainingDataSource.FIXTURE));
        runRepository.save(run);

        // 7. Rejects if recorded PID is still present
        run.setServerPid(99999L);
        runRepository.save(run);
        assertThrows(ProjectStateException.class, () -> projectService.resumeRun(runId));
    }

    @Test
    void resumeRun_appendsAttemptAndTransitionsToRunning() throws IOException {
        Run run = runService.createForStart(testProject, "FedAvg", 3, 1, 1);
        testProject.setActiveRunId(run.getId());
        projectRepository.save(testProject);

        UUID attempt1 = run.getCurrentAttemptId();
        runService.markRunning(run.getId(), attempt1, 50001);
        runService.markFailed(run.getId(), attempt1, "Test failure");

        createSessionFile(run.getId());

        RunStatusDto dto = projectService.resumeRun(run.getId());

        assertEquals("RUNNING", dto.getStatus());
        Run resumed = runRepository.findById(run.getId()).orElseThrow();
        assertEquals(RunStatus.RUNNING, resumed.getStatus());
        assertEquals(2, resumed.getAttemptCount());
        assertNotEquals(attempt1, resumed.getCurrentAttemptId());

        List<RunExecutionAttempt> attempts = attemptRepository.findByRunIdOrderByAttemptNumberAsc(run.getId());
        assertEquals(2, attempts.size());

        RunExecutionAttempt first = attempts.get(0);
        assertEquals(1, first.getAttemptNumber());
        assertEquals(RunStatus.FAILED, first.getStatus());
        assertEquals("Test failure", first.getFailureReason());

        RunExecutionAttempt second = attempts.get(1);
        assertEquals(2, second.getAttemptNumber());
        assertEquals(RunStatus.RUNNING, second.getStatus());
        assertEquals(resumed.getCurrentAttemptId(), second.getId());
    }
}
