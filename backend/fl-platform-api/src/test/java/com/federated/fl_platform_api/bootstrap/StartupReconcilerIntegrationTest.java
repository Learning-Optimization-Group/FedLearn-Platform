package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * BA-3 done-when, end to end against real OS processes and the Testcontainers DB: a live+identity
 * matched server is re-adopted and remains stoppable; a dead PID leaves no run stuck RUNNING and no
 * leaked port. Unix-only (spawns bash); the startup ApplicationRunner is disabled under test
 * (app.fl.reconcile-on-startup=false) so this drives reconcile() directly.
 */
@SpringBootTest
@ActiveProfiles("test")
@DisabledOnOs(OS.WINDOWS)
class StartupReconcilerIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired StartupReconciler reconciler;
    @Autowired RunRepository runRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired RunService runService;
    @Autowired FlServerManager serverManager;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;
    @Autowired RunTokenRegistry runTokenRegistry;

    private Project seedProject() {
        User owner = userRepository.save(new User(
                "recon-" + System.nanoTime(), "recon-" + System.nanoTime() + "@example.com",
                passwordEncoder.encode("Password1!")));
        Project p = new Project();
        p.setName("recon-" + System.nanoTime());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(ProjectVisibility.PRIVATE);
        return projectRepository.save(p);
    }

    private Run seedRunningRun(Project project, long pid, Instant startInstant) {
        Run run = runService.createForStart(project, "FedAvg", 1, 1, 1);
        run.setStatus(RunStatus.RUNNING);
        run.setServerPid(pid);
        run.setProcessStartedAt(startInstant);
        run = runRepository.save(run);
        project.setActiveRunId(run.getId());
        projectRepository.save(project);
        return run;
    }

    @Test
    void liveServer_isReadopted_andRemainsStoppable() throws Exception {
        Process proc = new ProcessBuilder("bash", "-c", "sleep 60").start();
        try {
            Project project = seedProject();
            Run run = seedRunningRun(project, proc.pid(),
                    proc.info().startInstant().orElseThrow());

            reconciler.reconcile();

            // Re-adopted: tracked as running, run still RUNNING (not reaped).
            assertTrue(serverManager.isServerRunning(project.getId()), "live server should be re-adopted");
            assertEquals(RunStatus.RUNNING, runRepository.findById(run.getId()).orElseThrow().getStatus());

            // And a subsequent stop terminates the re-adopted child.
            assertTrue(serverManager.stopServerForProject(project.getId()));
            assertTrue(proc.waitFor(5, TimeUnit.SECONDS), "stop should terminate the adopted process");
            assertFalse(serverManager.isServerRunning(project.getId()));
        } finally {
            proc.destroyForcibly();
        }
    }

    @Test
    void reAdoptedServer_tokenIsRehydrated_soCallbacksKeepAuthorizing() throws Exception {
        Process proc = new ProcessBuilder("bash", "-c", "sleep 60").start();
        try {
            Project project = seedProject();

            // Mint the run's token, then evict it to simulate the in-memory registry being empty after
            // a restart — the run row keeps only its SHA-256 hash.
            String token = runTokenRegistry.mint(project.getId(), UUID.randomUUID());
            String tokenHash = runTokenRegistry.hash(token);
            runTokenRegistry.evictForProject(project.getId());
            assertTrue(runTokenRegistry.resolve(token).isEmpty(), "token is gone after the simulated restart");

            Run run = seedRunningRun(project, proc.pid(), proc.info().startInstant().orElseThrow());
            run.setInternalTokenHash(tokenHash);
            runRepository.save(run);

            reconciler.reconcile();

            var scope = runTokenRegistry.resolve(token);
            assertTrue(scope.isPresent(), "a re-adopted server's token must be rehydrated");
            assertEquals(project.getId(), scope.get().projectId());
        } finally {
            proc.destroyForcibly();
        }
    }

    @Test
    void deadServer_isReaped_projectReturnedToIdle() throws Exception {
        Process proc = new ProcessBuilder("bash", "-c", "exit 0").start();
        proc.waitFor(5, TimeUnit.SECONDS);   // now dead — its PID no longer resolves to a live process
        long deadPid = proc.pid();

        Project project = seedProject();
        // A made-up past start-instant: even in the astronomically unlikely event the PID was reused,
        // the identity check fails and the run is still reaped (never adopted).
        Run run = seedRunningRun(project, deadPid, Instant.parse("2020-01-01T00:00:00Z"));

        reconciler.reconcile();

        assertEquals(RunStatus.FAILED, runRepository.findById(run.getId()).orElseThrow().getStatus(),
                "orphaned run with a dead PID must be reset to FAILED");
        Project reconciled = projectRepository.findById(project.getId()).orElseThrow();
        assertNull(reconciled.getActiveRunId(), "project should be returned to idle");
        assertFalse(serverManager.isServerRunning(project.getId()));
    }
}
