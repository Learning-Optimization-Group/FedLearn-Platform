package com.federated.fl_platform_api;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import com.federated.fl_platform_api.service.ProjectService;
import com.federated.fl_platform_api.service.RunService;
import com.federated.fl_platform_api.service.WebSocketService;
import java.time.Instant;
import java.util.ArrayList;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Retry-safe result callbacks must bind logical run and current execution authorization. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class RunResultIdempotencyIntegrationTest {
    @Autowired TestRestTemplate rest;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired UserRepository users;
    @Autowired RoundResultRepository results;
    @Autowired RunTokenRegistry tokens;
    @SpyBean RunService lifecycle;
    @Autowired ProjectService projectService;
    @Autowired PlatformTransactionManager transactionManager;
    @Autowired JdbcTemplate jdbc;
    @Autowired org.springframework.security.crypto.password.PasswordEncoder passwordEncoder;
    @Autowired com.federated.fl_platform_api.orchestration.FlServerManager serverManager;
    @Autowired com.federated.fl_platform_api.orchestration.RunProcessIdentityStore processIdentities;
    @SpyBean WebSocketService webSockets;

    record ActiveRun(Project project, Run run, String token) {}

    private ActiveRun activeRun() {
        String suffix = UUID.randomUUID().toString();
        User owner = users.save(new User("result-" + suffix, suffix + "@example.com", "test-only-hash"));
        Project project = new Project();
        project.setName("result-" + suffix);
        project.setUser(owner);
        project.setModelType("CNN");
        project.setModelName("net");
        project.setStatus("RUNNING");
        project.setOrgId(UUID.fromString("00000000-0000-0000-0000-000000000001"));
        project.setVisibility(ProjectVisibility.PRIVATE);
        return newRun(projects.saveAndFlush(project));
    }

    private ActiveRun newRun(Project project) {
        Run run = new Run();
        run.setProjectId(project.getId());
        run.setStrategy("FedAvg");
        run.setRecipeKey("CNN");
        run.setNumRounds(3);
        run.setMinClients(1);
        run.setClientsPerRound(1);
        run.setStatus(RunStatus.RUNNING);
        run.setCreatedAt(Instant.now());
        run = runs.saveAndFlush(run);
        String token = tokens.mint(project.getId(), run.getId());
        run.setInternalTokenHash(tokens.hash(token));
        run = runs.saveAndFlush(run);
        project.setActiveRunId(run.getId());
        return new ActiveRun(projects.saveAndFlush(project), run, token);
    }

    private HttpStatus post(ActiveRun active, int round, double loss) {
        String body = "{\"serverRound\":" + round + ",\"loss\":" + loss
                + ",\"accuracy\":0.8,\"gpuUtilization\":0.0}";
        return postBody(active, body);
    }

    private HttpStatus postBody(ActiveRun active, String body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("X-Internal-Key", "test-internal-key");
        headers.set("X-Internal-Run-Token", active.token());
        return HttpStatus.valueOf(rest.exchange("/api/internal/results/" + active.project().getId(),
                HttpMethod.POST, new HttpEntity<>(body, headers), String.class).getStatusCode().value());
    }

    private HttpStatus finish(ActiveRun active) {
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-Internal-Key", "test-internal-key");
        headers.set("X-Internal-Run-Token", active.token());
        return HttpStatus.valueOf(rest.exchange("/api/internal/results/" + active.project().getId() + "/finished",
                HttpMethod.POST, new HttpEntity<>(headers), String.class).getStatusCode().value());
    }

    @Test
    void identicalRetryCreatesOneRowAndOneLiveUpdate() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        UUID firstId = results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst().getId();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        var stored = results.findByProjectIdOrderByServerRoundAsc(active.project().getId());
        assertEquals(1, stored.size());
        assertEquals(firstId, stored.getFirst().getId());
        verify(webSockets, times(1)).sendResultUpdate(eq(active.project().getId()), any());
    }

    @Test
    void roundResultWithTimingBreakdownIsPersistedAndBroadcasted() {
        ActiveRun active = activeRun();
        String body = "{\"serverRound\":1,\"loss\":0.42,\"accuracy\":0.88,\"gpuUtilization\":0.0,"
                + "\"time_data_loading_s\":0.123,\"time_forward_backward_s\":0.456,\"time_optimizer_step_s\":0.078,"
                + "\"time_wire_encode_s\":0.025,\"time_aggregation_s\":0.089,\"time_wire_decode_s\":0.015,\"time_eval_s\":0.234}";
        assertEquals(HttpStatus.OK, postBody(active, body));
        var stored = results.findByProjectIdOrderByServerRoundAsc(active.project().getId());
        assertEquals(1, stored.size());
        var res = stored.getFirst();
        assertEquals(0.123, res.getTimeDataLoadingS(), 1e-4);
        assertEquals(0.456, res.getTimeForwardBackwardS(), 1e-4);
        assertEquals(0.078, res.getTimeOptimizerStepS(), 1e-4);
        assertEquals(0.025, res.getTimeWireEncodeS(), 1e-4);
        assertEquals(0.089, res.getTimeAggregationS(), 1e-4);
        assertEquals(0.015, res.getTimeWireDecodeS(), 1e-4);
        assertEquals(0.234, res.getTimeEvalS(), 1e-4);

        org.mockito.ArgumentCaptor<com.federated.fl_platform_api.dto.RoundResultDto> captor =
                org.mockito.ArgumentCaptor.forClass(com.federated.fl_platform_api.dto.RoundResultDto.class);
        verify(webSockets).sendResultUpdate(eq(active.project().getId()), captor.capture());
        var broadcast = captor.getValue();
        assertEquals(0.123, broadcast.getTimeDataLoadingS(), 1e-4);
        assertEquals(0.456, broadcast.getTimeForwardBackwardS(), 1e-4);
        assertEquals(0.078, broadcast.getTimeOptimizerStepS(), 1e-4);
        assertEquals(0.025, broadcast.getTimeWireEncodeS(), 1e-4);
        assertEquals(0.089, broadcast.getTimeAggregationS(), 1e-4);
        assertEquals(0.015, broadcast.getTimeWireDecodeS(), 1e-4);
        assertEquals(0.234, broadcast.getTimeEvalS(), 1e-4);
        assertEquals(0.123, broadcast.getTimeDataLoadingS(), 1e-4);
        assertEquals(0.234, broadcast.getTimeEvalS(), 1e-4);
    }

    @Test
    void conflictingRetryCannotReplaceTheFirstResult() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        assertEquals(HttpStatus.CONFLICT, post(active, 1, .25));
        var stored = results.findByProjectIdOrderByServerRoundAsc(active.project().getId());
        assertEquals(1, stored.size());
        assertEquals(.5, stored.getFirst().getLoss());
    }

    @Test
    void concurrentRetriesStillHaveOneLogicalResult() throws Exception {
        ActiveRun active = activeRun();
        CountDownLatch start = new CountDownLatch(1);
        try (var pool = Executors.newFixedThreadPool(4)) {
            var tasks = new ArrayList<java.util.concurrent.Future<HttpStatus>>();
            for (int index = 0; index < 4; index++) {
                tasks.add(pool.submit(() -> { assertTrue(start.await(10, TimeUnit.SECONDS)); return post(active, 1, .5); }));
            }
            start.countDown();
            for (var task : tasks) assertEquals(HttpStatus.OK, task.get(20, TimeUnit.SECONDS));
        }
        assertEquals(1, results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).size());
        verify(webSockets, times(1)).sendResultUpdate(eq(active.project().getId()), any());
    }

    @Test
    void aDifferentRunMayRecordTheSameRoundWithoutReusingTheOldRow() {
        ActiveRun first = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        ActiveRun second = newRun(first.project());
        assertEquals(HttpStatus.OK, post(second, 1, .25));
        assertEquals(2, results.findByProjectIdOrderByServerRoundAsc(first.project().getId()).size());
        assertEquals(HttpStatus.CONFLICT, post(first, 2, .5));
    }

    @Test
    void aRetiredAttemptTokenCannotWriteAfterAuthorizationRotates() {
        ActiveRun active = activeRun();
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        active.run().setInternalTokenHash(tokens.hash(replacement));
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.FORBIDDEN, post(active, 1, .5));
        assertTrue(results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).isEmpty());
        assertEquals(HttpStatus.OK, post(new ActiveRun(active.project(), active.run(), replacement), 1, .5));
    }

    @Test
    void failedRunCannotAcceptNewResultsOrRoundsBeyondItsBudget() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, post(active, 4, .5));
        active.run().setStatus(RunStatus.FAILED);
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.CONFLICT, post(active, 1, .5));
        assertTrue(results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).isEmpty());
    }

    @Test
    void unevaluatedRoundMayReplayNullMetricsButCannotLaterChangeThem() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1}"));
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1,\"loss\":null}"));
        assertEquals(HttpStatus.CONFLICT, post(active, 1, .5));
        assertEquals(1, results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).size());
    }

    @Test
    void callbackCannotSupplyItsOwnResultOrRunIdentity() {
        ActiveRun active = activeRun();
        UUID suppliedId = UUID.randomUUID();
        UUID otherRun = activeRun().run().getId();
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1,\"id\":\"" + suppliedId
                + "\",\"runId\":\"" + otherRun + "\"}"));
        var saved = results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst();
        assertNotEquals(suppliedId, saved.getId());
        assertEquals(active.run().getId(), saved.getRunId());
    }

    @Test
    void missingZeroNegativeOrNonfiniteInputsCannotCreateAResult() {
        ActiveRun active = activeRun();
        for (String body : new String[] {"{}", "{\"serverRound\":0}", "{\"serverRound\":-1}",
                "{\"serverRound\":1,\"loss\":\"NaN\"}", "{\"serverRound\":1,\"accuracy\":\"Infinity\"}"}) {
            assertEquals(HttpStatus.BAD_REQUEST, postBody(active, body), body);
        }
        assertTrue(results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(strings = {"1.9", "1.0", "1e0", "\"1\""})
    void coercedRoundCannotClaimAnIntegerRound(String round) {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, postBody(active, "{\"serverRound\":" + round + "}"));
        assertTrue(results.findByProjectIdOrderByServerRoundAsc(active.project().getId()).isEmpty());
        verify(webSockets, never()).sendResultUpdate(eq(active.project().getId()), any());
    }

    @Test
    void oldRunCannotCompleteTheProjectsNewRun() {
        ActiveRun first = activeRun();
        ActiveRun current = newRun(first.project());
        assertEquals(HttpStatus.CONFLICT, finish(first));
        assertEquals(RunStatus.RUNNING, runs.findById(current.run().getId()).orElseThrow().getStatus());
        assertEquals(RunStatus.RUNNING, runs.findById(first.run().getId()).orElseThrow().getStatus());
        assertEquals("RUNNING", projects.findById(first.project().getId()).orElseThrow().getStatus());
    }

    @Test
    void retiredAttemptCannotFinishTheCurrentAttempt() {
        ActiveRun active = activeRun();
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        active.run().setInternalTokenHash(tokens.hash(replacement));
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.FORBIDDEN, finish(active));
        assertEquals(RunStatus.RUNNING, runs.findById(active.run().getId()).orElseThrow().getStatus());
        assertEquals(HttpStatus.OK, finish(new ActiveRun(active.project(), active.run(), replacement)));
        assertEquals(RunStatus.COMPLETED, runs.findById(active.run().getId()).orElseThrow().getStatus());
    }

    @Test
    void completionCannotReclassifyAFailedProject() {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.FAILED);
        runs.saveAndFlush(active.run());
        active.project().setStatus("FAILED");
        projects.saveAndFlush(active.project());
        assertEquals(HttpStatus.CONFLICT, finish(active));
        assertEquals("FAILED", projects.findById(active.project().getId()).orElseThrow().getStatus());
        assertEquals(RunStatus.FAILED, runs.findById(active.run().getId()).orElseThrow().getStatus());
    }

    @Test
    void currentRunMayFinishAndRetiresItsAuthorization() {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.STARTING);
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.OK, finish(active));
        var run = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(RunStatus.COMPLETED, run.getStatus());
        assertNull(run.getInternalTokenHash());
        assertEquals("COMPLETED", projects.findById(active.project().getId()).orElseThrow().getStatus());
        assertEquals(HttpStatus.FORBIDDEN, post(active, 1, .5));
    }

    @ParameterizedTest
    @CsvSource({"running,false", "failed,false", "stopped,false", "running,true", "failed,true", "stopped,true"})
    void waitingLifecycleWriterCannotOverwriteTheFirstTerminalTransition(String transition, boolean preloaded) throws Exception {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.STARTING);
        runs.saveAndFlush(active.run());
        var pending = new java.util.concurrent.atomic.AtomicReference<java.util.concurrent.Future<?>>();
        try (var worker = Executors.newSingleThreadExecutor()) {
            new TransactionTemplate(transactionManager).executeWithoutResult(transaction -> {
                // Hold the row while the other transaction reaches its read/write boundary.
                runs.lockById(active.run().getId()).orElseThrow();
                pending.set(worker.submit(() -> {
                    Runnable write = () -> {
                        switch (transition) {
                            case "running" -> lifecycle.markRunning(active.run().getId(), 50001);
                            case "failed" -> lifecycle.markFailed(active.run().getId());
                            case "stopped" -> lifecycle.markStopped(active.run().getId());
                            default -> throw new AssertionError(transition);
                        }
                    };
                    if (preloaded) {
                        new TransactionTemplate(transactionManager).executeWithoutResult(outer -> {
                            runs.findById(active.run().getId()).orElseThrow();
                            write.run();
                        });
                    } else {
                        write.run();
                    }
                }));
                awaitRunLockWait();
                projectService.markProjectAsCompleted(active.project().getId(), new InternalCallbackContext(
                        active.project().getId(), active.run().getId(), tokens.hash(active.token())));
            });
            pending.get().get(20, TimeUnit.SECONDS);
        }
        var completed = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(RunStatus.COMPLETED, completed.getStatus());
        assertNull(completed.getInternalTokenHash());
        assertNull(completed.getServerPort());
    }

    @Test
    void cachedProjectCannotAuthorizeCompletionAfterAnotherRunBecomesActive() {
        ActiveRun original = activeRun();
        var replacement = new java.util.concurrent.atomic.AtomicReference<ActiveRun>();
        try (var worker = Executors.newSingleThreadExecutor()) {
            assertThrows(IllegalStateException.class, () -> new TransactionTemplate(transactionManager)
                    .executeWithoutResult(transaction -> {
                        projects.findById(original.project().getId()).orElseThrow();
                        runs.findById(original.run().getId()).orElseThrow();
                        replacement.set(java.util.concurrent.CompletableFuture.supplyAsync(() ->
                                newRun(projects.findById(original.project().getId()).orElseThrow()), worker).join());
                        projectService.markProjectAsCompleted(original.project().getId(), new InternalCallbackContext(
                                original.project().getId(), original.run().getId(), tokens.hash(original.token())));
                    }));
        }
        assertEquals(replacement.get().run().getId(), projects.findById(original.project().getId()).orElseThrow().getActiveRunId());
        assertEquals(RunStatus.RUNNING, runs.findById(original.run().getId()).orElseThrow().getStatus());
        assertEquals(RunStatus.RUNNING, runs.findById(replacement.get().run().getId()).orElseThrow().getStatus());
    }

    @Test
    void stoppingAndAnAlreadyAuthenticatedCompletionDoNotDeadlock() throws Exception {
        ActiveRun active = activeRun();
        var owner = users.findById(active.project().getUser().getId()).orElseThrow();
        owner.setPassword(passwordEncoder.encode("Password1!"));
        users.saveAndFlush(owner);
        var loginHeaders = new HttpHeaders();
        loginHeaders.setContentType(MediaType.APPLICATION_JSON);
        var login = rest.postForEntity("/api/auth/login", new HttpEntity<>(java.util.Map.of(
                "username", owner.getUsername(), "password", "Password1!"), loginHeaders), String.class);
        assertEquals(HttpStatus.OK, login.getStatusCode());
        var stopHeaders = new HttpHeaders();
        stopHeaders.set(HttpHeaders.COOKIE, login.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0]);
        var runLocked = new CountDownLatch(1);
        var allowStopToCommit = new CountDownLatch(1);
        doAnswer(call -> {
            Object result = call.callRealMethod();
            runLocked.countDown();
            assertTrue(allowStopToCommit.await(10, TimeUnit.SECONDS));
            return result;
        }).when(lifecycle).markStopped(active.run().getId());
        try (var workers = Executors.newFixedThreadPool(2)) {
            var stopped = workers.submit(() -> rest.exchange("/api/projects/" + active.project().getId() + "/stop",
                    HttpMethod.POST, new HttpEntity<>(stopHeaders), String.class));
            try {
                assertTrue(runLocked.await(10, TimeUnit.SECONDS));
                var completion = workers.submit(() -> {
                    try {
                        // Represents a callback admitted before stop evicted its token from the filter registry.
                        projectService.markProjectAsCompleted(active.project().getId(), new InternalCallbackContext(
                                active.project().getId(), active.run().getId(), tokens.hash(active.token())));
                        return false;
                    } catch (org.springframework.security.access.AccessDeniedException | IllegalStateException retired) {
                        return true;
                    }
                });
                awaitRunLockWait();
                allowStopToCommit.countDown();
                assertEquals(HttpStatus.OK, stopped.get(20, TimeUnit.SECONDS).getStatusCode());
                assertTrue(completion.get(20, TimeUnit.SECONDS));
            } finally {
                allowStopToCommit.countDown();
            }
        }
        assertEquals(RunStatus.STOPPED, runs.findById(active.run().getId()).orElseThrow().getStatus());
        assertEquals("STOPPED", projects.findById(active.project().getId()).orElseThrow().getStatus());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateProcessIdentityCannotRestoreCompletedRun(boolean preloaded) {
        ActiveRun active = activeRun();
        Runnable attach = () -> {
            assertEquals(HttpStatus.OK, finish(active));
            attachIdentity(active, tokens.hash(active.token()));
        };
        if (preloaded) {
            new TransactionTemplate(transactionManager).executeWithoutResult(transaction -> {
                runs.findById(active.run().getId()).orElseThrow();
                attach.run();
            });
        } else {
            attach.run();
        }
        var completed = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(RunStatus.COMPLETED, completed.getStatus());
        assertNull(completed.getServerPid());
        assertNull(completed.getProcessStartedAt());
        assertNull(completed.getServerPort());
        assertNull(completed.getInternalTokenHash());
    }

    @Test
    void processIdentityCannotRestoreRetiredCallbackAuthorization() {
        ActiveRun active = activeRun();
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        active.run().setInternalTokenHash(tokens.hash(replacement));
        runs.saveAndFlush(active.run());
        assertThrows(IllegalStateException.class, () -> attachIdentity(active, tokens.hash(active.token())));
        var current = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(tokens.hash(replacement), current.getInternalTokenHash());
        assertNull(current.getServerPid());
        assertEquals(HttpStatus.FORBIDDEN, post(active, 1, .5));
        assertEquals(HttpStatus.OK, post(new ActiveRun(active.project(), current, replacement), 1, .5));
    }

    @Test
    void currentProcessMayRecordItsIdentityWithoutChangingItsAuthorization() {
        ActiveRun active = activeRun();
        String hash = tokens.hash(active.token());
        attachIdentity(active, hash);
        var current = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(RunStatus.RUNNING, current.getStatus());
        assertEquals(12345L, current.getServerPid());
        assertEquals(50001, current.getServerPort());
        assertEquals(hash, current.getInternalTokenHash());
        assertEquals(HttpStatus.OK, post(active, 1, .5));
    }

    private void attachIdentity(ActiveRun active, String hash) {
        org.springframework.test.util.ReflectionTestUtils.invokeMethod(serverManager, "recordProcessIdentity",
                active.run().getId(), 12345L, Instant.now(), 50001, hash);
    }

    @Test
    void launchAuthorizationCommitsBeforeAnEnclosingRequestCanRollBack() {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.STARTING);
        runs.saveAndFlush(active.run());
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        new TransactionTemplate(transactionManager).executeWithoutResult(transaction -> {
            runs.findById(active.run().getId()).orElseThrow();
            processIdentities.authorizeLaunch(active.run().getId(), tokens.hash(replacement));
            assertEquals(HttpStatus.OK, post(new ActiveRun(active.project(), active.run(), replacement), 1, .5));
            transaction.setRollbackOnly();
        });
        assertEquals(tokens.hash(replacement), runs.findById(active.run().getId()).orElseThrow().getInternalTokenHash());
        assertEquals(HttpStatus.FORBIDDEN, post(active, 2, .5));
    }

    @ParameterizedTest
    @ValueSource(strings = {"COMPLETED", "FAILED", "STOPPED", "RUNNING"})
    void onlyAStartingRunCanAuthorizeALaunch(String status) {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.valueOf(status));
        runs.saveAndFlush(active.run());
        assertThrows(IllegalStateException.class, () -> processIdentities.authorizeLaunch(active.run().getId(), "replacement"));
        assertEquals(tokens.hash(active.token()), runs.findById(active.run().getId()).orElseThrow().getInternalTokenHash());
    }

    @ParameterizedTest
    @ValueSource(strings = {"FAILED", "STOPPED"})
    void inactiveProcessCannotAttachAnIdentity(String status) {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.valueOf(status));
        active.run().setInternalTokenHash(null);
        runs.saveAndFlush(active.run());
        assertThrows(IllegalStateException.class, () -> attachIdentity(active, tokens.hash(active.token())));
        var run = runs.findById(active.run().getId()).orElseThrow();
        assertEquals(RunStatus.valueOf(status), run.getStatus());
        assertNull(run.getServerPid());
        assertNull(run.getInternalTokenHash());
    }

    private void awaitRunLockWait() {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (System.nanoTime() < deadline) {
            jdbc.execute("SELECT pg_stat_clear_snapshot()");
            Boolean waiting = jdbc.queryForObject("SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                    + "WHERE datname = current_database() AND wait_event_type = 'Lock' "
                    + "AND (lower(query) LIKE '%runs%' OR lower(query) LIKE '%projects%'))", Boolean.class);
            if (Boolean.TRUE.equals(waiting)) return;
            try {
                Thread.sleep(10);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new AssertionError(interrupted);
            }
        }
        fail("The competing lifecycle transaction did not reach the held database lock");
    }
}
