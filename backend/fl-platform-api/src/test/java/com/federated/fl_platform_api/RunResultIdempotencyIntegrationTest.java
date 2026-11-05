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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;

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
}
