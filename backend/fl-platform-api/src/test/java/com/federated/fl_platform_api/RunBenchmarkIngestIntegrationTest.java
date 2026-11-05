package com.federated.fl_platform_api;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.BenchmarkService;
import com.fasterxml.jackson.databind.JsonNode;
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
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;

/** Benchmark evidence must retain each run and the first accepted round payload. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class RunBenchmarkIngestIntegrationTest {
    @Autowired TestRestTemplate rest;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired UserRepository users;
    @Autowired BenchmarkRoundRepository rounds;
    @Autowired BenchmarkRunRepository summaries;
    @Autowired RunTokenRegistry tokens;
    @Autowired BenchmarkService benchmarks;
    @Autowired org.springframework.security.crypto.password.PasswordEncoder passwordEncoder;
    @Autowired org.springframework.jdbc.core.JdbcTemplate jdbc;

    record ActiveRun(Project project, Run run, String token) {}

    private ActiveRun activeRun() {
        String suffix = UUID.randomUUID().toString();
        User owner = users.save(new User("benchmark-" + suffix, suffix + "@example.com", "test-only-hash"));
        Project project = new Project();
        project.setName("benchmark-" + suffix);
        project.setUser(owner);
        project.setModelType("CNN");
        project.setModelName("net");
        project.setTaskType("SEQ_CLASSIFICATION");
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
        run.setIntent(RunIntent.capture(project, false, false, 120_000L, TrainingDataSource.FIXTURE));
        run.setCreatedAt(Instant.now());
        run = runs.saveAndFlush(run);
        String token = tokens.mint(project.getId(), run.getId());
        run.setInternalTokenHash(tokens.hash(token));
        run = runs.saveAndFlush(run);
        project.setActiveRunId(run.getId());
        return new ActiveRun(projects.saveAndFlush(project), run, token);
    }

    private HttpStatus post(ActiveRun active, int round, double loss) {
        return postRaw(active, Integer.toString(round), loss);
    }

    private HttpStatus postRaw(ActiveRun active, String round, double loss) {
        String body = "{\"serverRound\":" + round + ",\"loss\":" + loss
                + ",\"accuracy\":0.8,\"modelType\":\"CNN\",\"taskType\":\"SEQ_CLASSIFICATION\","
                + "\"roundDurationMs\":100,\"evalDurationMs\":20,\"clientCount\":1,\"samplesEvaluated\":8}";
        return postBody(active, body);
    }

    private HttpStatus postBody(ActiveRun active, String body) {
        var headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("X-Internal-Key", "test-internal-key");
        headers.set("X-Internal-Run-Token", active.token());
        return HttpStatus.valueOf(rest.exchange("/api/internal/benchmarks/" + active.project().getId(),
                HttpMethod.POST, new HttpEntity<>(body, headers), String.class).getStatusCode().value());
    }

    @Test
    void currentRunRecordsItsOwnBenchmark() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        var saved = rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst();
        assertEquals(active.run().getId(), saved.getRunId());
        assertEquals(.5, saved.getLoss());
    }

    @Test
    void identicalReplayPreservesTheOriginalRecordingTimes() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        var first = rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst();
        var summaryTime = summaries.findByProjectId(active.project().getId()).orElseThrow().getLastRecordedAt();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        var replayed = rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId());
        assertEquals(1, replayed.size());
        assertEquals(first.getId(), replayed.getFirst().getId());
        assertEquals(first.getRecordedAt(), replayed.getFirst().getRecordedAt());
        assertEquals(summaryTime, summaries.findByProjectId(active.project().getId()).orElseThrow().getLastRecordedAt());
    }

    @Test
    void conflictingReplayCannotReplaceTheFirstMeasurements() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        assertEquals(HttpStatus.CONFLICT, post(active, 1, .25));
        assertEquals(.5, rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst().getLoss());
        assertEquals(.5, summaries.findByProjectId(active.project().getId()).orElseThrow().getFinalLoss());
    }

    @Test
    void differentRunsRetainTheirOwnCopyOfTheSameRound() {
        ActiveRun first = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        ActiveRun second = newRun(first.project());
        assertEquals(HttpStatus.OK, post(second, 1, .25));
        var stored = rounds.findByProjectIdOrderByServerRoundAsc(first.project().getId());
        assertEquals(2, stored.size());
        assertEquals(.5, stored.stream().filter(r -> first.run().getId().equals(r.getRunId())).findFirst().orElseThrow().getLoss());
        assertEquals(.25, stored.stream().filter(r -> second.run().getId().equals(r.getRunId())).findFirst().orElseThrow().getLoss());
    }

    @Test
    void oldRunCannotOverwriteTheNewRunsMeasurements() {
        ActiveRun first = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        ActiveRun second = newRun(first.project());
        assertEquals(HttpStatus.OK, post(second, 1, .25));
        assertEquals(HttpStatus.CONFLICT, post(first, 1, 9.0));
        assertEquals(.25, summaries.findByProjectId(first.project().getId()).orElseThrow().getFinalLoss());
    }

    @Test
    void retiredAttemptCannotWriteButItsReplacementCan() {
        ActiveRun active = activeRun();
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        active.run().setInternalTokenHash(tokens.hash(replacement));
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.FORBIDDEN, post(active, 1, .5));
        assertEquals(0, rounds.countByProjectId(active.project().getId()));
        assertEquals(HttpStatus.OK, post(new ActiveRun(active.project(), active.run(), replacement), 1, .5));
    }

    @Test
    void currentSummaryAndChartNeverMixDifferentRuns() {
        ActiveRun first = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        assertEquals(HttpStatus.OK, post(first, 2, .4));
        ActiveRun second = newRun(first.project());
        assertEquals(HttpStatus.OK, post(second, 1, .8));
        var view = benchmarks.getProjectBenchmark(first.project().getId());
        assertEquals(1, view.getRounds().size());
        assertEquals(1, view.getSummary().getRoundsCompleted());
        assertEquals(.8, view.getSummary().getFinalLoss());
        assertEquals(.8, view.getRounds().getFirst().getLoss());
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "-1", "4", "1.9", "\"1\"", "null"})
    void invalidRoundCannotEnterBenchmarkEvidence(String round) {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, postRaw(active, round, .5));
        assertEquals(0, rounds.countByProjectId(active.project().getId()));
    }

    @Test
    void concurrentRetriesRetainOneRound() throws Exception {
        ActiveRun active = activeRun();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(4)) {
            var futures = new ArrayList<java.util.concurrent.Future<HttpStatus>>();
            for (int index = 0; index < 4; index++) {
                futures.add(workers.submit(() -> {
                    assertTrue(start.await(10, TimeUnit.SECONDS));
                    return post(active, 1, .5);
                }));
            }
            start.countDown();
            for (var future : futures) assertEquals(HttpStatus.OK, future.get(20, TimeUnit.SECONDS));
        }
        assertEquals(1, rounds.countByProjectId(active.project().getId()));
    }

    private HttpHeaders login(User user) {
        user.setPassword(passwordEncoder.encode("Password1!"));
        users.saveAndFlush(user);
        var response = rest.postForEntity("/api/auth/login", java.util.Map.of(
                "username", user.getUsername(), "password", "Password1!"), String.class);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        var headers = new HttpHeaders();
        headers.set(HttpHeaders.COOKIE, response.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0]);
        return headers;
    }

    private HttpHeaders adminHeaders() {
        String suffix = UUID.randomUUID().toString();
        var admin = new User("badmin-" + suffix, suffix + "@example.com", "test-only-hash");
        admin.setPlatformRole(PlatformRole.PLATFORM_ADMIN);
        return login(users.saveAndFlush(admin));
    }

    private ResponseEntity<JsonNode> selected(ActiveRun active, UUID runId, HttpHeaders headers) {
        return rest.exchange("/api/admin/benchmarks/projects/" + active.project().getId() + "?runId=" + runId,
                HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class);
    }

    @Test
    void adminCanSelectAnEarlierRunWithoutChangingTheLatestOverviewCache() {
        ActiveRun first = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        assertEquals(HttpStatus.OK, post(first, 2, .4));
        ActiveRun current = newRun(first.project());
        assertEquals(HttpStatus.OK, post(current, 1, .8));
        var response = selected(first, first.run().getId(), adminHeaders());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(first.run().getId().toString(), response.getBody().path("runId").asText());
        assertEquals(2, response.getBody().path("rounds").size());
        assertEquals(.5, response.getBody().path("rounds").get(0).path("loss").asDouble());
        assertTrue(response.getBody().path("summary").path("runIdentityVerified").asBoolean());
        assertEquals(current.run().getId(), summaries.findByProjectId(first.project().getId()).orElseThrow().getRunId());
        assertEquals(.8, summaries.findByProjectId(first.project().getId()).orElseThrow().getFinalLoss());
    }

    @Test
    void selectedRunMustBelongToTheRequestedProject() {
        ActiveRun first = activeRun(), other = activeRun();
        assertEquals(HttpStatus.OK, post(first, 1, .5));
        assertEquals(HttpStatus.OK, post(other, 1, .8));
        assertEquals(HttpStatus.NOT_FOUND, selected(first, other.run().getId(), adminHeaders()).getStatusCode());
    }

    @Test
    void selectedRunReadRemainsRestrictedToPlatformAdmins() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, post(active, 1, .5));
        var owner = users.findById(active.project().getUser().getId()).orElseThrow();
        assertEquals(HttpStatus.FORBIDDEN, selected(active, active.run().getId(), login(owner)).getStatusCode());
    }

    @Test
    void legacyReportedIdentityIsRetainedAndExplicitlyUnverified() {
        ActiveRun active = activeRun();
        UUID reportedRun = UUID.randomUUID(), id = UUID.randomUUID();
        jdbc.update("INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,loss,recorded_at) "
                + "VALUES (?,?,?,1,0.42,CURRENT_TIMESTAMP)", id, active.project().getId(), reportedRun);
        var response = selected(active, reportedRun, adminHeaders());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(reportedRun.toString(), response.getBody().path("runId").asText());
        assertFalse(response.getBody().path("summary").path("runIdentityVerified").asBoolean());
        assertEquals(.42, response.getBody().path("rounds").get(0).path("loss").asDouble());
        assertEquals(reportedRun, rounds.findById(id).orElseThrow().getRunId());
        assertNull(rounds.findById(id).orElseThrow().getPayloadSha256());
    }

    @Test
    void unverifiableLegacyPayloadCannotBeOverwrittenByAReplay() {
        ActiveRun active = activeRun();
        jdbc.update("INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,loss,recorded_at) "
                + "VALUES (?,?,?,1,0.5,CURRENT_TIMESTAMP)", UUID.randomUUID(), active.project().getId(), active.run().getId());
        assertEquals(HttpStatus.CONFLICT, post(active, 1, .5));
        assertNull(rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst().getPayloadSha256());
    }

    @Test
    void keyOrderNullOmissionAndClassificationAliasDoNotChangeTheAcceptedPayload() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1,\"extraMetrics\":{\"z\":2,\"a\":1}}"));
        var first = rounds.findByProjectIdOrderByServerRoundAsc(active.project().getId()).getFirst();
        assertEquals(HttpStatus.OK, postBody(active, "{\"loss\":null,\"taskType\":\"CLASSIFICATION\","
                + "\"modelType\":\"CNN\",\"extraMetrics\":{\"a\":1,\"z\":2},\"serverRound\":1}"));
        assertEquals(first.getRecordedAt(), rounds.findById(first.getId()).orElseThrow().getRecordedAt());
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "{\"serverRound\":1,\"loss\":\"NaN\"}",
            "{\"serverRound\":1,\"extraMetrics\":{\"compute\":\"Infinity\"}}",
            "{\"serverRound\":1,\"perClass\":[{\"label\":\"x\",\"f1\":\"-Infinity\"}]}",
            "{\"serverRound\":1,\"roundDurationMs\":-1}",
            "{\"serverRound\":1,\"clientCount\":-1}"
    })
    void invalidMeasurementsCannotEnterEvidence(String body) {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, postBody(active, body));
        assertEquals(0, rounds.countByProjectId(active.project().getId()));
    }

    @Test
    void incompleteSeriesDoesNotInventATimeToTarget() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1,\"accuracy\":0.5,\"targetAccuracy\":0.8,\"roundDurationMs\":100}"));
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":3,\"accuracy\":0.9,\"targetAccuracy\":0.8,\"roundDurationMs\":100}"));
        var summary = benchmarks.getProjectBenchmark(active.project().getId()).getSummary();
        assertNull(summary.getRoundsToTarget());
        assertNull(summary.getMsToTarget());
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "\"roundDurationMs\":-0.5", "\"roundDurationMs\":100.9", "\"evalDurationMs\":\"20\"",
            "\"paramCount\":1.9", "\"clientCount\":-0.5", "\"samplesEvaluated\":2.8", "\"numClasses\":2.1",
            "\"perClass\":[{\"label\":\"x\",\"support\":1.5}]", "\"confusionMatrix\":[[1.5]]",
            "\"confusionMatrix\":[[\"2\"]]", "\"confusionMatrix\":[[-1]]"
    })
    void integralMeasurementsCannotBeTruncatedOrCoerced(String field) {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, postBody(active, "{\"serverRound\":1," + field + "}"));
        assertEquals(0, rounds.countByProjectId(active.project().getId()));
    }

    @Test
    void integerDurationsAndStructuredCountsRetainTheirExactAcceptedValues() {
        ActiveRun active = activeRun();
        String body = "{\"serverRound\":1,\"roundDurationMs\":4294967296,\"evalDurationMs\":20,"
                + "\"paramCount\":4294967296,\"clientCount\":1,\"samplesEvaluated\":3,\"numClasses\":2,"
                + "\"perClass\":[{\"label\":\"x\",\"support\":3}],\"confusionMatrix\":[[2,1],[0,0]]}";
        assertEquals(HttpStatus.OK, postBody(active, body));
        assertEquals(HttpStatus.OK, postBody(active, body));
        var row = rounds.findByRunIdAndServerRound(active.run().getId(), 1).orElseThrow();
        assertEquals(4294967296L, row.getRoundDurationMs());
        assertEquals(4294967296L, row.getParamCount());
        assertEquals("[[2,1],[0,0]]", row.getConfusionMatrixJson());
    }

    @Test
    void missingDurationPreservesKnownTargetRoundWithoutInventingElapsedTime() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":1,\"accuracy\":0.5,\"targetAccuracy\":0.8}"));
        assertEquals(HttpStatus.OK, postBody(active, "{\"serverRound\":2,\"accuracy\":0.9,\"targetAccuracy\":0.8,\"roundDurationMs\":200}"));
        var summary = benchmarks.getProjectBenchmark(active.project().getId()).getSummary();
        assertEquals(2, summary.getRoundsToTarget());
        assertNull(summary.getMsToTarget());
    }

    @Test
    void legacyCachedRollupsCannotClaimTimeToTargetOrChangeVerifiedAggregates() {
        var before = benchmarks.getOverview();
        ActiveRun active = activeRun();
        BenchmarkRun legacy = new BenchmarkRun();
        legacy.setProjectId(active.project().getId());
        legacy.setRunId(active.run().getId());
        legacy.setTaskType("SEQ_CLASSIFICATION");
        legacy.setRoundsCompleted(99);
        legacy.setBestAccuracy(.999);
        legacy.setFinalAccuracy(.999);
        legacy.setFinalF1Macro(.999);
        legacy.setRoundsToTarget(17);
        legacy.setMsToTarget(123456L);
        legacy.setAvgRoundMs(123456L);
        legacy.setModelSizeMb(999.);
        summaries.saveAndFlush(legacy);
        var after = benchmarks.getOverview();
        var exposed = after.getRuns().stream().filter(r -> active.project().getId().toString().equals(r.getProjectId()))
                .findFirst().orElseThrow();
        assertFalse(exposed.isRunIdentityVerified());
        assertAll(
                () -> assertNull(exposed.getRoundsToTarget()),
                () -> assertNull(exposed.getMsToTarget()),
                () -> assertEquals(before.getClassificationRuns(), after.getClassificationRuns()),
                () -> assertEquals(before.getTotalRoundsRecorded(), after.getTotalRoundsRecorded()),
                () -> assertEquals(before.getAvgFinalAccuracy(), after.getAvgFinalAccuracy()),
                () -> assertEquals(before.getBestAccuracy(), after.getBestAccuracy()),
                () -> assertEquals(before.getAvgFinalF1Macro(), after.getAvgFinalF1Macro()),
                () -> assertEquals(before.getAvgRoundDurationMs(), after.getAvgRoundDurationMs()),
                () -> assertEquals(before.getAvgModelSizeMb(), after.getAvgModelSizeMb()));
        assertEquals(17, summaries.findById(legacy.getId()).orElseThrow().getRoundsToTarget());
    }

    @Test
    void explicitLegacySelectionKeepsUnknownRunsSeparateFromTheActiveRun() {
        ActiveRun active = activeRun();
        for (double loss : new double[] {.3, .4}) {
            jdbc.update("INSERT INTO benchmark_rounds (id,project_id,server_round,loss,recorded_at) "
                    + "VALUES (?,?,1,?,CURRENT_TIMESTAMP)", UUID.randomUUID(), active.project().getId(), loss);
        }
        assertEquals(HttpStatus.OK, post(active, 1, .9));
        var response = rest.exchange("/api/admin/benchmarks/projects/" + active.project().getId() + "?legacy=true",
                HttpMethod.GET, new HttpEntity<>(adminHeaders()), JsonNode.class);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertTrue(response.getBody().path("runId").isNull());
        assertEquals(2, response.getBody().path("rounds").size());
        assertFalse(response.getBody().path("summary").path("runIdentityVerified").asBoolean());
        assertTrue(response.getBody().path("rounds").findValues("loss").stream().noneMatch(n -> n.asDouble() == .9));
    }
}
