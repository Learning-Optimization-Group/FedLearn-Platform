package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

import static org.assertj.core.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class SimulationRuntimeIntegrationTest {
    static final Path ROOT = fixture();
    static final String PIN = "b".repeat(64);
    @Autowired TestRestTemplate http;
    @Autowired UserRepository users;
    @Autowired PasswordEncoder passwords;
    @Autowired SimulationRunRepository runs;
    @Autowired SimulationService service;
    @Autowired com.federated.fl_platform_api.service.AuthorizationService authorization;
    @Autowired SimulationProperties properties;
    @MockitoSpyBean SimulationProcessRunner runner;
    private final java.util.List<String> cookies = new java.util.ArrayList<>();

    static Path fixture() {
        try {
            Path root = Files.createTempDirectory("lab-contract-");
            Path source = root.resolve("source/fedlearn/simulation"); Files.createDirectories(source);
            Files.writeString(source.resolve("__init__.py"), ""); Files.writeString(source.getParent().resolve("__init__.py"), "");
            try (var worker = SimulationRuntimeIntegrationTest.class.getResourceAsStream("/simulation/contract_worker.py");
                 var cache = SimulationRuntimeIntegrationTest.class.getResourceAsStream("/simulation/contract_cache.py")) {
                Files.copy(worker, source.getParent().resolve("lab_worker.py")); Files.copy(cache, source.resolve("nih_cache.py"));
            }
            Files.createDirectory(root.resolve("cache"));
            Files.writeString(root.resolve("cache/report.json"), "{\"pin\":\"" + "b".repeat(64) + "\",\"pretrained\":true,\"dataset\":\"synthetic-contract-control\",\"splits\":{\"validation\":{\"images\":8,\"patients\":4}}}");
            return root;
        } catch (Exception error) { throw new ExceptionInInitializerError(error); }
    }

    @DynamicPropertySource static void properties(DynamicPropertyRegistry registry) {
        registry.add("app.simulation.enabled", () -> true);
        registry.add("app.simulation.interpreter", () -> "/usr/bin/python3");
        registry.add("app.simulation.source-root", () -> ROOT.resolve("source").toString());
        registry.add("app.simulation.cache", () -> ROOT.resolve("cache").toString());
        registry.add("app.simulation.storage", () -> ROOT.resolve("jobs").toString());
        registry.add("app.simulation.report-sha256", () -> PIN);
        registry.add("app.simulation.deadline-seconds", () -> 2);
        registry.add("app.simulation.reconcile-on-startup", () -> false);
    }

    String owner() {
        String name = "lab-runtime-" + UUID.randomUUID();
        User user = new User(name, name + "@example.com", passwords.encode("Password1!"));
        user.setPlatformRole(PlatformRole.PROJECT_OWNER); users.save(user);
        var response = http.postForEntity("/api/auth/login", Map.of("username", name, "password", "Password1!"), JsonNode.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        String cookie = response.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];
        cookies.add(cookie); return cookie;
    }

    ResponseEntity<JsonNode> request(String cookie, HttpMethod method, String path, String body) {
        HttpHeaders headers = new HttpHeaders(); headers.setContentType(MediaType.APPLICATION_JSON); headers.set(HttpHeaders.COOKIE, cookie);
        return http.exchange("/api/simulations" + path, method, new HttpEntity<>(body, headers), JsonNode.class);
    }

    ResponseEntity<JsonNode> start(String cookie, int seed) { return request(cookie, HttpMethod.POST, "/runs", "{\"choices\":{\"rounds\":2,\"seed\":" + seed + "}}"); }

    JsonNode terminal(String cookie, String id) throws Exception {
        long until = System.nanoTime() + Duration.ofSeconds(9).toNanos();
        JsonNode result;
        do {
            result = request(cookie, HttpMethod.GET, "/runs/" + id, null).getBody();
            if (java.util.Set.of("COMPLETED", "FAILED", "STOPPED", "TIMED_OUT", "INTERRUPTED").contains(result.path("status").asText())) return result;
            Thread.sleep(25);
        } while (System.nanoTime() < until);
        throw new AssertionError("Run did not terminate: " + result);
    }

    @AfterEach void drain() throws Exception {
        for (String cookie : cookies) {
            var listed = request(cookie, HttpMethod.GET, "/runs", null).getBody();
            for (JsonNode run : listed.path("content")) {
                String id = run.path("id").asText();
                request(cookie, HttpMethod.POST, "/runs/" + id + "/stop", "{}"); terminal(cookie, id);
            }
        }
        assertThat(runs.findByStatusIn(java.util.List.of(SimulationStatus.RUNNING, SimulationStatus.STARTING, SimulationStatus.STOPPING))).isEmpty();
    }

    @Test void realProcessRoundEvidencePersistsAndDownloadIsOwnedAndRedacted() throws Exception {
        String owner = owner(); String other = owner();
        assertThat(request(owner, HttpMethod.GET, "/workloads", null).getBody().path("workloads").get(0).path("ready").asBoolean()).isTrue();
        assertThat(request(owner, HttpMethod.GET, "/workloads", null).getBody().path("workloads").get(0)
                .path("limits").path("deadlineSeconds").asInt()).isEqualTo(2);
        var resolved = request(owner, HttpMethod.POST, "/resolve", "{\"choices\":{\"rounds\":2}}");
        assertThat(resolved.getStatusCode()).isEqualTo(HttpStatus.OK);
        var accepted = start(owner, 42);
        assertThat(accepted.getStatusCode()).isEqualTo(HttpStatus.ACCEPTED);
        String id = accepted.getBody().path("id").asText();
        JsonNode completed = terminal(owner, id);
        assertThat(completed.path("status").asText()).isEqualTo("COMPLETED");
        assertThat(completed.path("completedRounds").asInt()).isEqualTo(2);
        assertThat(completed.path("rounds").size()).isEqualTo(2);
        assertThat(completed.path("configHash")).isEqualTo(resolved.getBody().path("configHash"));
        var download = request(owner, HttpMethod.GET, "/runs/" + id + "/record", null);
        assertThat(download.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(download.getHeaders().getFirst(HttpHeaders.CONTENT_DISPOSITION)).isEqualTo("attachment; filename=\"simulation-" + id + "-research.json\"");
        assertThat(download.getBody().path("qualification").path("scope").asText()).isEqualTo("simulation_execution_only");
        assertThat(download.getBody().toString()).doesNotContain(ROOT.toString(), "private-contract-host", "features_path");
        assertThat(download.getBody().path("resolvedConfig").path("seeds").path("attack").asInt()).isEqualTo(1042);
        assertThat(download.getBody().path("resolvedConfig").path("federation").path("network").path("uplink_mbps").asDouble()).isEqualTo(10);
        assertThat(download.getBody().path("source").path("runtime").path("python").asText()).isEqualTo("3.12.7");
        assertThat(download.getBody().path("execution").path("peakRssBytes").asLong()).isEqualTo(100000000);
        assertThat(download.getBody().path("rounds").get(0).path("participants").path("accepted").asInt()).isEqualTo(10);
        assertThat(download.getBody().toString()).doesNotContain("private-client-identity", "per_client_bytes");
        Path job = runner.jobDirectory(UUID.fromString(id)); Path archived = ROOT.resolve("archived-" + id);
        Files.move(job, archived);
        try { assertThat(request(owner, HttpMethod.GET, "/runs/" + id + "/record", null).getBody()).isEqualTo(download.getBody()); }
        finally { Files.move(archived, job); }
        for (String suffix : new String[]{"", "/record", "/stop"}) assertThat(request(other, suffix.equals("/stop") ? HttpMethod.POST : HttpMethod.GET,
                "/runs/" + id + suffix, "{}").getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        assertThat(request(owner, HttpMethod.POST, "/runs/" + id + "/stop", "{}").getBody()).isEqualTo(completed);
        assertThat(runs.findById(UUID.fromString(id)).orElseThrow().getExitCode()).isZero();
    }

    @Test void simultaneousStartsAdmitOneAndStopDrainsBeforeNextAdmission() throws Exception {
        String owner = owner();
        var first = CompletableFuture.supplyAsync(() -> start(owner, 43));
        var second = CompletableFuture.supplyAsync(() -> start(owner, 43));
        var a = first.get(); var b = second.get();
        assertThat(java.util.List.of(a.getStatusCode(), b.getStatusCode())).containsExactlyInAnyOrder(HttpStatus.ACCEPTED, HttpStatus.CONFLICT);
        String id = (a.getStatusCode() == HttpStatus.ACCEPTED ? a : b).getBody().path("id").asText();
        var stop = request(owner, HttpMethod.POST, "/runs/" + id + "/stop", "{}");
        assertThat(stop.getBody().path("status").asText()).isIn("STOPPING", "STOPPED");
        assertThat(terminal(owner, id).path("status").asText()).isEqualTo("STOPPED");
        var run = runs.findById(UUID.fromString(id)).orElseThrow();
        assertThat(ProcessHandle.of(run.getWorkerPid()).map(ProcessHandle::isAlive).orElse(false)).isFalse();
        assertThat(start(owner, 42).getStatusCode()).isEqualTo(HttpStatus.ACCEPTED);
    }

    @Test void actualTimeoutAndUnqualifiedExitCannotAppearCompleted() throws Exception {
        String owner = owner();
        String hung = start(owner, 46).getBody().path("id").asText();
        assertThat(terminal(owner, hung).path("status").asText()).isEqualTo("TIMED_OUT");
        String unqualified = start(owner, 44).getBody().path("id").asText();
        var detail = terminal(owner, unqualified);
        assertThat(detail.path("status").asText()).isEqualTo("FAILED");
        assertThat(detail.path("completedRounds").asInt()).isEqualTo(2);
        assertThat(detail.path("qualification").isNull()).isTrue();
        String foreign = start(owner, 45).getBody().path("id").asText();
        assertThat(terminal(owner, foreign).path("status").asText()).isEqualTo("FAILED");
    }

    @Test void resolverTracebackIsPrivateAndUnsupportedChoicesDoNotCreateJobs() {
        String owner = owner(); long before = runs.count();
        for (String choices : new String[]{"{\"features_path\":\"/private\"}", "{\"epsilon\":" + "9".repeat(400) + "}"}) {
            var response = request(owner, HttpMethod.POST, "/resolve", "{\"choices\":" + choices + "}");
            assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
            assertThat(response.getBody().toString()).doesNotContain("/private", "Traceback", "OverflowError");
        }
        assertThat(runs.count()).isEqualTo(before);
    }

    @Test void privateStderrWarningsDoNotCorruptSuccessfulResolverJson() {
        var response = request(owner(), HttpMethod.POST, "/resolve", "{\"choices\":{\"seed\":50}}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().toString()).doesNotContain("/private/control-only-diagnostic");
    }

    @Test void failedLaunchIsRetainedAndDoesNotOccupyAdmission() throws Exception {
        String owner = owner();
        org.mockito.Mockito.doThrow(new java.io.IOException("/private/launch-error"))
                .when(runner).launch(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.any());
        try {
            var response = start(owner, 42);
            assertThat(response.getStatusCode()).isEqualTo(HttpStatus.ACCEPTED);
            JsonNode failed = terminal(owner, response.getBody().path("id").asText());
            assertThat(failed.path("status").asText()).isEqualTo("FAILED");
            assertThat(failed.path("errorCode").asText()).isEqualTo("LAUNCH_FAILED");
            assertThat(failed.toString()).doesNotContain("/private/launch-error");
        } finally { org.mockito.Mockito.reset(runner); }
        String id = start(owner, 42).getBody().path("id").asText();
        assertThat(terminal(owner, id).path("status").asText()).isEqualTo("COMPLETED");
    }

    SimulationRun oldJob(SimulationStatus status) {
        owner();
        SimulationRun run = new SimulationRun(); run.setId(UUID.randomUUID()); run.setCreatedBy(users.findAll().getFirst().getId());
        run.setStatus(status); run.setConfigHash("a".repeat(64)); run.setRecipeJson("{}"); run.setCreatedAt(java.time.Instant.now());
        run.setRequestedRounds(2); return run;
    }

    @Test void restartReapsOnlyExactOwnedProcessAndRetainsInterruptedOutcome() throws Exception {
        var resolution = runner.resolve(SimulationJson.stored("{\"rounds\":2,\"seed\":43}"));
        SimulationRun old = oldJob(SimulationStatus.RUNNING);
        var child = runner.launch(old.getId(), resolution.config());
        try {
            Path progress = runner.jobDirectory(old.getId()).resolve("execution/progress.json");
            long until = System.nanoTime() + Duration.ofSeconds(4).toNanos();
            while (!Files.exists(progress) && System.nanoTime() < until) Thread.sleep(10);
            assertThat(progress).exists();
            old.setConfigHash(resolution.hash()); old.setRecipeJson(resolution.recipe().toString());
            old.setWorkerPid(child.process().pid()); old.setWorkerStartIdentity(child.startIdentity());
            old.setWorkerExecutableIdentity(child.executableIdentity()); old = runs.saveAndFlush(old);
            service.reconcile();
            assertThat(child.awaitDrain(Duration.ofSeconds(5))).isTrue();
            assertThat(runs.findById(old.getId()).orElseThrow().getStatus()).isEqualTo(SimulationStatus.INTERRUPTED);
        } finally { child.stop(); child.awaitDrain(Duration.ofSeconds(6)); runs.deleteById(old.getId()); }
    }

    @Test void reusedPidIsNeverKilledAndAmbiguousOwnershipBlocksNewLaunch() throws Exception {
        Process foreign = new ProcessBuilder("/bin/sleep", "30").start();
        SimulationRun old = oldJob(SimulationStatus.RUNNING);
        try {
            old.setWorkerPid(foreign.pid()); old.setWorkerStartIdentity("1970-01-01T00:00:00Z");
            old.setWorkerExecutableIdentity(foreign.info().command().orElseThrow()); old = runs.saveAndFlush(old);
            service.reconcile();
            assertThat(foreign.isAlive()).isTrue();
            assertThat(runs.findById(old.getId()).orElseThrow().getStatus()).isEqualTo(SimulationStatus.INTERRUPTED);
            runs.deleteById(old.getId());
            old = oldJob(SimulationStatus.RUNNING); old.setWorkerPid(foreign.pid());
            old.setWorkerStartIdentity(foreign.info().startInstant().orElseThrow().toString());
            old.setWorkerExecutableIdentity(foreign.info().command().orElseThrow()); old = runs.saveAndFlush(old);
            service.reconcile();
            assertThat(foreign.isAlive()).isTrue();
            assertThat(runs.findById(old.getId()).orElseThrow().getErrorCode()).isEqualTo("OWNERSHIP_UNRESOLVED");
            assertThat(start(owner(), 42).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        } finally { foreign.destroyForcibly(); foreign.waitFor(); runs.deleteById(old.getId()); }
    }

    @Test void unknownLaunchWindowBlocksInsteadOfAssumingNoChildExists() {
        SimulationRun old = runs.saveAndFlush(oldJob(SimulationStatus.STARTING));
        try {
            service.reconcile();
            assertThat(runs.findById(old.getId()).orElseThrow().getErrorCode()).isEqualTo("OWNERSHIP_UNRESOLVED");
            assertThat(start(owner(), 42).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        } finally { runs.deleteById(old.getId()); }
    }

    @Test void backendShutdownDrainsItsWorkerAndPersistsInterruption() throws Exception {
        String cookie = owner();
        HttpHeaders headers = new HttpHeaders(); headers.set(HttpHeaders.COOKIE, cookie);
        String name = http.exchange("/api/auth/me", HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class).getBody().path("username").asText();
        org.springframework.security.core.context.SecurityContextHolder.getContext().setAuthentication(
                new org.springframework.security.authentication.UsernamePasswordAuthenticationToken(name, "", java.util.List.of()));
        try (var isolatedRunner = new SimulationProcessRunner(properties)) {
            var isolatedService = new SimulationService(runs, authorization, isolatedRunner, properties);
            try {
                var accepted = isolatedService.start(SimulationJson.stored("{\"seed\":43,\"rounds\":2}"));
                isolatedService.shutdown();
                SimulationRun stopped = runs.findById(accepted.id()).orElseThrow();
                assertThat(stopped.getStatus()).isEqualTo(SimulationStatus.INTERRUPTED);
                assertThat(ProcessHandle.of(stopped.getWorkerPid()).map(ProcessHandle::isAlive).orElse(false)).isFalse();
            } finally { isolatedService.shutdown(); }
        } finally { org.springframework.security.core.context.SecurityContextHolder.clearContext(); }
    }

    Path slowProductionStartup() throws Exception {
        Path source = Files.createTempDirectory(ROOT, "slow-production-startup-");
        Path packageRoot = Files.createDirectories(source.resolve("fedlearn/simulation"));
        Files.copy(ROOT.resolve("source/fedlearn/simulation/__init__.py"), packageRoot.resolve("__init__.py"));
        Files.copy(ROOT.resolve("source/fedlearn/simulation/nih_cache.py"), packageRoot.resolve("nih_cache.py"));
        Files.copy(ROOT.resolve("source/fedlearn/__init__.py"), packageRoot.getParent().resolve("__init__.py"));
        Path production = Path.of("../../framework/src/fedlearn/lab_worker.py").toAbsolutePath().normalize();
        Files.copy(production, source.resolve("production_worker.py"));
        Files.copy(ROOT.resolve("source/fedlearn/lab_worker.py"), source.resolve("contract_worker.py"));
        Files.writeString(packageRoot.getParent().resolve("lab_worker.py"), """
                import runpy,sys
                from pathlib import Path
                root=Path(__file__).resolve().parents[1]
                runpy.run_path(str(root/('production_worker.py' if sys.argv[1]=='run' else 'contract_worker.py')),run_name='__main__')
                """);
        Files.writeString(source.resolve("torch.py"), """
                from pathlib import Path
                import time
                Path(__file__).with_name('import-entered').write_text('heavy import control')
                time.sleep(10)
                """);
        return source;
    }

    void importing(Path source) throws Exception {
        long until = System.nanoTime() + Duration.ofSeconds(3).toNanos();
        while (!Files.exists(source.resolve("import-entered")) && System.nanoTime() < until) Thread.sleep(10);
        assertThat(source.resolve("import-entered")).exists();
    }

    @Test void stoppingDuringActualWorkerStartupRetainsStoppedRatherThanInvalidEvidence() throws Exception {
        String owner = owner(); Path original = properties.getSourceRoot();
        Path slow = slowProductionStartup(); properties.setSourceRoot(slow);
        try {
            String id = start(owner, 42).getBody().path("id").asText(); importing(slow);
            request(owner, HttpMethod.POST, "/runs/" + id + "/stop", "{}");
            var detail = terminal(owner, id);
            assertThat(detail.path("status").asText()).isEqualTo("STOPPED");
            assertThat(detail.path("completedRounds").asInt(-1)).isZero();
            assertThat(detail.path("qualification").isNull()).isTrue();
            assertThat(detail.path("recordAvailable").asBoolean()).isFalse();
        } finally { properties.setSourceRoot(original); }
        assertThat(terminal(owner, start(owner, 42).getBody().path("id").asText()).path("status").asText()).isEqualTo("COMPLETED");
    }

    @Test void deadlineDuringActualWorkerStartupRetainsTimedOutRatherThanInvalidEvidence() throws Exception {
        String owner = owner(); Path original = properties.getSourceRoot();
        Path slow = slowProductionStartup(); properties.setSourceRoot(slow);
        try {
            String id = start(owner, 42).getBody().path("id").asText(); importing(slow);
            var detail = terminal(owner, id);
            assertThat(detail.path("status").asText()).isEqualTo("TIMED_OUT");
            assertThat(detail.path("completedRounds").asInt(-1)).isZero();
            assertThat(detail.path("qualification").isNull()).isTrue();
            assertThat(detail.path("recordAvailable").asBoolean()).isFalse();
        } finally { properties.setSourceRoot(original); }
        assertThat(terminal(owner, start(owner, 42).getBody().path("id").asText()).path("status").asText()).isEqualTo("COMPLETED");
    }

    @Test void shutdownDuringActualWorkerStartupRetainsInterruptedRatherThanInvalidEvidence() throws Exception {
        String cookie = owner(); HttpHeaders headers = new HttpHeaders(); headers.set(HttpHeaders.COOKIE, cookie);
        String name = http.exchange("/api/auth/me", HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class).getBody().path("username").asText();
        org.springframework.security.core.context.SecurityContextHolder.getContext().setAuthentication(
                new org.springframework.security.authentication.UsernamePasswordAuthenticationToken(name, "", java.util.List.of()));
        Path original = properties.getSourceRoot(); Path slow = slowProductionStartup(); properties.setSourceRoot(slow);
        try (var isolatedRunner = new SimulationProcessRunner(properties)) {
            var isolatedService = new SimulationService(runs, authorization, isolatedRunner, properties);
            try {
                var accepted = isolatedService.start(SimulationJson.stored("{\"seed\":42,\"rounds\":2}")); importing(slow);
                isolatedService.shutdown();
                var interrupted = runs.findById(accepted.id()).orElseThrow();
                assertThat(interrupted.getStatus()).isEqualTo(SimulationStatus.INTERRUPTED);
                assertThat(ProcessHandle.of(interrupted.getWorkerPid()).map(ProcessHandle::isAlive).orElse(false)).isFalse();
                assertThat(interrupted.getCompletedRounds()).isZero();
                assertThat(interrupted.getRecordJson()).isNull();
            } finally { isolatedService.shutdown(); }
        } finally { properties.setSourceRoot(original); org.springframework.security.core.context.SecurityContextHolder.clearContext(); }
        assertThat(terminal(cookie, start(cookie, 42).getBody().path("id").asText()).path("status").asText()).isEqualTo("COMPLETED");
    }

    @Test void partialDownloadPreservesObservedParticipantsSeedsAndControlledTerminalMetadata() throws Exception {
        String owner = owner(); String id = start(owner, 43).getBody().path("id").asText();
        long until = System.nanoTime() + Duration.ofSeconds(3).toNanos();
        JsonNode progress;
        do {
            progress = request(owner, HttpMethod.GET, "/runs/" + id, null).getBody();
            if (progress.path("completedRounds").asInt() == 1) break;
            Thread.sleep(10);
        } while (System.nanoTime() < until);
        assertThat(progress.path("completedRounds").asInt()).isEqualTo(1);
        request(owner, HttpMethod.POST, "/runs/" + id + "/stop", "{}");
        assertThat(terminal(owner, id).path("status").asText()).isEqualTo("STOPPED");
        JsonNode record = request(owner, HttpMethod.GET, "/runs/" + id + "/record", null).getBody();
        assertThat(record.path("qualification").isNull()).isTrue();
        assertThat(record.path("rounds").size()).isEqualTo(1);
        assertThat(record.path("rounds").get(0).path("participants").path("accepted").asInt()).isEqualTo(10);
        assertThat(record.path("resolvedConfig").path("seeds").path("noise").asInt()).isEqualTo(2043);
        assertThat(record.path("terminal").path("status").asText()).isEqualTo("STOPPED");
        assertThat(record.path("source").path("runtime").path("git_commit").isNull()).isTrue();
        assertThat(record.toString()).doesNotContain(ROOT.toString(), "private-contract-host", "private-client-identity", "features_path");
    }
}
