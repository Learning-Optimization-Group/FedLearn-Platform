package com.federated.fl_platform_api;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SE-7 "Done when": an {@code /api/internal/**} call authenticated with run A's token cannot mutate
 * project B. The static {@code X-Internal-Key} is the outer gate; the scoped per-run token binds each
 * callback to exactly one project, so a leaked/compromised run token can affect only its own project.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class InternalRunTokenScopeIntegrationTest {

    private static final String INTERNAL_KEY = "test-internal-key";  // application-test.properties

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired private TestRestTemplate rest;
    @Autowired private RunTokenRegistry runTokenRegistry;
    @Autowired private ProjectRepository projectRepository;
    @Autowired private UserRepository userRepository;
    @Autowired private PasswordEncoder passwordEncoder;
    @Autowired private RoundResultRepository roundResultRepository;

    private HttpHeaders headers(String runToken) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.set("X-Internal-Key", INTERNAL_KEY);      // valid outer gate
        if (runToken != null) {
            h.set("X-Internal-Run-Token", runToken);
        }
        return h;
    }

    private Project seedProject() {
        User owner = userRepository.save(new User(
                "ing-" + System.nanoTime(), "ing-" + System.nanoTime() + "@example.com",
                passwordEncoder.encode("Password1!")));
        Project p = new Project();
        p.setName("ing-" + System.nanoTime());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(ProjectVisibility.PRIVATE);
        return projectRepository.save(p);
    }

    @Test
    void runTokenScopedToProjectA_cannotMutateProjectB() {
        UUID projectA = UUID.randomUUID();
        UUID projectB = UUID.randomUUID();
        String tokenA = runTokenRegistry.mint(projectA, UUID.randomUUID());

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/results/" + projectB, HttpMethod.POST,
            new HttpEntity<>("{\"round\":1,\"loss\":0.5,\"accuracy\":0.9}", headers(tokenA)),
            String.class);

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode(),
            "run A's token must be forbidden from acting on project B — the whole point of SE-7");
    }

    @Test
    void staticKeyAloneWithoutRunToken_isNoLongerSufficient() {
        UUID projectA = UUID.randomUUID();
        runTokenRegistry.mint(projectA, UUID.randomUUID());

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/results/" + projectA, HttpMethod.POST,
            new HttpEntity<>("{}", headers(null)), String.class);

        assertEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode(),
            "the broad static key alone no longer grants unscoped internal access");
    }

    @Test
    void matchingProjectToken_passesTheScopeGate() {
        UUID projectA = UUID.randomUUID();
        String tokenA = runTokenRegistry.mint(projectA, UUID.randomUUID());

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/results/" + projectA, HttpMethod.POST,
            new HttpEntity<>("{\"round\":1,\"loss\":0.5,\"accuracy\":0.9}", headers(tokenA)),
            String.class);

        // Scope matches, so the internal filter admits the call; it then reaches the controller
        // (which 404s because project A was never persisted). The point: NOT blocked by the gate.
        assertNotEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode());
        assertNotEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
    }

    // BA-7 done-when: the same project-scope guarantee holds for the benchmark ingest endpoint (parity
    // with results), and legitimate ingestion for the token's OWN project actually succeeds end to end.

    @Test
    void benchmarkRunTokenScopedToProjectA_cannotMutateProjectB() {
        UUID projectA = UUID.randomUUID();
        UUID projectB = UUID.randomUUID();
        String tokenA = runTokenRegistry.mint(projectA, UUID.randomUUID());

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/benchmarks/" + projectB, HttpMethod.POST,
            new HttpEntity<>("{\"serverRound\":1,\"modelType\":\"CNN\",\"loss\":0.5,\"accuracy\":0.9}", headers(tokenA)),
            String.class);

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode(),
            "a run A token must not ingest benchmarks for project B");
    }

    @Test
    void legitimateResultIngestionForOwnProject_succeeds() {
        Project project = seedProject();
        String token = runTokenRegistry.mint(project.getId(), UUID.randomUUID());
        long before = roundResultRepository.count();

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/results/" + project.getId(), HttpMethod.POST,
            new HttpEntity<>("{\"serverRound\":1,\"loss\":0.5,\"accuracy\":0.9,\"gpuUtilization\":0.4}", headers(token)),
            String.class);

        assertEquals(HttpStatus.OK, resp.getStatusCode(),
            "a run's token ingesting a result for its own project must succeed");
        assertEquals(before + 1, roundResultRepository.count(), "the round result was persisted");
    }

    @Test
    void legitimateBenchmarkIngestionForOwnProject_succeeds() {
        Project project = seedProject();
        String token = runTokenRegistry.mint(project.getId(), UUID.randomUUID());

        ResponseEntity<String> resp = rest.exchange(
            "/api/internal/benchmarks/" + project.getId(), HttpMethod.POST,
            new HttpEntity<>("{\"serverRound\":1,\"modelType\":\"CNN\",\"loss\":0.5,\"accuracy\":0.9}", headers(token)),
            String.class);

        assertEquals(HttpStatus.OK, resp.getStatusCode(),
            "a run's token ingesting a benchmark for its own project must succeed");
    }
}
