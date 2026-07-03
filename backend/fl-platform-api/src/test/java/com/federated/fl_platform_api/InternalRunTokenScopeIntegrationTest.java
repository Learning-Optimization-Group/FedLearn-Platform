package com.federated.fl_platform_api;

import com.federated.fl_platform_api.security.RunTokenRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
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

    @Autowired private TestRestTemplate rest;
    @Autowired private RunTokenRegistry runTokenRegistry;

    private HttpHeaders headers(String runToken) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.set("X-Internal-Key", INTERNAL_KEY);      // valid outer gate
        if (runToken != null) {
            h.set("X-Internal-Run-Token", runToken);
        }
        return h;
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
}
