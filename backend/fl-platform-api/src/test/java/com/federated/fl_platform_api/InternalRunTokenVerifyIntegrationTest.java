package com.federated.fl_platform_api;

import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.context.ActiveProfiles;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * BA-6 — end-to-end coverage of {@code POST /api/internal/runs/{projectId}/{runId}/verify-connection-token}
 * through the REAL filter chain. Proves two things:
 *   1. the endpoint is unreachable without the internal-API-key + per-run-token gate
 *      ({@link com.federated.fl_platform_api.security.InternalApiKeyFilter}); and
 *   2. once past the gate, a valid connection token minted for the matching run resolves to
 *      the enrolled identity (200), while a token scoped to another run/project is rejected.
 *
 * <p>Runs on the shared Testcontainers-Postgres context; the endpoint itself is stateless
 * (pure token verification) so no DB rows are needed.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class InternalRunTokenVerifyIntegrationTest {

    private static final String INTERNAL_KEY = "test-internal-key";  // application-test.properties

    @Autowired private TestRestTemplate rest;
    @Autowired private RunTokenRegistry runTokenRegistry;
    @Autowired private ConnectionTokenService tokenService;

    private HttpHeaders headers(String internalKey, String runToken) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        if (internalKey != null) {
            h.set("X-Internal-Key", internalKey);
        }
        if (runToken != null) {
            h.set("X-Internal-Run-Token", runToken);
        }
        return h;
    }

    private String url(UUID projectId, UUID runId) {
        return "/api/internal/runs/" + projectId + "/" + runId + "/verify-connection-token";
    }

    private String body(String connectionToken) {
        return "{\"connectionToken\":\"" + connectionToken + "\"}";
    }

    private String mintConnectionToken(long userId, UUID runId, UUID projectId, int partitionId, String kind) {
        return tokenService.mint(new ConnectionTokenService.Claims(
                userId, runId, projectId, partitionId, "localhost:50000", null, kind)).token();
    }

    @Test
    void withoutInternalApiKey_isUnauthorized() {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String runToken = runTokenRegistry.mint(projectId, runId);
        String connToken = mintConnectionToken(1L, runId, projectId, 0, "SHARD");

        ResponseEntity<String> resp = rest.exchange(
                url(projectId, runId), HttpMethod.POST,
                new HttpEntity<>(body(connToken), headers(null, runToken)), String.class);

        assertEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode(),
                "the internal filter must reject a call with no X-Internal-Key");
    }

    @Test
    void withoutRunToken_isUnauthorized() {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String connToken = mintConnectionToken(1L, runId, projectId, 0, "SHARD");

        ResponseEntity<String> resp = rest.exchange(
                url(projectId, runId), HttpMethod.POST,
                new HttpEntity<>(body(connToken), headers(INTERNAL_KEY, null)), String.class);

        assertEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode(),
                "the static key alone is not sufficient — a scoped per-run token is required");
    }

    @Test
    void runTokenScopedToAnotherProject_isForbiddenByTheGate() {
        UUID projectA = UUID.randomUUID();
        UUID projectB = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String runTokenForA = runTokenRegistry.mint(projectA, runId);
        String connToken = mintConnectionToken(1L, runId, projectB, 0, "SHARD");

        // Path targets project B, but the run token is scoped to project A → filter forbids it
        // before the controller is ever reached.
        ResponseEntity<String> resp = rest.exchange(
                url(projectB, runId), HttpMethod.POST,
                new HttpEntity<>(body(connToken), headers(INTERNAL_KEY, runTokenForA)), String.class);

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode(),
                "a run token scoped to project A cannot verify tokens under project B's path");
    }

    @Test
    void validConnectionTokenForMatchingRun_returns200_withIdentity() {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String runToken = runTokenRegistry.mint(projectId, runId);
        String connToken = mintConnectionToken(99L, runId, projectId, 2, "SHARD");

        ResponseEntity<String> resp = rest.exchange(
                url(projectId, runId), HttpMethod.POST,
                new HttpEntity<>(body(connToken), headers(INTERNAL_KEY, runToken)), String.class);

        assertEquals(HttpStatus.OK, resp.getStatusCode());
        String b = resp.getBody();
        org.junit.jupiter.api.Assertions.assertNotNull(b);
        org.junit.jupiter.api.Assertions.assertTrue(b.contains("\"userId\":99"), b);
        org.junit.jupiter.api.Assertions.assertTrue(b.contains("\"partitionId\":2"), b);
        org.junit.jupiter.api.Assertions.assertTrue(b.contains("\"clientKind\":\"SHARD\""), b);
        org.junit.jupiter.api.Assertions.assertTrue(b.contains(runId.toString()), b);
    }

    @Test
    void connectionTokenForDifferentRun_returns403() {
        UUID projectId = UUID.randomUUID();
        UUID tokenRunId = UUID.randomUUID();
        UUID pathRunId = UUID.randomUUID();   // != token's runId
        // Run token must be scoped to the SAME project so the filter admits the call; the
        // controller then rejects because the connection token belongs to a different run.
        String runToken = runTokenRegistry.mint(projectId, pathRunId);
        String connToken = mintConnectionToken(1L, tokenRunId, projectId, 0, "SHARD");

        ResponseEntity<String> resp = rest.exchange(
                url(projectId, pathRunId), HttpMethod.POST,
                new HttpEntity<>(body(connToken), headers(INTERNAL_KEY, runToken)), String.class);

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode(),
                "an authentic token minted for another run must be rejected with 403");
    }
}
