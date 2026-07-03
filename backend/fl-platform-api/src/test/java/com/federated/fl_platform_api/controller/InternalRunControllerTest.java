package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.security.ConnectionTokenService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.UUID;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * BA-6 — standalone MockMvc (no Spring context, no database) for the
 * connection-token verification endpoint. The {@code X-Internal-Key} +
 * per-run-token gate is a cross-cutting filter over all {@code /api/internal/**}
 * (wired in SecurityConfig) and is exercised separately by
 * {@link com.federated.fl_platform_api.InternalRunTokenVerifyIntegrationTest};
 * here we isolate the controller's own token-verification logic.
 *
 * <p>Tokens are minted with a real {@link ConnectionTokenService} keyed on the
 * same secret the test profile uses, so we never hand-roll JWTs.
 */
class InternalRunControllerTest {

    // Matches app.jwt.secret in application-test.properties (app.fl.token-secret defaults to it).
    private static final String TEST_SECRET =
            "bXlfc2VjcmV0X2tleV90aGF0X2lzX2F0X2xlYXN0XzMyX2J5dGVzX2xvbmc=";

    private MockMvc mvc;
    private ConnectionTokenService goodSigner;   // valid, long-lived tokens
    private ConnectionTokenService expiredSigner; // ttl in the past → already-expired tokens
    private ConnectionTokenService forgedSigner;  // different key → signature will not verify

    @BeforeEach
    void setUp() {
        goodSigner = signer(TEST_SECRET, 3600L);
        expiredSigner = signer(TEST_SECRET, -3600L);
        // A structurally-valid but WRONG HMAC key (40 bytes → base64) — a token it signs is,
        // relative to the real key, indistinguishable from a tampered/forged token.
        forgedSigner = signer(base64Of("x".repeat(40)), 3600L);

        mvc = MockMvcBuilders.standaloneSetup(new InternalRunController(goodSigner)).build();
    }

    private static ConnectionTokenService signer(String secret, long ttlSeconds) {
        ConnectionTokenService svc = new ConnectionTokenService();
        ReflectionTestUtils.setField(svc, "secretString", secret);
        ReflectionTestUtils.setField(svc, "ttlSeconds", ttlSeconds);
        svc.init();
        return svc;
    }

    private static String base64Of(String raw) {
        return java.util.Base64.getEncoder().encodeToString(raw.getBytes());
    }

    private static String body(String token) {
        return "{\"connectionToken\":\"" + token + "\"}";
    }

    private static ConnectionTokenService.Claims claims(long userId, UUID runId, UUID projectId,
                                                        int partitionId, String clientKind) {
        return new ConnectionTokenService.Claims(
                userId, runId, projectId, partitionId,
                "localhost:50000", null, clientKind);
    }

    @Test
    void validTokenForMatchingRun_returns200_withEnrolledIdentity() throws Exception {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String token = goodSigner.mint(claims(42L, runId, projectId, 3, "SHARD")).token();

        mvc.perform(post("/api/internal/runs/{projectId}/{runId}/verify-connection-token",
                        projectId, runId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body(token)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.userId").value(42))
                .andExpect(jsonPath("$.runId").value(runId.toString()))
                .andExpect(jsonPath("$.projectId").value(projectId.toString()))
                .andExpect(jsonPath("$.partitionId").value(3))
                .andExpect(jsonPath("$.clientKind").value("SHARD"));
    }

    @Test
    void tamperedOrForgedSignature_returns401() throws Exception {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String forged = forgedSigner.mint(claims(1L, runId, projectId, 0, "SHARD")).token();

        mvc.perform(post("/api/internal/runs/{projectId}/{runId}/verify-connection-token",
                        projectId, runId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body(forged)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void expiredToken_returns401() throws Exception {
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String expired = expiredSigner.mint(claims(1L, runId, projectId, 0, "SHARD")).token();

        mvc.perform(post("/api/internal/runs/{projectId}/{runId}/verify-connection-token",
                        projectId, runId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body(expired)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void validTokenForDifferentRun_returns403() throws Exception {
        UUID projectId = UUID.randomUUID();
        UUID tokenRunId = UUID.randomUUID();
        UUID pathRunId = UUID.randomUUID();   // != token's runId
        String token = goodSigner.mint(claims(7L, tokenRunId, projectId, 1, "SHARD")).token();

        mvc.perform(post("/api/internal/runs/{projectId}/{runId}/verify-connection-token",
                        projectId, pathRunId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body(token)))
                .andExpect(status().isForbidden());
    }

    @Test
    void validTokenForDifferentProject_returns403() throws Exception {
        UUID tokenProjectId = UUID.randomUUID();
        UUID pathProjectId = UUID.randomUUID();  // != token's projectId
        UUID runId = UUID.randomUUID();
        String token = goodSigner.mint(claims(7L, runId, tokenProjectId, 1, "SHARD")).token();

        mvc.perform(post("/api/internal/runs/{projectId}/{runId}/verify-connection-token",
                        pathProjectId, runId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body(token)))
                .andExpect(status().isForbidden());
    }
}
