package com.federated.fl_platform_api.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import io.jsonwebtoken.Claims;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SE-1 Java-&gt;Python cross-language pin. Mints a real JJWT connection token with a FIXED secret +
 * fixed claims + a ~100-year TTL, asserts it round-trips on the Java side, and emits it to build/ for
 * (re)generating framework/tests/fixtures/golden_connection_token.json — the committed static artifact
 * the Python PyJWT verifier checks (test_token_verify_golden.py). If the JJWT token format ever drifts
 * from what PyJWT accepts, that Python test fails. To regenerate: run this test, then copy
 * build/golden_connection_token.json over the committed fixture.
 */
class GoldenConnectionTokenFixtureTest {

    // 32 bytes -> HS256. MUST stay in sync with the committed fixture's secret_base64.
    private static final String SECRET_B64 =
            Base64.getEncoder().encodeToString("fedlearn-golden-token-secret-32b".getBytes(StandardCharsets.UTF_8));
    private static final String RUN_ID = "11111111-1111-1111-1111-111111111111";
    private static final String PROJECT_ID = "22222222-2222-2222-2222-222222222222";

    @Test
    void emitAndRoundTripGoldenConnectionToken() throws Exception {
        ConnectionTokenService svc = new ConnectionTokenService();
        ReflectionTestUtils.setField(svc, "secretString", SECRET_B64);
        ReflectionTestUtils.setField(svc, "ttlSeconds", 3_153_600_000L); // ~100 years — committed token won't expire
        svc.init();

        ConnectionTokenService.Minted minted = svc.mint(new ConnectionTokenService.Claims(
                42L, UUID.fromString(RUN_ID), UUID.fromString(PROJECT_ID),
                3, "127.0.0.1:50051", null, "SHARD"));

        // Round-trips on the Java side.
        Claims c = svc.verify(minted.token());
        assertEquals("42", c.getSubject());
        assertEquals(RUN_ID, c.get("runId", String.class));
        assertEquals(3, c.get("partitionId", Integer.class));

        // SE-13: a real cross-language contract guard, not just a self-round-trip. Load the COMMITTED
        // golden (the exact bytes the Python PyJWT verifier checks in test_token_verify_golden.py) and
        // assert the CURRENT ConnectionTokenService still (a) VERIFIES it and (b) mints the SAME claim
        // shape. (a) catches verify-side drift (a stricter/renamed-claim verify path); (b) catches
        // minting-side format drift (an added/renamed/dropped claim a stale fixture would otherwise
        // hide). Either failure means Java has drifted from what Python accepts -> regenerate the
        // fixture per the javadoc. This runs on the backend gradle job, which now triggers on
        // backend/** (SE-13 done-when #2), so a ConnectionTokenService edit that breaks Java<->Python
        // compat fails CI here rather than silently at an FL client's first connect.
        Path fixture = Path.of("..", "..", "framework", "tests", "fixtures", "golden_connection_token.json");
        JsonNode golden = new ObjectMapper().readTree(Files.readString(fixture));
        Claims committed = svc.verify(golden.get("token").asText());   // (a) still verifies
        JsonNode goldenClaims = golden.get("claims");
        assertEquals(goldenClaims.get("sub").asText(), committed.getSubject());
        assertEquals(goldenClaims.get("runId").asText(), committed.get("runId", String.class));
        assertEquals(goldenClaims.get("projectId").asText(), committed.get("projectId", String.class));
        assertEquals(goldenClaims.get("partitionId").asInt(), committed.get("partitionId", Integer.class));
        assertEquals(goldenClaims.get("clientKind").asText(), committed.get("clientKind", String.class));
        assertEquals(goldenClaims.get("grpcEndpoint").asText(), committed.get("grpcEndpoint", String.class));
        // grpcEndpoint value-format guard on the FRESH MINT -- NOT the frozen committed token, which by
        // construction can't exhibit minting drift, so an instanceof check on it would be
        // non-falsifiable. If the current ConnectionTokenService ever emitted grpcEndpoint as a
        // non-String (e.g. a nested object), the Python golden's value-check would become meaningless;
        // assert the LIVE mint keeps it a "host:port" String.
        Object freshGrpc = c.get("grpcEndpoint");
        assertTrue(freshGrpc instanceof String,
                "fresh mint grpcEndpoint must be a String, was: "
                        + (freshGrpc == null ? "null" : freshGrpc.getClass()));
        assertEquals("127.0.0.1:50051", freshGrpc);
        // (b) a fresh mint carries the SAME claim names as the committed golden (iat/exp differ in
        // value but are present in both; the JJWT array-form `aud` interop is exercised Python-side).
        assertEquals(committed.keySet(), c.keySet(),
                "fresh mint claim-set drifted from the committed golden — regenerate the fixture "
                        + "(run this test, copy build/golden_connection_token.json over the committed one)");

        // Emit for regeneration (build/ is gitignored). The committed fixture is copied from this.
        String json = "{\n"
                + "  \"secret_base64\": \"" + SECRET_B64 + "\",\n"
                + "  \"token\": \"" + minted.token() + "\",\n"
                + "  \"claims\": {\"sub\": \"42\", \"runId\": \"" + RUN_ID + "\", \"projectId\": \""
                + PROJECT_ID + "\", \"partitionId\": 3, \"grpcEndpoint\": \"127.0.0.1:50051\", "
                + "\"clientKind\": \"SHARD\"}\n"
                + "}\n";
        Files.writeString(Path.of("build", "golden_connection_token.json"), json);
    }
}
