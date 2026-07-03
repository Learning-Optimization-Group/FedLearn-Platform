package com.federated.fl_platform_api.run;

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

        // Emit for regeneration (build/ is gitignored). The committed fixture is copied from this.
        String json = "{\n"
                + "  \"secret_base64\": \"" + SECRET_B64 + "\",\n"
                + "  \"token\": \"" + minted.token() + "\",\n"
                + "  \"claims\": {\"sub\": \"42\", \"runId\": \"" + RUN_ID + "\", \"projectId\": \""
                + PROJECT_ID + "\", \"partitionId\": 3, \"clientKind\": \"SHARD\"}\n"
                + "}\n";
        Files.writeString(Path.of("build", "golden_connection_token.json"), json);
    }
}
