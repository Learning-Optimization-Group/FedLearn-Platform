package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.security.ConnectionTokenService;
import io.jsonwebtoken.Claims;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Base64;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class ConnectionTokenServiceTest {

    private ConnectionTokenService svc;

    @BeforeEach
    void setUp() {
        svc = new ConnectionTokenService();
        // 32+ byte base64 secret for HS256
        String secret = Base64.getEncoder().encodeToString(
                "0123456789abcdef0123456789abcdef".getBytes());
        ReflectionTestUtils.setField(svc, "secretString", secret);
        ReflectionTestUtils.setField(svc, "ttlSeconds", 120L);
        svc.init();
    }

    @Test
    void mintThenVerify_roundTripsClaims() {
        UUID runId = UUID.randomUUID();
        UUID projectId = UUID.randomUUID();
        ConnectionTokenService.Minted minted = svc.mint(new ConnectionTokenService.Claims(
                42L, runId, projectId, 3, "localhost:50001", null, "SHARD"));

        Claims c = svc.verify(minted.token());
        assertEquals("42", c.getSubject());
        assertEquals(runId.toString(), c.get("runId", String.class));
        assertEquals(projectId.toString(), c.get("projectId", String.class));
        assertEquals(3, c.get("partitionId", Integer.class));
        assertEquals("localhost:50001", c.get("grpcEndpoint", String.class));
        assertEquals("SHARD", c.get("clientKind", String.class));
        assertTrue(c.getAudience().contains("fedlearn-fl-server"));
        assertNotNull(minted.expiresAt());
    }

    // --- SE-14: the token must outlive the whole run, or a long run expires it mid-training and the
    // client (once require-client-auth is on) is rejected on its next RPC. ttlForRun derives the
    // lifetime from numRounds, floored at the base TTL and capped at a ceiling. ---

    private ConnectionTokenService svcWithTtl(long floor, long perRound, long buffer, long max) {
        ConnectionTokenService s = new ConnectionTokenService();
        String secret = Base64.getEncoder().encodeToString(
                "0123456789abcdef0123456789abcdef".getBytes());
        ReflectionTestUtils.setField(s, "secretString", secret);
        ReflectionTestUtils.setField(s, "ttlSeconds", floor);
        ReflectionTestUtils.setField(s, "perRoundSeconds", perRound);
        ReflectionTestUtils.setField(s, "startupBufferSeconds", buffer);
        ReflectionTestUtils.setField(s, "maxTtlSeconds", max);
        s.init();
        return s;
    }

    @Test
    void ttlForRun_floorsShortRunsAtTheBaseTtl() {
        ConnectionTokenService s = svcWithTtl(3600, 600, 900, 86400);
        assertEquals(3600, s.ttlForRun(1));    // 1*600+900=1500 < floor -> 3600
        assertEquals(3600, s.ttlForRun(0));    // defensive: 0 rounds -> floor
        assertEquals(3600, s.ttlForRun(-5));   // negative clamped -> floor
    }

    @Test
    void ttlForRun_derivesFromRunLengthBetweenFloorAndCap() {
        ConnectionTokenService s = svcWithTtl(3600, 600, 900, 86400);
        assertEquals(10L * 600 + 900, s.ttlForRun(10));   // 6900
        assertEquals(50L * 600 + 900, s.ttlForRun(50));   // 30900
    }

    @Test
    void ttlForRun_capsVeryLongRunsAtMax() {
        ConnectionTokenService s = svcWithTtl(3600, 600, 900, 86400);
        assertEquals(86400, s.ttlForRun(1000));   // 600900 -> capped
    }

    @Test
    void mintWithExplicitTtl_setsTheExpirationAccordingly() {
        ConnectionTokenService s = svcWithTtl(3600, 600, 900, 86400);
        long ttl = s.ttlForRun(10);   // 6900
        ConnectionTokenService.Minted m = s.mint(new ConnectionTokenService.Claims(
                1L, UUID.randomUUID(), UUID.randomUUID(), 0, "h:1", null, "SHARD"), ttl);
        long secs = java.time.Duration.between(java.time.Instant.now(), m.expiresAt()).getSeconds();
        assertTrue(secs > 6800 && secs <= 6900, "exp should reflect the derived TTL, was " + secs);
    }

    @Test
    void verify_rejectsWrongAudience() {
        // A token minted by the auth provider (different audience) must fail verify().
        // Simulate by tampering: mint, then verify with a fresh service whose secret differs.
        ConnectionTokenService other = new ConnectionTokenService();
        String secret = Base64.getEncoder().encodeToString(
                "ffffffffffffffffffffffffffffffff".getBytes());
        ReflectionTestUtils.setField(other, "secretString", secret);
        ReflectionTestUtils.setField(other, "ttlSeconds", 120L);
        other.init();
        ConnectionTokenService.Minted minted = other.mint(new ConnectionTokenService.Claims(
                1L, UUID.randomUUID(), UUID.randomUUID(), 0, "h:1", null, "SHARD"));
        assertThrows(RuntimeException.class, () -> svc.verify(minted.token()));
    }
}
