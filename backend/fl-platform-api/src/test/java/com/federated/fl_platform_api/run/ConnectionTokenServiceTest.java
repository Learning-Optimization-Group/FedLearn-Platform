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
