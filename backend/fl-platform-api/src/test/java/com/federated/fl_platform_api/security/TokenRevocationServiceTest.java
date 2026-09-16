package com.federated.fl_platform_api.security;

import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** SE-8: the in-memory JWT revocation denylist. Pure — no Spring context. */
class TokenRevocationServiceTest {

    @Test
    void aRevokedJtiIsReportedRevoked() {
        TokenRevocationService svc = new TokenRevocationService();
        assertFalse(svc.isRevoked("jti-1"));
        svc.revoke("jti-1", Instant.now().plusSeconds(3600));
        assertTrue(svc.isRevoked("jti-1"));
    }

    @Test
    void anUnknownJtiIsNotRevoked() {
        assertFalse(new TokenRevocationService().isRevoked("never-seen"));
    }

    @Test
    void nullJtiIsNeverRevoked() {
        TokenRevocationService svc = new TokenRevocationService();
        svc.revoke(null, Instant.now().plusSeconds(60));   // no-op, must not blow up
        assertFalse(svc.isRevoked(null));
    }

    @Test
    void anAlreadyExpiredRevocationIsPrunedNotRetained() {
        // The token is dead by its own expiry; the denylist entry must not linger (bounded memory).
        TokenRevocationService svc = new TokenRevocationService();
        svc.revoke("jti-old", Instant.now().minusSeconds(10));
        assertFalse(svc.isRevoked("jti-old"));
    }
}
