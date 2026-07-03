package com.federated.fl_platform_api.security;

import org.springframework.stereotype.Component;

import java.time.Clock;
import java.time.Instant;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory JWT revocation denylist keyed by {@code jti} (SE-8). Logout revokes the current token's
 * jti so a leaked-but-not-yet-expired token stops working immediately, instead of remaining valid for
 * the full TTL. A revoked jti only needs remembering until the token would expire anyway, so entries
 * are pruned by their expiry — the set stays bounded.
 *
 * <p>Per-instance (not shared across replicas); a distributed store is a later hardening, same as the
 * login rate limiter. The {@link Clock} is an injectable test seam.</p>
 */
@Component
public class TokenRevocationService {

    private final ConcurrentHashMap<String, Instant> revoked = new ConcurrentHashMap<>();
    private final Clock clock;

    public TokenRevocationService() {
        this(Clock.systemUTC());
    }

    TokenRevocationService(Clock clock) {
        this.clock = clock;
    }

    /** Revoke {@code jti} until {@code expiresAt} (after which the token is dead anyway). */
    public void revoke(String jti, Instant expiresAt) {
        if (jti == null) {
            return;
        }
        prune();
        revoked.put(jti, expiresAt != null ? expiresAt : Instant.MAX);
    }

    /** True if {@code jti} is currently revoked. */
    public boolean isRevoked(String jti) {
        if (jti == null) {
            return false;
        }
        prune();
        return revoked.containsKey(jti);
    }

    private void prune() {
        Instant now = clock.instant();
        revoked.entrySet().removeIf(e -> e.getValue().isBefore(now));
    }
}
