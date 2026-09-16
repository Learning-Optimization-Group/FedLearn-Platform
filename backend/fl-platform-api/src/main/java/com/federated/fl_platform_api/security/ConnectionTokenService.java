package com.federated.fl_platform_api.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.time.Instant;
import java.util.Date;
import java.util.UUID;

/**
 * Mints short-lived, HMAC-signed connection tokens a client presents to the FL
 * server when joining a run. Phase 1 only ISSUES these (and can verify them);
 * the Python FL server begins validating them in Phase 2 (alongside gRPC TLS).
 * Reuses {@code app.jwt.secret} so no new required secret is introduced.
 */
@Service
public class ConnectionTokenService {

    public static final String AUDIENCE = "fedlearn-fl-server";

    // SE-7: signed with the dedicated FL secret (app.fl.token-secret), which defaults to app.jwt.secret
    // but should be set distinctly in production so the FL trust domain is isolated from web auth.
    @Value("${app.fl.token-secret}")
    private String secretString;

    // SE-14: the base/floor lifetime. A token minted for a run is derived from the run length (below)
    // but never shorter than this.
    @Value("${app.fl.connection-token.ttl-seconds:120}")
    private long ttlSeconds;

    // SE-14: run-scoped TTL inputs. The token must outlive the whole run (numRounds rounds run
    // sequentially, each bounded by the FL server's round timeout), or a long run expires the token
    // mid-training and — once require-client-auth is on — the client is rejected on its next RPC.
    // perRoundSeconds should track the FL server's FEDLEARN_ROUND_TIMEOUT_S; startupBufferSeconds
    // covers enrollment + the initial O(d) global-model download before round 1.
    @Value("${app.fl.connection-token.per-round-seconds:600}")
    private long perRoundSeconds;

    @Value("${app.fl.connection-token.startup-buffer-seconds:900}")
    private long startupBufferSeconds;

    // SE-14: hard ceiling so a huge numRounds can't mint an absurdly long-lived token.
    @Value("${app.fl.connection-token.max-ttl-seconds:86400}")
    private long maxTtlSeconds;

    private SecretKey key;

    @PostConstruct
    public void init() {
        this.key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(secretString));
    }

    public record Claims(Long userId, UUID runId, UUID projectId, int partitionId,
                         String grpcEndpoint, String caFingerprint, String clientKind) {}

    public record Minted(String token, Instant expiresAt) {}

    /**
     * SE-14: the connection-token lifetime (seconds) a run of {@code numRounds} needs — derived as
     * {@code numRounds * perRoundSeconds + startupBufferSeconds}, floored at the base {@code ttlSeconds}
     * (short runs still get a sane minimum) and capped at {@code maxTtlSeconds} (a very long run can't
     * mint an unbounded token). Negative/zero rounds clamp to the floor. Pure + testable.
     */
    public long ttlForRun(int numRounds) {
        long rounds = Math.max(0L, (long) numRounds);
        long derived = rounds * perRoundSeconds + startupBufferSeconds;
        return Math.min(maxTtlSeconds, Math.max(ttlSeconds, derived));
    }

    /** Mint with the default base TTL (callers with no run context, e.g. the golden fixture). */
    public Minted mint(Claims c) {
        return mint(c, ttlSeconds);
    }

    /** Mint with an explicit lifetime — use {@link #ttlForRun(int)} to size it to the run. */
    public Minted mint(Claims c, long ttlSeconds) {
        Instant now = Instant.now();
        Instant exp = now.plusSeconds(ttlSeconds);
        String token = Jwts.builder()
                .audience().add(AUDIENCE).and()
                .subject(String.valueOf(c.userId()))
                .claim("runId", c.runId().toString())
                .claim("projectId", c.projectId().toString())
                .claim("partitionId", c.partitionId())
                .claim("grpcEndpoint", c.grpcEndpoint())
                .claim("caFingerprint", c.caFingerprint())
                .claim("clientKind", c.clientKind())
                .issuedAt(Date.from(now))
                .expiration(Date.from(exp))
                .signWith(key)
                .compact();
        return new Minted(token, exp);
    }

    public io.jsonwebtoken.Claims verify(String token) {
        return Jwts.parser()
                .requireAudience(AUDIENCE)
                .verifyWith(key)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }
}
