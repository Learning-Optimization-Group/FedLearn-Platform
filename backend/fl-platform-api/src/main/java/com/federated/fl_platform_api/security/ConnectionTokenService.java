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

    @Value("${app.fl.connection-token.ttl-seconds:120}")
    private long ttlSeconds;

    private SecretKey key;

    @PostConstruct
    public void init() {
        this.key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(secretString));
    }

    public record Claims(Long userId, UUID runId, UUID projectId, int partitionId,
                         String grpcEndpoint, String caFingerprint, String clientKind) {}

    public record Minted(String token, Instant expiresAt) {}

    public Minted mint(Claims c) {
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
