package com.federated.fl_platform_api.security;

import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * SE-7 — per-run internal tokens. When the backend spawns an FL server it mints a random, opaque
 * token here, bound to that run's {@code {projectId, runId}}, and hands ONLY that token to the
 * child (never the raw internal secret — so a compromised child cannot forge another run's token).
 *
 * {@code /api/internal/**} callbacks present the token; {@link InternalApiKeyFilter} resolves it to
 * its scope and rejects any call whose target project doesn't match. The blast radius of a leaked
 * run token is therefore one project, not the whole platform.
 *
 * State is in-memory and per-instance. A backend restart empties it — but BA-3 re-adopts FL-server
 * children that survived the crash, so those runs' token entries are {@link #rehydrate rehydrated}
 * from the SHA-256 hash persisted on the run (the plaintext lives only in the still-running child).
 * Runs that are reaped rather than re-adopted are never rehydrated, so their tokens stay dead.
 *
 * The map is keyed by the token's SHA-256 hash, never the plaintext: neither the heap nor the
 * persisted run row ever holds a usable token, only an irreversible digest.
 */
@Component
public class RunTokenRegistry {

    /** The project/run a token authorizes callbacks for. */
    public record Scope(UUID projectId, UUID runId) {}

    private static final SecureRandom RNG = new SecureRandom();
    private static final int TOKEN_BYTES = 32;  // 256 bits

    /** hash(token) -> scope. Keyed by digest so the plaintext token is never stored server-side. */
    private final ConcurrentHashMap<String, Scope> tokens = new ConcurrentHashMap<>();

    /** Mint a fresh opaque token bound to {@code (projectId, runId)}; only its hash is retained. */
    public String mint(UUID projectId, UUID runId) {
        byte[] raw = new byte[TOKEN_BYTES];
        RNG.nextBytes(raw);
        String token = Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
        tokens.put(hash(token), new Scope(projectId, runId));
        return token;
    }

    /** Resolve a presented token to its scope, or empty if unknown/blank. */
    public Optional<Scope> resolve(String token) {
        if (token == null || token.isEmpty()) {
            return Optional.empty();
        }
        return Optional.ofNullable(tokens.get(hash(token)));
    }

    /**
     * BA-3 re-adoption: restore a surviving run's token entry from the hash persisted on its run, so
     * the still-running child's callbacks keep authorizing after a backend restart. No-op on a
     * blank/null hash; never overwrites a live entry.
     */
    public void rehydrate(String tokenHash, Scope scope) {
        if (tokenHash == null || tokenHash.isEmpty() || scope == null) {
            return;
        }
        tokens.putIfAbsent(tokenHash, scope);
    }

    /** Drop every token for a project — called when that project's FL server stops. */
    public void evictForProject(UUID projectId) {
        tokens.values().removeIf(scope -> scope.projectId().equals(projectId));
    }

    /**
     * SHA-256 hex of a token — the value stored in this registry and persisted on the run for
     * re-adoption. A 256-bit random token needs no salt: the digest is irreversible and the input
     * space is not brute-forceable.
     */
    public String hash(String token) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(token.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);   // never on a standard JRE
        }
    }

    /** Test seam: number of live tokens. */
    int size() {
        return tokens.size();
    }
}
