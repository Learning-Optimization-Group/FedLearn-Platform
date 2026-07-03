package com.federated.fl_platform_api.security;

import org.springframework.stereotype.Component;

import java.security.SecureRandom;
import java.util.Base64;
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
 * State is in-memory and per-instance, consistent with {@code FlowerServerManager.runningServers}:
 * a backend restart orphans the spawned processes anyway, so losing the token map with them is fine.
 */
@Component
public class RunTokenRegistry {

    /** The project/run a token authorizes callbacks for. */
    public record Scope(UUID projectId, UUID runId) {}

    private static final SecureRandom RNG = new SecureRandom();
    private static final int TOKEN_BYTES = 32;  // 256 bits

    private final ConcurrentHashMap<String, Scope> tokens = new ConcurrentHashMap<>();

    /** Mint a fresh opaque token bound to {@code (projectId, runId)}. */
    public String mint(UUID projectId, UUID runId) {
        byte[] raw = new byte[TOKEN_BYTES];
        RNG.nextBytes(raw);
        String token = Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
        tokens.put(token, new Scope(projectId, runId));
        return token;
    }

    /** Resolve a presented token to its scope, or empty if unknown/blank. */
    public Optional<Scope> resolve(String token) {
        if (token == null || token.isEmpty()) {
            return Optional.empty();
        }
        return Optional.ofNullable(tokens.get(token));
    }

    /** Drop every token for a project — called when that project's FL server stops. */
    public void evictForProject(UUID projectId) {
        tokens.values().removeIf(scope -> scope.projectId().equals(projectId));
    }

    /** Test seam: number of live tokens. */
    int size() {
        return tokens.size();
    }
}
