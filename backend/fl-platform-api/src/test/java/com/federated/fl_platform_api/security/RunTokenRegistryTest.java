package com.federated.fl_platform_api.security;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SE-7: the per-run internal token registry. Each spawned FL server gets a random, opaque token
 * bound to its {projectId, runId}; the backend resolves the token back to that scope so an
 * /api/internal/** call can be rejected when its target project doesn't match the token's project.
 */
class RunTokenRegistryTest {

    @Test
    void mintThenResolve_returnsBoundScope() {
        RunTokenRegistry reg = new RunTokenRegistry();
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();

        String token = reg.mint(projectId, runId);
        assertNotNull(token);
        assertFalse(token.isBlank());

        RunTokenRegistry.Scope scope = reg.resolve(token).orElseThrow();
        assertEquals(projectId, scope.projectId());
        assertEquals(runId, scope.runId());
    }

    @Test
    void resolve_unknownOrEmpty_isEmpty() {
        RunTokenRegistry reg = new RunTokenRegistry();
        assertTrue(reg.resolve("not-a-real-token").isEmpty());
        assertTrue(reg.resolve(null).isEmpty());
        assertTrue(reg.resolve("").isEmpty());
    }

    @Test
    void mintedTokens_areUniqueAndHighEntropy() {
        RunTokenRegistry reg = new RunTokenRegistry();
        String a = reg.mint(UUID.randomUUID(), UUID.randomUUID());
        String b = reg.mint(UUID.randomUUID(), UUID.randomUUID());
        assertNotEquals(a, b, "tokens must be random, not derived");
        assertTrue(a.length() >= 40, "≥256 bits base64url ≈ 43 chars");
    }

    @Test
    void evictForProject_dropsOnlyThatProjectsTokens() {
        RunTokenRegistry reg = new RunTokenRegistry();
        UUID projectA = UUID.randomUUID();
        UUID projectB = UUID.randomUUID();
        String tokenA = reg.mint(projectA, UUID.randomUUID());
        String tokenB = reg.mint(projectB, UUID.randomUUID());

        reg.evictForProject(projectA);

        assertTrue(reg.resolve(tokenA).isEmpty(), "project A's token is gone after its server stops");
        assertTrue(reg.resolve(tokenB).isPresent(), "project B is unaffected");
    }

    // BA-3: tokens are stored/persisted by SHA-256 hash (never plaintext) so a re-adopted server's
    // token can be rehydrated after a restart from the hash alone.

    @Test
    void hash_isDeterministicSha256Hex() {
        RunTokenRegistry reg = new RunTokenRegistry();
        String token = reg.mint(UUID.randomUUID(), UUID.randomUUID());

        String h1 = reg.hash(token);
        assertEquals(h1, reg.hash(token), "hash is deterministic");
        assertEquals(64, h1.length(), "SHA-256 hex is 64 chars");
        assertTrue(h1.matches("[0-9a-f]{64}"));
    }

    @Test
    void rehydrate_restoresATokenAfterARestart_fromItsHashAlone() {
        // A survivor's plaintext token lives only in the still-running child; the backend kept its hash.
        RunTokenRegistry original = new RunTokenRegistry();
        UUID projectId = UUID.randomUUID();
        UUID runId = UUID.randomUUID();
        String token = original.mint(projectId, runId);
        String persistedHash = original.hash(token);

        // A restart: a brand-new, empty registry that never saw the plaintext.
        RunTokenRegistry afterRestart = new RunTokenRegistry();
        assertTrue(afterRestart.resolve(token).isEmpty(), "token is unknown before rehydration");

        afterRestart.rehydrate(persistedHash, new RunTokenRegistry.Scope(projectId, runId));

        RunTokenRegistry.Scope scope = afterRestart.resolve(token).orElseThrow();
        assertEquals(projectId, scope.projectId());
        assertEquals(runId, scope.runId());
    }

    @Test
    void rehydrate_ignoresBlankHashOrNullScope() {
        RunTokenRegistry reg = new RunTokenRegistry();
        reg.rehydrate(null, new RunTokenRegistry.Scope(UUID.randomUUID(), UUID.randomUUID()));
        reg.rehydrate("", new RunTokenRegistry.Scope(UUID.randomUUID(), UUID.randomUUID()));
        reg.rehydrate("abc", null);
        assertEquals(0, reg.size(), "no-op rehydrations add nothing");
    }
}
