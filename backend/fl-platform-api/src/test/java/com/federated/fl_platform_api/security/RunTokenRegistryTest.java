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
}
