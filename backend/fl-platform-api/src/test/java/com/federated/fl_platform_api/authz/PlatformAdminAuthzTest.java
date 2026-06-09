package com.federated.fl_platform_api.authz;

import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.context.SecurityContextImpl;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Pins down the v1→v2 authorization drift fix: {@code isPlatformAdmin()} must
 * key off {@code ROLE_PLATFORM_ADMIN} and must NOT be satisfied by the legacy
 * {@code ROLE_ADMIN} authority.
 */
class PlatformAdminAuthzTest {

    private final AuthorizationService authz = new AuthorizationService();

    @AfterEach
    void clearContext() {
        SecurityContextHolder.clearContext();
    }

    private void authenticateWith(String authority) {
        GrantedAuthority ga = new SimpleGrantedAuthority(authority);
        // Install a fresh real context (not relying on the ambient one, which a
        // sibling test may have replaced with a Mockito mock whose setter no-ops).
        SecurityContextImpl ctx = new SecurityContextImpl();
        ctx.setAuthentication(new UsernamePasswordAuthenticationToken("u", "p", List.of(ga)));
        SecurityContextHolder.setContext(ctx);
    }

    @Test
    void platformAdminAuthority_isPlatformAdmin() {
        authenticateWith("ROLE_PLATFORM_ADMIN");
        assertThat(authz.isPlatformAdmin()).isTrue();
    }

    @Test
    void legacyAdminAuthority_isNotPlatformAdmin() {
        authenticateWith("ROLE_ADMIN");
        assertThat(authz.isPlatformAdmin()).isFalse();
    }

    @Test
    void userAuthority_isNotPlatformAdmin() {
        authenticateWith("ROLE_USER");
        assertThat(authz.isPlatformAdmin()).isFalse();
    }
}
