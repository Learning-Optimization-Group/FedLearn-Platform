package com.federated.fl_platform_api.authz;

import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.Test;
import org.springframework.security.access.AccessDeniedException;

import java.util.Set;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

/**
 * Plain unit coverage for the request-scoped {@link OrgScope} and
 * {@link AuthorizationService#requireOrgScope(UUID)}. No Spring context —
 * OrgScope is constructed directly via its setter, and the AuthorizationService
 * is given the OrgScope via its test setter.
 */
class OrgScopeTest {

    private static final UUID ORG_A = UUID.fromString("00000000-0000-0000-0000-00000000000a");
    private static final UUID ORG_B = UUID.fromString("00000000-0000-0000-0000-00000000000b");

    @Test
    void allows_trueForInScopeOrg() {
        OrgScope scope = new OrgScope();
        scope.set(Set.of(ORG_A), false);
        assertThat(scope.allows(ORG_A)).isTrue();
    }

    @Test
    void allows_falseForOutOfScopeOrg() {
        OrgScope scope = new OrgScope();
        scope.set(Set.of(ORG_A), false);
        assertThat(scope.allows(ORG_B)).isFalse();
    }

    @Test
    void allows_trueForAnyOrgWhenUnrestricted() {
        OrgScope scope = new OrgScope();
        scope.set(Set.of(), true);
        assertThat(scope.isUnrestricted()).isTrue();
        assertThat(scope.allows(ORG_A)).isTrue();
        assertThat(scope.allows(ORG_B)).isTrue();
    }

    @Test
    void requireOrgScope_throwsWhenOutOfScope() {
        AuthorizationService authz = new AuthorizationService();
        OrgScope scope = new OrgScope();
        scope.set(Set.of(ORG_A), false);
        authz.setOrgScope(scope);

        assertThatThrownBy(() -> authz.requireOrgScope(ORG_B))
                .isInstanceOf(AccessDeniedException.class);
    }

    @Test
    void requireOrgScope_passesWhenInScope() {
        AuthorizationService authz = new AuthorizationService();
        OrgScope scope = new OrgScope();
        scope.set(Set.of(ORG_A), false);
        authz.setOrgScope(scope);

        assertDoesNotThrow(() -> authz.requireOrgScope(ORG_A));
    }

    @Test
    void requireOrgScope_passesWhenUnrestricted() {
        AuthorizationService authz = new AuthorizationService();
        OrgScope scope = new OrgScope();
        scope.set(Set.of(), true);
        authz.setOrgScope(scope);

        assertDoesNotThrow(() -> authz.requireOrgScope(ORG_B));
    }
}
