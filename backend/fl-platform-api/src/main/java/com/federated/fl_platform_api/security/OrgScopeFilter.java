package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.lang.NonNull;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Populates the request-scoped {@link OrgScope} for the authenticated caller.
 * Runs after {@link JwtAuthenticationFilter} (registered in
 * {@code SecurityConfig}) so the {@code SecurityContext} is already set when we
 * resolve the user's org memberships.
 *
 * <p>Transitional-safety rule (critical): regular registered users are NOT
 * auto-added to {@code organization_memberships} — only the bootstrap admin is
 * seeded. A naive {@code org_id IN (visibleOrgIds)} filter would therefore make
 * {@code visibleOrgIds} empty for normal users and return ZERO projects on their
 * dashboard. So when a user has no memberships we fall back to the single
 * bootstrap org ({@link ProjectService#DEFAULT_ORG_ID}), preserving today's
 * single-org behaviour while making multi-org isolation real once memberships
 * exist. Platform admins are marked {@code unrestricted} and see every org.
 */
@Component
public class OrgScopeFilter extends OncePerRequestFilter {

    private static final String ROLE_PLATFORM_ADMIN = "ROLE_PLATFORM_ADMIN";

    private final UserRepository userRepository;
    private final OrganizationMembershipRepository orgMembershipRepository;
    private final OrgScope orgScope;

    public OrgScopeFilter(UserRepository userRepository,
                          OrganizationMembershipRepository orgMembershipRepository,
                          OrgScope orgScope) {
        this.userRepository = userRepository;
        this.orgMembershipRepository = orgMembershipRepository;
        this.orgScope = orgScope;
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain) throws ServletException, IOException {

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.isAuthenticated() && auth.getName() != null) {
            if (isPlatformAdmin(auth)) {
                // Platform admins bypass org isolation entirely.
                orgScope.set(Set.of(), true);
            } else {
                userRepository.findByUsername(auth.getName()).ifPresent(user -> {
                    orgScope.set(resolveVisibleOrgIds(user), false);
                });
            }
        }
        filterChain.doFilter(request, response);
    }

    private boolean isPlatformAdmin(Authentication auth) {
        for (GrantedAuthority a : auth.getAuthorities()) {
            if (ROLE_PLATFORM_ADMIN.equals(a.getAuthority())) return true;
        }
        return false;
    }

    /**
     * The org ids the user belongs to, or {@code {DEFAULT_ORG_ID}} when the user
     * has no memberships (the transitional fallback that keeps the
     * single-org dashboard working — see class javadoc).
     */
    private Set<UUID> resolveVisibleOrgIds(User user) {
        Set<UUID> orgIds = orgMembershipRepository.findByUserId(user.getId()).stream()
                .map(OrganizationMembership::getOrgId)
                .collect(Collectors.toCollection(HashSet::new));
        if (orgIds.isEmpty()) {
            orgIds.add(ProjectService.DEFAULT_ORG_ID);
        }
        return orgIds;
    }
}
