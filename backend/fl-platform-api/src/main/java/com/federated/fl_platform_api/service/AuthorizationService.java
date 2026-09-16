package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.OrgScope;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Central authorization helpers. All project-scoped checks across the
 * controllers / services route through here so the rules in spec §3.3 have
 * exactly one implementation.
 */
@Service
public class AuthorizationService {

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private OrganizationMembershipRepository orgMembershipRepository;
    @Autowired private OrgScope orgScope;

    /**
     * Test seam: lets unit tests inject a plain {@link OrgScope} without a Spring
     * request context. Not part of the public contract.
     */
    public void setOrgScope(OrgScope orgScope) {
        this.orgScope = orgScope;
    }

    public User currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) throw new AccessDeniedException("No authenticated principal");
        return userRepository.findByUsername(auth.getName())
            .orElseThrow(() -> new UsernameNotFoundException(
                "Authenticated principal has no matching user row: " + auth.getName()));
    }

    public boolean isPlatformAdmin() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) return false;
        for (GrantedAuthority a : auth.getAuthorities()) {
            if ("ROLE_PLATFORM_ADMIN".equals(a.getAuthority())) return true;
        }
        return false;
    }

    public boolean isOwner(Project project) {
        User self = currentUser();
        return project.getUser() != null && project.getUser().getId().equals(self.getId());
    }

    public boolean hasMembership(Project project, MembershipRole role) {
        User self = currentUser();
        return membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(
            project.getId(), self.getId(), role);
    }

    public Optional<ProjectMembership> myMembership(Project project) {
        User self = currentUser();
        return membershipRepository.findByIdProjectIdAndIdUserId(project.getId(), self.getId());
    }

    /**
     * Enforces org-level multi-tenant isolation: the caller's request-scoped
     * {@link OrgScope} must include the given org (or be unrestricted, i.e. a
     * platform admin). Throws {@link AccessDeniedException} otherwise. This is
     * an additional gate layered on top of the ownership/membership checks.
     */
    public void requireOrgScope(UUID orgId) {
        if (orgScope != null && orgScope.allows(orgId)) return;
        throw new AccessDeniedException("Project is outside your organization scope");
    }

    /**
     * Non-throwing form of {@link #requireOrgScope(UUID)}: returns whether the
     * caller's {@link OrgScope} includes the given org (or is unrestricted).
     * Read/list paths use this to translate an out-of-scope project into a 404
     * (existence not leaked) instead of a 403, mirroring
     * {@code ProjectService.getProject}.
     */
    public boolean isInOrgScope(UUID orgId) {
        return orgScope != null && orgScope.allows(orgId);
    }

    public void requireOwnerOrAdmin(Project project) {
        if (isPlatformAdmin() || isOwner(project)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }

    public void requireOwnerOrMemberOrAdmin(Project project) {
        if (isPlatformAdmin() || isOwner(project)) return;
        if (hasMembership(project, MembershipRole.MEMBER)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }

    /**
     * Non-throwing participant check: {@code true} if the caller is an owner, member, or client of
     * the project (or a platform admin). This is the read-side gate for endpoints that must not leak
     * a project's existence — a non-participant is turned into a 404, not a 403 (SE-16).
     */
    public boolean isParticipant(Project project) {
        if (isPlatformAdmin() || isOwner(project)) return true;
        Optional<ProjectMembership> m = myMembership(project);
        return m.isPresent() && (m.get().getRole() == MembershipRole.MEMBER
                              || m.get().getRole() == MembershipRole.CLIENT);
    }

    /**
     * Pass if caller is an owner, member, or client of the project (or admin).
     * Used for read endpoints that any project participant may see.
     */
    public void requireParticipant(Project project) {
        if (!isParticipant(project)) {
            throw new AccessDeniedException("You do not have access to this project");
        }
    }

    /**
     * STOMP subscription gate for a project-scoped topic. Enforces the exact same
     * rules as the REST read path ({@code ProjectService.getProject} /
     * {@code resolveInferenceTarget}): a project outside the caller's visible orgs
     * is denied first (cross-tenant isolation), then a non-participant is denied.
     *
     * <p>Runs on the STOMP inbound-channel thread, where there is no bound HTTP
     * request — so it resolves org scope directly from
     * {@code organization_memberships} instead of the request-scoped
     * {@link OrgScope} bean (which is only populated by {@code OrgScopeFilter}
     * during a servlet request). The participant check reuses
     * {@link #requireParticipant(Project)}. The caller must have set the
     * {@code SecurityContext} from the STOMP session principal before invoking.
     */
    public void requireSubscribable(Project project) {
        requireOrgVisible(project);
        requireParticipant(project);
    }

    /**
     * Off-request-thread equivalent of the {@code orgScope.allows(orgId)} gate.
     * Mirrors {@code OrgScopeFilter}: platform admins are unrestricted; a user
     * with no org memberships falls back to the transitional bootstrap org so the
     * single-org deployment keeps working, and multi-org isolation becomes real
     * once memberships exist.
     */
    private void requireOrgVisible(Project project) {
        if (isPlatformAdmin()) {
            return;
        }
        User self = currentUser();
        Set<UUID> visibleOrgIds = orgMembershipRepository.findByUserId(self.getId()).stream()
                .map(OrganizationMembership::getOrgId)
                .collect(Collectors.toCollection(HashSet::new));
        if (visibleOrgIds.isEmpty()) {
            visibleOrgIds.add(ProjectService.DEFAULT_ORG_ID);
        }
        if (!visibleOrgIds.contains(project.getOrgId())) {
            throw new AccessDeniedException("Project is outside your organization scope");
        }
    }

    /** True iff the caller holds the given {@code ROLE_*} authority. */
    private boolean hasAuthority(String authority) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) return false;
        for (GrantedAuthority a : auth.getAuthorities()) {
            if (authority.equals(a.getAuthority())) return true;
        }
        return false;
    }

    /** Throws unless the caller is a platform admin. */
    public void requirePlatformAdmin() {
        if (!isPlatformAdmin()) {
            throw new AccessDeniedException("Platform administrator role required");
        }
    }

    /**
     * Whether the caller may create/own projects: a PROJECT_OWNER (admin-granted
     * via the owner-promotion workflow) or a platform admin. Plain USERs cannot.
     */
    public boolean canCreateProjects() {
        return hasAuthority("ROLE_PROJECT_OWNER") || isPlatformAdmin();
    }

    /** Gate for project creation — see {@link #canCreateProjects()}. */
    public void requireCanCreateProject() {
        if (!canCreateProjects()) {
            throw new AccessDeniedException(
                "Only project owners can create projects. Request owner access from a platform admin.");
        }
    }
}
