package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.User;
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

import java.util.Optional;
import java.util.UUID;

/**
 * Central authorization helpers. All project-scoped checks across the
 * controllers / services route through here so the rules in spec §3.3 have
 * exactly one implementation.
 */
@Service
public class AuthorizationService {

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
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
     * Pass if caller is an owner, member, or client of the project (or admin).
     * Used for read endpoints that any project participant may see.
     */
    public void requireParticipant(Project project) {
        if (isPlatformAdmin() || isOwner(project)) return;
        Optional<ProjectMembership> m = myMembership(project);
        if (m.isPresent() && (m.get().getRole() == MembershipRole.MEMBER
                           || m.get().getRole() == MembershipRole.CLIENT)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }
}
