package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Optional;

/**
 * Central authorization helpers. All project-scoped checks across the
 * controllers / services route through here so the rules in spec §3.3 have
 * exactly one implementation.
 */
@Service
public class AuthorizationService {

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;

    public User currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) throw new AccessDeniedException("No authenticated principal");
        return userRepository.findByUsername(auth.getName())
            .orElseThrow(() -> new UsernameNotFoundException(
                "Authenticated principal has no matching user row: " + auth.getName()));
    }

    public boolean isAdmin() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null) return false;
        for (GrantedAuthority a : auth.getAuthorities()) {
            if ("ROLE_ADMIN".equals(a.getAuthority())) return true;
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

    public void requireOwnerOrAdmin(Project project) {
        if (isAdmin() || isOwner(project)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }

    public void requireOwnerOrMemberOrAdmin(Project project) {
        if (isAdmin() || isOwner(project)) return;
        if (hasMembership(project, MembershipRole.MEMBER)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }

    /**
     * Pass if caller is an owner, member, or client of the project (or admin).
     * Used for read endpoints that any project participant may see.
     */
    public void requireParticipant(Project project) {
        if (isAdmin() || isOwner(project)) return;
        Optional<ProjectMembership> m = myMembership(project);
        if (m.isPresent() && (m.get().getRole() == MembershipRole.MEMBER
                           || m.get().getRole() == MembershipRole.CLIENT)) return;
        throw new AccessDeniedException("You do not have access to this project");
    }
}
