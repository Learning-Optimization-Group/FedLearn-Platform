package com.federated.fl_platform_api.authz;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AuthorizationServiceTest {

    @Mock UserRepository userRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock SecurityContext securityContext;
    @Mock Authentication authentication;

    @InjectMocks AuthorizationService authz;

    private User owner;
    private User other;
    private User admin;
    private Project project;

    @BeforeEach
    void setUp() {
        owner = new User("alice", "alice@example.com", "x"); owner.setId(1L);
        other = new User("bob",   "bob@example.com",   "x"); other.setId(2L);
        admin = new User("admin", "admin@example.com", "x"); admin.setId(3L); admin.setPlatformRole("ADMIN");

        project = new Project();
        project.setId(UUID.randomUUID());
        project.setUser(owner);

        SecurityContextHolder.setContext(securityContext);
        when(securityContext.getAuthentication()).thenReturn(authentication);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private void loggedInAs(User u) {
        when(authentication.getName()).thenReturn(u.getUsername());
        when(userRepository.findByUsername(u.getUsername())).thenReturn(Optional.of(u));
        if ("ADMIN".equals(u.getPlatformRole())) {
            when(authentication.getAuthorities())
                .thenReturn((java.util.Collection) List.of(new SimpleGrantedAuthority("ROLE_ADMIN")));
        } else {
            when(authentication.getAuthorities())
                .thenReturn((java.util.Collection) List.of(new SimpleGrantedAuthority("ROLE_USER")));
        }
    }

    @Test
    void requireOwnerOrAdmin_passesForOwner() {
        loggedInAs(owner);
        assertDoesNotThrow(() -> authz.requireOwnerOrAdmin(project));
    }

    @Test
    void requireOwnerOrAdmin_passesForAdmin() {
        loggedInAs(admin);
        assertDoesNotThrow(() -> authz.requireOwnerOrAdmin(project));
    }

    @Test
    void requireOwnerOrAdmin_deniesForOthers() {
        loggedInAs(other);
        assertThrows(AccessDeniedException.class, () -> authz.requireOwnerOrAdmin(project));
    }

    @Test
    void requireOwnerOrMemberOrAdmin_passesForMember() {
        loggedInAs(other);
        when(membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(
            project.getId(), other.getId(), MembershipRole.MEMBER)).thenReturn(true);
        assertDoesNotThrow(() -> authz.requireOwnerOrMemberOrAdmin(project));
    }

    @Test
    void requireOwnerOrMemberOrAdmin_deniesPlainClient() {
        loggedInAs(other);
        when(membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(
            project.getId(), other.getId(), MembershipRole.MEMBER)).thenReturn(false);
        assertThrows(AccessDeniedException.class, () -> authz.requireOwnerOrMemberOrAdmin(project));
    }

    @Test
    void requireParticipant_passesForClient() {
        loggedInAs(other);
        when(membershipRepository.findByIdProjectIdAndIdUserId(project.getId(), other.getId()))
            .thenReturn(Optional.of(makeMembership(MembershipRole.CLIENT)));
        assertDoesNotThrow(() -> authz.requireParticipant(project));
    }

    private ProjectMembership makeMembership(MembershipRole role) {
        ProjectMembership m = new ProjectMembership();
        m.setRole(role);
        return m;
    }
}
