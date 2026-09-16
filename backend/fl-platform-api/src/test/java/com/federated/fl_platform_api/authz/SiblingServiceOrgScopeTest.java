package com.federated.fl_platform_api.authz;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.security.OrgScopeFilter;
import com.federated.fl_platform_api.service.AccessRequestService;
import com.federated.fl_platform_api.service.ClientApiService;
import com.federated.fl_platform_api.service.MembershipService;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.context.SecurityContextImpl;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Org-scope completeness coverage for the sibling tenant-owned services
 * (MembershipService / AccessRequestService / ClientApiService) — the paths
 * that {@code requireOrgScope} was NOT yet enforced on (P0-a follow-up).
 *
 * <p>Reuses the same seeding/scope-binding pattern as {@link OrgIsolationTest}:
 * persist a real Project in ORG_B, authenticate a user scoped only to ORG_A,
 * run the real {@link OrgScopeFilter}, then assert each cross-org mutation is
 * blocked by the org gate (403 {@link AccessDeniedException}).
 */
@SpringBootTest
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
class SiblingServiceOrgScopeTest {

    private static final UUID ORG_A = UUID.fromString("00000000-0000-0000-0000-0000000000a1");
    private static final UUID ORG_B = UUID.fromString("00000000-0000-0000-0000-0000000000b1");

    @Autowired MembershipService membershipService;
    @Autowired AccessRequestService accessRequestService;
    @Autowired ClientApiService clientApiService;
    @Autowired OrgScopeFilter orgScopeFilter;
    @Autowired OrgScope orgScope;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired OrganizationMembershipRepository orgMembershipRepository;

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
        RequestContextHolder.resetRequestAttributes();
    }

    private void bindRequestScope() {
        RequestContextHolder.setRequestAttributes(
                new ServletRequestAttributes(new MockHttpServletRequest()));
    }

    private void authenticate(User u, String authority) {
        SecurityContextImpl ctx = new SecurityContextImpl();
        ctx.setAuthentication(new UsernamePasswordAuthenticationToken(
                u.getUsername(), "x", List.of(new SimpleGrantedAuthority(authority))));
        SecurityContextHolder.setContext(ctx);
    }

    private void runFilter() throws Exception {
        FilterChain noop = mock(FilterChain.class);
        orgScopeFilter.doFilter(new MockHttpServletRequest(),
                new MockHttpServletResponse(), noop);
    }

    private Project newProject(String name, User owner, UUID orgId, ProjectVisibility vis) {
        Project p = new Project();
        p.setName(name + "-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(orgId);
        p.setVisibility(vis);
        return projectRepository.saveAndFlush(p);
    }

    @Test
    void userScopedToOrgA_cannotMutateOrConnectToProjectInOrgB() throws Exception {
        User alice = userRepository.save(new User("alice_sib", "alice_sib@example.com", "h"));
        User mallory = userRepository.save(new User("mallory_sib", "mallory_sib@example.com", "h"));
        User victim = userRepository.save(new User("victim_sib", "victim_sib@example.com", "h"));
        orgMembershipRepository.save(new OrganizationMembership(ORG_A, alice.getId(), OrgRole.OWNER));

        Project bProj = newProject("b-proj-sib", mallory, ORG_B, ProjectVisibility.PRIVATE);

        bindRequestScope();
        authenticate(alice, "ROLE_USER");
        runFilter();

        assertFalse(orgScope.isUnrestricted());
        assertTrue(orgScope.allows(ORG_A));
        assertFalse(orgScope.allows(ORG_B), "alice must not be scoped to ORG_B");

        // MembershipService.add — mutation, hard 403.
        assertThrows(AccessDeniedException.class,
                () -> membershipService.add(bProj.getId(), victim.getUsername(), MembershipRole.CLIENT),
                "adding a membership in an out-of-org project must be blocked");

        // AccessRequestService.submit — mutation, hard 403.
        assertThrows(AccessDeniedException.class,
                () -> accessRequestService.submit(bProj.getId(), "let me in"),
                "submitting an access request to an out-of-org project must be blocked");

        // ClientApiService.getConnection — mutation, hard 403 (gate fires before
        // the running-status / participant checks).
        assertThrows(AccessDeniedException.class,
                () -> clientApiService.getConnection(bProj.getId()),
                "getting a client connection for an out-of-org project must be blocked");

        // ClientApiService.list — the out-of-org project must NOT appear.
        List<UUID> visibleClientProjects = clientApiService.listForCurrentUser()
                .stream().map(d -> d.getProjectId()).toList();
        assertFalse(visibleClientProjects.contains(bProj.getId()),
                "client project list must not leak the out-of-org project");
    }

    @Test
    void membershipLessUser_isNotBlockedByOrgGate_onOwnDefaultOrgProject() throws Exception {
        // The default-org fallback must keep working: a membership-less user
        // owning a project in the DEFAULT org is NOT rejected by the org gate.
        User carol = userRepository.save(new User("carol_sib", "carol_sib@example.com", "h"));
        // No OrganizationMembership rows for carol -> scope falls back to {DEFAULT_ORG_ID}.
        Project defProj = newProject("default-proj-sib", carol,
                ProjectService.DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        bindRequestScope();
        authenticate(carol, "ROLE_USER");
        runFilter();

        assertFalse(orgScope.isUnrestricted());
        assertTrue(orgScope.allows(ProjectService.DEFAULT_ORG_ID));

        // listForProject on her own default-org project does not 404 on the org
        // gate (carol is owner -> passes the participant check too).
        assertDoesNotThrow(() -> accessRequestService.listForProject(defProj.getId(), null),
                "membership-less owner must still reach their own default-org project");
        assertDoesNotThrow(() -> membershipService.list(defProj.getId(), null),
                "membership-less owner must still list memberships on their own project");
    }
}
