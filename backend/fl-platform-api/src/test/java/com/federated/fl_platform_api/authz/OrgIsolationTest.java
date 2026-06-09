package com.federated.fl_platform_api.authz;

import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.security.OrgScopeFilter;
import com.federated.fl_platform_api.service.ProjectService;
import jakarta.servlet.FilterChain;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * End-to-end org-isolation coverage against a real {@link OrgScopeFilter},
 * request-scoped {@link OrgScope}, and {@link ProjectService}. Each scenario
 * binds a real request scope, authenticates a principal, runs the filter to
 * populate the OrgScope, then exercises the service in that same request.
 *
 * <p>Critically, the membership-less scenario exercises the transitional
 * empty-&gt;DEFAULT_ORG_ID fallback so the dashboard is NOT empty for normal
 * users (the design trap this work guards against).
 */
@SpringBootTest
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
class OrgIsolationTest {

    private static final UUID ORG_A = UUID.fromString("00000000-0000-0000-0000-0000000000aa");
    private static final UUID ORG_B = UUID.fromString("00000000-0000-0000-0000-0000000000bb");

    @Autowired ProjectService projectService;
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
        MockHttpServletRequest request = new MockHttpServletRequest();
        RequestContextHolder.setRequestAttributes(new ServletRequestAttributes(request));
    }

    private void authenticate(User u, String authority) {
        // Install a FRESH real context — do not mutate the ambient one. A sibling
        // unit test (AuthorizationServiceTest) installs a Mockito SecurityContext
        // mock whose setAuthentication(..) is a no-op; reusing it would silently
        // drop our token and leak its stubbed [ROLE_USER] authorities into here.
        SecurityContextImpl ctx = new SecurityContextImpl();
        ctx.setAuthentication(new UsernamePasswordAuthenticationToken(
                u.getUsername(), "x", List.of(new SimpleGrantedAuthority(authority))));
        SecurityContextHolder.setContext(ctx);
    }

    /** Runs the real OrgScopeFilter so the request-scoped OrgScope is populated. */
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
    void userScopedToOrgA_cannotSeeProjectInOrgB() throws Exception {
        User alice = userRepository.save(new User("alice_iso", "alice_iso@example.com", "h"));
        User mallory = userRepository.save(new User("mallory_iso", "mallory_iso@example.com", "h"));
        orgMembershipRepository.save(new OrganizationMembership(ORG_A, alice.getId(), OrgRole.OWNER));

        Project aProj = newProject("a-proj", alice, ORG_A, ProjectVisibility.PRIVATE);
        Project bProj = newProject("b-proj", mallory, ORG_B, ProjectVisibility.PRIVATE);

        bindRequestScope();
        authenticate(alice, "ROLE_USER");
        runFilter();

        // Scope resolved to exactly {ORG_A}, not unrestricted.
        assertFalse(orgScope.isUnrestricted());
        assertTrue(orgScope.allows(ORG_A));
        assertFalse(orgScope.allows(ORG_B));

        // Dashboard lists alice's ORG_A project only; the ORG_B project is excluded.
        List<ProjectResponseDto> dash = projectService.getProjectsForCurrentUser();
        List<UUID> ids = dash.stream().map(ProjectResponseDto::getId).toList();
        assertTrue(ids.contains(aProj.getId()), "alice should see her ORG_A project");
        assertFalse(ids.contains(bProj.getId()), "alice must not see the ORG_B project");

        // Direct get of the ORG_B project is a 404 (cross-tenant existence not leaked).
        assertThrows(ResourceNotFoundException.class,
                () -> projectService.getProject(bProj.getId()));
    }

    @Test
    void platformAdmin_seesProjectsInEveryOrg() throws Exception {
        User admin = new User("admin_iso", "admin_iso@example.com", "h");
        admin.setPlatformRole(PlatformRole.PLATFORM_ADMIN);
        admin = userRepository.save(admin);
        User mallory = userRepository.save(new User("mallory_iso2", "mallory_iso2@example.com", "h"));

        Project aProj = newProject("a-proj-admin", mallory, ORG_A, ProjectVisibility.PRIVATE);
        Project bProj = newProject("b-proj-admin", mallory, ORG_B, ProjectVisibility.PRIVATE);

        bindRequestScope();
        authenticate(admin, "ROLE_PLATFORM_ADMIN");
        runFilter();

        assertTrue(orgScope.isUnrestricted(), "platform admin scope must be unrestricted");

        // Admin can read a project in either org.
        assertEquals(aProj.getId(), projectService.getProject(aProj.getId()).getId());
        assertEquals(bProj.getId(), projectService.getProject(bProj.getId()).getId());
    }

    @Test
    void membershipLessUser_stillSeesOwnProjectInDefaultOrg() throws Exception {
        // Regression guard for the design trap: a normal user with NO org
        // memberships must fall back to {DEFAULT_ORG_ID} and still see their
        // own default-org project — the dashboard must NOT be empty.
        User carol = userRepository.save(new User("carol_iso", "carol_iso@example.com", "h"));
        // No OrganizationMembership rows for carol.
        Project defProj = newProject("default-proj", carol,
                ProjectService.DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        bindRequestScope();
        authenticate(carol, "ROLE_USER");
        runFilter();

        // Fallback: empty memberships -> {DEFAULT_ORG_ID}, not unrestricted.
        assertFalse(orgScope.isUnrestricted());
        assertEquals(java.util.Set.of(ProjectService.DEFAULT_ORG_ID), orgScope.visibleOrgIds());

        List<ProjectResponseDto> dash = projectService.getProjectsForCurrentUser();
        List<UUID> ids = dash.stream().map(ProjectResponseDto::getId).toList();
        assertTrue(ids.contains(defProj.getId()),
                "membership-less user must still see their default-org project");

        // And can read it directly.
        assertEquals(defProj.getId(), projectService.getProject(defProj.getId()).getId());
    }
}
