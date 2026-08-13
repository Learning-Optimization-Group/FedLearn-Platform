package com.federated.fl_platform_api.perf;

import com.federated.fl_platform_api.dto.DiscoverProjectDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.JoinedVia;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectAccessRequest;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectAccessRequestRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.security.OrgScopeFilter;
import com.federated.fl_platform_api.service.ProjectService;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.EntityManagerFactory;
import jakarta.servlet.FilterChain;
import org.hibernate.SessionFactory;
import org.hibernate.stat.Statistics;
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
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * BA-10 regression: the dashboard list endpoints ({@code getProjectsForCurrentUser},
 * {@code getDiscoverProjects}) must issue a CONSTANT number of SQL statements
 * regardless of how many projects the caller sees. Before the batch refactor each
 * listed project triggered its own membership (and, for discover, access-request)
 * SELECT — a classic N+1.
 *
 * <p>Two guarantees are asserted:
 * <ol>
 *   <li><b>Bounded queries</b> — measured with Hibernate {@link Statistics}
 *       ({@link Statistics#getPrepareStatementCount()}); the count for 20 projects
 *       equals the count for 3 (does not scale with N).</li>
 *   <li><b>Byte-identical payloads</b> — the serialized DTOs for the
 *       owner / member / client / discover(NONE|PENDING|APPROVED|DENIED) cases
 *       match a hand-authored golden, so the batch refactor cannot silently
 *       change any field value the frontend depends on
 *       (relationship / myRelationship, request-status, visibility).</li>
 * </ol>
 */
@SpringBootTest
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
@Transactional
class DashboardListN1Test {

    @Autowired ProjectService projectService;
    @Autowired OrgScopeFilter orgScopeFilter;
    @Autowired OrgScope orgScope;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired ProjectMembershipRepository membershipRepository;
    @Autowired ProjectAccessRequestRepository accessRequestRepository;
    @Autowired EntityManagerFactory emf;

    private final ObjectMapper json = new ObjectMapper();
    private static final AtomicLong SEQ = new AtomicLong();

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
        RequestContextHolder.resetRequestAttributes();
    }

    // ── request-scope / auth plumbing (mirrors OrgIsolationTest) ─────────────

    private void authAndScope(User u) throws Exception {
        RequestContextHolder.setRequestAttributes(
                new ServletRequestAttributes(new MockHttpServletRequest()));
        SecurityContextImpl ctx = new SecurityContextImpl();
        ctx.setAuthentication(new UsernamePasswordAuthenticationToken(
                u.getUsername(), "x", List.of(new SimpleGrantedAuthority("ROLE_USER"))));
        SecurityContextHolder.setContext(ctx);
        FilterChain noop = mock(FilterChain.class);
        orgScopeFilter.doFilter(new MockHttpServletRequest(), new MockHttpServletResponse(), noop);
        // A membership-less user falls back to exactly the default org, so all
        // seeded projects must live there to be visible.
        assertEquals(Set.of(ProjectService.DEFAULT_ORG_ID), orgScope.visibleOrgIds());
    }

    private User newUser(String base) {
        String tag = base + "_" + SEQ.incrementAndGet();
        return userRepository.saveAndFlush(new User(tag, tag + "@example.com", "h"));
    }

    private Project newProject(String name, User owner, ProjectVisibility vis) {
        Project p = new Project();
        p.setName(name);
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setOptimizer("adam");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(ProjectService.DEFAULT_ORG_ID);
        p.setVisibility(vis);
        return projectRepository.saveAndFlush(p);
    }

    private void addMember(Project p, User u, MembershipRole role, User addedBy) {
        membershipRepository.saveAndFlush(
                new ProjectMembership(p, u, role, JoinedVia.OWNER_ADD, addedBy));
    }

    private long prepareStatementCount(RunnableX action) throws Exception {
        Statistics stats = emf.unwrap(SessionFactory.class).getStatistics();
        stats.setStatisticsEnabled(true);
        stats.clear();
        action.run();
        return stats.getPrepareStatementCount();
    }

    @FunctionalInterface
    private interface RunnableX { void run() throws Exception; }

    // ── 1a. my-projects: bounded query count ─────────────────────────────────

    @Test
    void myProjects_queryCountDoesNotScaleWithProjectCount() throws Exception {
        long small = measureMyProjects(3);
        long large = measureMyProjects(20);

        assertEquals(small, large,
                "my-projects issued " + small + " queries for 3 projects but "
                        + large + " for 20 — the count scales with N (N+1 not fixed)");
        assertTrue(large <= 6,
                "expected a small constant query count for my-projects, got " + large);
    }

    /** Seeds N member-projects for a fresh caller and returns the query count of the list call. */
    private long measureMyProjects(int n) throws Exception {
        User owner = newUser("mp_owner");
        User caller = newUser("mp_caller");
        for (int i = 0; i < n; i++) {
            Project p = newProject("mp-" + n + "-" + i + "-" + SEQ.incrementAndGet(),
                    owner, ProjectVisibility.PRIVATE);
            addMember(p, caller, MembershipRole.MEMBER, owner);
        }
        authAndScope(caller);
        long[] size = new long[1];
        long queries = prepareStatementCount(() -> size[0] = projectService.getProjectsForCurrentUser().size());
        assertEquals(n, size[0], "sanity: caller should list exactly its N member-projects");
        return queries;
    }

    // ── 1b. discover: bounded query count ────────────────────────────────────

    @Test
    void discover_queryCountDoesNotScaleWithCandidateCount() throws Exception {
        long small = measureDiscover(3);
        long large = measureDiscover(20);

        assertEquals(small, large,
                "discover issued " + small + " queries for 3 candidates but "
                        + large + " for 20 — the count scales with N (N+1 not fixed)");
        assertTrue(large <= 7,
                "expected a small constant query count for discover, got " + large);
    }

    /** Seeds N public candidate projects (owned by others, no membership) for a fresh caller. */
    private long measureDiscover(int n) throws Exception {
        User owner = newUser("dc_owner");
        User caller = newUser("dc_caller");
        for (int i = 0; i < n; i++) {
            newProject("dc-" + n + "-" + i + "-" + SEQ.incrementAndGet(), owner, ProjectVisibility.PUBLIC);
        }
        authAndScope(caller);
        long[] size = new long[1];
        long queries = prepareStatementCount(() -> size[0] = projectService.getDiscoverProjects().size());
        // >= n, not == n: the public discover feed is org-wide, so a later caller in
        // the same tx also sees the earlier measurement's public candidates. The
        // query-count-does-not-scale guarantee is unaffected (counts are constant).
        assertTrue(size[0] >= n, "sanity: caller should discover at least its N candidate projects, got " + size[0]);
        return queries;
    }

    // ── 2a. my-projects: byte-identical payload for owner/member/client ──────

    @Test
    void myProjects_payloadIsByteIdenticalForOwnerMemberClient() throws Exception {
        User caller = newUser("pay_caller");
        User other = newUser("pay_other");

        // Names share a prefix + a/b/c suffix so the "order by p.name" is deterministic.
        String pre = "pay-" + SEQ.incrementAndGet() + "-";
        Project owned = newProject(pre + "a", caller, ProjectVisibility.PUBLIC);
        Project asMember = newProject(pre + "b", other, ProjectVisibility.RESTRICTED);
        Project asClient = newProject(pre + "c", other, ProjectVisibility.PRIVATE);
        addMember(asMember, caller, MembershipRole.MEMBER, other);
        addMember(asClient, caller, MembershipRole.CLIENT, other);

        authAndScope(caller);
        List<ProjectResponseDto> actual = projectService.getProjectsForCurrentUser();
        actual.forEach(d -> d.setId(null)); // id is a random UUID → excluded from the golden

        List<ProjectResponseDto> golden = List.of(
                projectDto(pre + "a", "OWNER", "PUBLIC"),
                projectDto(pre + "b", "MEMBER", "RESTRICTED"),
                projectDto(pre + "c", "CLIENT", "PRIVATE"));

        assertEquals(json.writeValueAsString(golden), json.writeValueAsString(actual));

        // Explicit per-case field asserts for readability.
        assertEquals("OWNER", actual.get(0).getMyRelationship());
        assertEquals("MEMBER", actual.get(1).getMyRelationship());
        assertEquals("CLIENT", actual.get(2).getMyRelationship());
        assertEquals("PUBLIC", actual.get(0).getVisibility());
        assertEquals("RESTRICTED", actual.get(1).getVisibility());
        assertEquals("PRIVATE", actual.get(2).getVisibility());
    }

    private ProjectResponseDto projectDto(String name, String relationship, String visibility) {
        ProjectResponseDto d = new ProjectResponseDto();
        d.setId(null);
        d.setName(name);
        d.setModelType("CNN-CIFAR10");
        // P1-4: the payload always states an arm (entity default FULL), never null — a frozen
        // project must be distinguishable from a full one in every list view.
        d.setTrainingArm("FULL");
        d.setModelName("resnet8");
        d.setServerPort(null);
        d.setOptimizer("adam");
        d.setStatus("CREATED");
        d.setMyRelationship(relationship);
        d.setVisibility(visibility);
        d.setOwnerUsername(null);
        d.setParticipantCount(null);
        return d;
    }

    // ── 2b. discover: byte-identical payload incl. request-status mapping ────

    @Test
    void discover_payloadIsByteIdenticalIncludingRequestStatus() throws Exception {
        User caller = newUser("dpay_caller");
        User owner = newUser("dpay_owner");
        String ownerName = owner.getUsername();

        String pre = "disc-" + SEQ.incrementAndGet() + "-";
        Project none = newProject(pre + "a", owner, ProjectVisibility.PUBLIC);
        Project pending = newProject(pre + "b", owner, ProjectVisibility.RESTRICTED);
        Project approved = newProject(pre + "c", owner, ProjectVisibility.RESTRICTED);
        Project denied = newProject(pre + "d", owner, ProjectVisibility.PUBLIC);
        none.setModelDescription("desc-a");
        pending.setModelDescription("desc-b");
        approved.setModelDescription("desc-c");
        denied.setModelDescription("desc-d");
        projectRepository.saveAllAndFlush(List.of(none, pending, approved, denied));

        request(pending, caller, AccessRequestStatus.PENDING);
        request(approved, caller, AccessRequestStatus.APPROVED);
        request(denied, caller, AccessRequestStatus.DENIED);

        authAndScope(caller);
        List<DiscoverProjectDto> all = projectService.getDiscoverProjects();
        // Restrict to this test's four projects (ignore any unrelated public seed).
        Set<UUID> mine = Set.of(none.getId(), pending.getId(), approved.getId(), denied.getId());
        List<DiscoverProjectDto> actual = all.stream()
                .filter(d -> mine.contains(d.getId()))
                .sorted((x, y) -> x.getName().compareTo(y.getName()))
                .toList();
        actual.forEach(d -> d.setId(null));

        List<DiscoverProjectDto> golden = List.of(
                discoverDto(pre + "a", "PUBLIC", ownerName, "desc-a", "NONE"),
                discoverDto(pre + "b", "RESTRICTED", ownerName, "desc-b", "PENDING"),
                discoverDto(pre + "c", "RESTRICTED", ownerName, "desc-c", "APPROVED"),
                discoverDto(pre + "d", "PUBLIC", ownerName, "desc-d", "DENIED"));

        assertEquals(json.writeValueAsString(golden), json.writeValueAsString(actual));

        assertEquals("NONE", actual.get(0).getMyRequestStatus());
        assertEquals("PENDING", actual.get(1).getMyRequestStatus());
        assertEquals("APPROVED", actual.get(2).getMyRequestStatus());
        assertEquals("DENIED", actual.get(3).getMyRequestStatus());
    }

    private void request(Project p, User u, AccessRequestStatus status) {
        ProjectAccessRequest r = new ProjectAccessRequest(p, u, "hi");
        r.setStatus(status);
        accessRequestRepository.saveAndFlush(r);
    }

    private DiscoverProjectDto discoverDto(String name, String visibility, String ownerUsername,
                                           String description, String requestStatus) {
        DiscoverProjectDto d = new DiscoverProjectDto();
        d.setId(null);
        d.setName(name);
        d.setVisibility(visibility);
        d.setOwnerUsername(ownerUsername);
        d.setModelType("CNN-CIFAR10");
        d.setMyRequestStatus(requestStatus);
        d.setLastAccuracy(null);
        d.setDescription(description);
        return d;
    }
}
