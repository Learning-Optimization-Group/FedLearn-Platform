package com.federated.fl_platform_api.admin;

import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * GET /api/admin/projects/search — the paginated, search-first projects
 * directory. Envelope contract: {@code {items, page, size, total}}; q matches
 * project name OR owner username case-insensitively; status/visibility filters
 * combine; sorted name asc. Status is derived from the active run (BA-4), so a
 * freshly seeded project (init DONE, no run) reads as CREATED.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AdminProjectSearchIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        return userRepository.save(u);
    }

    private Project createProject(String name, User owner, ProjectVisibility visibility) {
        Project p = new Project();
        p.setName(name);
        p.setModelType("CNN");
        p.setModelName("net");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(visibility);
        return projectRepository.save(p);
    }

    private String loginAs(String username) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/auth/login", HttpMethod.POST,
            new HttpEntity<>(Map.of("username", username, "password", "Password1!"), h),
            Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        return resp.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];
    }

    private HttpHeaders headers(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private ResponseEntity<Map> search(String cookie, String queryString) {
        return restTemplate.exchange(
            "/api/admin/projects/search" + queryString, HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), Map.class);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> items(ResponseEntity<Map> resp) {
        return (List<Map<String, Object>>) resp.getBody().get("items");
    }

    private static int total(ResponseEntity<Map> resp) {
        return ((Number) resp.getBody().get("total")).intValue();
    }

    private String seedThreeProjectsAndLoginAdmin() {
        createUser("admin_ps", PlatformRole.PLATFORM_ADMIN);
        User alice = createUser("alice_owner", PlatformRole.PROJECT_OWNER);
        User bob = createUser("bob_owner", PlatformRole.PROJECT_OWNER);
        createProject("ecg-transformer", bob, ProjectVisibility.PRIVATE);
        createProject("mnist-cnn", alice, ProjectVisibility.PUBLIC);
        createProject("pneumonia-detect", alice, ProjectVisibility.RESTRICTED);
        return loginAs("admin_ps");
    }

    @Test
    void q_matchesProjectName_caseInsensitively() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        ResponseEntity<Map> resp = search(cookie, "?q=MNIST");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(1, total(resp));
        assertEquals("mnist-cnn", items(resp).get(0).get("name"));
        assertEquals("alice_owner", items(resp).get(0).get("ownerUsername"));
    }

    @Test
    void q_matchesOwnerUsername() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        // "bob_owner" appears in no project name — only as the owner.
        ResponseEntity<Map> resp = search(cookie, "?q=bob_owner");
        assertEquals(1, total(resp));
        assertEquals("ecg-transformer", items(resp).get(0).get("name"));
    }

    @Test
    void visibilityFilter_narrows() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        ResponseEntity<Map> resp = search(cookie, "?visibility=PUBLIC");
        assertEquals(1, total(resp));
        assertEquals("mnist-cnn", items(resp).get(0).get("name"));
    }

    @Test
    void statusFilter_usesDerivedStatus() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        // Freshly seeded projects (init DONE, no active run) derive to CREATED.
        assertEquals(3, total(search(cookie, "?status=CREATED")));
        assertEquals(0, total(search(cookie, "?status=RUNNING")));
    }

    @Test
    void pagination_envelope_and_nameAscSort() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        ResponseEntity<Map> page0 = search(cookie, "?size=2&page=0");
        assertEquals(0, ((Number) page0.getBody().get("page")).intValue());
        assertEquals(2, ((Number) page0.getBody().get("size")).intValue());
        assertEquals(3, total(page0));
        assertEquals(List.of("ecg-transformer", "mnist-cnn"),
            items(page0).stream().map(i -> i.get("name")).toList());

        ResponseEntity<Map> page1 = search(cookie, "?size=2&page=1");
        assertEquals(List.of("pneumonia-detect"),
            items(page1).stream().map(i -> i.get("name")).toList());
    }

    @Test
    void items_carryStatusVisibilityAndParticipantCount() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        Map<String, Object> item = items(search(cookie, "?q=pneumonia")).get(0);
        assertEquals("CREATED", item.get("status"));
        assertEquals("RESTRICTED", item.get("visibility"));
        assertEquals(0, ((Number) item.get("participantCount")).intValue());
    }

    @Test
    void oversizedPageSize_isClampedTo200() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        ResponseEntity<Map> resp = search(cookie, "?size=100000");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(200, ((Number) resp.getBody().get("size")).intValue());
        assertEquals(3, total(resp));

        // Floor: a size below 1 clamps up to 1.
        ResponseEntity<Map> floored = search(cookie, "?size=0");
        assertEquals(HttpStatus.OK, floored.getStatusCode());
        assertEquals(1, ((Number) floored.getBody().get("size")).intValue());
        assertEquals(1, items(floored).size());
        assertEquals(3, total(floored));
    }

    @Test
    void invalidVisibilityFilter_returns400() {
        String cookie = seedThreeProjectsAndLoginAdmin();

        assertEquals(HttpStatus.BAD_REQUEST, search(cookie, "?visibility=SECRET").getStatusCode());
    }

    @Test
    void nonAdmin_gets403() {
        createUser("plain_ps", PlatformRole.USER);
        String cookie = loginAs("plain_ps");

        assertEquals(HttpStatus.FORBIDDEN, search(cookie, "?q=x").getStatusCode());
    }
}
