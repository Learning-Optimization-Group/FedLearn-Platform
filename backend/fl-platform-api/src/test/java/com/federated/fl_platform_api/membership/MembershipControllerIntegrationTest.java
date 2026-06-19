package com.federated.fl_platform_api.membership;

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

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class MembershipControllerIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private String loginAs(String username, String password) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/auth/login", HttpMethod.POST,
            new HttpEntity<>(Map.of("username", username, "password", password), h),
            Map.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        String cookie = resp.getHeaders().getFirst(HttpHeaders.SET_COOKIE);
        assertNotNull(cookie);
        return cookie.split(";")[0];
    }

    private User createUser(String username) {
        User u = new User(username, username + "@example.com",
            passwordEncoder.encode("Password1!"));
        return userRepository.save(u);
    }

    private Project createProject(User owner, ProjectVisibility v) {
        Project p = new Project();
        p.setName("p-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(UUID.fromString("00000000-0000-0000-0000-000000000001"));
        p.setVisibility(v);
        return projectRepository.save(p);
    }

    private HttpHeaders authHeaders(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @Test
    void owner_canAddClientByUsername_andOtherUserCannot() {
        User alice = createUser("alice");
        createUser("bob");
        createUser("carol");
        Project p = createProject(alice, ProjectVisibility.PRIVATE);

        String aliceCookie = loginAs("alice", "Password1!");
        String carolCookie = loginAs("carol", "Password1!");

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> add = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/memberships",
            HttpMethod.POST,
            new HttpEntity<>(Map.of("username", "bob", "role", "CLIENT"),
                authHeaders(aliceCookie)),
            Map.class);
        assertEquals(HttpStatus.CREATED, add.getStatusCode());

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> sneaky = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/memberships",
            HttpMethod.POST,
            new HttpEntity<>(Map.of("username", "carol", "role", "CLIENT"),
                authHeaders(carolCookie)),
            Map.class);
        assertEquals(HttpStatus.FORBIDDEN, sneaky.getStatusCode());
    }

    @Test
    void owner_canRemoveClient() {
        User alice = createUser("alice2");
        User bob = createUser("bob2");
        Project p = createProject(alice, ProjectVisibility.PRIVATE);
        String aliceCookie = loginAs("alice2", "Password1!");

        restTemplate.exchange(
            "/api/projects/" + p.getId() + "/memberships",
            HttpMethod.POST,
            new HttpEntity<>(Map.of("username", "bob2", "role", "CLIENT"),
                authHeaders(aliceCookie)),
            Map.class);

        ResponseEntity<Void> rm = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/memberships/" + bob.getId(),
            HttpMethod.DELETE,
            new HttpEntity<>(authHeaders(aliceCookie)),
            Void.class);
        assertEquals(HttpStatus.NO_CONTENT, rm.getStatusCode());
    }

    @Test
    void discover_returnsPublicAndRestricted_withRequestStatus() {
        User alice = createUser("alice_dis");
        createUser("bob_dis");
        createProject(alice, ProjectVisibility.PUBLIC);
        Project restricted = createProject(alice, ProjectVisibility.RESTRICTED);
        // A PRIVATE project must NOT surface in discovery (invite-only).
        createProject(alice, ProjectVisibility.PRIVATE);

        String bob = loginAs("bob_dis", "Password1!");

        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> first = restTemplate.exchange("/api/projects/discover",
            HttpMethod.GET, new HttpEntity<>(authHeaders(bob)), List.class);
        assertEquals(HttpStatus.OK, first.getStatusCode());
        assertNotNull(first.getBody());
        // PUBLIC + RESTRICTED are discoverable; the PRIVATE one is hidden.
        assertEquals(2, first.getBody().size());

        restTemplate.exchange("/api/projects/" + restricted.getId() + "/access-requests",
            HttpMethod.POST,
            new HttpEntity<>(Map.of(), authHeaders(bob)),
            Map.class);

        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> second = restTemplate.exchange("/api/projects/discover",
            HttpMethod.GET, new HttpEntity<>(authHeaders(bob)), List.class);
        assertNotNull(second.getBody());
        boolean foundPendingRestricted = second.getBody().stream().anyMatch(o -> {
            Map<?, ?> m = (Map<?, ?>) o;
            return "RESTRICTED".equals(m.get("visibility"))
                && "PENDING".equals(m.get("myRequestStatus"));
        });
        assertTrue(foundPendingRestricted);
    }
}
