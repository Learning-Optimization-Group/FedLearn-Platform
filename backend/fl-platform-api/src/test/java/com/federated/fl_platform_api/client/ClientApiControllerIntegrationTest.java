package com.federated.fl_platform_api.client;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
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
class ClientApiControllerIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired ProjectMembershipRepository membershipRepository;
    @Autowired PasswordEncoder passwordEncoder;

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

    private User createUser(String username) {
        return userRepository.save(new User(username, username + "@example.com",
            passwordEncoder.encode("Password1!")));
    }

    private Project createProject(User owner, String status, Integer port) {
        Project p = new Project();
        p.setName("p-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus(status);
        p.setServerPort(port);
        p.setUser(owner);
        p.setOrgId(UUID.fromString("00000000-0000-0000-0000-000000000001"));
        p.setVisibility(ProjectVisibility.PRIVATE);
        return projectRepository.save(p);
    }

    private HttpHeaders headers(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @Test
    void list_returnsOwnedAndClientMemberships() {
        User alice = createUser("alice_cl");
        User bob = createUser("bob_cl");
        Project p = createProject(alice, "CREATED", null);

        membershipRepository.save(new ProjectMembership(
            p, bob, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, alice));

        String aliceCookie = loginAs("alice_cl");
        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> aliceResp = restTemplate.exchange(
            "/api/client/projects", HttpMethod.GET,
            new HttpEntity<>(headers(aliceCookie)), List.class);
        assertEquals(HttpStatus.OK, aliceResp.getStatusCode());
        assertNotNull(aliceResp.getBody());
        assertEquals(1, aliceResp.getBody().size());

        String bobCookie = loginAs("bob_cl");
        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> bobResp = restTemplate.exchange(
            "/api/client/projects", HttpMethod.GET,
            new HttpEntity<>(headers(bobCookie)), List.class);
        assertEquals(HttpStatus.OK, bobResp.getStatusCode());
        assertNotNull(bobResp.getBody());
        assertEquals(1, bobResp.getBody().size());
    }

    @Test
    void connection_assignsAndPersistsPartitionId() {
        User alice = createUser("alice_cn");
        User bob = createUser("bob_cn");
        Project p = createProject(alice, "RUNNING", 50000);
        membershipRepository.save(new ProjectMembership(
            p, bob, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, alice));

        String bobCookie = loginAs("bob_cn");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> first = restTemplate.exchange(
            "/api/client/projects/" + p.getId() + "/connection",
            HttpMethod.GET, new HttpEntity<>(headers(bobCookie)), Map.class);
        assertEquals(HttpStatus.OK, first.getStatusCode());
        assertNotNull(first.getBody());
        assertEquals(0, ((Number) first.getBody().get("partitionId")).intValue());
        assertEquals("RUNNING", first.getBody().get("status"));
        assertTrue(((String) first.getBody().get("serverAddress")).endsWith(":50000"));

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> second = restTemplate.exchange(
            "/api/client/projects/" + p.getId() + "/connection",
            HttpMethod.GET, new HttpEntity<>(headers(bobCookie)), Map.class);
        assertNotNull(second.getBody());
        assertEquals(0, ((Number) second.getBody().get("partitionId")).intValue(),
            "Subsequent calls must return the same sticky partition_id");
    }

    @Test
    void connection_rejectsWhenProjectNotRunning() {
        User alice = createUser("alice_st");
        User bob = createUser("bob_st");
        Project p = createProject(alice, "CREATED", null);
        membershipRepository.save(new ProjectMembership(
            p, bob, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, alice));

        String bobCookie = loginAs("bob_st");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/client/projects/" + p.getId() + "/connection",
            HttpMethod.GET, new HttpEntity<>(headers(bobCookie)), Map.class);
        assertEquals(HttpStatus.CONFLICT, resp.getStatusCode());
    }

    @Test
    void connection_403ForNonClient() {
        User alice = createUser("alice_nc");
        createUser("carol_nc");
        Project p = createProject(alice, "RUNNING", 50000);

        String carolCookie = loginAs("carol_nc");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/client/projects/" + p.getId() + "/connection",
            HttpMethod.GET, new HttpEntity<>(headers(carolCookie)), Map.class);
        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
    }
}
