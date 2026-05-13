package com.federated.fl_platform_api.accessrequest;

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

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AccessRequestControllerIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
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

    private Project createProject(User owner, ProjectVisibility v) {
        Project p = new Project();
        p.setName("p-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setVisibility(v);
        return projectRepository.save(p);
    }

    private HttpHeaders headers(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @Test
    void publicProject_submit_autoJoinsAsClientMembership() {
        User alice = createUser("alice_pub");
        createUser("bob_pub");
        Project p = createProject(alice, ProjectVisibility.PUBLIC);

        String bobCookie = loginAs("bob_pub");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/access-requests",
            HttpMethod.POST,
            new HttpEntity<>(Map.of(), headers(bobCookie)),
            Map.class);
        assertEquals(HttpStatus.CREATED, resp.getStatusCode());
        Map<?, ?> body = resp.getBody();
        assertNotNull(body);
        assertNotNull(body.get("membership"));
        assertNull(body.get("request"));
        Map<?, ?> membership = (Map<?, ?>) body.get("membership");
        assertEquals("CLIENT", membership.get("role"));
        assertEquals("PUBLIC_JOIN", membership.get("joinedVia"));
    }

    @Test
    void privateProject_submit_createsPendingRequest_andApprovalCreatesMembership() {
        User alice = createUser("alice_priv");
        createUser("bob_priv");
        Project p = createProject(alice, ProjectVisibility.PRIVATE);

        String bobCookie = loginAs("bob_priv");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> postResp = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/access-requests",
            HttpMethod.POST,
            new HttpEntity<>(Map.of("message", "please"), headers(bobCookie)),
            Map.class);
        assertEquals(HttpStatus.CREATED, postResp.getStatusCode());
        Map<?, ?> request = (Map<?, ?>) postResp.getBody().get("request");
        assertNotNull(request);
        assertEquals("PENDING", request.get("status"));
        Long requestId = ((Number) request.get("id")).longValue();

        String aliceCookie = loginAs("alice_priv");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> putResp = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/access-requests/" + requestId,
            HttpMethod.PUT,
            new HttpEntity<>(Map.of("decision", "APPROVED"), headers(aliceCookie)),
            Map.class);
        assertEquals(HttpStatus.OK, putResp.getStatusCode());
        assertEquals("APPROVED", putResp.getBody().get("status"));
    }
}
