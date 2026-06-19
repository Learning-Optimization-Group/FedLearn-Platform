package com.federated.fl_platform_api.workflow;

import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.ModelInitializer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doNothing;

/**
 * End-to-end coverage for the two admin-approval workflows added in V7:
 * owner promotion (USER → PROJECT_OWNER) and project deletion. Exercises the
 * full HTTP path including the platform-admin gate on {@code /api/admin/**}.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class OwnerAndDeletionWorkflowIntegrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired PasswordEncoder passwordEncoder;

    @MockBean ModelInitializer modelInitializer;

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        return userRepository.save(u);
    }

    private Project createProject(User owner) {
        Project p = new Project();
        p.setName("p-" + System.nanoTime());
        p.setModelType("CNN");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG);
        p.setVisibility(ProjectVisibility.PRIVATE);
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

    private Map<String, Object> projectBody() {
        return Map.of(
            "name", "wf-project",
            "modelType", "CNN",
            "modelName", "simple-cnn",
            "optimizer", "adam",
            "pretrainEpochs", 0);
    }

    @Test
    void ownerPromotion_gatesCreation_thenApprovalUnlocksIt() throws Exception {
        doNothing().when(modelInitializer).initializeModelFile(any(), any(), any(), any(), anyInt());
        createUser("op_user", PlatformRole.USER);
        createUser("op_admin", PlatformRole.PLATFORM_ADMIN);

        String userCookie = loginAs("op_user");

        // A plain USER cannot create projects yet.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> blocked = restTemplate.exchange(
            "/api/projects", HttpMethod.POST,
            new HttpEntity<>(projectBody(), headers(userCookie)), Map.class);
        assertEquals(HttpStatus.FORBIDDEN, blocked.getStatusCode());

        // Submit an owner-promotion request.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> submit = restTemplate.exchange(
            "/api/owner-requests", HttpMethod.POST,
            new HttpEntity<>(Map.of("message", "I want to host a project"), headers(userCookie)),
            Map.class);
        assertEquals(HttpStatus.CREATED, submit.getStatusCode());
        assertEquals("PENDING", submit.getBody().get("status"));

        // A non-admin cannot reach the admin queue.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> forbiddenQueue = restTemplate.exchange(
            "/api/admin/owner-requests", HttpMethod.GET,
            new HttpEntity<>(headers(userCookie)), Map.class);
        assertEquals(HttpStatus.FORBIDDEN, forbiddenQueue.getStatusCode());

        // Admin lists the queue and approves.
        String adminCookie = loginAs("op_admin");
        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> queue = restTemplate.exchange(
            "/api/admin/owner-requests?status=PENDING", HttpMethod.GET,
            new HttpEntity<>(headers(adminCookie)), List.class);
        assertEquals(HttpStatus.OK, queue.getStatusCode());
        assertNotNull(queue.getBody());
        Map<?, ?> req = (Map<?, ?>) queue.getBody().stream()
            .filter(o -> "op_user".equals(((Map<?, ?>) o).get("username")))
            .findFirst().orElseThrow();
        Long requestId = ((Number) req.get("id")).longValue();

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> decide = restTemplate.exchange(
            "/api/admin/owner-requests/" + requestId, HttpMethod.PUT,
            new HttpEntity<>(Map.of("decision", "APPROVED"), headers(adminCookie)), Map.class);
        assertEquals(HttpStatus.OK, decide.getStatusCode());
        assertEquals("APPROVED", decide.getBody().get("status"));

        // The SAME session can now create a project — authorities are reloaded
        // from the DB per request, so promotion takes effect without re-login.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> created = restTemplate.exchange(
            "/api/projects", HttpMethod.POST,
            new HttpEntity<>(projectBody(), headers(userCookie)), Map.class);
        assertEquals(HttpStatus.CREATED, created.getStatusCode());
        assertNotNull(created.getBody().get("id"));
    }

    @Test
    void deletion_ownerRequests_adminApproves_projectIsGone() {
        User owner = createUser("del_owner", PlatformRole.PROJECT_OWNER);
        createUser("del_admin", PlatformRole.PLATFORM_ADMIN);
        Project p = createProject(owner);

        String ownerCookie = loginAs("del_owner");

        // Owner cannot delete directly.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> directDelete = restTemplate.exchange(
            "/api/projects/" + p.getId(), HttpMethod.DELETE,
            new HttpEntity<>(headers(ownerCookie)), Map.class);
        assertEquals(HttpStatus.FORBIDDEN, directDelete.getStatusCode());

        // Owner files a deletion request.
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> request = restTemplate.exchange(
            "/api/projects/" + p.getId() + "/deletion-request", HttpMethod.POST,
            new HttpEntity<>(Map.of("reason", "done with it"), headers(ownerCookie)), Map.class);
        assertEquals(HttpStatus.CREATED, request.getStatusCode());
        assertEquals("PENDING", request.getBody().get("status"));

        // Admin approves → project is hard-deleted.
        String adminCookie = loginAs("del_admin");
        @SuppressWarnings({"rawtypes", "unchecked"})
        ResponseEntity<List> queue = restTemplate.exchange(
            "/api/admin/deletion-requests?status=PENDING", HttpMethod.GET,
            new HttpEntity<>(headers(adminCookie)), List.class);
        assertEquals(HttpStatus.OK, queue.getStatusCode());
        Map<?, ?> req = (Map<?, ?>) queue.getBody().stream()
            .filter(o -> p.getId().toString().equals(((Map<?, ?>) o).get("projectId")))
            .findFirst().orElseThrow();
        Long requestId = ((Number) req.get("id")).longValue();

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> decide = restTemplate.exchange(
            "/api/admin/deletion-requests/" + requestId, HttpMethod.PUT,
            new HttpEntity<>(Map.of("decision", "APPROVED"), headers(adminCookie)), Map.class);
        assertEquals(HttpStatus.OK, decide.getStatusCode());
        assertEquals("APPROVED", decide.getBody().get("status"));

        assertFalse(projectRepository.findById(p.getId()).isPresent());
    }
}
