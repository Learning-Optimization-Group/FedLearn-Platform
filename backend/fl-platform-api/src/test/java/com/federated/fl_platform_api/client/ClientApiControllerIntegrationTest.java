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

import java.time.Instant;
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
    @Autowired RunRepository runRepository;
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

    /**
     * Seeds a Run row (with the given status and port), updates the project's
     * active_run_id to point to it, and persists both. Returns the saved Run.
     * clientsPerRound is set high enough (100) to never cap enrollment in tests.
     */
    private Run createRunForProject(Project project, RunStatus status, Integer port) {
        Run run = new Run();
        run.setProjectId(project.getId());
        run.setStrategy("FedAvg");
        run.setNumRounds(3);
        run.setMinClients(1);
        run.setClientsPerRound(100);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setStatus(status);
        run.setServerHost("localhost");
        run.setServerPort(port);
        run.setRecipeKey(project.getModelType());
        run.setCreatedBy(project.getUser() != null ? project.getUser().getId() : null);
        run.setCreatedAt(Instant.now());
        if (status == RunStatus.RUNNING) {
            run.setStartedAt(Instant.now());
        }
        run.setSeed(42L);
        run = runRepository.save(run);

        project.setActiveRunId(run.getId());
        projectRepository.save(project);

        return run;
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
        // Project starts with RUNNING status; a real Run row is required for the shim.
        Project p = createProject(alice, "RUNNING", null);
        membershipRepository.save(new ProjectMembership(
            p, bob, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, alice));
        createRunForProject(p, RunStatus.RUNNING, 50000);

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
        assertNotNull(first.getBody().get("connectionToken"),
            "connectionToken must be present in the response");

        // Second call must return the same sticky partition_id (idempotent enrollment).
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
        // Project has no active run → shim throws ProjectStateException → 409 CONFLICT.
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
        // Project must have an active RUNNING run so the shim reaches the authz check
        // inside RunService.enroll(), which then rejects carol (non-member) with 403.
        Project p = createProject(alice, "RUNNING", null);
        createRunForProject(p, RunStatus.RUNNING, 50000);

        String carolCookie = loginAs("carol_nc");
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/client/projects/" + p.getId() + "/connection",
            HttpMethod.GET, new HttpEntity<>(headers(carolCookie)), Map.class);
        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
    }
}
