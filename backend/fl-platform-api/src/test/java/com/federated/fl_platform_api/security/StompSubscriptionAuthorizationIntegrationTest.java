package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.JoinedVia;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.converter.StringMessageConverter;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompFrameHandler;
import org.springframework.messaging.simp.stomp.StompHeaders;
import org.springframework.messaging.simp.stomp.StompSession;
import org.springframework.messaging.simp.stomp.StompSessionHandlerAdapter;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.socket.WebSocketHttpHeaders;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.messaging.WebSocketStompClient;

import java.lang.reflect.Type;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end STOMP subscription-authorization tests. Drives a real
 * {@link WebSocketStompClient} against a RANDOM_PORT server so the JWT
 * handshake, the CONNECT principal promotion, and the per-destination
 * SUBSCRIBE gate all run exactly as in production.
 *
 * <p>The security invariant under test (BA-5): a SUBSCRIBE to a project-scoped
 * topic ({@code /topic/logs|status|results|inference/{projectId}}) is only
 * delivered if the authenticated principal passes the same org-scope +
 * participant check the REST read path enforces. Non-project destinations
 * ({@code /user/**}) are never gated.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class StompSubscriptionAuthorizationIntegrationTest {

    private static final UUID DEFAULT_ORG_ID =
            UUID.fromString("00000000-0000-0000-0000-000000000001");

    @LocalServerPort int port;
    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired ProjectMembershipRepository membershipRepository;
    @Autowired OrganizationMembershipRepository orgMembershipRepository;
    @Autowired PasswordEncoder passwordEncoder;
    @Autowired SimpMessagingTemplate messagingTemplate;

    // ─── Tests ────────────────────────────────────────────────────────────────

    @Test
    void nonParticipant_isBlockedFromSubscribingToProjectLogs() throws Exception {
        User owner = createUser("owner-np", PlatformRole.USER);
        User outsider = createUser("outsider-np", PlatformRole.USER);
        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        RecordingSessionHandler handler = new RecordingSessionHandler();
        StompSession session = connect(jwtFor("outsider-np"), handler);

        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/logs/" + p.getId(), stringFrames(received));

        // The unauthorized SUBSCRIBE is rejected: the server sends an ERROR frame
        // and tears down the session.
        assertTrue(handler.errorLatch.await(5, TimeUnit.SECONDS),
                "unauthorized SUBSCRIBE should be rejected (ERROR / transport error)");

        // And no broadcast on that topic ever reaches this session.
        for (int i = 0; i < 5; i++) {
            messagingTemplate.convertAndSend("/topic/logs/" + p.getId(), "leak-" + i);
        }
        assertNull(received.poll(1, TimeUnit.SECONDS),
                "a non-participant must never receive project broadcasts");
    }

    @Test
    void wildcardSubscription_isRejectedAndReceivesNoBroadcasts() throws Exception {
        // BA-5 wildcard bypass: the SimpleBroker matches SUBSCRIBE destinations as Ant patterns, so an
        // ungated /topic/** would receive every project's broadcasts across tenants. It must be rejected.
        User owner = createUser("owner-wc", PlatformRole.USER);
        createUser("attacker-wc", PlatformRole.USER);
        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        RecordingSessionHandler handler = new RecordingSessionHandler();
        StompSession session = connect(jwtFor("attacker-wc"), handler);

        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/**", stringFrames(received));

        assertTrue(handler.errorLatch.await(5, TimeUnit.SECONDS),
                "a wildcard SUBSCRIBE (/topic/**) must be rejected");

        for (int i = 0; i < 5; i++) {
            messagingTemplate.convertAndSend("/topic/logs/" + p.getId(), "wildcard-leak-" + i);
        }
        assertNull(received.poll(1, TimeUnit.SECONDS),
                "a wildcard subscriber must never receive project broadcasts across tenants");
    }

    @Test
    void participantClient_canSubscribeAndReceiveProjectLogs() throws Exception {
        User owner = createUser("owner-cl", PlatformRole.USER);
        User client = createUser("client-cl", PlatformRole.USER);
        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);
        addMembership(p, client, MembershipRole.CLIENT);

        StompSession session = connect(jwtFor("client-cl"), new RecordingSessionHandler());

        CountDownLatch got = new CountDownLatch(1);
        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/logs/" + p.getId(), countingFrames(received, got));

        assertTrue(broadcastUntilReceived("/topic/logs/" + p.getId(), "hello-client", got),
                "an authorized participant (CLIENT) must receive project broadcasts");
        assertEquals("hello-client", received.peek());
    }

    @Test
    void owner_canSubscribeToProjectLogs() throws Exception {
        User owner = createUser("owner-ok", PlatformRole.USER);
        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        StompSession session = connect(jwtFor("owner-ok"), new RecordingSessionHandler());

        CountDownLatch got = new CountDownLatch(1);
        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/status/" + p.getId(), countingFrames(received, got));

        assertTrue(broadcastUntilReceived("/topic/status/" + p.getId(), "owner-msg", got),
                "the project owner must be able to subscribe to project topics");
    }

    @Test
    void platformAdmin_canSubscribeToAnyProject() throws Exception {
        User owner = createUser("owner-ad", PlatformRole.USER);
        createUser("admin-ad", PlatformRole.PLATFORM_ADMIN);
        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);

        StompSession session = connect(jwtFor("admin-ad"), new RecordingSessionHandler());

        CountDownLatch got = new CountDownLatch(1);
        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/results/" + p.getId(), countingFrames(received, got));

        assertTrue(broadcastUntilReceived("/topic/results/" + p.getId(), "admin-msg", got),
                "a platform admin must be able to subscribe to any project topic");
    }

    @Test
    void crossOrgParticipant_isDeniedLikeRestOrgScope() throws Exception {
        // Carol is a participant of the project but is scoped to a DIFFERENT org,
        // so the org-isolation gate must deny her subscription — mirroring the REST
        // read path where an out-of-scope project reads as non-existent.
        User owner = createUser("owner-xo", PlatformRole.USER);
        User carol = createUser("carol-xo", PlatformRole.USER);
        UUID otherOrg = UUID.randomUUID();
        orgMembershipRepository.save(new OrganizationMembership(otherOrg, carol.getId(), OrgRole.MEMBER));

        Project p = createProject(owner, DEFAULT_ORG_ID, ProjectVisibility.PRIVATE);
        addMembership(p, carol, MembershipRole.CLIENT); // participant, but wrong org

        RecordingSessionHandler handler = new RecordingSessionHandler();
        StompSession session = connect(jwtFor("carol-xo"), handler);

        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/topic/logs/" + p.getId(), stringFrames(received));

        assertTrue(handler.errorLatch.await(5, TimeUnit.SECONDS),
                "a cross-org participant must be denied identically to REST org-scope");
        for (int i = 0; i < 5; i++) {
            messagingTemplate.convertAndSend("/topic/logs/" + p.getId(), "leak-" + i);
        }
        assertNull(received.poll(1, TimeUnit.SECONDS),
                "a cross-org caller must never receive project broadcasts");
    }

    @Test
    void userDestination_isNeverGated() throws Exception {
        createUser("notify-me", PlatformRole.USER);

        StompSession session = connect(jwtFor("notify-me"), new RecordingSessionHandler());

        CountDownLatch got = new CountDownLatch(1);
        BlockingQueue<String> received = new LinkedBlockingQueue<>();
        session.subscribe("/user/queue/notifications", countingFrames(received, got));

        // /user/** is a non-project destination and must pass through the gate
        // unchanged: the user still receives their own targeted messages.
        boolean ok = false;
        for (int i = 0; i < 50 && !ok; i++) {
            messagingTemplate.convertAndSendToUser("notify-me", "/queue/notifications", "ping");
            ok = got.await(100, TimeUnit.MILLISECONDS);
        }
        assertTrue(ok, "/user/** subscriptions must not be blocked by the project gate");
    }

    // ─── STOMP helpers ──────────────────────────────────────────────────────────

    private StompSession connect(String jwt, StompSessionHandlerAdapter handler) throws Exception {
        WebSocketStompClient client = new WebSocketStompClient(new StandardWebSocketClient());
        client.setMessageConverter(new StringMessageConverter());
        WebSocketHttpHeaders handshake = new WebSocketHttpHeaders();
        handshake.add("Authorization", "Bearer " + jwt);
        return client.connectAsync("ws://localhost:" + port + "/ws-logs", handshake, handler)
                .get(5, TimeUnit.SECONDS);
    }

    /** Broadcasts repeatedly (tolerating subscription-registration latency) until received or timeout. */
    private boolean broadcastUntilReceived(String destination, String payload, CountDownLatch got)
            throws InterruptedException {
        for (int i = 0; i < 50; i++) {
            messagingTemplate.convertAndSend(destination, payload);
            if (got.await(100, TimeUnit.MILLISECONDS)) {
                return true;
            }
        }
        return false;
    }

    private static StompFrameHandler stringFrames(BlockingQueue<String> sink) {
        return new StompFrameHandler() {
            @Override public Type getPayloadType(StompHeaders headers) { return String.class; }
            @Override public void handleFrame(StompHeaders headers, Object payload) {
                sink.add((String) payload);
            }
        };
    }

    private static StompFrameHandler countingFrames(BlockingQueue<String> sink, CountDownLatch got) {
        return new StompFrameHandler() {
            @Override public Type getPayloadType(StompHeaders headers) { return String.class; }
            @Override public void handleFrame(StompHeaders headers, Object payload) {
                sink.add((String) payload);
                got.countDown();
            }
        };
    }

    /** Counts down on any error signal (ERROR frame, exception, or transport close). */
    private static final class RecordingSessionHandler extends StompSessionHandlerAdapter {
        final CountDownLatch errorLatch = new CountDownLatch(1);

        @Override public void handleException(StompSession session, StompCommand command,
                                              StompHeaders headers, byte[] payload, Throwable exception) {
            errorLatch.countDown();
        }
        @Override public void handleTransportError(StompSession session, Throwable exception) {
            errorLatch.countDown();
        }
        @Override public void handleFrame(StompHeaders headers, Object payload) {
            // Session-level frames here are STOMP ERROR frames.
            errorLatch.countDown();
        }
    }

    // ─── Fixture helpers ────────────────────────────────────────────────────────

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        return userRepository.save(u);
    }

    private Project createProject(User owner, UUID orgId, ProjectVisibility visibility) {
        Project p = new Project();
        p.setName("p-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(orgId);
        p.setVisibility(visibility);
        return projectRepository.save(p);
    }

    private void addMembership(Project project, User user, MembershipRole role) {
        membershipRepository.save(
                new ProjectMembership(project, user, role, JoinedVia.OWNER_ADD, project.getUser()));
    }

    private String jwtFor(String username) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        ResponseEntity<Map<String, Object>> resp = restTemplate.exchange(
                "/api/auth/login", HttpMethod.POST,
                new HttpEntity<>(Map.of("username", username, "password", "Password1!"), h),
                new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {});
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        String setCookie = resp.getHeaders().getFirst(HttpHeaders.SET_COOKIE);
        assertNotNull(setCookie, "login must set a jwtToken cookie");
        String cookiePair = setCookie.split(";")[0];           // jwtToken=<jwt>
        assertTrue(cookiePair.startsWith("jwtToken="));
        return cookiePair.substring("jwtToken=".length());     // raw JWT for the Bearer handshake header
    }
}
