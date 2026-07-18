package com.federated.fl_platform_api.admin;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
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

import static org.junit.jupiter.api.Assertions.*;

/**
 * GET /api/admin/audit-events — the paginated audit explorer. Envelope
 * contract: {@code {items, page, size, total}}, newest first; {@code actor} is
 * a username resolved server-side; filters combine; {@code from}/{@code to}
 * are ISO-8601 instants (from inclusive, to exclusive).
 *
 * <p>Logging in writes USER_LOGIN_SUCCEEDED rows, so assertions are scoped by
 * action/targetType filters (never bare unfiltered counts).
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AdminAuditEventsIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired AuditEventRepository auditEventRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        return userRepository.save(u);
    }

    private AuditEvent seedEvent(AuditAction action, Long actorId, String targetType, String targetId) {
        return auditEventRepository.save(AuditEvent.builder()
            .action(action)
            .actorUserId(actorId)
            .targetType(targetType)
            .targetId(targetId)
            .requestIp("10.1.2.3")
            .metadata("{\"reason\":\"test\"}")
            .build());
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
    private ResponseEntity<Map> query(String cookie, String queryString) {
        return restTemplate.exchange(
            "/api/admin/audit-events" + queryString, HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), Map.class);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> items(ResponseEntity<Map> resp) {
        return (List<Map<String, Object>>) resp.getBody().get("items");
    }

    private static int total(ResponseEntity<Map> resp) {
        return ((Number) resp.getBody().get("total")).intValue();
    }

    private static void sleepMillis(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    @Test
    void newestFirst_withEnvelopeAndItemShape() {
        User admin = createUser("admin_ae1", PlatformRole.PLATFORM_ADMIN);
        seedEvent(AuditAction.PROJECT_CREATED, admin.getId(), "PROJECT", "p-1");
        sleepMillis(5);
        seedEvent(AuditAction.PROJECT_DELETED, admin.getId(), "PROJECT", "p-1");
        String cookie = loginAs("admin_ae1");

        ResponseEntity<Map> resp = query(cookie, "");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(0, ((Number) resp.getBody().get("page")).intValue());
        assertEquals(50, ((Number) resp.getBody().get("size")).intValue());   // default size
        assertTrue(total(resp) >= 2);

        // Newest first across the whole page (login events included).
        List<Map<String, Object>> list = items(resp);
        for (int i = 1; i < list.size(); i++) {
            Instant prev = Instant.parse((String) list.get(i - 1).get("occurredAt"));
            Instant curr = Instant.parse((String) list.get(i).get("occurredAt"));
            assertFalse(prev.isBefore(curr), "items must be sorted newest first");
        }

        // Full item shape on a seeded row.
        Map<String, Object> deleted = items(query(cookie, "?action=PROJECT_DELETED")).get(0);
        assertNotNull(deleted.get("id"));
        assertNotNull(deleted.get("occurredAt"));
        assertEquals(admin.getId().intValue(), ((Number) deleted.get("actorUserId")).intValue());
        assertEquals("admin_ae1", deleted.get("actorUsername"));
        assertEquals("PROJECT_DELETED", deleted.get("action"));
        assertEquals("PROJECT", deleted.get("targetType"));
        assertEquals("p-1", deleted.get("targetId"));
        assertEquals("10.1.2.3", deleted.get("requestIp"));
        assertTrue(((String) deleted.get("metadata")).contains("\"reason\""));
    }

    @Test
    void actorFilter_resolvesUsername_andMapsUsernamesInResponse() {
        createUser("admin_ae2", PlatformRole.PLATFORM_ADMIN);
        User alice = createUser("alice_ae2", PlatformRole.USER);
        User bob = createUser("bob_ae2", PlatformRole.USER);
        seedEvent(AuditAction.PROJECT_CREATED, alice.getId(), "PROJECT", "p-a");
        seedEvent(AuditAction.PROJECT_DELETED, alice.getId(), "PROJECT", "p-a");
        seedEvent(AuditAction.PROJECT_CREATED, bob.getId(), "PROJECT", "p-b");
        String cookie = loginAs("admin_ae2");

        ResponseEntity<Map> resp = query(cookie, "?actor=alice_ae2");
        assertEquals(2, total(resp));
        for (Map<String, Object> item : items(resp)) {
            assertEquals(alice.getId().intValue(), ((Number) item.get("actorUserId")).intValue());
            assertEquals("alice_ae2", item.get("actorUsername"));
        }
    }

    @Test
    void unknownActor_returnsEmptyPage() {
        createUser("admin_ae3", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_ae3");

        ResponseEntity<Map> resp = query(cookie, "?actor=no_such_user");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(0, total(resp));
        assertTrue(items(resp).isEmpty());
    }

    @Test
    void actionAndTargetTypeFilters_combine() {
        User admin = createUser("admin_ae4", PlatformRole.PLATFORM_ADMIN);
        seedEvent(AuditAction.PROJECT_CREATED, admin.getId(), "PROJECT", "p-1");
        seedEvent(AuditAction.PROJECT_CREATED, admin.getId(), "PROJECT", "p-2");
        seedEvent(AuditAction.USER_SUSPENDED, admin.getId(), "USER", "42");
        String cookie = loginAs("admin_ae4");

        assertEquals(2, total(query(cookie, "?action=PROJECT_CREATED")));
        assertEquals(1, total(query(cookie, "?action=USER_SUSPENDED")));
        assertEquals(2, total(query(cookie, "?targetType=PROJECT")));
        // Combined filters intersect: PROJECT_CREATED never targets USER here.
        assertEquals(0, total(query(cookie, "?action=PROJECT_CREATED&targetType=USER")));
        assertEquals(1, total(query(cookie, "?action=USER_SUSPENDED&targetType=USER&actor=admin_ae4")));
    }

    @Test
    void fromInclusive_toExclusive_windowing() {
        User admin = createUser("admin_ae5", PlatformRole.PLATFORM_ADMIN);
        seedEvent(AuditAction.PROJECT_CREATED, admin.getId(), "PROJECT", "p-old");
        sleepMillis(10);
        Instant cut = Instant.now();
        sleepMillis(10);
        seedEvent(AuditAction.PROJECT_DELETED, admin.getId(), "PROJECT", "p-new");
        String cookie = loginAs("admin_ae5");

        // from=cut keeps only the later event.
        assertEquals(0, total(query(cookie, "?from=" + cut + "&action=PROJECT_CREATED")));
        assertEquals(1, total(query(cookie, "?from=" + cut + "&action=PROJECT_DELETED")));
        // to=cut keeps only the earlier event (exclusive upper bound).
        assertEquals(1, total(query(cookie, "?to=" + cut + "&action=PROJECT_CREATED")));
        assertEquals(0, total(query(cookie, "?to=" + cut + "&action=PROJECT_DELETED")));
    }

    @Test
    void nullActor_yieldsNullActorUsername() {
        createUser("admin_ae6", PlatformRole.PLATFORM_ADMIN);
        seedEvent(AuditAction.BOOTSTRAP_ORG_CREATED, null, "ORG", "org-1");
        String cookie = loginAs("admin_ae6");

        Map<String, Object> item = items(query(cookie, "?action=BOOTSTRAP_ORG_CREATED")).get(0);
        assertNull(item.get("actorUserId"));
        assertNull(item.get("actorUsername"));
    }

    @Test
    void pagination_slicesFilteredSet() {
        User admin = createUser("admin_ae7", PlatformRole.PLATFORM_ADMIN);
        for (int i = 0; i < 3; i++) {
            seedEvent(AuditAction.PROJECT_CREATED, admin.getId(), "PROJECT", "p-" + i);
            sleepMillis(5);
        }
        String cookie = loginAs("admin_ae7");

        ResponseEntity<Map> page0 = query(cookie, "?action=PROJECT_CREATED&size=2&page=0");
        assertEquals(3, total(page0));
        assertEquals(2, items(page0).size());
        ResponseEntity<Map> page1 = query(cookie, "?action=PROJECT_CREATED&size=2&page=1");
        assertEquals(1, items(page1).size());
        // Newest first: page 0 holds p-2 and p-1, page 1 holds p-0.
        assertEquals("p-2", items(page0).get(0).get("targetId"));
        assertEquals("p-0", items(page1).get(0).get("targetId"));
    }

    @Test
    void invalidActionOrTimestamp_returns400() {
        createUser("admin_ae8", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_ae8");

        assertEquals(HttpStatus.BAD_REQUEST, query(cookie, "?action=NOT_AN_ACTION").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST, query(cookie, "?from=yesterday").getStatusCode());
    }

    @Test
    void nonAdmin_gets403() {
        createUser("plain_ae9", PlatformRole.USER);
        String cookie = loginAs("plain_ae9");

        assertEquals(HttpStatus.FORBIDDEN, query(cookie, "").getStatusCode());
    }
}
