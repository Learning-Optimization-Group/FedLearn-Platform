package com.federated.fl_platform_api.admin;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
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

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * PUT /api/admin/users/{id}/status — suspend / reactivate with both 409
 * guards, audit rows (USER_SUSPENDED / USER_REACTIVATED), and end-to-end
 * suspension enforcement: an already-issued session cookie stops working on
 * the next request, login is refused while suspended, and both work again
 * after reactivation.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AdminUserStatusIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired AuditEventRepository auditEventRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        return userRepository.save(u);
    }

    private String loginAs(String username) {
        ResponseEntity<Map> resp = loginRaw(username);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        return resp.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private ResponseEntity<Map> loginRaw(String username) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        return restTemplate.exchange(
            "/api/auth/login", HttpMethod.POST,
            new HttpEntity<>(Map.of("username", username, "password", "Password1!"), h),
            Map.class);
    }

    private HttpHeaders headers(String cookie) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private ResponseEntity<Map> putStatus(String cookie, Long userId, String status) {
        return restTemplate.exchange(
            "/api/admin/users/" + userId + "/status", HttpMethod.PUT,
            new HttpEntity<>(Map.of("status", status), headers(cookie)), Map.class);
    }

    private ResponseEntity<String> whoAmI(String cookie) {
        return restTemplate.exchange(
            "/api/auth/me", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), String.class);
    }

    private List<AuditEvent> eventsWithAction(AuditAction action) {
        return auditEventRepository.findAll().stream()
            .filter(e -> e.getAction() == action)
            .toList();
    }

    @Test
    void suspend_setsStatus_andWritesAuditRow() {
        User admin = createUser("admin_st1", PlatformRole.PLATFORM_ADMIN);
        User target = createUser("user_st1", PlatformRole.USER);
        String cookie = loginAs("admin_st1");

        ResponseEntity<Map> resp = putStatus(cookie, target.getId(), "SUSPENDED");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("SUSPENDED", resp.getBody().get("status"));
        assertEquals("user_st1", resp.getBody().get("username"));

        assertEquals(UserStatus.SUSPENDED,
            userRepository.findById(target.getId()).orElseThrow().getStatus());

        List<AuditEvent> rows = eventsWithAction(AuditAction.USER_SUSPENDED);
        assertEquals(1, rows.size());
        assertEquals("USER", rows.get(0).getTargetType());
        assertEquals(target.getId().toString(), rows.get(0).getTargetId());
        assertEquals(admin.getId(), rows.get(0).getActorUserId());
    }

    @Test
    void suspendedUser_sessionAndLoginRejected_untilReactivated() {
        createUser("admin_st2", PlatformRole.PLATFORM_ADMIN);
        User target = createUser("user_st2", PlatformRole.USER);
        String adminCookie = loginAs("admin_st2");

        // The user has a live session before suspension.
        String userCookie = loginAs("user_st2");
        assertEquals(HttpStatus.OK, whoAmI(userCookie).getStatusCode());

        // Suspend: the already-issued cookie dies on the very next request (the
        // filter reloads the user per request), and login is refused too.
        assertEquals(HttpStatus.OK, putStatus(adminCookie, target.getId(), "SUSPENDED").getStatusCode());
        assertEquals(HttpStatus.UNAUTHORIZED, whoAmI(userCookie).getStatusCode());
        assertEquals(HttpStatus.UNAUTHORIZED, loginRaw("user_st2").getStatusCode());

        // Reactivate: login works again and the session is usable.
        ResponseEntity<Map> reactivate = putStatus(adminCookie, target.getId(), "ACTIVE");
        assertEquals(HttpStatus.OK, reactivate.getStatusCode());
        assertEquals("ACTIVE", reactivate.getBody().get("status"));
        String freshCookie = loginAs("user_st2");
        assertEquals(HttpStatus.OK, whoAmI(freshCookie).getStatusCode());

        assertEquals(1, eventsWithAction(AuditAction.USER_REACTIVATED).size());
    }

    @Test
    void suspendingLastActiveAdmin_returns409() {
        User admin = createUser("admin_st3", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_st3");

        ResponseEntity<Map> resp = putStatus(cookie, admin.getId(), "SUSPENDED");
        assertEquals(HttpStatus.CONFLICT, resp.getStatusCode());
        assertEquals(UserStatus.ACTIVE,
            userRepository.findById(admin.getId()).orElseThrow().getStatus());
        // Guard failure writes no audit row (the aspect only fires on success).
        assertEquals(0, eventsWithAction(AuditAction.USER_SUSPENDED).size());
    }

    @Test
    void selfSuspension_returns409_evenWithAnotherActiveAdmin() {
        User self = createUser("admin_st4", PlatformRole.PLATFORM_ADMIN);
        createUser("admin_st4b", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_st4");

        // Two ACTIVE admins, so the last-admin guard passes — this 409 is the
        // self-suspension guard specifically.
        ResponseEntity<Map> resp = putStatus(cookie, self.getId(), "SUSPENDED");
        assertEquals(HttpStatus.CONFLICT, resp.getStatusCode());
        assertEquals(UserStatus.ACTIVE,
            userRepository.findById(self.getId()).orElseThrow().getStatus());
    }

    @Test
    void suspendingAnotherAdmin_succeeds_whenTwoActiveAdminsExist() {
        createUser("admin_st5", PlatformRole.PLATFORM_ADMIN);
        User other = createUser("admin_st5b", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_st5");

        ResponseEntity<Map> resp = putStatus(cookie, other.getId(), "SUSPENDED");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("SUSPENDED", resp.getBody().get("status"));
    }

    @Test
    void pendingOrUnknownStatusValue_returns400() {
        createUser("admin_st6", PlatformRole.PLATFORM_ADMIN);
        User target = createUser("user_st6", PlatformRole.USER);
        String cookie = loginAs("admin_st6");

        // PENDING is excluded from the accepted set; arbitrary strings too.
        assertEquals(HttpStatus.BAD_REQUEST, putStatus(cookie, target.getId(), "PENDING").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST, putStatus(cookie, target.getId(), "BANNED").getStatusCode());
    }

    @Test
    void unknownUser_returns404() {
        createUser("admin_st7", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_st7");

        assertEquals(HttpStatus.NOT_FOUND, putStatus(cookie, 999999L, "SUSPENDED").getStatusCode());
    }

    @Test
    void nonAdmin_gets403() {
        createUser("plain_st8", PlatformRole.USER);
        User target = createUser("user_st8", PlatformRole.USER);
        String cookie = loginAs("plain_st8");

        assertEquals(HttpStatus.FORBIDDEN, putStatus(cookie, target.getId(), "SUSPENDED").getStatusCode());
    }
}
