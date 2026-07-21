package com.federated.fl_platform_api.admin;

import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
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
 * GET /api/admin/users/search — the paginated, search-first users directory.
 * Envelope contract: {@code {items, page, size, total}}; q matches username OR
 * email case-insensitively; role/status filters combine with q; sorted
 * username asc; default size 25.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AdminUserSearchIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username, String email, PlatformRole role,
                            UserStatus status, String displayName) {
        User u = new User(username, email, passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
        u.setStatus(status);
        u.setDisplayName(displayName);
        return userRepository.save(u);
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
            "/api/admin/users/search" + queryString, HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), Map.class);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> items(ResponseEntity<Map> resp) {
        return (List<Map<String, Object>>) resp.getBody().get("items");
    }

    private static int total(ResponseEntity<Map> resp) {
        return ((Number) resp.getBody().get("total")).intValue();
    }

    @Test
    void q_matchesUsernameOrEmail_caseInsensitively() {
        createUser("admin_us1", "admin_us1@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        createUser("alice_dev", "alice@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        createUser("bob_qa", "bob@corp.io", PlatformRole.USER, UserStatus.ACTIVE, null);
        String cookie = loginAs("admin_us1");

        // Username match, uppercase query.
        ResponseEntity<Map> byUsername = search(cookie, "?q=ALICE");
        assertEquals(HttpStatus.OK, byUsername.getStatusCode());
        assertEquals(1, total(byUsername));
        assertEquals("alice_dev", items(byUsername).get(0).get("username"));

        // Email-only match (the string "corp.io" appears in no username).
        ResponseEntity<Map> byEmail = search(cookie, "?q=CORP.IO");
        assertEquals(1, total(byEmail));
        assertEquals("bob_qa", items(byEmail).get(0).get("username"));
    }

    @Test
    void roleAndStatusFilters_combineWithQ() {
        createUser("admin_us2", "admin_us2@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        createUser("carol_owner", "carol@example.com", PlatformRole.PROJECT_OWNER, UserStatus.ACTIVE, null);
        createUser("dave_user", "dave@example.com", PlatformRole.USER, UserStatus.SUSPENDED, null);
        String cookie = loginAs("admin_us2");

        ResponseEntity<Map> byRole = search(cookie, "?role=PROJECT_OWNER");
        assertEquals(1, total(byRole));
        assertEquals("carol_owner", items(byRole).get(0).get("username"));

        ResponseEntity<Map> byStatus = search(cookie, "?status=SUSPENDED");
        assertEquals(1, total(byStatus));
        assertEquals("dave_user", items(byStatus).get(0).get("username"));

        // Combined: q matches carol but she is not suspended.
        assertEquals(0, total(search(cookie, "?q=carol&status=SUSPENDED")));
        assertEquals(1, total(search(cookie, "?q=dave&status=SUSPENDED")));
    }

    @Test
    void pagination_envelope_and_usernameAscSort() {
        createUser("a_admin", "a_admin@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        createUser("b_user", "b_user@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        createUser("c_user", "c_user@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        createUser("d_user", "d_user@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        String cookie = loginAs("a_admin");

        ResponseEntity<Map> page0 = search(cookie, "?size=2&page=0");
        assertEquals(HttpStatus.OK, page0.getStatusCode());
        assertEquals(0, ((Number) page0.getBody().get("page")).intValue());
        assertEquals(2, ((Number) page0.getBody().get("size")).intValue());
        assertEquals(4, total(page0));
        assertEquals(List.of("a_admin", "b_user"),
            items(page0).stream().map(i -> i.get("username")).toList());

        ResponseEntity<Map> page1 = search(cookie, "?size=2&page=1");
        assertEquals(List.of("c_user", "d_user"),
            items(page1).stream().map(i -> i.get("username")).toList());
    }

    @Test
    void defaultSize_is25() {
        createUser("admin_us4", "admin_us4@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        String cookie = loginAs("admin_us4");

        ResponseEntity<Map> resp = search(cookie, "");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(0, ((Number) resp.getBody().get("page")).intValue());
        assertEquals(25, ((Number) resp.getBody().get("size")).intValue());
    }

    @Test
    void dto_carriesStatusDisplayNameAndLastLoginAt() {
        createUser("admin_us5", "admin_us5@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, "Admin Five");
        createUser("eve_never", "eve@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        String cookie = loginAs("admin_us5");   // sets admin_us5.lastLoginAt

        Map<String, Object> admin = items(search(cookie, "?q=admin_us5")).get(0);
        assertEquals("ACTIVE", admin.get("status"));
        assertEquals("Admin Five", admin.get("displayName"));
        assertNotNull(admin.get("lastLoginAt"), "admin just logged in — lastLoginAt must be set");

        // Nullable-safe on a user who never logged in and has no display name.
        Map<String, Object> eve = items(search(cookie, "?q=eve_never")).get(0);
        assertEquals("ACTIVE", eve.get("status"));
        assertTrue(eve.containsKey("displayName"));
        assertNull(eve.get("displayName"));
        assertNull(eve.get("lastLoginAt"));
    }

    @Test
    void oversizedPageSize_isClampedTo200() {
        createUser("admin_us8", "admin_us8@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        String cookie = loginAs("admin_us8");

        ResponseEntity<Map> resp = search(cookie, "?size=100000");
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals(200, ((Number) resp.getBody().get("size")).intValue());

        // Floor: a size below 1 clamps up to 1.
        ResponseEntity<Map> floored = search(cookie, "?size=0");
        assertEquals(HttpStatus.OK, floored.getStatusCode());
        assertEquals(1, ((Number) floored.getBody().get("size")).intValue());
        assertEquals(1, items(floored).size());
    }

    @Test
    void invalidRoleFilter_returns400() {
        createUser("admin_us6", "admin_us6@example.com", PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE, null);
        String cookie = loginAs("admin_us6");

        assertEquals(HttpStatus.BAD_REQUEST, search(cookie, "?role=NOT_A_ROLE").getStatusCode());
    }

    @Test
    void nonAdmin_gets403() {
        createUser("plain_us7", "plain_us7@example.com", PlatformRole.USER, UserStatus.ACTIVE, null);
        String cookie = loginAs("plain_us7");

        assertEquals(HttpStatus.FORBIDDEN, search(cookie, "?q=x").getStatusCode());
    }
}
