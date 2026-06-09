package com.federated.fl_platform_api.admin;

import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
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

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AdminControllerIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username, PlatformRole role) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode("Password1!"));
        u.setPlatformRole(role);
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

    @Test
    void nonAdmin_gets403OnAdminUsers() {
        createUser("user_a1", PlatformRole.USER);
        String cookie = loginAs("user_a1");

        ResponseEntity<String> resp = restTemplate.exchange(
            "/api/admin/users", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), String.class);
        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
    }

    @Test
    void admin_seesAllUsers() {
        createUser("admin_a2", PlatformRole.PLATFORM_ADMIN);
        createUser("user_a2", PlatformRole.USER);
        String cookie = loginAs("admin_a2");

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<List> resp = restTemplate.exchange(
            "/api/admin/users", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), List.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertNotNull(resp.getBody());
        assertEquals(2, resp.getBody().size());
    }

    @Test
    void demoteOnlyAdmin_returns409() {
        createUser("admin_a3", PlatformRole.PLATFORM_ADMIN);
        String cookie = loginAs("admin_a3");

        Long adminId = userRepository.findByUsername("admin_a3").orElseThrow().getId();

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/admin/users/" + adminId + "/role", HttpMethod.PUT,
            new HttpEntity<>(Map.of("role", "USER"), headers(cookie)), Map.class);
        assertEquals(HttpStatus.CONFLICT, resp.getStatusCode());
    }

    @Test
    void promoteUser_thenDemoteOriginalAdmin_succeeds() {
        User admin = createUser("admin_a4", PlatformRole.PLATFORM_ADMIN);
        User user = createUser("user_a4", PlatformRole.USER);
        String cookie = loginAs("admin_a4");

        // Promote user to PLATFORM_ADMIN
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> promote = restTemplate.exchange(
            "/api/admin/users/" + user.getId() + "/role", HttpMethod.PUT,
            new HttpEntity<>(Map.of("role", "PLATFORM_ADMIN"), headers(cookie)), Map.class);
        assertEquals(HttpStatus.OK, promote.getStatusCode());
        assertEquals("PLATFORM_ADMIN", promote.getBody().get("role"));

        // Now demote the original admin — should succeed since there are 2 admins
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> demote = restTemplate.exchange(
            "/api/admin/users/" + admin.getId() + "/role", HttpMethod.PUT,
            new HttpEntity<>(Map.of("role", "USER"), headers(cookie)), Map.class);
        assertEquals(HttpStatus.OK, demote.getStatusCode());
        assertEquals("USER", demote.getBody().get("role"));
    }
}
