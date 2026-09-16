package com.federated.fl_platform_api.profile;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests for {@code GET/PATCH /api/users/me/profile} — real HTTP,
 * real security filter chain, Testcontainers Postgres (test profile).
 *
 * <p>All principals here are plain {@link PlatformRole#USER}s: the endpoint
 * must be reachable without any admin role.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class ProfileControllerIntegrationTest {

    private static final String PASSWORD = "Password1!";

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired AuditEventRepository auditEventRepository;
    @Autowired PasswordEncoder passwordEncoder;

    @BeforeEach
    void enablePatch() {
        // The default SimpleClientHttpRequestFactory (HttpURLConnection) cannot
        // send PATCH; the JDK HttpClient-backed factory can.
        restTemplate.getRestTemplate().setRequestFactory(new JdkClientHttpRequestFactory());
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private User createUser(String username) {
        User u = new User(username, username + "@example.com", passwordEncoder.encode(PASSWORD));
        u.setPlatformRole(PlatformRole.USER);
        return userRepository.save(u);
    }

    private String loginAs(String username) {
        return loginAs(username, PASSWORD);
    }

    private String loginAs(String username, String password) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/auth/login", HttpMethod.POST,
            new HttpEntity<>(Map.of("username", username, "password", password), h),
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
    private ResponseEntity<Map> getProfile(String cookie) {
        return restTemplate.exchange(
            "/api/users/me/profile", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), Map.class);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private ResponseEntity<Map> patchProfile(String cookie, Map<String, Object> body) {
        return restTemplate.exchange(
            "/api/users/me/profile", HttpMethod.PATCH,
            new HttpEntity<>(body, headers(cookie)), Map.class);
    }

    // ─── GET ────────────────────────────────────────────────────────────────

    @Test
    void get_returnsFullProfileShape_forPlainUserRole() {
        User u = createUser("prof_shape");
        u.setDisplayName("Shape Tester");
        userRepository.save(u);
        String cookie = loginAs("prof_shape");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = getProfile(cookie);

        // A USER-role (non-admin) principal reaches the endpoint.
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        Map<?, ?> body = resp.getBody();
        assertNotNull(body);
        assertEquals("prof_shape", body.get("username"));
        assertEquals("prof_shape@example.com", body.get("email"));
        assertEquals("Shape Tester", body.get("displayName"));
        assertEquals("USER", body.get("role"));
        assertEquals(Boolean.FALSE, body.get("emailVerified"));
        assertNotNull(body.get("createdAt"));
        assertNotNull(body.get("lastLoginAt"), "login just succeeded, so lastLoginAt must be set");
    }

    @Test
    void get_withoutAuth_returns401() {
        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/users/me/profile", HttpMethod.GET, HttpEntity.EMPTY, Map.class);
        assertEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode());
    }

    // ─── PATCH: display name ────────────────────────────────────────────────

    @Test
    void patch_updatesDisplayName_andTrimsIt() {
        createUser("prof_dn");
        String cookie = loginAs("prof_dn");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of("displayName", "  Ada Lovelace  "));

        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("Ada Lovelace", resp.getBody().get("displayName"));
        assertEquals("Ada Lovelace",
            userRepository.findByUsername("prof_dn").orElseThrow().getDisplayName());
    }

    @Test
    void patch_displayNameOver80Chars_returns400() {
        createUser("prof_dn_long");
        String cookie = loginAs("prof_dn_long");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of("displayName", "x".repeat(81)));

        assertEquals(HttpStatus.BAD_REQUEST, resp.getStatusCode());
        assertNull(userRepository.findByUsername("prof_dn_long").orElseThrow().getDisplayName());
    }

    // ─── PATCH: email ───────────────────────────────────────────────────────

    @Test
    void patch_emailTakenByAnotherUser_returns409() {
        createUser("prof_em_a");
        createUser("prof_em_b");
        String cookie = loginAs("prof_em_a");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of("email", "prof_em_b@example.com"));

        assertEquals(HttpStatus.CONFLICT, resp.getStatusCode());
        assertEquals("prof_em_a@example.com",
            userRepository.findByUsername("prof_em_a").orElseThrow().getEmail());
    }

    @Test
    void patch_emailChange_resetsEmailVerified() {
        User u = createUser("prof_em_reset");
        u.setEmailVerified(true);
        userRepository.save(u);
        String cookie = loginAs("prof_em_reset");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of("email", "prof_em_new@example.com"));

        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("prof_em_new@example.com", resp.getBody().get("email"));
        assertEquals(Boolean.FALSE, resp.getBody().get("emailVerified"));

        User reloaded = userRepository.findByUsername("prof_em_reset").orElseThrow();
        assertEquals("prof_em_new@example.com", reloaded.getEmail());
        assertFalse(reloaded.getEmailVerified());
    }

    // ─── PATCH: password ────────────────────────────────────────────────────

    @Test
    void patch_passwordChange_happyPath_andAudited() {
        createUser("prof_pw");
        String cookie = loginAs("prof_pw");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of(
            "currentPassword", PASSWORD,
            "newPassword", "NewPassword2!"));

        assertEquals(HttpStatus.OK, resp.getStatusCode());
        // The new password is live: a fresh login with it succeeds.
        loginAs("prof_pw", "NewPassword2!");

        // Both audit trails exist: the PATCH itself and the dedicated password row.
        assertTrue(auditEventRepository.findAll().stream()
                .anyMatch(e -> e.getAction() == AuditAction.USER_PROFILE_UPDATED),
            "PATCH must write a USER_PROFILE_UPDATED audit row");
        assertTrue(auditEventRepository.findAll().stream()
                .anyMatch(e -> e.getAction() == AuditAction.USER_PASSWORD_CHANGED),
            "password change must write a USER_PASSWORD_CHANGED audit row");
    }

    @Test
    void patch_wrongCurrentPassword_returns403_andLeavesPasswordUnchanged() {
        createUser("prof_pw_wrong");
        String cookie = loginAs("prof_pw_wrong");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of(
            "currentPassword", "not-the-password",
            "newPassword", "NewPassword2!"));

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
        assertTrue(passwordEncoder.matches(PASSWORD,
                userRepository.findByUsername("prof_pw_wrong").orElseThrow().getPassword()),
            "old password must still be in effect");
    }

    @Test
    void patch_newPasswordWithoutCurrent_returns403() {
        createUser("prof_pw_nocur");
        String cookie = loginAs("prof_pw_nocur");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of("newPassword", "NewPassword2!"));

        assertEquals(HttpStatus.FORBIDDEN, resp.getStatusCode());
    }

    @Test
    void patch_weakNewPassword_returns400() {
        createUser("prof_pw_weak");
        String cookie = loginAs("prof_pw_weak");

        // 5 chars — below the registration minimum of 6, same rule reused here.
        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, Map.of(
            "currentPassword", PASSWORD,
            "newPassword", "12345"));

        assertEquals(HttpStatus.BAD_REQUEST, resp.getStatusCode());
        assertTrue(passwordEncoder.matches(PASSWORD,
                userRepository.findByUsername("prof_pw_weak").orElseThrow().getPassword()));
    }

    // ─── PATCH: hardening ───────────────────────────────────────────────────

    @Test
    void patch_usernameIsNotEditable() {
        createUser("prof_immutable");
        String cookie = loginAs("prof_immutable");

        Map<String, Object> body = new HashMap<>();
        body.put("username", "hijacked");
        body.put("displayName", "Still Me");

        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = patchProfile(cookie, body);

        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertEquals("prof_immutable", resp.getBody().get("username"));
        assertTrue(userRepository.findByUsername("prof_immutable").isPresent());
        assertTrue(userRepository.findByUsername("hijacked").isEmpty());
    }

    @Test
    void patch_withoutAuth_returns401() {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        @SuppressWarnings("rawtypes")
        ResponseEntity<Map> resp = restTemplate.exchange(
            "/api/users/me/profile", HttpMethod.PATCH,
            new HttpEntity<>(Map.of("displayName", "nope"), h), Map.class);
        assertEquals(HttpStatus.UNAUTHORIZED, resp.getStatusCode());
    }
}
