package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class AuthControllerIntegrationTest {

    @Autowired
    private TestRestTemplate restTemplate;

    private Map<String, Object> registerPayload(String username, String email, String password) {
        return Map.of("username", username, "email", email, "password", password);
    }

    @Test
    void register_withValidData_shouldReturn201() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(registerPayload("bob", "bob@example.com", "Password1!"), headers);

        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity("/api/auth/register", request, (Class<Map<String, Object>>) (Class<?>) Map.class);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        Map<String, Object> body = response.getBody();
        assertNotNull(body);
        assertTrue(body.containsKey("userId"));
    }

    @Test
    void register_duplicateUsername_shouldReturn409() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(registerPayload("bob2", "bob2@example.com", "Password1!"), headers);

        restTemplate.postForEntity("/api/auth/register", request, Map.class);
        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> second = restTemplate.postForEntity("/api/auth/register", request, (Class<Map<String, Object>>) (Class<?>) Map.class);

        assertEquals(HttpStatus.CONFLICT, second.getStatusCode());
    }

    @Test
    void login_withCorrectCredentials_shouldReturn200WithCookie() {
        // First register
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        restTemplate.postForEntity("/api/auth/register",
                new HttpEntity<>(registerPayload("charlie", "charlie@example.com", "Password1!"), headers),
                Map.class);

        // Now login
        HttpEntity<Map<String, Object>> loginRequest = new HttpEntity<>(
                Map.of("username", "charlie", "password", "Password1!"), headers);
        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity("/api/auth/login", loginRequest, (Class<Map<String, Object>>) (Class<?>) Map.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        // Cookie should be set
        assertTrue(response.getHeaders().containsKey(HttpHeaders.SET_COOKIE));
        String cookie = response.getHeaders().getFirst(HttpHeaders.SET_COOKIE);
        assertNotNull(cookie);
        assertTrue(cookie.contains("jwtToken"));
        // SE-8: this login carried NO X-FedLearn-Client marker (a browser SPA), which
        // authenticates via the HttpOnly cookie + GET /api/auth/me. The body must carry
        // identity only — no JS-readable accessToken.
        assertNotNull(response.getBody());
        assertEquals("charlie", response.getBody().get("username"));
        assertFalse(response.getBody().containsKey("accessToken"),
                "browser login (no native-client marker) must not return accessToken in the body (SE-8)");
    }

    @Test
    void login_withNativeClientMarker_returnsAccessTokenInBody() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        restTemplate.postForEntity("/api/auth/register",
                new HttpEntity<>(registerPayload("nat", "nat@example.com", "Password1!"), headers),
                Map.class);

        // A native client (desktop/mobile) self-identifies with the X-FedLearn-Client marker and
        // still needs the token in the body — it cannot read the HttpOnly Set-Cookie and replays
        // the JWT as a Bearer (the same marker SE-9 gates Bearer acceptance on).
        HttpHeaders nativeHeaders = new HttpHeaders();
        nativeHeaders.setContentType(MediaType.APPLICATION_JSON);
        nativeHeaders.add("X-FedLearn-Client", "fedlearn-desktop");
        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity("/api/auth/login",
                new HttpEntity<>(Map.of("username", "nat", "password", "Password1!"), nativeHeaders),
                (Class<Map<String, Object>>) (Class<?>) Map.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertNotNull(response.getBody().get("accessToken"), "native client must still get accessToken (SE-8)");
        assertFalse(response.getBody().get("accessToken").toString().isBlank());
        assertTrue(response.getHeaders().containsKey(HttpHeaders.SET_COOKIE));
    }

    @Test
    @SuppressWarnings("unchecked")
    void login_whenThrottled_returns429WithRetryAfterHeader() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        restTemplate.postForEntity("/api/auth/register",
                new HttpEntity<>(registerPayload("throttled", "throttled@example.com", "Password1!"), headers),
                Map.class);
        HttpEntity<Map<String, Object>> badLogin = new HttpEntity<>(
                Map.of("username", "throttled", "password", "WrongPass1!"), headers);

        // Exhaust the failure window (MAX_FAILURES=5) -> the account key locks.
        for (int i = 0; i < 5; i++) {
            restTemplate.postForEntity("/api/auth/login", badLogin, Map.class);
        }

        ResponseEntity<Map<String, Object>> throttled = restTemplate.postForEntity(
                "/api/auth/login", badLogin, (Class<Map<String, Object>>) (Class<?>) Map.class);
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, throttled.getStatusCode());
        String retryAfter = throttled.getHeaders().getFirst(HttpHeaders.RETRY_AFTER);
        assertNotNull(retryAfter, "429 must carry a Retry-After header (SE-4 done-when #1)");
        long seconds = Long.parseLong(retryAfter);
        assertTrue(seconds > 0 && seconds <= 15 * 60,
                "Retry-After should be a positive delay within the 15-min lockout, got " + seconds);
    }

    @Test
    @SuppressWarnings("unchecked")
    void logout_revokesTheToken_soTheSameCookieStopsWorking() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        restTemplate.postForEntity("/api/auth/register",
                new HttpEntity<>(registerPayload("revoker", "revoker@example.com", "Password1!"), headers),
                Map.class);
        ResponseEntity<Map<String, Object>> login = restTemplate.postForEntity("/api/auth/login",
                new HttpEntity<>(Map.of("username", "revoker", "password", "Password1!"), headers),
                (Class<Map<String, Object>>) (Class<?>) Map.class);
        String cookie = login.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];  // jwtToken=<value>

        HttpHeaders authed = new HttpHeaders();
        authed.add(HttpHeaders.COOKIE, cookie);

        // The token works before logout.
        ResponseEntity<Map<String, Object>> before = restTemplate.exchange("/api/auth/me",
                org.springframework.http.HttpMethod.GET, new HttpEntity<>(authed),
                (Class<Map<String, Object>>) (Class<?>) Map.class);
        assertEquals(HttpStatus.OK, before.getStatusCode());

        // Log out (revokes the token's jti).
        ResponseEntity<Void> logout = restTemplate.exchange("/api/auth/logout",
                org.springframework.http.HttpMethod.POST, new HttpEntity<>(authed), Void.class);
        assertEquals(HttpStatus.NO_CONTENT, logout.getStatusCode());

        // The SAME (unexpired) token is now rejected — clearing the cookie alone wouldn't do this.
        ResponseEntity<Map<String, Object>> after = restTemplate.exchange("/api/auth/me",
                org.springframework.http.HttpMethod.GET, new HttpEntity<>(authed),
                (Class<Map<String, Object>>) (Class<?>) Map.class);
        assertEquals(HttpStatus.UNAUTHORIZED, after.getStatusCode());
    }

    @Test
    void login_withWrongPassword_shouldReturn401() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(Map.of("username", "nobody", "password", "wrongpass"), headers);

        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> response = restTemplate.postForEntity("/api/auth/login", request, (Class<Map<String, Object>>) (Class<?>) Map.class);

        assertEquals(HttpStatus.UNAUTHORIZED, response.getStatusCode());
    }

    @Test
    void getMe_withoutAuth_shouldReturn401() {
        @SuppressWarnings("unchecked")
        ResponseEntity<Map<String, Object>> response = restTemplate.getForEntity("/api/auth/me", (Class<Map<String, Object>>) (Class<?>) Map.class);
        assertEquals(HttpStatus.UNAUTHORIZED, response.getStatusCode());
    }

    @Test
    void logout_shouldReturn204() {
        ResponseEntity<Void> response = restTemplate.postForEntity("/api/auth/logout", null, Void.class);
        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
