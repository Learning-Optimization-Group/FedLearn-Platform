package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.TestPropertySource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(properties = {
    "spring.datasource.url=jdbc:h2:mem:authtest;DB_CLOSE_DELAY=-1",
    "spring.datasource.driver-class-name=org.h2.Driver",
    "spring.jpa.database-platform=org.hibernate.dialect.H2Dialect",
    "spring.flyway.enabled=false",
    "spring.jpa.hibernate.ddl-auto=create-drop",
    "app.jwt.secret=dGVzdHNlY3JldGtleWZvcmp3dHRlc3RpbmcxMjM0NTY3ODk=",
    "app.jwt.expiration-ms=3600000",
    "app.auth.cookie.secure=false",
    "app.auth.cookie.same-site=Lax",
    "app.internal.api-key=test-internal-key",
    "app.cors.allowed-origins=http://localhost"
})
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

        ResponseEntity<Map> response = restTemplate.postForEntity("/api/auth/register", request, Map.class);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertNotNull(response.getBody());
        assertTrue(response.getBody().containsKey("userId"));
    }

    @Test
    void register_duplicateUsername_shouldReturn409() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(registerPayload("bob2", "bob2@example.com", "Password1!"), headers);

        restTemplate.postForEntity("/api/auth/register", request, Map.class);
        ResponseEntity<Map> second = restTemplate.postForEntity("/api/auth/register", request, Map.class);

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
        ResponseEntity<Map> response = restTemplate.postForEntity("/api/auth/login", loginRequest, Map.class);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        // Cookie should be set
        assertTrue(response.getHeaders().containsKey(HttpHeaders.SET_COOKIE));
        String cookie = response.getHeaders().getFirst(HttpHeaders.SET_COOKIE);
        assertNotNull(cookie);
        assertTrue(cookie.contains("jwtToken"));
    }

    @Test
    void login_withWrongPassword_shouldReturn401() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request =
                new HttpEntity<>(Map.of("username", "nobody", "password", "wrongpass"), headers);

        ResponseEntity<Map> response = restTemplate.postForEntity("/api/auth/login", request, Map.class);

        assertEquals(HttpStatus.UNAUTHORIZED, response.getStatusCode());
    }

    @Test
    void getMe_withoutAuth_shouldReturn401() {
        ResponseEntity<Map> response = restTemplate.getForEntity("/api/auth/me", Map.class);
        assertEquals(HttpStatus.UNAUTHORIZED, response.getStatusCode());
    }

    @Test
    void logout_shouldReturn204() {
        ResponseEntity<Void> response = restTemplate.postForEntity("/api/auth/logout", null, Void.class);
        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }
}
