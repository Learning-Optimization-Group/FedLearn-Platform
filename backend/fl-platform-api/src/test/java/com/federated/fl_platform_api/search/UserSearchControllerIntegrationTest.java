package com.federated.fl_platform_api.search;

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
class UserSearchControllerIntegrationTest {

    @Autowired TestRestTemplate restTemplate;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private User createUser(String username) {
        return userRepository.save(new User(username, username + "@example.com",
            passwordEncoder.encode("Password1!")));
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
        h.add(HttpHeaders.COOKIE, cookie);
        return h;
    }

    @Test
    void search_prefixMatch_caseInsensitive() {
        createUser("alice_srch");
        createUser("ALICE_upper");
        createUser("bob_srch");
        String cookie = loginAs("alice_srch");

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<List> resp = restTemplate.exchange(
            "/api/users/search?q=alice", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), List.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertNotNull(resp.getBody());
        assertEquals(2, resp.getBody().size());
    }

    @Test
    void search_queryTooShort_returnsEmpty() {
        createUser("user_short");
        String cookie = loginAs("user_short");

        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<List> resp = restTemplate.exchange(
            "/api/users/search?q=u", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), List.class);
        assertEquals(HttpStatus.OK, resp.getStatusCode());
        assertNotNull(resp.getBody());
        assertTrue(resp.getBody().isEmpty());
    }

    @Test
    void search_rateLimitExceeded_returns429() {
        createUser("user_rl");
        String cookie = loginAs("user_rl");

        // 30 allowed requests
        for (int i = 0; i < 30; i++) {
            @SuppressWarnings({"unchecked", "rawtypes"})
            ResponseEntity<List> resp = restTemplate.exchange(
                "/api/users/search?q=user", HttpMethod.GET,
                new HttpEntity<>(headers(cookie)), List.class);
            assertEquals(HttpStatus.OK, resp.getStatusCode(),
                "Request " + (i + 1) + " should succeed");
        }

        // 31st must be rejected
        @SuppressWarnings({"unchecked", "rawtypes"})
        ResponseEntity<Map> limited = restTemplate.exchange(
            "/api/users/search?q=user", HttpMethod.GET,
            new HttpEntity<>(headers(cookie)), Map.class);
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, limited.getStatusCode());
    }
}
