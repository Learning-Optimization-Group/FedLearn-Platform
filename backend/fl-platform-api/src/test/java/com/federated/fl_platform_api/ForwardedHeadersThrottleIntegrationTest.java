package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * OP-4 regression guard: once Spring Boot sits behind nginx, every request's TCP peer is the
 * loopback proxy (127.0.0.1). If the app does not honour the {@code X-Forwarded-*} headers nginx
 * sets, {@code HttpServletRequest.getRemoteAddr()} collapses to 127.0.0.1 for every request, and the
 * SE-4 per-IP login throttle — which keys on {@code "ip:" + getRemoteAddr()} in
 * {@code AuthController} — degrades into a single global bucket: five bad logins from any source
 * lock every user out (an unauthenticated, self-sustaining auth DoS) and audit rows all record
 * 127.0.0.1.
 *
 * <p>With {@code server.forward-headers-strategy=native} the embedded Tomcat installs
 * {@code RemoteIpValve}, which restores the real client IP from {@code X-Forwarded-For} — and trusts
 * only internal/loopback proxies by default, so a client hitting {@code :8081} directly cannot spoof
 * the header. This test proves the throttle then buckets per real client: one client's lockout must
 * NOT 429 a client arriving from a different forwarded IP. Without the strategy the final assertion
 * fails (both clients collapse to {@code ip:127.0.0.1}, so the bystander inherits the 429).
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@TestPropertySource(properties = "server.forward-headers-strategy=native")
class ForwardedHeadersThrottleIntegrationTest {

    @Autowired
    private TestRestTemplate restTemplate;

    /** A wrong-password login carrying an explicit forwarded client IP (as nginx would set). */
    private HttpStatus loginStatus(String username, String forwardedFor) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.add("X-Forwarded-For", forwardedFor);
        ResponseEntity<Map> resp = restTemplate.postForEntity(
                "/api/auth/login",
                new HttpEntity<>(Map.of("username", username, "password", "wrong-password"), headers),
                Map.class);
        return HttpStatus.valueOf(resp.getStatusCode().value());
    }

    @Test
    void perIpThrottleUsesTheForwardedClientIpNotTheProxyLoopback() {
        String attacker = "203.0.113.7";    // TEST-NET-3 — a stand-in public client IP
        String bystander = "198.51.100.9";  // TEST-NET-2 — a different public client IP

        // Five failed logins from the attacker's forwarded IP. Distinct usernames each time, so only
        // the per-IP bucket fills to the threshold — the per-username buckets never lock.
        for (int i = 1; i <= 5; i++) {
            assertEquals(HttpStatus.UNAUTHORIZED, loginStatus("spray-" + i, attacker),
                    "a wrong-password attempt is a 401");
        }

        // The attacker's IP bucket is now full: a sixth attempt (a brand-new username) is throttled.
        // This can only key on the FORWARDED IP — every request's real peer is the 127.0.0.1 loopback.
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, loginStatus("spray-6", attacker),
                "the attacker's own forwarded IP is locked out after 5 failures");

        // A client arriving from a DIFFERENT forwarded IP must get the normal 401, not 429 — it must
        // not inherit the attacker's lockout. Without forwarded-header handling both collapse to
        // ip:127.0.0.1 and this is a 429: the platform-wide collateral lockout OP-4 would introduce.
        assertEquals(HttpStatus.UNAUTHORIZED, loginStatus("innocent", bystander),
                "a different real client must not inherit another client's per-IP lockout");
    }
}
