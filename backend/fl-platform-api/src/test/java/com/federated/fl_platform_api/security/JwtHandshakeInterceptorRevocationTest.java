package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.service.CustomUserDetailsService;
import org.junit.jupiter.api.Test;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.http.server.ServletServerHttpResponse;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.socket.WebSocketHandler;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * SE-8: logout revokes a JWT's jti so every subsequent HTTP request is rejected (JwtAuthenticationFilter
 * checks {@code !isRevoked(jti)}). The WebSocket handshake must honor the same denylist — otherwise a
 * logged-out (or copied, still-unexpired) token could open a live socket after logout.
 */
class JwtHandshakeInterceptorRevocationTest {

    private final JwtTokenProvider provider = mock(JwtTokenProvider.class);
    private final CustomUserDetailsService userDetailsService = mock(CustomUserDetailsService.class);
    private final TokenRevocationService revocation = mock(TokenRevocationService.class);
    private final JwtHandshakeInterceptor interceptor =
            new JwtHandshakeInterceptor(provider, userDetailsService, revocation);

    private boolean handshake(String token, boolean revoked) {
        UserDetails ud = User.withUsername("alice").password("x").authorities("ROLE_USER").build();
        when(provider.getUsernameFromToken(token)).thenReturn("alice");
        when(userDetailsService.loadUserByUsername("alice")).thenReturn(ud);
        when(provider.validateToken(eq(token), any())).thenReturn(true);
        when(provider.getJti(token)).thenReturn("jti-1");
        when(revocation.isRevoked("jti-1")).thenReturn(revoked);

        MockHttpServletRequest req = new MockHttpServletRequest();
        req.addHeader("Authorization", "Bearer " + token);
        ServletServerHttpRequest shReq = new ServletServerHttpRequest(req);
        ServletServerHttpResponse shResp = new ServletServerHttpResponse(new MockHttpServletResponse());
        Map<String, Object> attrs = new HashMap<>();
        return interceptor.beforeHandshake(shReq, shResp, mock(WebSocketHandler.class), attrs);
    }

    @Test
    void revokedToken_isRejectedAtHandshake() {
        assertFalse(handshake("tok", true),
                "a revoked (logged-out) JWT must not be allowed to open a WebSocket");
    }

    @Test
    void validToken_isAcceptedAtHandshake() {
        assertTrue(handshake("tok", false),
                "a valid, non-revoked JWT must still complete the handshake");
    }
}
