package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.service.CustomUserDetailsService;
import jakarta.servlet.http.Cookie;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.http.server.ServletServerHttpResponse;
import org.springframework.lang.NonNull;
import org.springframework.lang.Nullable;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Map;

/**
 * Validates the JWT on the WebSocket upgrade (HTTP) request, before the
 * STOMP layer is even reachable. Rejecting at handshake time means an
 * unauthenticated client can never open a socket — there is no "subscribe
 * then we'll check" window.
 *
 * Reads the token from (in priority order):
 *   1. {@code Authorization: Bearer <jwt>} header
 *   2. {@code jwtToken} HttpOnly cookie (browser path; auto-sent by the
 *      browser on the upgrade request when the WS is same-origin)
 *
 * On success, stores the authenticated principal in the WebSocket session
 * attributes under {@link #PRINCIPAL_ATTR} so {@link JwtChannelInterceptor}
 * can promote it to the STOMP session principal at CONNECT time.
 */
@Component
public class JwtHandshakeInterceptor implements HandshakeInterceptor {

    public static final String PRINCIPAL_ATTR = "jwt.principal";

    private static final Logger log = LoggerFactory.getLogger(JwtHandshakeInterceptor.class);

    private final JwtTokenProvider jwtTokenProvider;
    private final CustomUserDetailsService userDetailsService;
    private final TokenRevocationService tokenRevocationService;

    public JwtHandshakeInterceptor(JwtTokenProvider jwtTokenProvider,
                                   CustomUserDetailsService userDetailsService,
                                   TokenRevocationService tokenRevocationService) {
        this.jwtTokenProvider = jwtTokenProvider;
        this.userDetailsService = userDetailsService;
        this.tokenRevocationService = tokenRevocationService;
    }

    @Override
    public boolean beforeHandshake(@NonNull ServerHttpRequest request,
                                   @NonNull ServerHttpResponse response,
                                   @NonNull WebSocketHandler wsHandler,
                                   @NonNull Map<String, Object> attributes) {

        String token = extractToken(request);
        if (token == null) {
            log.info("STOMP handshake rejected: no JWT presented");
            reject(response);
            return false;
        }

        try {
            String username = jwtTokenProvider.getUsernameFromToken(token);
            if (username == null) {
                reject(response);
                return false;
            }
            UserDetails userDetails = userDetailsService.loadUserByUsername(username);
            if (!jwtTokenProvider.validateToken(token, userDetails)) {
                log.info("STOMP handshake rejected: token validation failed");
                reject(response);
                return false;
            }
            // SE-8: honor the logout denylist here too. The HTTP path (JwtAuthenticationFilter) rejects a
            // revoked jti; without this check a logged-out (but not-yet-expired) token could still open a
            // live WebSocket, an inconsistency across the two authenticated surfaces.
            if (tokenRevocationService.isRevoked(jwtTokenProvider.getJti(token))) {
                log.info("STOMP handshake rejected: token revoked (logged out)");
                reject(response);
                return false;
            }

            UsernamePasswordAuthenticationToken auth = new UsernamePasswordAuthenticationToken(
                    userDetails, null, userDetails.getAuthorities());
            attributes.put(PRINCIPAL_ATTR, auth);
            return true;
        } catch (RuntimeException e) {
            // Never log the token itself; class name is enough to diagnose
            // (ExpiredJwtException, MalformedJwtException, etc.).
            log.info("STOMP handshake rejected: {}", e.getClass().getSimpleName());
            reject(response);
            return false;
        }
    }

    @Override
    public void afterHandshake(@NonNull ServerHttpRequest request,
                               @NonNull ServerHttpResponse response,
                               @NonNull WebSocketHandler wsHandler,
                               @Nullable Exception exception) {
        // No-op.
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private static String extractToken(ServerHttpRequest request) {
        // Authorization header takes priority over cookie. Both are accepted
        // so that programmatic STOMP clients (which don't speak cookies) and
        // browser clients (which can't read the HttpOnly cookie) both work.
        String authHeader = request.getHeaders().getFirst("Authorization");
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            return authHeader.substring(7);
        }
        if (request instanceof ServletServerHttpRequest servletRequest) {
            Cookie[] cookies = servletRequest.getServletRequest().getCookies();
            if (cookies != null) {
                for (Cookie c : cookies) {
                    if ("jwtToken".equals(c.getName())) {
                        return c.getValue();
                    }
                }
            }
        }
        return null;
    }

    private static void reject(ServerHttpResponse response) {
        if (response instanceof ServletServerHttpResponse servletResponse) {
            servletResponse.getServletResponse().setStatus(401);
        }
    }
}
