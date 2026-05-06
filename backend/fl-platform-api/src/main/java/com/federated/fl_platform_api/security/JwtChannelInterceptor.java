package com.federated.fl_platform_api.security;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.lang.NonNull;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Promotes the principal stashed by {@link JwtHandshakeInterceptor} (or
 * extracted from a STOMP {@code Authorization} CONNECT header) onto the
 * STOMP session, so {@code @MessageMapping} handlers and topic
 * subscriptions can authorize against the authenticated user.
 *
 * Also re-validates on CONNECT as defence-in-depth: if the handshake
 * interceptor was bypassed for any reason (misconfiguration), an unsigned
 * CONNECT will still be rejected here.
 */
@Component
public class JwtChannelInterceptor implements ChannelInterceptor {

    private static final Logger log = LoggerFactory.getLogger(JwtChannelInterceptor.class);

    private final JwtTokenProvider jwtTokenProvider;
    private final UserDetailsService userDetailsService;

    public JwtChannelInterceptor(JwtTokenProvider jwtTokenProvider,
                                 UserDetailsService userDetailsService) {
        this.jwtTokenProvider = jwtTokenProvider;
        this.userDetailsService = userDetailsService;
    }

    @Override
    public Message<?> preSend(@NonNull Message<?> message, @NonNull MessageChannel channel) {
        StompHeaderAccessor accessor =
                MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);
        if (accessor == null || !StompCommand.CONNECT.equals(accessor.getCommand())) {
            return message;
        }

        // 1. Try the principal cached by the handshake interceptor.
        UsernamePasswordAuthenticationToken handshakePrincipal = null;
        Map<String, Object> sessionAttributes = accessor.getSessionAttributes();
        if (sessionAttributes != null) {
            Object cached = sessionAttributes.get(JwtHandshakeInterceptor.PRINCIPAL_ATTR);
            if (cached instanceof UsernamePasswordAuthenticationToken auth) {
                handshakePrincipal = auth;
            }
        }
        if (handshakePrincipal != null) {
            accessor.setUser(handshakePrincipal);
            return message;
        }

        // 2. Fall back to STOMP-level Authorization header (programmatic clients
        //    that connect without a cookie or browser handshake).
        String token = stompAuthHeader(accessor);
        if (token == null) {
            log.info("STOMP CONNECT rejected: no JWT in handshake or CONNECT headers");
            throw new org.springframework.security.core.AuthenticationException("Missing JWT") {};
        }
        try {
            String username = jwtTokenProvider.getUsernameFromToken(token);
            UserDetails userDetails = userDetailsService.loadUserByUsername(username);
            if (!jwtTokenProvider.validateToken(token, userDetails)) {
                log.info("STOMP CONNECT rejected: token validation failed");
                throw new org.springframework.security.core.AuthenticationException("Invalid JWT") {};
            }
            UsernamePasswordAuthenticationToken auth = new UsernamePasswordAuthenticationToken(
                    userDetails, null, userDetails.getAuthorities());
            accessor.setUser(auth);
        } catch (RuntimeException e) {
            log.info("STOMP CONNECT rejected: {}", e.getClass().getSimpleName());
            throw e;
        }
        return message;
    }

    private static String stompAuthHeader(StompHeaderAccessor accessor) {
        List<String> values = accessor.getNativeHeader("Authorization");
        if (values == null || values.isEmpty()) {
            return null;
        }
        String first = values.get(0);
        return first != null && first.startsWith("Bearer ") ? first.substring(7) : null;
    }
}
