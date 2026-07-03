package com.federated.fl_platform_api.config;

import com.federated.fl_platform_api.security.JwtChannelInterceptor;
import com.federated.fl_platform_api.security.JwtHandshakeInterceptor;
import com.federated.fl_platform_api.security.StompSubscriptionInterceptor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.lang.NonNull;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

import java.util.Arrays;
import java.util.List;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    private final JwtHandshakeInterceptor jwtHandshakeInterceptor;
    private final JwtChannelInterceptor jwtChannelInterceptor;
    private final StompSubscriptionInterceptor stompSubscriptionInterceptor;
    private final String allowedOriginsCsv;

    public WebSocketConfig(JwtHandshakeInterceptor jwtHandshakeInterceptor,
                           JwtChannelInterceptor jwtChannelInterceptor,
                           StompSubscriptionInterceptor stompSubscriptionInterceptor,
                           @Value("${app.cors.allowed-origins}") String allowedOriginsCsv) {
        this.jwtHandshakeInterceptor = jwtHandshakeInterceptor;
        this.jwtChannelInterceptor = jwtChannelInterceptor;
        this.stompSubscriptionInterceptor = stompSubscriptionInterceptor;
        this.allowedOriginsCsv = allowedOriginsCsv;
    }

    @Override
    public void configureMessageBroker(@NonNull MessageBrokerRegistry config) {
        // In-memory STOMP broker — fine for single-replica deployments.
        // For multi-instance deploys, switch this to a relay (RabbitMQ/Redis).
        // /topic — public broadcast (logs, status). /queue — user-targeted via
        // /user/{username}/queue/... resolved by Spring's user-destination prefix.
        config.enableSimpleBroker("/topic", "/queue");
        config.setApplicationDestinationPrefixes("/app");
        config.setUserDestinationPrefix("/user");
    }

    @Override
    @SuppressWarnings("null")
    public void registerStompEndpoints(@NonNull StompEndpointRegistry registry) {
        // Origins are driven from the same allowlist as the REST CORS config so
        // there is exactly one place to update when adding a new frontend host.
        // Patterns (not literal origins) so wildcards like "http://localhost:*"
        // work; matches Spring's REST CORS behaviour exactly.
        List<String> origins = Arrays.stream(allowedOriginsCsv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
        if (origins.isEmpty()) {
            throw new IllegalStateException(
                    "app.cors.allowed-origins must be set to a non-empty allowlist for STOMP");
        }

        registry.addEndpoint("/ws-logs")
                .setAllowedOriginPatterns(origins.toArray(new String[0]))
                .addInterceptors(jwtHandshakeInterceptor);
    }

    @Override
    public void configureClientInboundChannel(@NonNull ChannelRegistration registration) {
        // Order matters: jwtChannelInterceptor runs first to promote the
        // handshake-cached principal onto the STOMP session at CONNECT time (and
        // reject any unauthenticated CONNECT). stompSubscriptionInterceptor runs
        // after it so the authenticated principal is already present when it
        // authorizes each SUBSCRIBE against project membership (BA-5).
        registration.interceptors(jwtChannelInterceptor, stompSubscriptionInterceptor);
    }
}
