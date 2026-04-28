package com.federated.fl_platform_api.config;

import com.federated.fl_platform_api.security.JwtChannelInterceptor;
import com.federated.fl_platform_api.security.JwtHandshakeInterceptor;
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
    private final String allowedOriginsCsv;

    public WebSocketConfig(JwtHandshakeInterceptor jwtHandshakeInterceptor,
                           JwtChannelInterceptor jwtChannelInterceptor,
                           @Value("${app.cors.allowed-origins}") String allowedOriginsCsv) {
        this.jwtHandshakeInterceptor = jwtHandshakeInterceptor;
        this.jwtChannelInterceptor = jwtChannelInterceptor;
        this.allowedOriginsCsv = allowedOriginsCsv;
    }

    @Override
    public void configureMessageBroker(@NonNull MessageBrokerRegistry config) {
        // In-memory STOMP broker — fine for single-replica deployments.
        // For multi-instance deploys, switch this to a relay (RabbitMQ/Redis).
        config.enableSimpleBroker("/topic");
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
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
        // Promote the handshake-cached principal onto the STOMP session at
        // CONNECT time, and reject any unauthenticated CONNECT as a backstop.
        registration.interceptors(jwtChannelInterceptor);
    }
}
