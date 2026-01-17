package com.federated.fl_platform_api.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // This sets up a simple in-memory message broker.
        // The broker will send messages to clients who are subscribed to destinations
        // starting with "/topic".
        config.enableSimpleBroker("/topic");

        // This defines the prefix for messages sent from clients TO the server.
        // We won't use this for logging, but it's good practice to define.
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // This is the HTTP endpoint that clients will connect to to upgrade to a WebSocket connection.
        // We allow all origins for local development. In production, you'd restrict this.
        registry.addEndpoint("/ws-logs")
                .setAllowedOrigins(
                        "http://localhost:5173",
                        "http://127.0.0.1:5173",
                        "https://federated-learning-platform-ui.vercel.app",
                        "https://zo-sl.vercel.app",// Your production URL
                        "https://adbd9c9fe9a7.ngrok-free.app"
                ); // Add your React dev server URL
//                .withSockJS();
    }
}