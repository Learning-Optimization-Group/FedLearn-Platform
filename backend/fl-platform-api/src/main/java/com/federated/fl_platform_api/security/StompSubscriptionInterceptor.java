package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.lang.NonNull;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

import java.security.Principal;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Per-destination authorization for STOMP {@code SUBSCRIBE} frames (BA-5).
 *
 * <p>The handshake ({@link JwtHandshakeInterceptor}) and CONNECT
 * ({@link JwtChannelInterceptor}) layers only establish <em>who</em> the caller
 * is. Without this interceptor any authenticated user could SUBSCRIBE to any
 * project's {@code /topic/logs|status|results|inference/{projectId}} stream —
 * across tenants — and receive its broadcasts. This gate closes that hole by
 * running the same org-scope + participant check the REST read path enforces
 * before the SUBSCRIBE is delivered to the broker.
 *
 * <p>Registered <em>after</em> {@link JwtChannelInterceptor} in
 * {@code WebSocketConfig.configureClientInboundChannel} so the session principal
 * is already set. Non-project destinations (e.g. {@code /user/**}, app
 * destinations) pass through untouched.
 */
@Component
public class StompSubscriptionInterceptor implements ChannelInterceptor {

    private static final Logger log = LoggerFactory.getLogger(StompSubscriptionInterceptor.class);

    /**
     * Project-scoped topic prefixes. The trailing segment after each prefix is the
     * project id. Mirrors every {@code /topic/**} destination published by
     * {@code WebSocketService} (logs / status / results / inference).
     */
    private static final List<String> PROJECT_TOPIC_PREFIXES = List.of(
            "/topic/logs/",
            "/topic/status/",
            "/topic/results/",
            "/topic/inference/");

    private final AuthorizationService authorizationService;
    private final ProjectRepository projectRepository;

    public StompSubscriptionInterceptor(AuthorizationService authorizationService,
                                        ProjectRepository projectRepository) {
        this.authorizationService = authorizationService;
        this.projectRepository = projectRepository;
    }

    @Override
    public Message<?> preSend(@NonNull Message<?> message, @NonNull MessageChannel channel) {
        StompHeaderAccessor accessor =
                MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);
        if (accessor == null || !StompCommand.SUBSCRIBE.equals(accessor.getCommand())) {
            return message; // only SUBSCRIBE frames are gated
        }

        String destination = accessor.getDestination();
        UUID projectId = projectIdFor(destination);
        if (projectId == null) {
            // Not a project-scoped topic (/user/**, app destinations, unknown
            // topics) — leave delivery unchanged.
            return message;
        }

        // A malformed / non-existent project topic is rejected rather than left
        // open. Deny without leaking existence.
        Optional<Project> project = projectRepository.findById(projectId);
        if (project.isEmpty()) {
            log.info("STOMP SUBSCRIBE rejected: unknown project for destination {}", destination);
            throw new AccessDeniedException("You do not have access to this project");
        }

        Authentication auth = principalAuthentication(accessor.getUser());
        if (auth == null) {
            // Should be unreachable: JwtChannelInterceptor rejects unauthenticated
            // CONNECT. Defensive backstop.
            throw new AccessDeniedException("Unauthenticated STOMP subscription");
        }

        // The authorization helpers read the SecurityContext; the STOMP inbound
        // channel runs off the servlet request thread, so bind the session
        // principal for the duration of the check and always clear it after.
        SecurityContext previous = SecurityContextHolder.getContext();
        try {
            SecurityContext ctx = SecurityContextHolder.createEmptyContext();
            ctx.setAuthentication(auth);
            SecurityContextHolder.setContext(ctx);
            authorizationService.requireSubscribable(project.get());
        } finally {
            SecurityContextHolder.setContext(previous);
        }
        return message;
    }

    /**
     * The project id encoded in a project-scoped topic destination, or {@code null}
     * when {@code destination} is not one (those pass through the gate unchanged).
     * A destination that carries a project-topic prefix but an empty or unparseable
     * id is rejected with {@link AccessDeniedException} rather than left open.
     */
    private static UUID projectIdFor(String destination) {
        if (destination == null) {
            return null;
        }
        for (String prefix : PROJECT_TOPIC_PREFIXES) {
            if (destination.startsWith(prefix)) {
                String tail = destination.substring(prefix.length());
                if (tail.isEmpty() || tail.indexOf('/') >= 0) {
                    // e.g. /topic/logs/  or  /topic/logs/<id>/extra — reject.
                    throw new AccessDeniedException("Malformed project topic destination");
                }
                try {
                    return UUID.fromString(tail);
                } catch (IllegalArgumentException ex) {
                    throw new AccessDeniedException("Malformed project topic destination");
                }
            }
        }
        return null;
    }

    private static Authentication principalAuthentication(Principal principal) {
        return principal instanceof Authentication auth ? auth : null;
    }
}
