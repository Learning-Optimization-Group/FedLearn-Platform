package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.Test;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;

import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit test for the BA-5 wildcard-subscription hole: the SimpleBroker matches SUBSCRIBE destinations as
 * Ant patterns, so an ungated wildcard destination (e.g. {@code /topic/**}) would receive every project's
 * broadcasts across tenants. {@link StompSubscriptionInterceptor} must reject pattern/wildcard and
 * unrecognized broker destinations BEFORE they reach the broker, while leaving concrete project topics
 * (which still get the per-project authz check) and non-broker destinations (/user/**) untouched.
 */
class StompSubscriptionInterceptorWildcardTest {

    private final AuthorizationService authz = mock(AuthorizationService.class);
    private final ProjectRepository projects = mock(ProjectRepository.class);
    private final StompSubscriptionInterceptor interceptor =
            new StompSubscriptionInterceptor(authz, projects);
    private final MessageChannel channel = mock(MessageChannel.class);

    private Message<byte[]> subscribe(String destination) {
        StompHeaderAccessor accessor = StompHeaderAccessor.create(StompCommand.SUBSCRIBE);
        accessor.setDestination(destination);
        accessor.setSessionId("s1");
        Authentication auth =
                new UsernamePasswordAuthenticationToken("alice", "n/a", java.util.List.of());
        accessor.setUser(auth);
        accessor.setLeaveMutable(true);
        return MessageBuilder.createMessage(new byte[0], accessor.getMessageHeaders());
    }

    @Test
    void wildcardTopicSubscription_isRejected() {
        // The confirmed exploit: /topic/** matches every /topic/logs/<projectId> broadcast in the broker.
        assertThrows(AccessDeniedException.class, () -> interceptor.preSend(subscribe("/topic/**"), channel));
        verify(authz, never()).requireSubscribable(any());
    }

    @Test
    void multiSegmentWildcardTopic_isRejected() {
        // /topic/*/* also matches /topic/logs/<id> under AntPathMatcher.
        assertThrows(AccessDeniedException.class, () -> interceptor.preSend(subscribe("/topic/*/*"), channel));
    }

    @Test
    void queueWildcardSubscription_isRejected() {
        // Defense-in-depth: /queue/** would pattern-match the user-destination notification queues.
        assertThrows(AccessDeniedException.class, () -> interceptor.preSend(subscribe("/queue/**"), channel));
    }

    @Test
    void unrecognizedConcreteTopic_isRejected() {
        // Deny-by-default in the project broadcast namespace: an unknown /topic destination is not allowed.
        assertThrows(AccessDeniedException.class, () -> interceptor.preSend(subscribe("/topic/random"), channel));
    }

    @Test
    void concreteProjectTopic_stillReachesAuthorizationCheck() {
        // Non-regression: a concrete project topic passes the wildcard guard and hits the per-project gate.
        UUID pid = UUID.randomUUID();
        when(projects.findById(pid)).thenReturn(Optional.of(new Project()));
        assertDoesNotThrow(() -> interceptor.preSend(subscribe("/topic/logs/" + pid), channel));
        verify(authz, times(1)).requireSubscribable(any());
    }

    @Test
    void userDestination_passesThroughUngated() {
        // Non-regression: /user/** is not a broker /topic destination and must pass through untouched.
        assertDoesNotThrow(() -> interceptor.preSend(subscribe("/user/queue/notifications"), channel));
        verify(authz, never()).requireSubscribable(any());
    }
}
