package com.federated.fl_platform_api.notification;

import com.federated.fl_platform_api.dto.NotificationDto;
import com.federated.fl_platform_api.service.WebSocketService;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.model.User;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.verify;

@SpringBootTest
@ActiveProfiles("test")
@Transactional
class StompNotificationIntegrationTest {

    @Autowired WebSocketService webSocketService;
    @Autowired UserRepository userRepository;
    @MockBean SimpMessagingTemplate messagingTemplate;

    @Test
    void sendUserNotification_resolvesUsernameAndSendsToUserDestination() {
        User u = userRepository.save(new User("frank", "frank@example.com", "hash"));

        NotificationDto payload = new NotificationDto();
        payload.setType(NotificationDto.Type.ACCESS_REQUEST_CREATED);
        payload.setProjectId(UUID.randomUUID());
        payload.setProjectName("demo");

        webSocketService.sendUserNotification(u.getId(), payload);

        ArgumentCaptor<String> user = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> dest = ArgumentCaptor.forClass(String.class);
        verify(messagingTemplate).convertAndSendToUser(user.capture(), dest.capture(), org.mockito.ArgumentMatchers.eq(payload));

        assertEquals("frank", user.getValue());
        assertEquals("/queue/notifications", dest.getValue());
    }

    @Test
    void sendUserNotification_unknownUser_isDropped() {
        NotificationDto payload = new NotificationDto();
        payload.setType(NotificationDto.Type.MEMBERSHIP_ADDED);
        webSocketService.sendUserNotification(999_999_999L, payload);
        org.mockito.Mockito.verifyNoInteractions(messagingTemplate);
    }
}
