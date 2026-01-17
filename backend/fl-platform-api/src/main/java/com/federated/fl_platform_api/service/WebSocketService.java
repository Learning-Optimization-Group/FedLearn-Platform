package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.ProjectStatusUpdateDto;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.util.UUID;

@Service
public class WebSocketService {

    // Spring automatically provides this template for sending messages.
    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    /**
     * Sends a log message to a project-specific WebSocket topic.
     * @param projectId The ID of the project the log belongs to.
     * @param logMessage The log message string to send.
     */
    public void sendLogs(UUID projectId, String logMessage) {
        if (projectId == null || logMessage == null) {
            return;
        }
        // The destination topic is unique for each project.
        // The front-end will subscribe to this exact path.
        String destination = "/topic/logs/" + projectId.toString();
        messagingTemplate.convertAndSend(destination, logMessage);
    }

    public void sendStatusUpdate(ProjectStatusUpdateDto statusUpdate) {
        UUID projectId = statusUpdate.getProjectId();
        String destination = "/topic/status/" + projectId.toString();
        messagingTemplate.convertAndSend(destination, statusUpdate);
    }
}