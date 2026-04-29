package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ProjectStatusUpdateDto;
import com.federated.fl_platform_api.model.ServerLog;
import com.federated.fl_platform_api.repository.ServerLogRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.UUID;

@Service
public class WebSocketService {

    private static final Logger log = LoggerFactory.getLogger(WebSocketService.class);

    // Spring automatically provides this template for sending messages.
    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    @Autowired
    private ServerLogRepository serverLogRepository;

    @org.springframework.beans.factory.annotation.Value("${feature.log-persistence.enabled:true}")
    private boolean logPersistenceEnabled;

    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Sends a log message to a project-specific WebSocket topic AND persists it
     * to the server_logs table for later retrieval / export.
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

        // Persist the log line to the database.
        // The FL server emits JSON-formatted log lines; extract level/message
        // from the JSON structure when possible, falling back to the raw line.
        if (logPersistenceEnabled) {
            persistLog(projectId, logMessage);
        }
    }

    public void sendStatusUpdate(ProjectStatusUpdateDto statusUpdate) {
        UUID projectId = statusUpdate.getProjectId();
        String destination = "/topic/status/" + projectId.toString();
        messagingTemplate.convertAndSend(destination, statusUpdate);
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    private void persistLog(UUID projectId, String rawLine) {
        ServerLog log = new ServerLog();
        log.setProjectId(projectId);
        log.setTimestamp(Instant.now());

        try {
            // FL server emits JSON: {"timestamp":"...","level":"INFO","message":"..."}
            JsonNode node = objectMapper.readTree(rawLine);
            log.setLevel(node.has("level") ? node.get("level").asText() : "INFO");
            log.setMessage(node.has("message") ? node.get("message").asText() : rawLine);
            if (node.has("stackTrace")) {
                log.setStackTrace(node.get("stackTrace").asText());
            }
        } catch (Exception e) {
            // Not JSON — store as a plain INFO message.
            log.setLevel("INFO");
            log.setMessage(rawLine);
        }

        try {
            serverLogRepository.save(log);
        } catch (RuntimeException e) {
            // Don't let a DB failure break the log stream — but make it visible to operators.
            WebSocketService.log.warn("Failed to persist FL log for project {}: {}",
                    projectId, e.getClass().getSimpleName());
        }
    }
}