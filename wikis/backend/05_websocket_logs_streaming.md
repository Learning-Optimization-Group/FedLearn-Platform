# 05 - WebSocket Logs Streaming

Providing real-time observability into the Federated Learning process is critical, as ML tasks can run for long periods and edge clients need assurance that the server is actively processing their model weights.

This is achieved via a pipeline consisting of process standard output capture, STOMP WebSockets, and asynchronous database persistence.

## 1. The Pipeline Architecture

1. **Log Generation:** The Python FL Server (whether running locally or on AWS) writes structured JSON logs to its standard output.
2. **Capture:** The Spring Boot backend reads these logs. For local execution, this is handled by a daemon thread attached to the `Process` output stream. For AWS, logs are shipped via CloudWatch or internal REST callbacks.
3. **Distribution (`WebSocketService`):** The `WebSocketService.sendLogs()` method routes the raw string.
4. **WebSocket Push:** Spring's `SimpMessagingTemplate` pushes the string over an open STOMP WebSocket channel.
5. **Persistence:** The backend parses the JSON string and saves the structured data to the `server_logs` table (H2 on this branch) for permanent storage.

---

## 2. Spring WebSocket Configuration

The WebSocket endpoints are registered in `WebSocketConfig.java`:

```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // Exposes the initial HTTP endpoint for the protocol upgrade
        registry.addEndpoint("/ws-logs")
                .setAllowedOrigins("http://localhost:5173", "https://fedlearn.production.com")
                .withSockJS(); // Fallback for browsers that don't support raw WebSockets
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // STOMP clients subscribe to topics starting with /topic
        config.enableSimpleBroker("/topic");
        // STOMP clients send messages to endpoints starting with /app
        config.setApplicationDestinationPrefixes("/app");
    }
}
```

The React frontend establishes a connection to `/ws-logs`. Security interceptors (documented in [02 - Security and Auth](02_security_and_auth.md)) ensure that the token is validated before the connection upgrades from HTTP to TCP.

---

## 3. The `WebSocketService`

The `WebSocketService` handles both distribution and persistence.

### Broadcasting via STOMP
Every project has a unique, dynamically generated topic endpoint.

```java
public void sendLogs(UUID projectId, String logMessage) {
    // The destination topic is unique for each project.
    String destination = "/topic/logs/" + projectId.toString();
    messagingTemplate.convertAndSend(destination, logMessage);
    
    // Concurrently persist to DB
    persistLog(projectId, logMessage);
}
```

When the React frontend opens the dashboard for project `1234`, it sends a STOMP `SUBSCRIBE` frame to `/topic/logs/1234`. It then passively receives all strings sent to that destination.

### JSON Parsing and Persistence
The backend attempts to parse the incoming log line. If the Python server successfully emitted valid JSON, the backend extracts the `level`, `message`, and optional `stackTrace`. If the line isn't JSON (e.g., raw error output from a crash), it falls back to saving the entire line as a plain `INFO` string.

```java
private void persistLog(UUID projectId, String rawLine) {
    ServerLog log = new ServerLog();
    log.setProjectId(projectId);
    log.setTimestamp(Instant.now());

    try {
        // Attempt to parse FL server JSON: {"timestamp":"...","level":"INFO","message":"..."}
        JsonNode node = objectMapper.readTree(rawLine);
        log.setLevel(node.has("level") ? node.get("level").asText() : "INFO");
        log.setMessage(node.has("message") ? node.get("message").asText() : rawLine);
    } catch (Exception e) {
        // Fallback: Not JSON — store as a plain INFO message.
        log.setLevel("INFO");
        log.setMessage(rawLine);
    }

    serverLogRepository.save(log);
}
```

## 4. Log Export

Once a project has finished, the user can download a complete text file of all logs generated during the training session. 

Because ML processes can generate enormous amounts of logging data, the `ProjectController` enforces a hard limit `MAX_LOGS_EXPORT_SIZE = 10_000` to prevent memory exhaustion on the JVM.

The export builds a downloadable `.txt` file constructed from the persisted `server_logs` table rows, mapping the `ServerLog` entities to formatted strings.
