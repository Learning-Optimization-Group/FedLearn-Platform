# 05 - WebSocket Logs Streaming

Providing real-time observability into the Federated Learning process is critical, as ML tasks can run for long periods and edge clients need assurance that the server is actively processing their model weights.

This is achieved via a pipeline consisting of process standard output capture, STOMP WebSockets, and asynchronous database persistence.

## 1. The Pipeline Architecture

1. **Log Generation:** The Python FL Server — always a **local OS process** on the API host (the only supported orchestration mode; see [04 - Federated Orchestration](04_federated_orchestration.md)) — writes structured JSON logs to its standard output.
2. **Capture:** The Spring Boot backend reads these logs via a daemon thread attached to the process's merged stdout/stderr stream.
3. **Distribution (`WebSocketService`):** The `WebSocketService.sendLogs()` method routes the raw string.
4. **WebSocket Push:** Spring's `SimpMessagingTemplate` pushes the string over an open STOMP WebSocket channel.
5. **Persistence:** The backend parses the JSON string and saves the structured data to the `server_logs` table (PostgreSQL — H2 has been retired) for permanent storage. Persistence is gated by `feature.log-persistence.enabled` (default `true`); broadcasting always happens.

---

## 2. Spring WebSocket Configuration

The WebSocket endpoints are registered in `WebSocketConfig.java`:

```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // Origins are driven from the same allowlist as the REST CORS config
        // (app.cors.allowed-origins, CSV) so there is exactly one place to update
        // when adding a frontend host. Patterns — not literal origins — so
        // wildcards like "http://localhost:*" work. Boot fails if the list is empty.
        registry.addEndpoint("/ws-logs")
                .setAllowedOriginPatterns(origins.toArray(new String[0]))
                .addInterceptors(jwtHandshakeInterceptor);
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // /topic — public broadcast (logs, status).
        // /queue — user-targeted via /user/{username}/queue/...
        config.enableSimpleBroker("/topic", "/queue");
        config.setApplicationDestinationPrefixes("/app");
        config.setUserDestinationPrefix("/user");
    }

    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        // Order matters: jwtChannelInterceptor promotes the handshake-cached
        // principal onto the STOMP session at CONNECT (rejecting unauthenticated
        // CONNECTs); stompSubscriptionInterceptor then authorizes each SUBSCRIBE
        // against project membership (BA-5).
        registration.interceptors(jwtChannelInterceptor, stompSubscriptionInterceptor);
    }
}
```

The React frontend establishes a connection to `/ws-logs` using `@stomp/stompjs` over a raw WebSocket — there is **no SockJS fallback** on either side. Security interceptors (documented in [02 - Security and Auth](02_security_and_auth.md)) validate the token during the handshake, before the connection upgrades from HTTP, and again at CONNECT and SUBSCRIBE time.

---

## 3. The `WebSocketService`

The `WebSocketService` handles both distribution and persistence.

### Broadcasting via STOMP
Every project has a unique, dynamically generated topic endpoint.

```java
public void sendLogs(UUID projectId, String logMessage) {
    if (projectId == null || logMessage == null) {
        return;
    }
    // The destination topic is unique for each project.
    String destination = "/topic/logs/" + projectId.toString();
    messagingTemplate.convertAndSend(destination, logMessage);

    // Then persist to the DB — feature-gated by feature.log-persistence.enabled.
    if (logPersistenceEnabled) {
        persistLog(projectId, logMessage);
    }
}
```

`WebSocketService` carries sibling broadcasters on the same pattern: `sendStatusUpdate(...)`
→ `/topic/status/{projectId}`, `sendResultUpdate(...)` → `/topic/results/{projectId}`, and
`sendInferenceToken(...)` → `/topic/inference/{projectId}`.

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

Because ML processes can generate enormous amounts of logging data, the export is capped server-side at `ProjectService.MAX_LOGS_EXPORT_SIZE = 10_000` — applied as a `PageRequest` bound on the query — to prevent memory exhaustion on the JVM.

The export builds a downloadable `.txt` file constructed from the persisted `server_logs` table rows, mapping the `ServerLog` entities to formatted strings.
