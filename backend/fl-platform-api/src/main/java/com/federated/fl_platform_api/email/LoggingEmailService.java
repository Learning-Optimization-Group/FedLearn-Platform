package com.federated.fl_platform_api.email;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;

public class LoggingEmailService implements EmailService {

    private static final Logger LOG = LoggerFactory.getLogger(LoggingEmailService.class);

    private final Path outDir;

    public LoggingEmailService() { this(Path.of("target", "sent-emails")); }

    public LoggingEmailService(Path outDir) { this.outDir = outDir; }

    @Override
    public void send(EmailMessage msg) {
        LOG.info("[email:logging] to={} subject=\"{}\"", msg.to(), msg.subject());
        try {
            Files.createDirectories(outDir);
            String filename = Instant.now().toEpochMilli() + "-" + UUID.randomUUID() + ".eml";
            Files.writeString(outDir.resolve(filename), render(msg));
        } catch (Exception e) {
            // Dev-only adapter; never throw.
            LOG.warn("[email:logging] failed to write .eml: {}", e.getMessage());
        }
    }

    private String render(EmailMessage m) {
        StringBuilder sb = new StringBuilder();
        sb.append("To: ").append(m.to()).append('\n');
        sb.append("Subject: ").append(m.subject()).append('\n');
        for (Map.Entry<String, String> h : m.headers().entrySet()) {
            sb.append(h.getKey()).append(": ").append(h.getValue()).append('\n');
        }
        sb.append("Content-Type: multipart/alternative; boundary=BOUNDARY\n\n");
        sb.append("--BOUNDARY\nContent-Type: text/plain; charset=UTF-8\n\n");
        sb.append(m.textBody()).append("\n\n");
        sb.append("--BOUNDARY\nContent-Type: text/html; charset=UTF-8\n\n");
        sb.append(m.htmlBody()).append("\n--BOUNDARY--\n");
        return sb.toString();
    }
}
