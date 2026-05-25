package com.federated.fl_platform_api.email;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class LoggingEmailServiceTest {

    @Test
    void writes_eml_file_with_subject_and_recipient(@TempDir Path tmp) throws Exception {
        LoggingEmailService svc = new LoggingEmailService(tmp);

        svc.send(new EmailMessage(
                "alice@example.com",
                "Welcome to FedLearn",
                "<p>Hello</p>",
                "Hello",
                Map.of("X-FedLearn-Category", "invitation")));

        List<Path> emls = Files.list(tmp)
                .filter(p -> p.toString().endsWith(".eml"))
                .toList();
        assertThat(emls).hasSize(1);

        String content = Files.readString(emls.get(0));
        assertThat(content).contains("To: alice@example.com");
        assertThat(content).contains("Subject: Welcome to FedLearn");
        assertThat(content).contains("X-FedLearn-Category: invitation");
        assertThat(content).contains("Hello");
        assertThat(content).contains("<p>Hello</p>");
    }

    @Test
    void never_throws_on_send() {
        LoggingEmailService svc = new LoggingEmailService(Path.of("/this/path/does/not/exist/x"));
        // Even with an unwritable dir, sending must not throw.
        svc.send(new EmailMessage("x@x", "s", "h", "t", Map.of()));
    }
}
