package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.email.EmailMessage;
import com.federated.fl_platform_api.email.EmailService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Smoke endpoint that triggers a single test email through the configured {@link EmailService}.
 *
 * <p>Flag-gated by {@code app.email.test-endpoint.enabled}; the bean is absent unless the flag is
 * {@code true}, so the endpoint disappears entirely in default deployments. Restricted to
 * {@code PLATFORM_ADMIN} authorities (which Spring resolves as {@code ROLE_PLATFORM_ADMIN} via
 * {@link org.springframework.security.access.prepost.PreAuthorize#value() hasRole}).
 */
@RestController
@RequestMapping("/api/admin")
@ConditionalOnProperty(name = "app.email.test-endpoint.enabled", havingValue = "true")
public class TestEmailController {

    private final EmailService email;

    public TestEmailController(EmailService email) {
        this.email = email;
    }

    @PostMapping("/test-email")
    @PreAuthorize("hasRole('PLATFORM_ADMIN')")
    public ResponseEntity<Void> send(@RequestParam String to) {
        email.send(new EmailMessage(
                to,
                "FedLearn email smoke test",
                "<p>If you can read this, email delivery is working.</p>",
                "If you can read this, email delivery is working.",
                Map.of("X-FedLearn-Category", "smoke-test")));
        return ResponseEntity.noContent().build();
    }
}
