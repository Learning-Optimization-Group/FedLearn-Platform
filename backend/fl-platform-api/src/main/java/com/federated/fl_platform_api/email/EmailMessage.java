package com.federated.fl_platform_api.email;

import java.util.Map;

public record EmailMessage(
        String to,
        String subject,
        String htmlBody,
        String textBody,
        Map<String, String> headers
) {
    public EmailMessage {
        if (to == null || to.isBlank())      throw new IllegalArgumentException("to required");
        if (subject == null)                  throw new IllegalArgumentException("subject required");
        if (textBody == null)                 throw new IllegalArgumentException("textBody required for deliverability");
        if (headers == null)                  headers = Map.of();
    }
}
