package com.federated.fl_platform_api.email;

import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.Test;
import org.springframework.mail.MailSendException;
import org.springframework.mail.javamail.JavaMailSender;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SmtpEmailServiceTest {

    @Test
    void calls_javamailsender_with_mapped_fields() throws Exception {
        JavaMailSender sender = mock(JavaMailSender.class);
        MimeMessage mime = mock(MimeMessage.class);
        when(sender.createMimeMessage()).thenReturn(mime);

        SmtpEmailService svc = new SmtpEmailService(sender, "noreply@fedlearn.io");
        svc.send(new EmailMessage("a@b.com", "Hi", "<p>html</p>", "text", Map.of()));

        verify(sender).send(any(MimeMessage.class));
    }

    @Test
    void wraps_transport_failure_in_email_delivery_exception() {
        JavaMailSender sender = mock(JavaMailSender.class);
        when(sender.createMimeMessage()).thenReturn(mock(MimeMessage.class));
        doThrow(new MailSendException("boom")).when(sender).send(any(MimeMessage.class));

        SmtpEmailService svc = new SmtpEmailService(sender, "noreply@fedlearn.io");
        assertThatThrownBy(() ->
                svc.send(new EmailMessage("a@b.com", "Hi", "<p>html</p>", "text", Map.of())))
                .isInstanceOf(EmailDeliveryException.class);
    }
}
