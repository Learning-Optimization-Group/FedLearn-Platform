package com.federated.fl_platform_api.email;

import jakarta.mail.internet.MimeMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.mail.MailException;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;

import java.nio.charset.StandardCharsets;
import java.util.Map;

public class SmtpEmailService implements EmailService {

    private static final Logger LOG = LoggerFactory.getLogger(SmtpEmailService.class);

    private final JavaMailSender sender;
    private final String fromAddress;

    public SmtpEmailService(JavaMailSender sender, String fromAddress) {
        this.sender = sender;
        this.fromAddress = fromAddress;
    }

    @Override
    public void send(EmailMessage msg) {
        try {
            MimeMessage mime = sender.createMimeMessage();
            MimeMessageHelper helper =
                    new MimeMessageHelper(mime, true, StandardCharsets.UTF_8.name());
            helper.setFrom(fromAddress);
            helper.setTo(msg.to());
            helper.setSubject(msg.subject());
            helper.setText(msg.textBody(), msg.htmlBody());
            for (Map.Entry<String, String> h : msg.headers().entrySet()) {
                mime.setHeader(h.getKey(), h.getValue());
            }
            sender.send(mime);
        } catch (MailException | jakarta.mail.MessagingException e) {
            LOG.error("[email:smtp] delivery failed to={}", msg.to(), e);
            throw new EmailDeliveryException("SMTP delivery failed", e);
        }
    }
}
