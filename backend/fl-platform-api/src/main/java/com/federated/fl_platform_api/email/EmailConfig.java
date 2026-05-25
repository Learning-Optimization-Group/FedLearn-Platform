package com.federated.fl_platform_api.email;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.mail.javamail.JavaMailSender;

@Configuration
public class EmailConfig {

    @Bean
    @ConditionalOnProperty(name = "app.email.provider", havingValue = "smtp")
    public EmailService smtpEmailService(
            JavaMailSender sender,
            @Value("${app.email.from:noreply@fedlearn.io}") String from) {
        return new SmtpEmailService(sender, from);
    }

    @Bean
    @ConditionalOnMissingBean(EmailService.class)
    public EmailService loggingEmailService() {
        return new LoggingEmailService();
    }
}
