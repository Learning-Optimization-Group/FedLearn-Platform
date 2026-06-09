package com.federated.fl_platform_api.email;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@ActiveProfiles("test")
@TestPropertySource(properties = "app.email.provider=")
class EmailConfigProfileTest {

    @Autowired EmailService bean;

    @Test
    void selects_logging_adapter_when_provider_unset() {
        assertThat(bean).isInstanceOf(LoggingEmailService.class);
    }
}
