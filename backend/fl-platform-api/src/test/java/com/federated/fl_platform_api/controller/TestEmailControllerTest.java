package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.email.EmailMessage;
import com.federated.fl_platform_api.email.EmailService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@TestPropertySource(properties = "app.email.test-endpoint.enabled=true")
class TestEmailControllerTest {

    @Autowired MockMvc mvc;
    @MockitoBean EmailService emailService;

    @Test
    @WithMockUser(roles = "PLATFORM_ADMIN")
    void admin_can_trigger_test_email() throws Exception {
        mvc.perform(post("/api/admin/test-email")
                        .param("to", "ops@fedlearn.io"))
                .andExpect(status().isNoContent());
        verify(emailService).send(any(EmailMessage.class));
    }

    @Test
    @WithMockUser(roles = "USER")
    void regular_user_is_forbidden() throws Exception {
        mvc.perform(post("/api/admin/test-email")
                        .param("to", "x@y.com"))
                .andExpect(status().isForbidden());
    }
}
