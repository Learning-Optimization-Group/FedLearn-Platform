package com.federated.fl_platform_api;

import com.federated.fl_platform_api.exception.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.context.annotation.Import;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

// A tiny controller that throws each exception on demand
@RestController
class ExceptionTestController {
    @GetMapping("/test/not-found")
    public void notFound() { throw ResourceNotFoundException.project(UUID.randomUUID()); }

    @GetMapping("/test/conflict-user")
    public void conflictUser() { throw new UserAlreadyExistsException("taken"); }

    @GetMapping("/test/project-state")
    public void projectState() { throw new ProjectStateException("already running"); }

    @GetMapping("/test/server-process")
    public void serverProcess() { throw new ServerProcessException("process failed"); }

    @GetMapping("/test/illegal-arg")
    public void illegalArg() { throw new IllegalArgumentException("bad input"); }
}

@WebMvcTest(controllers = ExceptionTestController.class)
@Import({GlobalExceptionHandler.class, com.federated.fl_platform_api.config.SecurityConfig.class})
@TestPropertySource(properties = {
    "app.cors.allowed-origins=http://localhost",
    "app.jwt.secret=dGVzdHNlY3JldGtleWZvcmp3dHRlc3RpbmcxMjM0NTY3ODk=",
    "app.jwt.expiration-ms=3600000",
    "app.internal.api-key=test-internal-key"
})
class GlobalExceptionHandlerTest {

    @Autowired
    private MockMvc mockMvc;

    // SecurityConfig needs these beans
    @MockitoBean
    private com.federated.fl_platform_api.service.CustomUserDetailsService customUserDetailsService;
    @MockitoBean
    private com.federated.fl_platform_api.security.JwtTokenProvider jwtTokenProvider;

    @MockitoBean
    private com.federated.fl_platform_api.security.TokenRevocationService tokenRevocationService;
    // SecurityConfig's @Bean wiring for the auditing handlers transitively requires these.
    @MockitoBean
    private com.federated.fl_platform_api.repository.UserRepository userRepository;
    @MockitoBean
    private com.federated.fl_platform_api.repository.AuditEventRepository auditEventRepository;
    // SecurityConfig now wires OrgScopeFilter, whose constructor needs these
    // (the JPA repo + request-scoped OrgScope aren't present in the MVC slice).
    @MockitoBean
    private com.federated.fl_platform_api.repository.OrganizationMembershipRepository organizationMembershipRepository;
    @MockitoBean
    private com.federated.fl_platform_api.security.OrgScope orgScope;

    @Test
    @WithMockUser
    void resourceNotFound_shouldReturn404() throws Exception {
        mockMvc.perform(get("/test/not-found"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }

    @Test
    @WithMockUser
    void userAlreadyExists_shouldReturn409() throws Exception {
        mockMvc.perform(get("/test/conflict-user"))
                .andExpect(status().isConflict());
    }

    @Test
    @WithMockUser
    void projectStateException_shouldReturn409() throws Exception {
        mockMvc.perform(get("/test/project-state"))
                .andExpect(status().isConflict());
    }

    @Test
    @WithMockUser
    void serverProcessException_shouldReturn502() throws Exception {
        mockMvc.perform(get("/test/server-process"))
                .andExpect(status().isBadGateway())
                .andExpect(jsonPath("$.correlationId").exists());
    }

    @Test
    @WithMockUser
    void illegalArgument_shouldReturn400() throws Exception {
        mockMvc.perform(get("/test/illegal-arg"))
                .andExpect(status().isBadRequest());
    }
}
