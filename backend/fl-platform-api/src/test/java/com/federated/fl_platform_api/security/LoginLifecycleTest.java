package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Verifies the side effects of login attempts:
 * <ul>
 *     <li>Successful login updates {@code users.last_login_at} and writes a
 *         {@link AuditAction#USER_LOGIN_SUCCEEDED} audit row.</li>
 *     <li>Login by a non-ACTIVE user is rejected with 401 (via the
 *         {@code DisabledException} thrown in {@code CustomUserDetailsService})
 *         and writes a {@link AuditAction#USER_LOGIN_FAILED} audit row.</li>
 * </ul>
 *
 * The audit row for a failed login deliberately stores only the submitted
 * username (target_type=USERNAME, target_id=&lt;username&gt;) — never the
 * user id — to avoid leaking account existence via a timing oracle.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
class LoginLifecycleTest {

    @Autowired MockMvc mvc;
    @Autowired UserRepository users;
    @Autowired AuditEventRepository audits;
    @Autowired PasswordEncoder encoder;

    @BeforeEach
    void clearAudits() {
        audits.deleteAll();
    }

    @Test
    void successful_login_updates_last_login_at_and_emits_audit_event() throws Exception {
        User u = newUser("alice", "secret");
        users.saveAndFlush(u);

        assertThat(u.getLastLoginAt()).isNull();

        mvc.perform(post("/api/auth/login")
                        .contentType("application/json")
                        .content("{\"username\":\"alice\",\"password\":\"secret\"}"))
                .andExpect(status().isOk());

        User after = users.findByUsername("alice").orElseThrow();
        assertThat(after.getLastLoginAt()).isNotNull();
        assertThat(audits.findAll())
                .anyMatch(e -> e.getAction() == AuditAction.USER_LOGIN_SUCCEEDED
                        && "USER".equals(e.getTargetType())
                        && after.getId().toString().equals(e.getTargetId()));
    }

    @Test
    void suspended_user_gets_401_and_login_failed_audit() throws Exception {
        User u = newUser("bob", "secret");
        u.setStatus(UserStatus.SUSPENDED);
        users.saveAndFlush(u);

        mvc.perform(post("/api/auth/login")
                        .contentType("application/json")
                        .content("{\"username\":\"bob\",\"password\":\"secret\"}"))
                .andExpect(status().isUnauthorized());

        assertThat(audits.findAll())
                .anyMatch(e -> e.getAction() == AuditAction.USER_LOGIN_FAILED
                        && "USERNAME".equals(e.getTargetType())
                        && "bob".equals(e.getTargetId()));
    }

    private User newUser(String username, String password) {
        // Do NOT call setId — User.id is @GeneratedValue(IDENTITY); JPA assigns it.
        User u = new User();
        u.setUsername(username);
        u.setEmail(username + "@example.com");
        u.setPassword(encoder.encode(password));
        u.setPlatformRole(com.federated.fl_platform_api.model.PlatformRole.USER);
        u.setStatus(UserStatus.ACTIVE);
        return u;
    }
}
