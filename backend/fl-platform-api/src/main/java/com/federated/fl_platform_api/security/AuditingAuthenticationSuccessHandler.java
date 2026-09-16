package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import jakarta.servlet.http.HttpServletRequest;

import java.time.Instant;

/**
 * Helper invoked from {@link com.federated.fl_platform_api.controller.AuthController}
 * after a successful login. Updates {@code users.last_login_at} and writes a
 * {@link AuditAction#USER_LOGIN_SUCCEEDED} row.
 *
 * <p>Not a Spring Security {@code AuthenticationSuccessHandler} — login is handled
 * by a regular controller rather than a security filter, so this is a plain
 * collaborator injected into the controller.
 */
public class AuditingAuthenticationSuccessHandler {

    private final UserRepository users;
    private final AuditEventRepository audits;

    public AuditingAuthenticationSuccessHandler(UserRepository users, AuditEventRepository audits) {
        this.users = users;
        this.audits = audits;
    }

    public void onSuccess(String username, HttpServletRequest req) {
        users.findByUsername(username).ifPresent(u -> {
            u.setLastLoginAt(Instant.now());
            users.save(u);

            audits.save(AuditEvent.builder()
                    .action(AuditAction.USER_LOGIN_SUCCEEDED)
                    .actorUserId(u.getId())
                    .targetType("USER")
                    .targetId(u.getId().toString())
                    .requestIp(req.getRemoteAddr())
                    .userAgent(req.getHeader("User-Agent"))
                    .build());
        });
    }
}
