package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import jakarta.servlet.http.HttpServletRequest;

/**
 * Helper invoked from {@link com.federated.fl_platform_api.controller.AuthController}
 * after a failed authentication attempt (bad credentials, disabled account, etc.).
 *
 * <p>Stores only the submitted username — never a user id — on
 * {@code target_id}. We deliberately avoid looking up the user here both to
 * sidestep a timing oracle (existence vs. non-existence) and because the
 * username may not correspond to any account.
 */
public class AuditingAuthenticationFailureHandler {

    private final AuditEventRepository audits;

    public AuditingAuthenticationFailureHandler(AuditEventRepository audits) {
        this.audits = audits;
    }

    public void onFailure(String username, HttpServletRequest req) {
        audits.save(AuditEvent.builder()
                .action(AuditAction.USER_LOGIN_FAILED)
                .targetType("USERNAME")
                .targetId(username == null ? "?" : username)
                .requestIp(req.getRemoteAddr())
                .userAgent(req.getHeader("User-Agent"))
                .build());
    }
}
