package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.audit.AuditContext;
import com.federated.fl_platform_api.audit.Auditable;
import com.federated.fl_platform_api.dto.UpdateProfileRequest;
import com.federated.fl_platform_api.dto.UserProfileDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

/**
 * Self-service profile reads and updates for the signed-in user
 * ({@code /api/users/me/profile}). Identity is resolved by the controller from
 * the SecurityContext and passed in as the username — a caller can only ever
 * touch their own row.
 */
@Service
public class ProfileService {

    /** Contract limit for {@code displayName}, applied after trimming. */
    static final int DISPLAY_NAME_MAX_LENGTH = 80;

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final AuditEventRepository auditEventRepository;

    public ProfileService(UserRepository userRepository,
                          PasswordEncoder passwordEncoder,
                          AuditEventRepository auditEventRepository) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.auditEventRepository = auditEventRepository;
    }

    @Transactional(readOnly = true)
    public UserProfileDto getProfile(String username) {
        return UserProfileDto.from(loadUser(username));
    }

    /**
     * Applies a partial profile update. All checks run before any mutation so
     * the PATCH is all-or-nothing:
     * <ul>
     *   <li>{@code newPassword} requires a matching {@code currentPassword}
     *       (403 otherwise); strength rules are the registration constraints,
     *       enforced by bean validation on {@link UpdateProfileRequest}.</li>
     *   <li>an email change must not collide with another account (409) and
     *       resets {@code emailVerified} to false.</li>
     *   <li>{@code displayName} is trimmed; max {@value #DISPLAY_NAME_MAX_LENGTH}
     *       chars; blank clears it.</li>
     * </ul>
     *
     * <p>Audited as {@link AuditAction#USER_PROFILE_UPDATED} via the aspect,
     * with changed-field flags in the metadata. A password change additionally
     * writes a dedicated {@link AuditAction#USER_PASSWORD_CHANGED} row in the
     * same transaction (direct-repository write, mirroring
     * {@code AuditingAuthenticationSuccessHandler}) — the aspect supports one
     * action per method, and password changes warrant their own trail.
     */
    @Transactional
    @Auditable(action = AuditAction.USER_PROFILE_UPDATED, targetType = "USER")
    public UserProfileDto updateProfile(String username, UpdateProfileRequest request) {
        User user = loadUser(username);

        // ── Validate everything up front ────────────────────────────────────
        boolean passwordChange = request.getNewPassword() != null;
        if (passwordChange
                && (request.getCurrentPassword() == null
                    || !passwordEncoder.matches(request.getCurrentPassword(), user.getPassword()))) {
            // AccessDeniedException → 403 via GlobalExceptionHandler.
            throw new AccessDeniedException("Current password is incorrect");
        }

        String requestedEmail = request.getEmail() == null ? null : request.getEmail().trim();
        boolean emailChange = requestedEmail != null && !requestedEmail.isEmpty()
                && !requestedEmail.equals(user.getEmail());
        if (emailChange && userRepository.findByEmail(requestedEmail)
                .filter(other -> !other.getId().equals(user.getId()))
                .isPresent()) {
            // Same exception (→ 409) registration raises for a taken email.
            throw new UserAlreadyExistsException(
                    "Email " + requestedEmail + " is already registered.");
        }

        String trimmedDisplayName = request.getDisplayName() == null
                ? null : request.getDisplayName().trim();
        if (trimmedDisplayName != null && trimmedDisplayName.length() > DISPLAY_NAME_MAX_LENGTH) {
            // IllegalArgumentException → 400 via GlobalExceptionHandler.
            throw new IllegalArgumentException(
                    "Display name cannot exceed " + DISPLAY_NAME_MAX_LENGTH + " characters");
        }

        // ── Apply ───────────────────────────────────────────────────────────
        if (trimmedDisplayName != null) {
            user.setDisplayName(trimmedDisplayName.isEmpty() ? null : trimmedDisplayName);
            AuditContext.put("displayNameChanged", "true");
        }
        if (emailChange) {
            user.setEmail(requestedEmail);
            user.setEmailVerified(false); // the new address has never been verified
            AuditContext.put("emailChanged", "true");
        }
        if (passwordChange) {
            user.setPassword(passwordEncoder.encode(request.getNewPassword()));
            AuditContext.put("passwordChanged", "true");
            auditEventRepository.save(AuditEvent.builder()
                    .action(AuditAction.USER_PASSWORD_CHANGED)
                    .actorUserId(user.getId())
                    .targetType("USER")
                    .targetId(user.getId().toString())
                    .build());
        }
        user.setUpdatedAt(Instant.now());

        return UserProfileDto.from(userRepository.save(user));
    }

    private User loadUser(String username) {
        return userRepository.findByUsername(username)
                .orElseThrow(() -> ResourceNotFoundException.forEntity("User", username));
    }
}
