package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.User;

import java.time.Instant;

/**
 * Self-service profile view of the signed-in user, returned by
 * {@code GET/PATCH /api/users/me/profile}.
 *
 * <p>Deliberately excludes internal fields (numeric id, status, platform-role
 * internals) — this is the user's own view, not the admin directory row.
 */
public record UserProfileDto(
        String username,
        String email,
        String displayName,
        String avatarUrl,
        String role,
        Instant createdAt,
        Instant lastLoginAt,
        boolean emailVerified) {

    public static UserProfileDto from(User user) {
        return new UserProfileDto(
                user.getUsername(),
                user.getEmail(),
                user.getDisplayName(),
                user.getAvatarUrl(),
                user.getPlatformRole().name(),
                user.getCreatedAt(),
                user.getLastLoginAt(),
                Boolean.TRUE.equals(user.getEmailVerified()));
    }
}
