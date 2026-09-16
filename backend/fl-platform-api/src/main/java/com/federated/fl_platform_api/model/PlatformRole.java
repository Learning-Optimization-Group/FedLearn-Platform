package com.federated.fl_platform_api.model;

/**
 * Coarse platform-level role used for endpoint authorization.
 *
 * <p>Mapped onto a Spring Security {@link org.springframework.security.core.GrantedAuthority}
 * of {@code ROLE_<name>} via {@link #authority()} by
 * {@link com.federated.fl_platform_api.service.CustomUserDetailsService}.</p>
 */
public enum PlatformRole {
    /** Default tier. May join/train projects (as a CLIENT) but may not create them. */
    USER,
    /**
     * May create and own projects (admin-granted via the owner-promotion workflow).
     * Per-project ownership of a specific project is still tracked by
     * {@code projects.user_id}; this role only gates the capability to create one.
     */
    PROJECT_OWNER,
    /** Platform administrator. Unrestricted across orgs; approves owner/deletion requests. */
    PLATFORM_ADMIN;

    /** The Spring Security authority string ({@code ROLE_USER} / {@code ROLE_PLATFORM_ADMIN}). */
    public String authority() {
        return "ROLE_" + name();
    }
}
