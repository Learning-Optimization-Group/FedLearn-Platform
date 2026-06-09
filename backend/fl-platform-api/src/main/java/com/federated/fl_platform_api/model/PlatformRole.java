package com.federated.fl_platform_api.model;

/**
 * Coarse platform-level role used for endpoint authorization.
 *
 * <p>Mapped onto a Spring Security {@link org.springframework.security.core.GrantedAuthority}
 * of {@code ROLE_<name>} via {@link #authority()} by
 * {@link com.federated.fl_platform_api.service.CustomUserDetailsService}.</p>
 */
public enum PlatformRole {
    USER,
    PLATFORM_ADMIN;

    /** The Spring Security authority string ({@code ROLE_USER} / {@code ROLE_PLATFORM_ADMIN}). */
    public String authority() {
        return "ROLE_" + name();
    }
}
