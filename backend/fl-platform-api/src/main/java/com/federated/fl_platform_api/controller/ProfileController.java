package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.UpdateProfileRequest;
import com.federated.fl_platform_api.dto.UserProfileDto;
import com.federated.fl_platform_api.service.ProfileService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Self-service profile endpoints for the signed-in user. Reachable by any
 * authenticated principal regardless of platform role — identity always comes
 * from the SecurityContext, never from the request, so a caller can only read
 * or update their own profile.
 *
 * <p>Authentication is enforced here rather than by the filter-chain matcher
 * (same pattern as {@code GET /api/auth/me}): an anonymous caller gets a clean
 * 401 — the signal the SPA's interceptor treats as "session expired" — instead
 * of the security chain's default 403.
 */
@RestController
@RequestMapping("/api/users/me/profile")
public class ProfileController {

    private final ProfileService profileService;

    public ProfileController(ProfileService profileService) {
        this.profileService = profileService;
    }

    @GetMapping
    public ResponseEntity<UserProfileDto> getMyProfile() {
        return ResponseEntity.ok(profileService.getProfile(authenticatedUsername()));
    }

    @PatchMapping
    public ResponseEntity<UserProfileDto> updateMyProfile(
            @Valid @RequestBody UpdateProfileRequest request) {
        return ResponseEntity.ok(profileService.updateProfile(authenticatedUsername(), request));
    }

    /**
     * Resolves the caller's username from the SecurityContext, or throws
     * {@link BadCredentialsException} (→ 401 via GlobalExceptionHandler) when
     * the request is anonymous. Mirrors {@code AuthController#currentUser()}.
     */
    private static String authenticatedUsername() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()
                || "anonymousUser".equals(authentication.getPrincipal())) {
            throw new BadCredentialsException("Not authenticated");
        }
        return authentication.getName();
    }
}
