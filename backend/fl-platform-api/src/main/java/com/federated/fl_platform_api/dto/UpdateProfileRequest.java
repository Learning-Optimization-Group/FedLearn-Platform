package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Size;

/**
 * Body of {@code PATCH /api/users/me/profile}. All fields are optional —
 * absent (null) fields are left unchanged.
 *
 * <p>The email and password constraints mirror {@link RegisterRequest} exactly
 * (minus {@code @NotBlank}, since every field here is optional) so a profile
 * update can never set a value that registration would have rejected.
 * Username is deliberately not editable.
 */
public class UpdateProfileRequest {

    /** Trimmed server-side; max 80 chars after trimming (checked in ProfileService). */
    private String displayName;

    @Email(message = "Email should be valid")
    @Size(max = 100, message = "Email cannot exceed 100 characters")
    private String email;

    /** Required (and verified) whenever {@link #newPassword} is present. */
    private String currentPassword;

    @Size(min = 6, max = 100, message = "Password must be between 6 and 100 characters")
    private String newPassword;

    public UpdateProfileRequest() {
    }

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getCurrentPassword() {
        return currentPassword;
    }

    public void setCurrentPassword(String currentPassword) {
        this.currentPassword = currentPassword;
    }

    public String getNewPassword() {
        return newPassword;
    }

    public void setNewPassword(String newPassword) {
        this.newPassword = newPassword;
    }

    @Override
    public String toString() {
        // Never include password material.
        return "UpdateProfileRequest{displayName='" + displayName + "', email='" + email + "'}";
    }
}
