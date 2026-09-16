package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.PlatformRole;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * BA-15 follow-up: {@code UpdateUserRoleRequest.role} is validated against the
 * {@link PlatformRole} enum (was a hand-written {@code @Pattern}). Pins that every
 * role name is accepted, an unknown value is rejected with the enum-listing message,
 * and the field stays required.
 */
class UpdateUserRoleRequestValidationTest {

    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private UpdateUserRoleRequest req(String role) {
        UpdateUserRoleRequest r = new UpdateUserRoleRequest();
        r.setRole(role);
        return r;
    }

    @ParameterizedTest
    @EnumSource(PlatformRole.class)
    void everyPlatformRoleNameIsAccepted(PlatformRole role) {
        assertTrue(v.validate(req(role.name())).isEmpty(),
                () -> role.name() + " is a valid PlatformRole and must be accepted");
    }

    @Test
    void unknownRoleIsRejectedWithMessageListingEveryRole() {
        Set<ConstraintViolation<UpdateUserRoleRequest>> violations = v.validate(req("SUPERUSER"));
        assertEquals(1, violations.size());
        String message = violations.iterator().next().getMessage();
        assertTrue(message.contains("must be one of"),
                () -> "expected the enum-listing message but was: " + message);
        for (PlatformRole role : PlatformRole.values()) {
            assertTrue(message.contains(role.name()),
                    () -> "message should list " + role.name() + " but was: " + message);
        }
    }

    @Test
    void nullRoleIsRejected() {
        assertFalse(v.validate(req(null)).isEmpty(), "role is required (@NotNull)");
    }
}
