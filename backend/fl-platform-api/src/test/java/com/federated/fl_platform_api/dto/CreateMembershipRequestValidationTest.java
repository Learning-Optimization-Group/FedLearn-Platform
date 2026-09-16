package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.MembershipRole;
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
 * BA-15 follow-up: {@code CreateMembershipRequest.role} is validated against the
 * {@link MembershipRole} enum minus the reserved {@code OWNER} constant — a member may be
 * granted {@code MEMBER} or {@code CLIENT} via this endpoint but never {@code OWNER}
 * (project ownership is not assignable here). Pins that the allowed subset passes, that
 * {@code OWNER} is rejected and absent from the message, and that the field stays required.
 */
class CreateMembershipRequestValidationTest {

    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private CreateMembershipRequest req(String role) {
        CreateMembershipRequest r = new CreateMembershipRequest();
        r.setUsername("alice");
        r.setRole(role);
        return r;
    }

    @ParameterizedTest
    @EnumSource(value = MembershipRole.class, names = {"MEMBER", "CLIENT"})
    void allowedRolesAreAccepted(MembershipRole role) {
        assertTrue(v.validate(req(role.name())).isEmpty(),
                () -> role.name() + " is assignable and must be accepted");
    }

    @Test
    void ownerRoleIsRejected_evenThoughItIsAMembershipRole() {
        Set<ConstraintViolation<CreateMembershipRequest>> violations = v.validate(req("OWNER"));
        assertEquals(1, violations.size(), "OWNER must not be grantable via this endpoint");
        String message = violations.iterator().next().getMessage();
        assertFalse(message.contains("OWNER"),
                () -> "the excluded OWNER must not appear in the message but was: " + message);
    }

    @Test
    void unknownRoleIsRejectedWithMessageListingAllowedRoles() {
        Set<ConstraintViolation<CreateMembershipRequest>> violations = v.validate(req("ADMIN"));
        assertEquals(1, violations.size());
        String message = violations.iterator().next().getMessage();
        assertTrue(message.contains("must be one of"),
                () -> "expected the enum-listing message but was: " + message);
        assertTrue(message.contains("MEMBER") && message.contains("CLIENT"),
                () -> "message should list the allowed roles but was: " + message);
    }

    @Test
    void nullRoleIsRejected() {
        assertFalse(v.validate(req(null)).isEmpty(), "role is required (@NotNull)");
    }
}
