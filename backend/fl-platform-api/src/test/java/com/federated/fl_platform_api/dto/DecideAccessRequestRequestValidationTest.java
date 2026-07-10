package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.AccessRequestStatus;
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
 * BA-15 follow-up: {@code DecideAccessRequestRequest.decision} is validated against the
 * {@link AccessRequestStatus} enum minus the initial {@code PENDING} constant — a decision
 * is {@code APPROVED} or {@code DENIED}, never the not-yet-decided {@code PENDING} state.
 * Pins that both terminal decisions pass, that {@code PENDING} is rejected and absent from
 * the message, and that the field stays required.
 */
class DecideAccessRequestRequestValidationTest {

    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private DecideAccessRequestRequest req(String decision) {
        DecideAccessRequestRequest r = new DecideAccessRequestRequest();
        r.setDecision(decision);
        return r;
    }

    @ParameterizedTest
    @EnumSource(value = AccessRequestStatus.class, names = {"APPROVED", "DENIED"})
    void terminalDecisionsAreAccepted(AccessRequestStatus status) {
        assertTrue(v.validate(req(status.name())).isEmpty(),
                () -> status.name() + " is a valid decision and must be accepted");
    }

    @Test
    void pendingIsRejected_itIsNotADecision() {
        Set<ConstraintViolation<DecideAccessRequestRequest>> violations = v.validate(req("PENDING"));
        assertEquals(1, violations.size(), "PENDING is the initial state, not a decision");
        String message = violations.iterator().next().getMessage();
        assertFalse(message.contains("PENDING"),
                () -> "the excluded PENDING must not appear in the message but was: " + message);
    }

    @Test
    void unknownDecisionIsRejectedWithMessageListingValidDecisions() {
        Set<ConstraintViolation<DecideAccessRequestRequest>> violations = v.validate(req("MAYBE"));
        assertEquals(1, violations.size());
        String message = violations.iterator().next().getMessage();
        assertTrue(message.contains("must be one of"),
                () -> "expected the enum-listing message but was: " + message);
        assertTrue(message.contains("APPROVED") && message.contains("DENIED"),
                () -> "message should list APPROVED and DENIED but was: " + message);
    }

    @Test
    void nullDecisionIsRejected() {
        assertFalse(v.validate(req(null)).isEmpty(), "decision is required (@NotNull)");
    }
}
