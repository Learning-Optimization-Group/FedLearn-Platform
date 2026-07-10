package com.federated.fl_platform_api.validation;

import com.federated.fl_platform_api.model.ProjectVisibility;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit coverage for {@link ValueOfEnum}/{@link ValueOfEnumValidator} — the durable,
 * enum-backed replacement for hand-written {@code @Pattern} enum regexes (BA-15).
 * Exercises the constraint directly via the Jakarta Validation API, decoupled from
 * any Spring/web wiring.
 */
class ValueOfEnumValidatorTest {

    private static ValidatorFactory factory;
    private static Validator validator;

    @BeforeAll
    static void setUp() {
        factory = Validation.buildDefaultValidatorFactory();
        validator = factory.getValidator();
    }

    @AfterAll
    static void tearDown() {
        factory.close();
    }

    /** Field under test: the accepted set is derived from ProjectVisibility. */
    private static final class Holder {
        @ValueOfEnum(enumClass = ProjectVisibility.class)
        private final String visibility;

        Holder(String visibility) {
            this.visibility = visibility;
        }
    }

    /** Field under test with a caller-supplied message override. */
    private static final class CustomMessageHolder {
        @ValueOfEnum(enumClass = ProjectVisibility.class, message = "pick a real tier")
        private final String visibility;

        CustomMessageHolder(String visibility) {
            this.visibility = visibility;
        }
    }

    @Test
    void nullIsValid() {
        assertTrue(validator.validate(new Holder(null)).isEmpty(),
                "null must be accepted so the constraint composes on optional fields");
    }

    @ParameterizedTest
    @EnumSource(ProjectVisibility.class)
    void everyEnumNameIsValid(ProjectVisibility v) {
        assertTrue(validator.validate(new Holder(v.name())).isEmpty(),
                () -> v.name() + " is a valid ProjectVisibility and must be accepted");
    }

    @Test
    void unknownValueIsRejectedWithAMessageListingEveryTier() {
        Set<ConstraintViolation<Holder>> violations = validator.validate(new Holder("BOGUS"));
        assertEquals(1, violations.size());
        String message = violations.iterator().next().getMessage();
        for (ProjectVisibility v : ProjectVisibility.values()) {
            assertTrue(message.contains(v.name()),
                    () -> "rejection message should list " + v.name() + " but was: " + message);
        }
    }

    @Test
    void caseSensitiveExactMatchOnly() {
        // Enum name()s are upper-case; a lower-case variant must not slip through.
        assertEquals(1, validator.validate(new Holder("public")).size());
    }

    @Test
    void customMessageOverrideIsHonored() {
        Set<ConstraintViolation<CustomMessageHolder>> violations =
                validator.validate(new CustomMessageHolder("BOGUS"));
        assertEquals(1, violations.size());
        assertEquals("pick a real tier", violations.iterator().next().getMessage());
    }
}
