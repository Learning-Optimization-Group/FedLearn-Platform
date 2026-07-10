package com.federated.fl_platform_api.validation;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Bean-validation constraint that accepts exactly the {@link Enum#name()} values
 * of a given enum.
 *
 * <p>Motivation (BA-15): DTO string fields that mirror an enum were validated
 * with hand-written {@code @Pattern(regexp = "A|B")} strings. Those regexes drift
 * silently from the enum — e.g. {@code UpdateProjectRequest.visibility} listed
 * {@code PUBLIC|PRIVATE} while {@code ProjectVisibility} had grown a third
 * {@code RESTRICTED} tier, so a legitimate value was rejected with a 400. Deriving
 * the accepted set from the enum at runtime makes that class of drift impossible:
 * adding a constant automatically widens what the DTO accepts, with zero DTO edits.
 *
 * <p>{@code null} is considered valid so the constraint composes cleanly on
 * optional fields (e.g. a partial {@code PATCH}); combine with {@code @NotNull}
 * when the field is required.
 *
 * <p>Unless a custom {@link #message()} is supplied, the rejection message lists
 * the accepted values, e.g. {@code must be one of [PUBLIC, RESTRICTED, PRIVATE]}.
 */
@Target({ElementType.FIELD, ElementType.PARAMETER, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = ValueOfEnumValidator.class)
@Documented
public @interface ValueOfEnum {

    /** The enum whose {@code name()} values define the accepted set. */
    Class<? extends Enum<?>> enumClass();

    /**
     * Optional message override. When left blank (the default), the validator
     * substitutes a message that enumerates the accepted enum values.
     */
    String message() default "";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
