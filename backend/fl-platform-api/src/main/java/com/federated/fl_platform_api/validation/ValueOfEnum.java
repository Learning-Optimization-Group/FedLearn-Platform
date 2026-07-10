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
 *
 * <p>{@link #exclude()} narrows the accepted set to a proper subset of the enum by
 * removing named constants — for fields that mirror an enum but must reject a
 * reserved constant (e.g. a membership may be granted as {@code MEMBER}/{@code CLIENT}
 * but never {@code OWNER}; an access decision is {@code APPROVED}/{@code DENIED} but
 * never the initial {@code PENDING}). The field stays enum-derived — a newly added
 * constant is still auto-accepted — while the excluded constants stay off the set and
 * out of the rejection message.
 */
@Target({ElementType.FIELD, ElementType.PARAMETER, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = ValueOfEnumValidator.class)
@Documented
public @interface ValueOfEnum {

    /** The enum whose {@code name()} values define the accepted set. */
    Class<? extends Enum<?>> enumClass();

    /**
     * Enum constant {@code name()}s to remove from the accepted set, yielding a proper
     * subset of the enum. Empty (the default) accepts every constant. Names that are not
     * constants of {@link #enumClass()} are simply ignored.
     */
    String[] exclude() default {};

    /**
     * Optional message override. When left blank (the default), the validator
     * substitutes a message that enumerates the accepted enum values.
     */
    String message() default "";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
