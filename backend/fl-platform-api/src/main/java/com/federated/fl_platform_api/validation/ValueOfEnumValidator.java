package com.federated.fl_platform_api.validation;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

import java.util.Arrays;
import java.util.List;

/**
 * Validator backing {@link ValueOfEnum}. Accepts {@code null} and any string
 * equal to one of the target enum's {@link Enum#name()} values; everything else
 * is rejected with a message that lists the accepted values (unless the
 * annotation supplied an explicit {@code message}).
 */
public class ValueOfEnumValidator implements ConstraintValidator<ValueOfEnum, CharSequence> {

    private List<String> acceptedValues;
    private boolean hasCustomMessage;

    @Override
    public void initialize(ValueOfEnum annotation) {
        Enum<?>[] constants = annotation.enumClass().getEnumConstants();
        this.acceptedValues = Arrays.stream(constants).map(Enum::name).toList();
        this.hasCustomMessage = annotation.message() != null && !annotation.message().isBlank();
    }

    @Override
    public boolean isValid(CharSequence value, ConstraintValidatorContext context) {
        // Optional-field semantics: presence is enforced by @NotNull, not here.
        if (value == null) {
            return true;
        }
        if (acceptedValues.contains(value.toString())) {
            return true;
        }
        // Replace the (blank) default template with one that names the valid tiers,
        // e.g. "must be one of [PUBLIC, RESTRICTED, PRIVATE]". A caller-supplied
        // message is left untouched.
        if (!hasCustomMessage) {
            context.disableDefaultConstraintViolation();
            context.buildConstraintViolationWithTemplate("must be one of " + acceptedValues)
                    .addConstraintViolation();
        }
        return false;
    }
}
