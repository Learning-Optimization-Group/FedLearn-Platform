package com.federated.fl_platform_api.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Field-level bounds for secure aggregation on {@code POST /api/projects/{id}/start}. The threshold floor of 2
 * mirrors {@code build_servicer}: a threshold of 1 admits a one-survivor round, whose "aggregate" is that client's
 * own contribution. The rules that need the resolved request - DeComFL only, client auth required, threshold no
 * larger than minClients - live in the service.
 */
class StartProjectSecureAggregationValidationTest {

    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private boolean rejected(String field, Consumer<StartProject> set) {
        StartProject p = new StartProject();
        set.accept(p);
        return v.validate(p).stream().anyMatch(cv -> field.equals(cv.getPropertyPath().toString()));
    }

    @Test
    void secureAggregationFieldsAreOptional() {
        assertThat(v.validate(new StartProject())).isEmpty();
    }

    @Test
    void theSwitchTakesEitherValue() {
        assertThat(rejected("secureAggregation", p -> p.setSecureAggregation(true))).isFalse();
        assertThat(rejected("secureAggregation", p -> p.setSecureAggregation(false))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 100})
    void aThresholdFromTwoIsAccepted(int t) {
        assertThat(rejected("secureAggThreshold", p -> p.setSecureAggThreshold(t))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 0, -1, 101})
    void aThresholdBelowTwoOrAboveTheClientCapIsRejected(int t) {
        assertThat(rejected("secureAggThreshold", p -> p.setSecureAggThreshold(t))).isTrue();
    }
}
