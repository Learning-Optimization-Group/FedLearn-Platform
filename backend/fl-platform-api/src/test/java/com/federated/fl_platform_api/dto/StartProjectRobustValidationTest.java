package com.federated.fl_platform_api.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Field-level bounds for the Robust aggregation settings on {@code POST /api/projects/{id}/start}.
 *
 * <p>These mirror the Python constructor's own checks ({@code RobustAggregator.__init__}): trim ratio in
 * [0, 0.5), centered-clip tau &gt; 0. The Byzantine fraction has no range check in Python at all, so the
 * DTO is its only bound. Cross-field rules - a rule that cannot run at {@code minClients}, or a robust
 * field on a non-Robust strategy - need the resolved request and live in the service.
 */
class StartProjectRobustValidationTest {

    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private boolean rejected(String field, Consumer<StartProject> set) {
        StartProject p = new StartProject();
        set.accept(p);
        return v.validate(p).stream().anyMatch(cv -> field.equals(cv.getPropertyPath().toString()));
    }

    @Test
    void robustSettingsAreOptional() {
        assertThat(v.validate(new StartProject())).isEmpty();
    }

    @ParameterizedTest
    @ValueSource(strings = {"MEDIAN", "TRIMMED_MEAN", "KRUM", "MULTI_KRUM", "BULYAN", "CENTERED_CLIP"})
    void everyRobustMethodNameIsAccepted(String name) {
        assertThat(rejected("robustMethod", p -> p.setRobustMethod(name))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(strings = {"krum", "multi_krum", "Median", "FEDAVG", "BULYAN;rm -rf /"})
    void anyOtherRobustMethodIsRejected(String name) {
        // The API takes enum names, like trainingArm; the lowercase server name is an argv detail.
        assertThat(rejected("robustMethod", p -> p.setRobustMethod(name))).isTrue();
    }

    @ParameterizedTest
    @ValueSource(doubles = {0.0, 0.1, 0.49})
    void byzantineFractionInsideZeroToHalfIsAccepted(double f) {
        assertThat(rejected("byzantineFraction", p -> p.setByzantineFraction(f))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(doubles = {-0.01, 0.5, 0.9})
    void byzantineFractionOutsideZeroToHalfIsRejected(double f) {
        assertThat(rejected("byzantineFraction", p -> p.setByzantineFraction(f))).isTrue();
    }

    @ParameterizedTest
    @ValueSource(doubles = {0.0, 0.2, 0.49})
    void trimRatioInsideZeroToHalfIsAccepted(double t) {
        assertThat(rejected("trimRatio", p -> p.setTrimRatio(t))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(doubles = {-0.1, 0.5})
    void trimRatioOutsideZeroToHalfIsRejected(double t) {
        assertThat(rejected("trimRatio", p -> p.setTrimRatio(t))).isTrue();
    }

    @ParameterizedTest
    @ValueSource(doubles = {0.0001, 1.0, 50.0})
    void positiveClippingRadiusIsAccepted(double tau) {
        assertThat(rejected("centeredClipTau", p -> p.setCenteredClipTau(tau))).isFalse();
    }

    @ParameterizedTest
    @ValueSource(doubles = {0.0, -1.0})
    void nonPositiveClippingRadiusIsRejected(double tau) {
        // fl_server.py reads tau with `or 1.0`, so 0.0 would silently become 1.0 instead of failing.
        assertThat(rejected("centeredClipTau", p -> p.setCenteredClipTau(tau))).isTrue();
    }
}
