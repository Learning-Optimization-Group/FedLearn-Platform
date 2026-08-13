package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * A model "recipe" from the framework's catalog (produced by {@code recipes.py
 * --describe}). Surfaced verbatim by {@code GET /api/model-recipes} so the
 * frontend can drive its model-picker, input collection, and label rendering
 * from a single source of truth instead of hardcoded tables.
 *
 * <p>The JSON contract on the wire is camelCase. The Python side emits
 * snake_case (e.g. {@code display_name}, {@code input_kind}); the
 * {@link JsonAlias} hints let Jackson read those without a separate parse DTO,
 * while the component names define the camelCase serialization back to clients.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ModelRecipeDto(
        @JsonProperty("key") String key,
        @JsonProperty("displayName") @JsonAlias("display_name") String displayName,
        @JsonProperty("inputKind") @JsonAlias("input_kind") String inputKind,
        @JsonProperty("classes") List<String> classes,
        @JsonProperty("baseModels") @JsonAlias("base_models") List<String> baseModels,
        @JsonProperty("optimizers") List<String> optimizers,
        @JsonProperty("requirements") DeviceRequirements requirements,
        /** Training arms this recipe supports; {@code null} for recipes that declare none. */
        @JsonProperty("supportedArms") @JsonAlias("supported_arms") List<String> supportedArms,
        /**
         * The measured frozen-vs-full trade-off, attached only to recipes that offer a CHOICE of
         * arms. Generated from the campaign's verdict record by {@code scripts/build_arm_tradeoff.py}
         * — never hand-written — so what the picker shows cannot drift from the measurement.
         */
        @JsonProperty("armTradeoff") @JsonAlias("arm_tradeoff") ArmTradeoff armTradeoff
) {
    /**
     * Pre-P1 arity: a recipe with no declared arms and no trade-off. Kept so the many call sites
     * that construct a recipe fixture do not each have to restate "no arms" — and so that adding
     * a future catalog field stays a one-line change here rather than a sweep through the tests.
     */
    public ModelRecipeDto(String key, String displayName, String inputKind, List<String> classes,
                          List<String> baseModels, List<String> optimizers,
                          DeviceRequirements requirements) {
        this(key, displayName, inputKind, classes, baseModels, optimizers, requirements, null, null);
    }

    /**
     * The subset of the trade-off the picker renders. Deliberately not the whole record: the
     * caveats are carried because a number shown without them is a claim the measurement does not
     * support (the communication ratio is round-budget dependent, and accuracy and on-device
     * latency were measured on different hardware).
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ArmTradeoff(
            @JsonProperty("headline") String headline,
            /**
             * Communication saving from freezing, as a ratio. {@code Double}, not {@code Integer}:
             * these were all large whole numbers (3,321x) until PNEUMONIA_CNN was measured on the
             * product path at <b>1.004x</b> — that recipe's classifier is 99.6% of its parameters,
             * so freezing saves almost nothing. Jackson coerces a JSON float into an Integer by
             * TRUNCATING, which would have rounded an honest 1.004 to 1 on its way to the picker.
             * {@code null} means not measured, and must never become 0.
             */
            @JsonProperty("commRatio") @JsonAlias("comm_ratio") Double commRatio,
            @JsonProperty("ondeviceRatio") @JsonAlias("ondevice_ratio") Double ondeviceRatio,
            @JsonProperty("measuredOn") @JsonAlias("measured_on") java.util.Map<String, String> measuredOn,
            @JsonProperty("arms") java.util.Map<String, java.util.Map<String, Object>> arms,
            @JsonProperty("caveats") List<String> caveats
    ) {
    }
}
