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
        @JsonProperty("requirements") DeviceRequirements requirements
) {
}
