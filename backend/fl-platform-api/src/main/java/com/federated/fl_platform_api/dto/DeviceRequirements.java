package com.federated.fl_platform_api.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Per-project (and per-recipe) device requirements. All fields nullable — a null
 * field means "no constraint on that dimension". Effective requirements = recipe
 * default merged most-restrictive-wins with an optional owner override (see merge).
 *
 * Recipe-only fields (maxTrainableParams, estimatedRoundTimeSeconds, acceleratorBackends)
 * are informational constraints set by the recipe author; they are always taken from the
 * recipe default and cannot be overridden by a project owner.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public record DeviceRequirements(
        @JsonProperty("minRamGb")               @JsonAlias("min_ram_gb")                    Double minRamGb,
        @JsonProperty("minStorageGb")           @JsonAlias("min_storage_gb")                Double minStorageGb,
        @JsonProperty("minOsAndroid")           @JsonAlias("min_os_android")                Integer minOsAndroid,
        @JsonProperty("minOsIos")               @JsonAlias("min_os_ios")                    String minOsIos,
        @JsonProperty("mobileSafe")             @JsonAlias("mobile_safe")                   Boolean mobileSafe,
        @JsonProperty("maxTrainableParams")     @JsonAlias("max_trainable_params")          Long maxTrainableParams,
        @JsonProperty("minNpuTops")             @JsonAlias("min_npu_tops")                  Double minNpuTops,
        @JsonProperty("estimatedRoundTimeSeconds") @JsonAlias("estimated_round_time_seconds") Integer estimatedRoundTimeSeconds,
        @JsonProperty("minBatteryPct")          @JsonAlias("min_battery_pct")               Integer minBatteryPct,
        @JsonProperty("requiresWifi")           @JsonAlias("requires_wifi")                 Boolean requiresWifi,
        @JsonProperty("acceleratorBackends")    @JsonAlias("accelerator_backends")          List<String> acceleratorBackends
) {

    /**
     * Most-restrictive-wins merge. Combines a recipe default with an optional owner
     * override:
     * <ul>
     *   <li>Numeric minimums (RAM, storage, Android OS, NPU) → {@code max(base, override)}</li>
     *   <li>iOS version string → stricter (higher) semantic version wins</li>
     *   <li>mobileSafe → {@code false} if either side is {@code false} (AND semantics)</li>
     *   <li>requiresWifi → {@code true} if either side is {@code true} (OR semantics)</li>
     *   <li>Recipe-only fields (maxTrainableParams, estimatedRoundTimeSeconds,
     *       acceleratorBackends) → always taken from {@code base}</li>
     * </ul>
     *
     * @param base     recipe default (source of truth for recipe-only fields)
     * @param override owner-supplied constraints; {@code null} returns {@code base} unchanged
     * @return effective {@link DeviceRequirements} instance
     */
    public static DeviceRequirements merge(DeviceRequirements base, DeviceRequirements override) {
        if (override == null) return base;
        if (base == null) base = empty();
        return new DeviceRequirements(
                maxD(base.minRamGb, override.minRamGb),
                maxD(base.minStorageGb, override.minStorageGb),
                maxI(base.minOsAndroid, override.minOsAndroid),
                stricterIos(base.minOsIos, override.minOsIos),
                andSafe(base.mobileSafe, override.mobileSafe),
                base.maxTrainableParams,                        // recipe-only
                maxD(base.minNpuTops, override.minNpuTops),
                base.estimatedRoundTimeSeconds,                 // recipe-only
                maxI(base.minBatteryPct, override.minBatteryPct),
                orWifi(base.requiresWifi, override.requiresWifi),
                base.acceleratorBackends                        // recipe-only
        );
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static DeviceRequirements empty() {
        return new DeviceRequirements(null, null, null, null, null, null, null, null, null, null, null);
    }

    /** Returns the larger of two nullable Doubles; null treated as "no constraint". */
    private static Double maxD(Double a, Double b) {
        if (a == null) return b;
        if (b == null) return a;
        return Math.max(a, b);
    }

    /** Returns the larger of two nullable Integers; null treated as "no constraint". */
    private static Integer maxI(Integer a, Integer b) {
        if (a == null) return b;
        if (b == null) return a;
        return Math.max(a, b);
    }

    /**
     * AND semantics for mobileSafe: {@code false} is stricter than {@code true}.
     * Null means "no constraint" (treated as safe / true for merge purposes).
     */
    private static Boolean andSafe(Boolean a, Boolean b) {
        if (Boolean.FALSE.equals(a) || Boolean.FALSE.equals(b)) return Boolean.FALSE;
        return (a != null) ? a : b;
    }

    /**
     * OR semantics for requiresWifi: requiring wifi is stricter than not requiring it.
     * Null means "no constraint" (treated as false for merge purposes).
     */
    private static Boolean orWifi(Boolean a, Boolean b) {
        if (Boolean.TRUE.equals(a) || Boolean.TRUE.equals(b)) return Boolean.TRUE;
        return (a != null) ? a : b;
    }

    /**
     * Returns the stricter (higher) iOS minimum version string. Parses "major.minor"
     * numerically; falls back to lexicographic if parsing fails. Null-safe.
     */
    private static String stricterIos(String a, String b) {
        if (a == null) return b;
        if (b == null) return a;
        long va = iosKey(a), vb = iosKey(b);
        return (vb > va) ? b : a;
    }

    private static long iosKey(String v) {
        try {
            String[] parts = v.split("\\.");
            int major = Integer.parseInt(parts[0].trim());
            int minor = parts.length > 1 ? Integer.parseInt(parts[1].trim()) : 0;
            return major * 1000L + minor;
        } catch (RuntimeException e) {
            return -1;
        }
    }
}
