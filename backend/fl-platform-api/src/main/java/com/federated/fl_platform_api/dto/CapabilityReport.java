package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.constraints.Size;

import java.util.List;

/**
 * What a device reports about itself when it enrolls in a run (Stage 3 D1): platform and OS level, ABIs, memory,
 * storage, the app build, the native bridge, and its thermal and battery state. Informational only: static facts never
 * approve training; the qualification probe and the contract gate do. Every field but the platform is optional, and
 * every field is bounded so an enrollment cannot carry an arbitrary document.
 */
public record CapabilityReport(
        @NotBlank @Pattern(regexp = "android|ios") String platform,
        @Size(max = 32) String osVersion,
        @PositiveOrZero Integer apiLevel,
        @Size(max = 8) List<@Size(max = 32) String> abis,
        @PositiveOrZero Long totalRamBytes,
        @PositiveOrZero Long freeStorageBytes,
        @Size(max = 64) String deviceModel,
        @Size(max = 32) String appVersion,
        @Size(max = 32) String appBuild,
        @PositiveOrZero Integer bridgeAbiVersion,
        @PositiveOrZero Integer protocolVersion,
        @Size(max = 32) String thermalState,
        @Min(0) @Max(100) Integer batteryPct) {
}
