package com.federated.fl_platform_api.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * OP-14: hardened single-VM (local-process FL servers) is the officially supported deployed
 * architecture; the ECS/Fargate managed-task orchestration path is NOT implemented (tracked as
 * OP-12). Setting {@code ecs.cluster-name} previously only surfaced as a confusing mid-run
 * {@code UnsupportedOperationException} from FlServerManager when a run was started. This validator
 * fails the boot early and clearly instead, so an operator who wires ECS by mistake finds out at
 * startup, not on the first federation. A blank cluster name (the default) is the single-VM path and
 * always boots.
 */
class FlOrchestrationModeValidatorTest {

    // ecs.cluster-name set -> ECS is unsupported -> fail boot closed
    @Test
    void ecsClusterNameSet_failsBoot() {
        assertThrows(IllegalStateException.class,
                () -> FlOrchestrationModeValidator.check("fedlearn-cluster"));
    }

    // blank / whitespace / null -> the supported single-VM local-process path -> always boots
    @Test
    void blankClusterName_ok() {
        assertDoesNotThrow(() -> FlOrchestrationModeValidator.check(""));
        assertDoesNotThrow(() -> FlOrchestrationModeValidator.check("   "));
        assertDoesNotThrow(() -> FlOrchestrationModeValidator.check(null));
    }
}
