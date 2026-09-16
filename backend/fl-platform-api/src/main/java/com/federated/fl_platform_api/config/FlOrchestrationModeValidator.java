package com.federated.fl_platform_api.config;

import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * OP-14: the hardened single-VM deployment (FL servers run as local processes) is the officially
 * supported deployed architecture. The ECS/Fargate managed-task orchestration path is <b>not
 * implemented</b> — it exists only as a runtime {@link UnsupportedOperationException} in
 * {@code FlServerManager} that fires when a run is started with {@code ecs.cluster-name} set (tracked
 * as OP-12).
 *
 * <p>Leaving that as a mid-run failure is a footgun: an operator who wires {@code ECS_CLUSTER_NAME}
 * boots fine and only discovers the gap on the first federation. This validator moves the failure to
 * boot — fail early and clearly — mirroring the fail-closed boot checks
 * {@link FlBoundaryAuthPolicyValidator} (SE-14) and {@link FlSecretDistinctnessValidator} (SE-20). It
 * gates in every profile because ECS is unsupported everywhere; the blank default
 * ({@code ecs.cluster-name=${ECS_CLUSTER_NAME:}}) is the supported single-VM path and always boots.
 */
@Component
public class FlOrchestrationModeValidator {

    private final String ecsClusterName;

    public FlOrchestrationModeValidator(@Value("${ecs.cluster-name:}") String ecsClusterName) {
        this.ecsClusterName = ecsClusterName;
    }

    @PostConstruct
    void validateOnBoot() {
        check(ecsClusterName);
    }

    /**
     * Throw when {@code ecs.cluster-name} is set, since the ECS/Fargate path is unimplemented.
     * Package-private + static so the policy is unit-testable without a Spring context.
     */
    static void check(String ecsClusterName) {
        if (ecsClusterName != null && !ecsClusterName.isBlank()) {
            throw new IllegalStateException(
                    "OP-14: ecs.cluster-name is set (\"" + ecsClusterName + "\") but ECS/Fargate "
                            + "FL-server orchestration is not implemented (tracked as OP-12). This build "
                            + "supports the hardened single-VM architecture only — FL servers run as local "
                            + "processes. Unset ECS_CLUSTER_NAME to boot.");
        }
    }
}
