package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.dto.RunManifestDto;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.Partitioning;
import com.fedlearn.contract.v1.SecureAggregation;
import com.fedlearn.contract.v1.SecurityPolicy;
import com.fedlearn.contract.v1.Strategy;
import com.fedlearn.contract.v1.UpdateProtocol;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Compares the decisions a run's legacy manifest and its execution contract both describe. During the compatibility
 * window old clients act on the legacy fields and updated clients on the contract, so a contract that disagrees with
 * them must not be published.
 */
public final class LegacyManifestEquivalence {

    private static final Map<String, Strategy> STRATEGIES = Map.of(
            "DeComFL", Strategy.STRATEGY_DECOMFL,
            "FedAvg", Strategy.STRATEGY_FEDAVG,
            "FedProx", Strategy.STRATEGY_FEDPROX,
            "FedOpt", Strategy.STRATEGY_FEDOPT,
            "Robust", Strategy.STRATEGY_ROBUST);

    private LegacyManifestEquivalence() {
    }

    /** The contract fields whose decision differs from the legacy manifest's; empty when they agree. */
    public static List<String> disagreements(RunManifestDto legacy, ExecutionContract contract) {
        List<String> out = new ArrayList<>();
        check(out, "runId", String.valueOf(legacy.getRunId()).equals(contract.getRunId()));
        check(out, "projectId", String.valueOf(legacy.getProjectId()).equals(contract.getProjectId()));
        check(out, "recipe", ("RECIPE_" + legacy.getRecipeKey()).equals(contract.getRecipe().name()));
        check(out, "strategy", STRATEGIES.get(legacy.getStrategy()) == contract.getStrategy());
        check(out, "numRounds", Integer.toUnsignedLong(contract.getNumRounds()) == legacy.getNumRounds());
        check(out, "clientsPerRound",
                Integer.toUnsignedLong(contract.getClientsPerRound()) == legacy.getClientsPerRound());
        check(out, "partitioning",
                ("PARTITIONING_" + legacy.getPartitioningMode()).equals(contract.getPartitioning().name())
                        && contract.getPartitioning() != Partitioning.PARTITIONING_UNSPECIFIED);
        check(out, "seed", contract.hasSeed()
                ? legacy.getSeed() != null && legacy.getSeed() == contract.getSeed()
                : legacy.getSeed() == null);
        check(out, "secureAggregation", secureAggregationAgrees(legacy, contract.getSecurity()));
        // A weight update needs the trainable program the legacy flag reports as staged.
        check(out, "updateProtocol",
                contract.getModelTraining().getUpdateProtocol() != UpdateProtocol.UPDATE_TRAINABLE_STATE_F32
                        || legacy.isFirstOrderSupported());
        return out;
    }

    private static boolean secureAggregationAgrees(RunManifestDto legacy, SecurityPolicy security) {
        boolean secured = security.getSecureAggregation() == SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR;
        if (legacy.isSecureAggregation() != secured) {
            return false;
        }
        return !secured || (security.hasSecureAggThreshold()
                && Objects.equals(legacy.getSecureAggThreshold(), security.getSecureAggThreshold()));
    }

    private static void check(List<String> out, String field, boolean agrees) {
        if (!agrees) {
            out.add(field);
        }
    }
}
