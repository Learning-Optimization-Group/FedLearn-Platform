package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.dto.RunManifestDto;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.SecureAggregation;
import com.fedlearn.contract.v1.Strategy;
import com.fedlearn.contract.v1.UpdateProtocol;
import org.junit.jupiter.api.Test;

import java.util.UUID;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * During the compatibility window a run is described twice: by the legacy manifest fields old clients read and by
 * the execution contract. Before a contract is published, every decision both describe must agree.
 */
class LegacyManifestEquivalenceTest {

    private static ExecutionContract contract() throws Exception {
        return ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes());
    }

    /** The legacy manifest a backend would emit for the golden contract's run. */
    private static RunManifestDto legacy(ExecutionContract c) {
        RunManifestDto m = new RunManifestDto();
        m.setRunId(UUID.fromString(c.getRunId()));
        m.setProjectId(UUID.fromString(c.getProjectId()));
        m.setRecipeKey("TINYNET_GOLDEN");
        m.setStrategy("FedAvg");
        m.setNumRounds(c.getNumRounds());
        m.setClientsPerRound(c.getClientsPerRound());
        m.setPartitioningMode("SHARDED");
        m.setSeed(c.getSeed());
        m.setSecureAggregation(false);
        m.setFirstOrderSupported(true);
        return m;
    }

    private static java.util.List<String> disagreements(Consumer<RunManifestDto> edit) throws Exception {
        ExecutionContract c = contract();
        RunManifestDto m = legacy(c);
        edit.accept(m);
        return LegacyManifestEquivalence.disagreements(m, c);
    }

    @Test
    void theGoldenContractAgreesWithItsLegacyManifest() throws Exception {
        assertThat(disagreements(m -> { })).isEmpty();
    }

    @Test
    void eachSharedDecisionIsCompared() throws Exception {
        assertThat(disagreements(m -> m.setRunId(UUID.randomUUID()))).containsExactly("runId");
        assertThat(disagreements(m -> m.setProjectId(UUID.randomUUID()))).containsExactly("projectId");
        assertThat(disagreements(m -> m.setRecipeKey("CNN"))).containsExactly("recipe");
        assertThat(disagreements(m -> m.setStrategy("FedOpt"))).containsExactly("strategy");
        assertThat(disagreements(m -> m.setNumRounds(4))).containsExactly("numRounds");
        assertThat(disagreements(m -> m.setClientsPerRound(5))).containsExactly("clientsPerRound");
        assertThat(disagreements(m -> m.setPartitioningMode("LOCAL"))).containsExactly("partitioning");
        assertThat(disagreements(m -> m.setSeed(7L))).containsExactly("seed");
        assertThat(disagreements(m -> m.setSeed(null))).containsExactly("seed");
        assertThat(disagreements(m -> {
            m.setSecureAggregation(true);
            m.setSecureAggThreshold(2);
        })).containsExactly("secureAggregation");
        assertThat(disagreements(m -> m.setFirstOrderSupported(false))).containsExactly("updateProtocol");
    }

    @Test
    void secureAggregationAgreesOnlyWithTheSameThreshold() throws Exception {
        ExecutionContract.Builder b = contract().toBuilder().setStrategy(Strategy.STRATEGY_DECOMFL);
        b.getModelTrainingBuilder().setUpdateProtocol(UpdateProtocol.UPDATE_DECOMFL_SCALAR);
        b.getSecurityBuilder().setSecureAggregation(SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR)
                .setSecureAggThreshold(3);
        ExecutionContract secured = b.build();
        RunManifestDto m = legacy(secured);
        m.setStrategy("DeComFL");
        m.setSecureAggregation(true);
        m.setSecureAggThreshold(3);
        assertThat(LegacyManifestEquivalence.disagreements(m, secured)).isEmpty();

        m.setSecureAggThreshold(2);
        assertThat(LegacyManifestEquivalence.disagreements(m, secured)).containsExactly("secureAggregation");
    }

    @Test
    void aScalarUpdateNeedsNoTrainableProgram() throws Exception {
        ExecutionContract.Builder b = contract().toBuilder().setStrategy(Strategy.STRATEGY_DECOMFL);
        b.getModelTrainingBuilder().setUpdateProtocol(UpdateProtocol.UPDATE_DECOMFL_SCALAR);
        RunManifestDto m = legacy(b.build());
        m.setStrategy("DeComFL");
        m.setFirstOrderSupported(false);
        assertThat(LegacyManifestEquivalence.disagreements(m, b.build())).isEmpty();
    }
}
