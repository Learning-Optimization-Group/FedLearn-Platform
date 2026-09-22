package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.PartitioningMode;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.federated.fl_platform_api.model.TrainingArm;
import com.fedlearn.contract.v1.ArtifactBackend;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.ClientAuth;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import com.fedlearn.contract.v1.Partitioning;
import com.fedlearn.contract.v1.Recipe;
import com.fedlearn.contract.v1.SecureAggregation;
import com.fedlearn.contract.v1.Strategy;
import com.fedlearn.contract.v1.Transport;
import org.junit.jupiter.api.Test;

import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The publisher's assembly step: a run's record and intent, the Python-resolved plan and the staged bundle's facts
 * become one execution contract, or the run is reported not representable with a reason. Nothing is inferred.
 */
class ExecutionContractAssemblerTest {

    private static final ExecutionContractAssembler.Policy POLICY =
            new ExecutionContractAssembler.Policy(2, 3, 1_000L);

    private static Run run(RunIntent intent, String strategy) {
        Run run = new Run();
        run.setId(UUID.fromString("4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17"));
        run.setProjectId(UUID.fromString("9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64"));
        run.setRecipeKey("TINYNET_GOLDEN");
        run.setStrategy(strategy);
        run.setNumRounds(3);
        run.setMinClients(4);
        run.setClientsPerRound(4);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setSeed(-2L);
        run.setIntent(intent);
        return run;
    }

    private static RunIntent intent(boolean tls, boolean auth) {
        Project p = new Project();
        p.setModelName("tinynet_golden");
        p.setTrainingArm(TrainingArm.FULL);
        return RunIntent.capture(p, tls, auth, 900_000L);
    }

    /** The golden contract's ModelTraining stands in for the Python plan: its Python-owned fields. */
    private static ModelTraining plan() throws Exception {
        ModelTraining golden = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes())
                .getModelTraining();
        return golden.toBuilder().clearModelRevision().clearArtifacts().build();
    }

    private static StagedBundle bundle() throws Exception {
        ArtifactVariant golden = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes())
                .getModelTraining().getArtifacts(0);
        List<String> names = List.of("loss.pte", "infer.pte", "trainable.pte");
        List<StagedBundle.ModelFile> files = new java.util.ArrayList<>();
        for (int i = 0; i < names.size(); i++) {
            files.add(new StagedBundle.ModelFile(names.get(i), golden.getFiles(i).getSha256(),
                    golden.getFiles(i).getByteSize()));
        }
        return new StagedBundle(files, golden.getRequiredOperatorsList(),
                new StagedBundle.ResourceEnvelope(8L << 20, 20_732L, 1_000L, 5_000L),
                List.of(new StagedBundle.LayoutEntry("fc1.weight", List.of(5L, 4L)),
                        new StagedBundle.LayoutEntry("fc1.bias", List.of(5L))));
    }

    private static ExecutionContract assemble(Run run, ModelTraining plan, StagedBundle bundle) throws Exception {
        return ExecutionContractAssembler.assemble(run, plan, bundle, POLICY);
    }

    @Test
    void aTinyNetFedAvgRunAssemblesIntoAValidContract() throws Exception {
        Run run = run(intent(true, true), "FedAvg");
        ExecutionContract c = assemble(run, plan(), bundle());

        assertThat(ExecutionContractValidator.validate(c, 2, run.getId().toString(), run.getProjectId().toString()))
                .isEmpty();
        assertThat(c.getRecipe()).isEqualTo(Recipe.RECIPE_TINYNET_GOLDEN);
        assertThat(c.getStrategy()).isEqualTo(Strategy.STRATEGY_FEDAVG);
        assertThat(c.getNumRounds()).isEqualTo(3);
        assertThat(c.getClientsPerRound()).isEqualTo(4);
        assertThat(c.getPartitioning()).isEqualTo(Partitioning.PARTITIONING_SHARDED);
        assertThat(c.getRound().getTimeoutMs()).isEqualTo(900_000L);
        assertThat(c.getRound().getMaxTransientRetries()).isEqualTo(3);
        assertThat(c.getRound().getRetryBackoffMs()).isEqualTo(1_000L);
        assertThat(c.getModelTraining().getLocalTraining()).isEqualTo(plan().getLocalTraining());
    }

    @Test
    void theRunSeedKeepsAllSixtyFourBits() throws Exception {
        ExecutionContract c = assemble(run(intent(true, true), "FedAvg"), plan(), bundle());
        assertThat(c.getSeed()).isEqualTo(-2L);   // uint64 18446744073709551614
    }

    @Test
    void theEffectiveDeploymentSettingsDecideTheSecurityPolicy() throws Exception {
        ExecutionContract secured = assemble(run(intent(true, true), "FedAvg"), plan(), bundle());
        assertThat(secured.getSecurity().getTransport()).isEqualTo(Transport.TRANSPORT_TLS_REQUIRED);
        assertThat(secured.getSecurity().getClientAuth()).isEqualTo(ClientAuth.CLIENT_AUTH_CONNECTION_TOKEN);

        ExecutionContract dev = assemble(run(intent(false, false), "FedAvg"), plan(), bundle());
        assertThat(dev.getSecurity().getTransport()).isEqualTo(Transport.TRANSPORT_PLAINTEXT_DEV);
        assertThat(dev.getSecurity().getClientAuth()).isEqualTo(ClientAuth.CLIENT_AUTH_DISABLED_DEV);
        assertThat(dev.getSecurity().getSecureAggregation()).isEqualTo(SecureAggregation.SECAGG_NONE);
        assertThat(dev.getSecurity().hasCentralDp()).isFalse();
    }

    @Test
    void centralDpIsDisclosedWithTheRecordedSettings() throws Exception {
        RunIntent dp = new RunIntent(TrainingArm.FULL, "tinynet_golden", null, true, 4.0, 1e-5, 1.0, true, true,
                900_000L);
        ExecutionContract c = assemble(run(dp, "FedAvg"), plan(), bundle());
        assertThat(c.getSecurity().getCentralDp().getTargetEpsilon()).isEqualTo(4.0);
        assertThat(c.getSecurity().getCentralDp().getDelta()).isEqualTo(1e-5);
        assertThat(c.getSecurity().getCentralDp().getClipNorm()).isEqualTo(1.0);
    }

    @Test
    void secureAggregationIsDisclosedWithItsThreshold() throws Exception {
        Run run = run(intent(true, true), "DeComFL");
        run.setSecureAggregation(true);
        run.setSecureAggThreshold(3);
        ExecutionContract c = assemble(run, plan(), bundle());
        assertThat(c.getSecurity().getSecureAggregation()).isEqualTo(SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR);
        assertThat(c.getSecurity().getSecureAggThreshold()).isEqualTo(3);
    }

    @Test
    void theStagedProgramsBecomeOnePortableCpuVariant() throws Exception {
        ArtifactVariant v = assemble(run(intent(true, true), "FedAvg"), plan(), bundle())
                .getModelTraining().getArtifacts(0);
        assertThat(v.getBackend()).isEqualTo(ArtifactBackend.BACKEND_EXECUTORCH_CPU);
        assertThat(v.getAbi()).isEqualTo("arm64-v8a");
        assertThat(v.getFilesList()).extracting(f -> f.getRelativePath())
                .containsExactly("loss.pte", "infer.pte", "trainable.pte");
        assertThat(v.getRequiredOperatorsList()).isEqualTo(bundle().requiredOperators());
        assertThat(v.getDeclaredStorageBytes()).isEqualTo(20_732L);
        assertThat(v.getDeclaredPeakMemoryBytes()).isEqualTo(8L << 20);
    }

    @Test
    void theModelRevisionIsTheDigestOfTheProgramSet() throws Exception {
        StagedBundle b = bundle();
        StringBuilder lines = new StringBuilder();
        b.modelFiles().stream().sorted((x, y) -> x.file().compareTo(y.file()))
                .forEach(f -> lines.append(f.file()).append(' ').append(f.sha256()).append('\n'));
        String expected = "sha256:" + HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(lines.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)));

        assertThat(assemble(run(intent(true, true), "FedAvg"), plan(), b).getModelTraining().getModelRevision())
                .isEqualTo(expected);
    }

    @Test
    void anIntentWithoutARecordedRoundTimeoutIsNotRepresentable() {
        Project p = new Project();
        p.setModelName("tinynet_golden");
        RunIntent v1 = RunIntent.capture(p, true, true);
        assertThatThrownBy(() -> assemble(run(v1, "FedAvg"), plan(), bundle()))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("round timeout");
    }

    @Test
    void aRunWithoutAnIntentIsNotRepresentable() {
        Run legacy = run(intent(true, true), "FedAvg");
        Run bare = new Run();
        bare.setId(legacy.getId());
        bare.setProjectId(legacy.getProjectId());
        bare.setRecipeKey("TINYNET_GOLDEN");
        bare.setStrategy("FedAvg");
        assertThatThrownBy(() -> assemble(bare, plan(), bundle())).isInstanceOf(NotRepresentableException.class);
    }

    @Test
    void aStrategyOutsideTheContractVocabularyIsNotRepresentable() {
        assertThatThrownBy(() -> assemble(run(intent(true, true), "FoT"), plan(), bundle()))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("FoT");
    }

    @Test
    void aRecipeOutsideTheContractVocabularyIsNotRepresentable() {
        Run run = run(intent(true, true), "FedAvg");
        run.setRecipeKey("FROZEN_DEMO");
        assertThatThrownBy(() -> assemble(run, plan(), bundle()))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("FROZEN_DEMO");
    }

    @Test
    void aBundleWhoseLayoutDisagreesWithThePlanIsNotRepresentable() throws Exception {
        StagedBundle b = bundle();
        StagedBundle reordered = new StagedBundle(b.modelFiles(), b.requiredOperators(), b.envelope(),
                List.of(b.paramLayout().get(1), b.paramLayout().get(0)));
        assertThatThrownBy(() -> assemble(run(intent(true, true), "FedAvg"), plan(), reordered))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("layout");
    }

    @Test
    void aWeightUpdateWithoutATrainableProgramIsNotRepresentable() throws Exception {
        StagedBundle b = bundle();
        StagedBundle noTrainable = new StagedBundle(b.modelFiles().subList(0, 2), b.requiredOperators(),
                b.envelope(), b.paramLayout());
        assertThatThrownBy(() -> assemble(run(intent(true, true), "FedAvg"), plan(), noTrainable))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("trainable.pte");
    }

    @Test
    void aPlanWithoutStateDigestsIsNotRepresentable() throws Exception {
        ModelTraining noInitial = plan().toBuilder().clearInitialStateSha256().build();
        assertThatThrownBy(() -> assemble(run(intent(true, true), "FedAvg"), noInitial, bundle()))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("initial");
    }

    @Test
    void anIncompleteCentralDpSettingIsNotRepresentable() {
        RunIntent dp = new RunIntent(TrainingArm.FULL, "tinynet_golden", null, true, 4.0, null, 1.0, true, true,
                900_000L);
        assertThatThrownBy(() -> assemble(run(dp, "FedAvg"), plan(), bundle()))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("central DP");
    }
}
