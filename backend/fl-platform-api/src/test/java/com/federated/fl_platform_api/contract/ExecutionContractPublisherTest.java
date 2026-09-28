package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.PartitioningMode;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.federated.fl_platform_api.model.TrainingDataSource;
import com.federated.fl_platform_api.model.TrainingArm;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.service.RegistryModelResolver;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.contains;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * The publisher turns each staging outcome into exactly one contract decision: a published contract when every
 * source agrees, otherwise UNAVAILABLE with the reason and detail of the first thing that did not.
 */
class ExecutionContractPublisherTest {

    @TempDir Path dir;

    private final RunRepository runs = mock(RunRepository.class);
    private final ProjectRepository projects = mock(ProjectRepository.class);
    private final RegistryModelResolver registry = mock(RegistryModelResolver.class);
    private final StagedBundleReader reader = mock(StagedBundleReader.class);
    private final ExecutionPlanResolver resolver = mock(ExecutionPlanResolver.class);
    private final ExecutionContractStore store = mock(ExecutionContractStore.class);
    private final com.federated.fl_platform_api.service.RunService runService =
            mock(com.federated.fl_platform_api.service.RunService.class);
    private ExecutionContractPublisher publisher;
    private Run run;
    private Project project;
    private Path initialModel;

    @BeforeEach
    void setUp() throws Exception {
        publisher = new ExecutionContractPublisher(runs, projects, registry, reader, resolver, store, runService);
        ReflectionTestUtils.setField(publisher, "maxTransientRetries", 3);
        ReflectionTestUtils.setField(publisher, "retryBackoffMs", 1_000L);

        initialModel = Files.writeString(dir.resolve("model.npz"), "initial");
        Files.setLastModifiedTime(initialModel, FileTime.from(Instant.now().minusSeconds(3600)));
        project = new Project();
        project.setId(UUID.fromString("9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64"));
        project.setModelName("tinynet_golden");
        project.setModelPath(initialModel.toString());
        project.setTrainingArm(TrainingArm.FULL);

        run = new Run();
        run.setId(UUID.fromString("4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17"));
        run.setProjectId(project.getId());
        run.setRecipeKey("TINYNET_GOLDEN");
        run.setStrategy("FedAvg");
        run.setNumRounds(3);
        run.setMinClients(4);
        run.setClientsPerRound(4);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setCreatedAt(Instant.now().minusSeconds(60));
        run.setIntent(RunIntent.capture(project, true, true, 900_000L));

        when(runs.findById(run.getId())).thenReturn(Optional.of(run));
        when(projects.findById(project.getId())).thenReturn(Optional.of(project));
        when(registry.resolveModelPath(project)).thenReturn(Optional.empty());
        when(reader.read(run.getId())).thenReturn(bundle());
        when(resolver.resolve(eq("TINYNET_GOLDEN"), eq("FedAvg"), eq("FULL"), any(), any())).thenReturn(plan());
        when(runService.legacyManifest(run)).thenReturn(legacyManifest());
    }

    /** The legacy manifest the backend emits for this run: it agrees with the contract being published. */
    private com.federated.fl_platform_api.dto.RunManifestDto legacyManifest() {
        var m = new com.federated.fl_platform_api.dto.RunManifestDto();
        m.setRunId(run.getId());
        m.setProjectId(run.getProjectId());
        m.setRecipeKey(run.getRecipeKey());
        m.setStrategy(run.getStrategy());
        m.setNumRounds(run.getNumRounds());
        m.setClientsPerRound(run.getClientsPerRound());
        m.setPartitioningMode(run.getPartitioningMode().name());
        m.setSeed(run.getSeed());
        m.setFirstOrderSupported(true);
        return m;
    }

    private static ModelTraining plan() throws Exception {
        return ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes()).getModelTraining()
                .toBuilder().clearModelRevision().clearArtifacts().build();
    }

    private static ModelTraining planOn(com.fedlearn.contract.v1.DataSource source) throws Exception {
        ModelTraining p = plan();
        return p.toBuilder().setData(p.getData().toBuilder().setSource(source)).build();
    }

    private static StagedBundle bundle() throws Exception {
        ArtifactVariant golden = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes())
                .getModelTraining().getArtifacts(0);
        List<String> names = List.of("loss.pte", "infer.pte", "trainable.pte");
        List<StagedBundle.ModelFile> files = new ArrayList<>();
        for (int i = 0; i < names.size(); i++) {
            files.add(new StagedBundle.ModelFile(names.get(i), golden.getFiles(i).getSha256(),
                    golden.getFiles(i).getByteSize()));
        }
        return new StagedBundle(files, golden.getRequiredOperatorsList(),
                new StagedBundle.ResourceEnvelope(8L << 20, 20_732L, 1_000L, 5_000L),
                List.of(new StagedBundle.LayoutEntry("fc1.weight", List.of(5L, 4L)),
                        new StagedBundle.LayoutEntry("fc1.bias", List.of(5L))));
    }

    @Test
    void aStagedRunIsPublishedFromItsSources() throws Exception {
        publisher.onStaged(run.getId());

        ArgumentCaptor<ExecutionContract> contract = ArgumentCaptor.forClass(ExecutionContract.class);
        verify(store).publish(eq(run), contract.capture());
        assertThat(contract.getValue().getRunId()).isEqualTo(run.getId().toString());
        assertThat(contract.getValue().getRound().getMaxTransientRetries()).isEqualTo(3);
        assertThat(ExecutionContractValidator.validate(contract.getValue(), ExecutionContractStore.SERVER_PROTOCOL_VERSION,
                run.getId().toString(), project.getId().toString())).isEmpty();
        verify(resolver).resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, initialModel);
    }

    @Test
    void aRunOnParticipantsOwnDataIsResolvedWithItsDataSource() throws Exception {
        run.setIntent(RunIntent.capture(project, true, true, 900_000L, TrainingDataSource.LOCAL_SNAPSHOT));
        when(resolver.resolve(eq("TINYNET_GOLDEN"), eq("FedAvg"), eq("FULL"), eq(TrainingDataSource.LOCAL_SNAPSHOT),
                any())).thenReturn(planOn(com.fedlearn.contract.v1.DataSource.DATA_SOURCE_LOCAL_SNAPSHOT));

        publisher.onStaged(run.getId());

        ArgumentCaptor<ExecutionContract> contract = ArgumentCaptor.forClass(ExecutionContract.class);
        verify(store).publish(eq(run), contract.capture());
        assertThat(contract.getValue().getModelTraining().getData().getSource())
                .isEqualTo(com.fedlearn.contract.v1.DataSource.DATA_SOURCE_LOCAL_SNAPSHOT);
    }

    // The resolver is a separate script: a plan that states another data source than the run's must not reach phones,
    // which would train the fixture batch on a run meant for their own data (or the reverse).
    @Test
    void aPlanStatingAnotherDataSourceThanTheRunsIsNotPublished() throws Exception {
        run.setIntent(RunIntent.capture(project, true, true, 900_000L, TrainingDataSource.LOCAL_SNAPSHOT));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.INVALID_CONTRACT),
                contains("data source"));
        verify(store, never()).publish(any(), any());
    }

    // A run started before intents recorded a data source trained the recipe's fixture batch; it still does.
    @Test
    void aRunWhoseIntentPredatesDataSourcesIsResolvedAsAFixtureRun() throws Exception {
        assertThat(run.getIntent().orElseThrow().dataSource()).isNull();

        publisher.onStaged(run.getId());

        verify(resolver).resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, initialModel);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"FedOpt, STRATEGY_FEDOPT", "Robust, STRATEGY_ROBUST"})
    void aTinyNetFedOptOrRobustRunIsPublished(String strategy, com.fedlearn.contract.v1.Strategy expected)
            throws Exception {
        run.setStrategy(strategy);
        when(resolver.resolve(eq("TINYNET_GOLDEN"), eq(strategy), eq("FULL"), any(), any())).thenReturn(plan());
        when(runService.legacyManifest(run)).thenReturn(legacyManifest());

        publisher.onStaged(run.getId());

        ArgumentCaptor<ExecutionContract> contract = ArgumentCaptor.forClass(ExecutionContract.class);
        verify(store).publish(eq(run), contract.capture());
        assertThat(contract.getValue().getStrategy()).isEqualTo(expected);
        assertThat(ExecutionContractValidator.validate(contract.getValue(), ExecutionContractStore.SERVER_PROTOCOL_VERSION,
                run.getId().toString(), project.getId().toString())).isEmpty();
    }

    @Test
    void aTinyNetDeComFLRunIsPublishedWithItsZerothOrderTraining() throws Exception {
        run.setStrategy("DeComFL");
        com.fedlearn.contract.v1.LocalTraining zeroth = plan().getLocalTraining().toBuilder()
                .clearLocalEpochs()
                .setZerothOrderSgd(com.fedlearn.contract.v1.ZerothOrderSgd.newBuilder()
                        .setLearningRate(0.001).setSmoothing(0.001).setNumLocalSteps(1).setNumPerturbations(10)
                        .setEstimator(com.fedlearn.contract.v1.GradientEstimator.ESTIMATOR_FORWARD)
                        .setRng(com.fedlearn.contract.v1.PerturbationRng.RNG_TORCH_CPU_RANDN_F32))
                .build();
        ModelTraining decomfl = plan().toBuilder()
                .setUpdateProtocol(com.fedlearn.contract.v1.UpdateProtocol.UPDATE_DECOMFL_SCALAR)
                .setLocalTraining(zeroth)
                .build();
        when(resolver.resolve(eq("TINYNET_GOLDEN"), eq("DeComFL"), eq("FULL"), any(), any())).thenReturn(decomfl);
        when(runService.legacyManifest(run)).thenReturn(legacyManifest());

        publisher.onStaged(run.getId());

        ArgumentCaptor<ExecutionContract> contract = ArgumentCaptor.forClass(ExecutionContract.class);
        verify(store).publish(eq(run), contract.capture());
        assertThat(contract.getValue().getStrategy()).isEqualTo(com.fedlearn.contract.v1.Strategy.STRATEGY_DECOMFL);
        assertThat(contract.getValue().getModelTraining().getLocalTraining().getZerothOrderSgd().getNumPerturbations())
                .isEqualTo(10);
        assertThat(ExecutionContractValidator.validate(contract.getValue(), ExecutionContractStore.SERVER_PROTOCOL_VERSION,
                run.getId().toString(), project.getId().toString())).isEmpty();
    }

    // FedProx's coefficient is part of the resolver's plan; the published contract carries it unchanged.
    @Test
    void aTinyNetFedProxRunIsPublishedWithItsProximalCoefficient() throws Exception {
        run.setStrategy("FedProx");
        when(resolver.resolve(eq("TINYNET_GOLDEN"), eq("FedProx"), eq("FULL"), any(), any()))
                .thenReturn(plan().toBuilder().setFedproxMu(0.1).build());
        when(runService.legacyManifest(run)).thenReturn(legacyManifest());

        publisher.onStaged(run.getId());

        ArgumentCaptor<ExecutionContract> contract = ArgumentCaptor.forClass(ExecutionContract.class);
        verify(store).publish(eq(run), contract.capture());
        assertThat(contract.getValue().getStrategy()).isEqualTo(com.fedlearn.contract.v1.Strategy.STRATEGY_FEDPROX);
        assertThat(contract.getValue().getModelTraining().getFedproxMu()).isEqualTo(0.1);
        assertThat(ExecutionContractValidator.validate(contract.getValue(), ExecutionContractStore.SERVER_PROTOCOL_VERSION,
                run.getId().toString(), project.getId().toString())).isEmpty();
    }

    @Test
    void aContinuedRunDigestsTheRegistryModelTheServerLoads() throws Exception {
        Path head = Files.writeString(dir.resolve("head.npz"), "registry head");
        Files.setLastModifiedTime(head, FileTime.from(Instant.now().minusSeconds(3600)));
        when(registry.resolveModelPath(project)).thenReturn(Optional.of(head.toString()));

        publisher.onStaged(run.getId());

        verify(resolver).resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, head);
    }

    @Test
    void anInitialModelChangedAfterTheRunStartedIsNotRepresentable() throws Exception {
        Files.setLastModifiedTime(initialModel, FileTime.from(Instant.now()));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.NOT_REPRESENTABLE),
                contains("changed after the run started"));
        verify(resolver, never()).resolve(anyString(), anyString(), anyString(), any(), any());
    }

    @Test
    void aProjectWithoutAnInitialModelFileIsNotRepresentable() throws Exception {
        project.setModelPath(null);

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.NOT_REPRESENTABLE),
                contains("no initial model file"));
    }

    @Test
    void anUnreadableBundleIsAStagingFailure() throws Exception {
        when(reader.read(run.getId())).thenThrow(new IOException("the staged loss.pte no longer matches"));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(run, ContractUnavailableReason.STAGING_FAILED,
                "the staged loss.pte no longer matches");
        verify(store, never()).publish(any(), any());
    }

    @Test
    void anUnrepresentablePlanIsRecordedWithItsReason() throws Exception {
        when(resolver.resolve(anyString(), anyString(), anyString(), any(), any()))
                .thenThrow(new NotRepresentableException("no execution contract v1 plan"));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(run, ContractUnavailableReason.NOT_REPRESENTABLE,
                "no execution contract v1 plan");
    }

    @Test
    void aResolverThatCannotRunIsRecordedAsNotRepresentable() throws Exception {
        when(resolver.resolve(anyString(), anyString(), anyString(), any(), any()))
                .thenThrow(new IOException("the execution-plan resolver exited 1"));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.NOT_REPRESENTABLE),
                contains("exited 1"));
    }

    @Test
    void anAssemblyDisagreementIsRecordedAsNotRepresentable() throws Exception {
        StagedBundle b = bundle();
        when(reader.read(run.getId())).thenReturn(new StagedBundle(b.modelFiles(), b.requiredOperators(),
                b.envelope(), List.of(b.paramLayout().get(1), b.paramLayout().get(0))));

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.NOT_REPRESENTABLE), contains("layout"));
    }

    @Test
    void aContractThatDisagreesWithTheLegacyManifestIsNotPublished() throws Exception {
        var legacy = legacyManifest();
        legacy.setFirstOrderSupported(false);
        when(runService.legacyManifest(run)).thenReturn(legacy);

        publisher.onStaged(run.getId());

        verify(store).markUnavailable(eq(run), eq(ContractUnavailableReason.INVALID_CONTRACT),
                contains("updateProtocol"));
        verify(store, never()).publish(any(), any());
    }

    @Test
    void aTimedOutStagingIsRecordedAsATimeout() {
        publisher.onStagingFailed(run.getId(), true, "stage timed out after 120s");
        verify(store).markUnavailable(run, ContractUnavailableReason.STAGING_TIMED_OUT, "stage timed out after 120s");
    }

    @Test
    void aFailedStagingIsRecordedAsAFailure() {
        publisher.onStagingFailed(run.getId(), false, "model-bundle auto-staging is disabled");
        verify(store).markUnavailable(run, ContractUnavailableReason.STAGING_FAILED,
                "model-bundle auto-staging is disabled");
    }

    @Test
    void aLegacyRunIsLeftAlone() {
        Run legacy = new Run();
        legacy.setId(UUID.randomUUID());
        when(runs.findById(legacy.getId())).thenReturn(Optional.of(legacy));

        publisher.onStaged(legacy.getId());
        publisher.onStagingFailed(legacy.getId(), false, "failed");

        verifyNoInteractions(store);
    }

    @Test
    void anUnknownRunIsLeftAlone() {
        UUID unknown = UUID.randomUUID();
        when(runs.findById(unknown)).thenReturn(Optional.empty());

        publisher.onStaged(unknown);

        verifyNoInteractions(store);
    }
}
