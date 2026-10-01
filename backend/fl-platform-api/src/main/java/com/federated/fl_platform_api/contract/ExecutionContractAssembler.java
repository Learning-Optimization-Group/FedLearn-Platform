package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.PartitioningMode;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.fedlearn.contract.v1.ArtifactBackend;
import com.fedlearn.contract.v1.ArtifactRef;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.CentralDp;
import com.fedlearn.contract.v1.ClientAuth;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import com.fedlearn.contract.v1.Partitioning;
import com.fedlearn.contract.v1.Recipe;
import com.fedlearn.contract.v1.RoundPolicy;
import com.fedlearn.contract.v1.SecureAggregation;
import com.fedlearn.contract.v1.SecurityPolicy;
import com.fedlearn.contract.v1.Strategy;
import com.fedlearn.contract.v1.TensorSpec;
import com.fedlearn.contract.v1.Transport;
import com.fedlearn.contract.v1.UpdateProtocol;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Assembles a run's execution contract from three sources, each authoritative for its part: the run record and its
 * intent snapshot (identity, rounds, security and deployment settings), the plan resolved by
 * fl-runtime/execution_plan.py (training, layout, data and state digests), and the staged bundle (the programs
 * clients load). Anything missing or contradictory makes the run not representable; nothing is inferred.
 * The result is validated when it is stored ({@link ExecutionContractStore#publish}).
 */
public final class ExecutionContractAssembler {

    /** The one v1 artifact variant: the staged ExecuTorch programs on the portable CPU runtime. */
    static final String PORTABLE_CPU_VARIANT = "portable-cpu-arm64-v8a";
    static final String ANDROID_ABI = "arm64-v8a";
    static final String TRAINABLE_PROGRAM = "trainable.pte";
    private static final List<String> MODEL_PROGRAMS = List.of("loss.pte", "infer.pte", TRAINABLE_PROGRAM);

    private static final Map<String, Strategy> STRATEGIES = Map.of(
            "DeComFL", Strategy.STRATEGY_DECOMFL,
            "FedAvg", Strategy.STRATEGY_FEDAVG,
            "FedProx", Strategy.STRATEGY_FEDPROX,
            "FedOpt", Strategy.STRATEGY_FEDOPT,
            "Robust", Strategy.STRATEGY_ROBUST);

    /** Server policy that the run record does not carry. */
    public record Policy(int clientProtocolVersion, int maxTransientRetries, long retryBackoffMs) {
    }

    private ExecutionContractAssembler() {
    }

    public static ExecutionContract assemble(Run run, ModelTraining plan, StagedBundle bundle, Policy policy)
            throws NotRepresentableException {
        RunIntent intent = run.getIntent().orElseThrow(() -> new NotRepresentableException(
                "the run predates the run-intent snapshot"));
        if (intent.roundTimeoutMs() == null) {
            throw new NotRepresentableException("the run intent predates the recorded round timeout");
        }
        ExecutionContract.Builder contract = ExecutionContract.newBuilder()
                .setContractVersion(ExecutionContractValidator.CONTRACT_VERSION)
                .setMinClientProtocolVersion(policy.clientProtocolVersion())
                .setRunId(run.getId().toString())
                .setProjectId(run.getProjectId().toString())
                .setRecipe(recipe(run.getRecipeKey()))
                .setStrategy(strategy(run.getStrategy()))
                .setNumRounds(run.getNumRounds())
                .setClientsPerRound(run.getClientsPerRound())
                .setPartitioning(run.getPartitioningMode() == PartitioningMode.LOCAL
                        ? Partitioning.PARTITIONING_LOCAL : Partitioning.PARTITIONING_SHARDED)
                .setRound(RoundPolicy.newBuilder()
                        .setTimeoutMs(intent.roundTimeoutMs())
                        .setOneAcceptedUpdatePerRound(true)
                        .setMaxTransientRetries(policy.maxTransientRetries())
                        .setRetryBackoffMs(policy.retryBackoffMs()))
                .setSecurity(security(run, intent))
                .setModelTraining(modelTraining(plan, bundle));
        if (run.getSeed() != null) {
            contract.setSeed(run.getSeed());   // the run seed's 64 bits, read as uint64
        }
        return contract.build();
    }

    private static Recipe recipe(String recipeKey) throws NotRepresentableException {
        try {
            return Recipe.valueOf("RECIPE_" + recipeKey);
        } catch (IllegalArgumentException | NullPointerException e) {
            throw new NotRepresentableException("recipe " + recipeKey + " has no execution contract identifier");
        }
    }

    private static Strategy strategy(String strategy) throws NotRepresentableException {
        Strategy s = strategy == null ? null : STRATEGIES.get(strategy);
        if (s == null) {
            throw new NotRepresentableException("strategy " + strategy + " has no execution contract identifier");
        }
        return s;
    }

    private static SecurityPolicy security(Run run, RunIntent intent) throws NotRepresentableException {
        SecurityPolicy.Builder security = SecurityPolicy.newBuilder()
                .setTransport(intent.tlsRequired() ? Transport.TRANSPORT_TLS_REQUIRED : Transport.TRANSPORT_PLAINTEXT_DEV)
                .setClientAuth(intent.clientAuthRequired()
                        ? ClientAuth.CLIENT_AUTH_CONNECTION_TOKEN : ClientAuth.CLIENT_AUTH_DISABLED_DEV);
        if (run.isSecureAggregation()) {
            if (run.getSecureAggThreshold() == null) {
                throw new NotRepresentableException("secure aggregation is on without a threshold");
            }
            security.setSecureAggregation(SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR)
                    .setSecureAggThreshold(run.getSecureAggThreshold());
        } else {
            security.setSecureAggregation(SecureAggregation.SECAGG_NONE);
        }
        if (intent.dpEnabled()) {
            if (!Project.isCompleteDpConfig(intent.dpTargetEpsilon(), intent.dpDelta(), intent.dpClipNorm())) {
                throw new NotRepresentableException("central DP is on with an incomplete configuration");
            }
            security.setCentralDp(CentralDp.newBuilder()
                    .setTargetEpsilon(intent.dpTargetEpsilon())
                    .setDelta(intent.dpDelta())
                    .setClipNorm(intent.dpClipNorm()));
        }
        return security.build();
    }

    private static ModelTraining modelTraining(ModelTraining plan, StagedBundle bundle)
            throws NotRepresentableException {
        if (plan.getFrozenStateSha256().isEmpty()) {
            throw new NotRepresentableException("the resolved plan has no frozen-state digest");
        }
        if (plan.getInitialStateSha256().isEmpty()) {
            throw new NotRepresentableException("the resolved plan has no initial-state digest");
        }
        requireLayoutAgreement(plan.getTrainableList(), bundle.paramLayout());
        requireBatchFits(plan.getLocalTraining().getBatchSize(), bundle.maxBatch());
        List<StagedBundle.ModelFile> programs = programs(bundle,
                plan.getUpdateProtocol() == UpdateProtocol.UPDATE_TRAINABLE_STATE_F32);
        ArtifactVariant.Builder variant = ArtifactVariant.newBuilder()
                .setVariantId(PORTABLE_CPU_VARIANT)
                .setBackend(ArtifactBackend.BACKEND_EXECUTORCH_CPU)
                .setAbi(ANDROID_ABI)
                .addAllRequiredOperators(bundle.requiredOperators())
                .setDeclaredPeakMemoryBytes(bundle.envelope().peakMemoryBytes())
                .setDeclaredStorageBytes(bundle.envelope().storageBytes())
                .setDeclaredProbeMs(bundle.envelope().probeMs())
                .setDeclaredTrainMs(bundle.envelope().trainMs());
        for (StagedBundle.ModelFile program : programs) {
            variant.addFiles(ArtifactRef.newBuilder()
                    .setRelativePath(program.file())
                    .setSha256(program.sha256())
                    .setByteSize(program.byteSize()));
        }
        return plan.toBuilder()
                .setModelRevision(modelRevision(programs))
                .clearArtifacts()
                .addArtifacts(variant)
                .build();
    }

    /** Every training step feeds batch_size examples through the staged programs, which take 1..maxBatch. */
    private static void requireBatchFits(int batchSize, int maxBatch) throws NotRepresentableException {
        if (batchSize > maxBatch) {
            throw new NotRepresentableException("the plan trains batches of " + batchSize
                    + " examples, but the staged programs take at most " + maxBatch + " examples per step");
        }
    }

    private static void requireLayoutAgreement(List<TensorSpec> planned, List<StagedBundle.LayoutEntry> staged)
            throws NotRepresentableException {
        List<StagedBundle.LayoutEntry> expected = new ArrayList<>();
        for (TensorSpec spec : planned) {
            expected.add(new StagedBundle.LayoutEntry(spec.getName(), spec.getShapeList()));
        }
        if (!expected.equals(staged)) {
            throw new NotRepresentableException("the staged bundle's trainable layout " + staged
                    + " disagrees with the resolved plan's " + expected);
        }
    }

    /** The staged programs in their canonical order; a weight update needs the trainable program. */
    private static List<StagedBundle.ModelFile> programs(StagedBundle bundle, boolean needsTrainable)
            throws NotRepresentableException {
        List<StagedBundle.ModelFile> programs = new ArrayList<>();
        for (String name : MODEL_PROGRAMS) {
            Optional<StagedBundle.ModelFile> file = bundle.modelFiles().stream()
                    .filter(f -> f.file().equals(name)).findFirst();
            if (file.isPresent()) {
                programs.add(file.get());
            } else if (!name.equals(TRAINABLE_PROGRAM) || needsTrainable) {
                throw new NotRepresentableException("the staged bundle has no " + name);
            }
        }
        return programs;
    }

    /** "sha256:" and the digest of the sorted "file digest" lines: an identity for the exact program set. */
    static String modelRevision(List<StagedBundle.ModelFile> programs) {
        StringBuilder lines = new StringBuilder();
        programs.stream().sorted(Comparator.comparing(StagedBundle.ModelFile::file))
                .forEach(f -> lines.append(f.file()).append(' ').append(f.sha256()).append('\n'));
        return "sha256:" + ExecutionContractStore.sha256(lines.toString().getBytes(StandardCharsets.UTF_8));
    }
}
