package com.federated.fl_platform_api.contract;

import com.fedlearn.contract.v1.Adam;
import com.fedlearn.contract.v1.AdamW;
import com.fedlearn.contract.v1.Arm;
import com.fedlearn.contract.v1.ArtifactBackend;
import com.fedlearn.contract.v1.ArtifactRef;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.BatchOrder;
import com.fedlearn.contract.v1.CentralDp;
import com.fedlearn.contract.v1.ClientAuth;
import com.fedlearn.contract.v1.ContractIssueCode;
import com.fedlearn.contract.v1.DType;
import com.fedlearn.contract.v1.DataSource;
import com.fedlearn.contract.v1.DataRequirement;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.GradientEstimator;
import com.fedlearn.contract.v1.IdentityVector;
import com.fedlearn.contract.v1.ImageToUnitTensor;
import com.fedlearn.contract.v1.LocalTraining;
import com.fedlearn.contract.v1.ModelTraining;
import com.fedlearn.contract.v1.NormalizeChannels;
import com.fedlearn.contract.v1.Objective;
import com.fedlearn.contract.v1.Partitioning;
import com.fedlearn.contract.v1.PerturbationRng;
import com.fedlearn.contract.v1.Recipe;
import com.fedlearn.contract.v1.Rmsprop;
import com.fedlearn.contract.v1.RoundPolicy;
import com.fedlearn.contract.v1.SecureAggregation;
import com.fedlearn.contract.v1.SecurityPolicy;
import com.fedlearn.contract.v1.Sgd;
import com.fedlearn.contract.v1.Strategy;
import com.fedlearn.contract.v1.Task;
import com.fedlearn.contract.v1.TensorSpec;
import com.fedlearn.contract.v1.Transform;
import com.fedlearn.contract.v1.Transport;
import com.fedlearn.contract.v1.UpdateProtocol;
import com.fedlearn.contract.v1.ZerothOrderSgd;
import com.google.protobuf.Internal;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.regex.Pattern;

import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_IDENTITY_MISMATCH;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_ARTIFACT;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_HASH;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_IDENTIFIER;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_OPTIMIZER;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_PATH;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_SECURITY;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_INVALID_STRATEGY_SETTINGS;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_MALFORMED_LAYOUT;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_MISSING_ARTIFACT;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_MISSING_FIELD;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_OUT_OF_RANGE;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNKNOWN_ENUM;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNSUPPORTED_CLIENT_PROTOCOL;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNSUPPORTED_COMBINATION;
import static com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNSUPPORTED_CONTRACT_VERSION;

/**
 * Execution contract v1 validation rules, shared with the Python and TypeScript readers.
 *
 * <p>The rules, limits and issue paths are specified in
 * framework/tests/fixtures/execution_contract_v1/README.md and pinned by the conformance corpus beside
 * it. The proto's unsigned integers arrive as Java {@code int}/{@code long}, so every bound treats them
 * as unsigned: a value at or above 2^31 (uint32) or 2^63 (uint64) reads as negative and is out of range.
 */
public final class ExecutionContractValidator {

    public static final int CONTRACT_VERSION = 1;

    static final long MAX_ROUNDS = 10_000;
    static final long MAX_CLIENTS_PER_ROUND = 10_000;
    static final long MAX_TIMEOUT_MS = 86_400_000;
    static final long MAX_DECLARED_MS = 86_400_000;
    static final long MAX_TRANSIENT_RETRIES = 10;
    static final long MAX_RETRY_BACKOFF_MS = 3_600_000;
    static final int MAX_TENSORS = 4096;
    static final int MAX_RANK = 8;
    static final long MAX_ELEMENTS = 2_147_483_647L;
    static final long MAX_LOCAL_EPOCHS = 1000;
    static final long MAX_LOCAL_STEPS = 1_000_000;
    static final long MAX_PERTURBATIONS = 10_000;
    static final long MAX_BATCH_SIZE = 65_536;
    static final long MAX_CLASSES = 1_000_000;
    static final int MAX_TRANSFORMS = 16;
    static final long MAX_IMAGE_SIDE = 4096;
    static final int MAX_VARIANTS = 16;
    static final int MAX_FILES = 64;
    static final int MAX_OPERATORS = 4096;
    static final long MAX_FILE_BYTES = 1L << 36;
    static final long MAX_DECLARED_BYTES = 1L << 40;
    static final int MAX_TENSOR_NAME_LENGTH = 256;
    static final int MAX_OPERATOR_LENGTH = 128;
    static final int MAX_PATH_LENGTH = 255;
    static final int MAX_PATH_SEGMENTS = 8;

    private static final Pattern UUID = Pattern.compile(
            "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");
    private static final Pattern MODEL_ID = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,127}");
    private static final Pattern REVISION = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private static final Pattern VARIANT_ID = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._-]{0,63}");
    private static final Pattern ABI = Pattern.compile("[a-z0-9][a-z0-9_-]{0,31}");
    private static final Pattern TENSOR_NAME = Pattern.compile("[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)*");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final Pattern PATH_SEGMENT = Pattern.compile("[A-Za-z0-9_-][A-Za-z0-9._-]*");
    private static final Pattern OPERATOR = Pattern.compile(
            "[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_][A-Za-z0-9_]*)?");

    /** The approved v1 matrix: (recipe, strategy, arm, task, objective, update protocol). */
    // FedOpt and Robust are first-order client training too; their server-side work is not client behavior.
    // FedProx is first-order client training plus the proximal term its fedproxMu states.
    private static final Set<List<Integer>> APPROVED_MATRIX = Set.of(
            tinyNetFirstOrder(Strategy.STRATEGY_FEDAVG_VALUE),
            tinyNetFirstOrder(Strategy.STRATEGY_FEDOPT_VALUE),
            tinyNetFirstOrder(Strategy.STRATEGY_ROBUST_VALUE),
            tinyNetFirstOrder(Strategy.STRATEGY_FEDPROX_VALUE),
            List.of(Recipe.RECIPE_TINYNET_GOLDEN_VALUE, Strategy.STRATEGY_DECOMFL_VALUE,
                    Arm.ARM_FULL_VALUE, Task.TASK_VECTOR_CLASSIFICATION_VALUE,
                    Objective.OBJECTIVE_CROSS_ENTROPY_VALUE, UpdateProtocol.UPDATE_DECOMFL_SCALAR_VALUE),
            // MLP (Stage 4): first-order training under the strategies whose servers send no client settings.
            mlpFirstOrder(Strategy.STRATEGY_FEDAVG_VALUE),
            mlpFirstOrder(Strategy.STRATEGY_ROBUST_VALUE),
            // CNN (Stage 4): the same first-order training on images, prepared by the contract's image transforms.
            cnnFirstOrder(Strategy.STRATEGY_FEDAVG_VALUE),
            cnnFirstOrder(Strategy.STRATEGY_ROBUST_VALUE));

    private static List<Integer> cnnFirstOrder(int strategy) {
        return List.of(Recipe.RECIPE_CNN_VALUE, strategy, Arm.ARM_FULL_VALUE,
                Task.TASK_IMAGE_CLASSIFICATION_VALUE, Objective.OBJECTIVE_CROSS_ENTROPY_VALUE,
                UpdateProtocol.UPDATE_TRAINABLE_STATE_F32_VALUE);
    }

    private static List<Integer> mlpFirstOrder(int strategy) {
        return List.of(Recipe.RECIPE_MLP_VALUE, strategy, Arm.ARM_FULL_VALUE,
                Task.TASK_VECTOR_CLASSIFICATION_VALUE, Objective.OBJECTIVE_CROSS_ENTROPY_VALUE,
                UpdateProtocol.UPDATE_TRAINABLE_STATE_F32_VALUE);
    }

    private static List<Integer> tinyNetFirstOrder(int strategy) {
        return List.of(Recipe.RECIPE_TINYNET_GOLDEN_VALUE, strategy, Arm.ARM_FULL_VALUE,
                Task.TASK_VECTOR_CLASSIFICATION_VALUE, Objective.OBJECTIVE_CROSS_ENTROPY_VALUE,
                UpdateProtocol.UPDATE_TRAINABLE_STATE_F32_VALUE);
    }

    private static final Set<Integer> CLASSIFICATION_TASKS = Set.of(
            Task.TASK_VECTOR_CLASSIFICATION_VALUE, Task.TASK_IMAGE_CLASSIFICATION_VALUE,
            Task.TASK_SEQUENCE_CLASSIFICATION_VALUE);
    private static final Set<Integer> TEXT_TASKS = Set.of(
            Task.TASK_SEQUENCE_CLASSIFICATION_VALUE, Task.TASK_CAUSAL_LM_VALUE);

    private final ExecutionContract contract;
    private final int readerProtocolVersion;
    private final String expectedRunId;
    private final String expectedProjectId;
    private final List<ContractIssue> issues = new ArrayList<>();

    private ExecutionContractValidator(ExecutionContract contract, int readerProtocolVersion,
                                       String expectedRunId, String expectedProjectId) {
        this.contract = contract;
        this.readerProtocolVersion = readerProtocolVersion;
        this.expectedRunId = expectedRunId;
        this.expectedProjectId = expectedProjectId;
    }

    /**
     * Every v1 issue in {@code contract}; an empty list means the contract is accepted.
     *
     * @param expectedRunId     the run the contract was delivered for, or null when not known
     * @param expectedProjectId the project the contract was delivered for, or null when not known
     */
    public static List<ContractIssue> validate(ExecutionContract contract, int readerProtocolVersion,
                                               String expectedRunId, String expectedProjectId) {
        if (contract.getContractVersion() != CONTRACT_VERSION) {
            return List.of(new ContractIssue(ISSUE_UNSUPPORTED_CONTRACT_VERSION, "contractVersion"));
        }
        return new ExecutionContractValidator(contract, readerProtocolVersion, expectedRunId,
                expectedProjectId).run();
    }

    private List<ContractIssue> run() {
        ExecutionContract c = contract;
        long minProtocol = Integer.toUnsignedLong(c.getMinClientProtocolVersion());
        if (minProtocol == 0 || minProtocol > readerProtocolVersion) {
            add(ISSUE_UNSUPPORTED_CLIENT_PROTOCOL, "minClientProtocolVersion");
        }
        identity(c.getRunId(), expectedRunId, "runId");
        identity(c.getProjectId(), expectedProjectId, "projectId");
        enumValue(Recipe::forNumber, c.getRecipeValue(), "recipe");
        enumValue(Strategy::forNumber, c.getStrategyValue(), "strategy");
        enumValue(Partitioning::forNumber, c.getPartitioningValue(), "partitioning");
        bounded(Integer.toUnsignedLong(c.getNumRounds()), 1, MAX_ROUNDS, "numRounds");
        bounded(Integer.toUnsignedLong(c.getClientsPerRound()), 1, MAX_CLIENTS_PER_ROUND, "clientsPerRound");
        if (c.hasRound()) {
            roundPolicy(c.getRound());
        } else {
            add(ISSUE_MISSING_FIELD, "round");
        }
        if (c.hasSecurity()) {
            security(c.getSecurity());
        } else {
            add(ISSUE_MISSING_FIELD, "security");
        }
        if (c.hasModelTraining()) {
            modelTraining(c.getModelTraining());
            matrix(c.getModelTraining());
        } else {
            add(ISSUE_MISSING_FIELD, "modelTraining");
        }
        return List.copyOf(issues);
    }

    private void add(ContractIssueCode code, String path) {
        issues.add(new ContractIssue(code, path));
    }

    private void check(boolean ok, ContractIssueCode code, String path) {
        if (!ok) {
            add(code, path);
        }
    }

    /** Signed comparison is unsigned-safe here: every bound is below 2^63 and every low is >= 0. */
    private void bounded(long value, long low, long high, String path) {
        check(value >= low && value <= high, ISSUE_OUT_OF_RANGE, path);
    }

    private static <E extends Internal.EnumLite> boolean known(IntFunction<E> forNumber, int value) {
        return value != 0 && forNumber.apply(value) != null;
    }

    private <E extends Internal.EnumLite> void enumValue(IntFunction<E> forNumber, int value, String path) {
        check(known(forNumber, value), ISSUE_UNKNOWN_ENUM, path);
    }

    private static boolean matches(Pattern pattern, String value) {
        return pattern.matcher(value).matches();
    }

    private static boolean positiveFinite(double value) {
        return Double.isFinite(value) && value > 0;
    }

    private static boolean nonnegativeFinite(double value) {
        return Double.isFinite(value) && value >= 0;
    }

    private static boolean openUnit(double value) {
        return Double.isFinite(value) && value > 0 && value < 1;
    }

    /** The element count of a valid shape, or -1 when the shape is not valid. */
    private static long elementCount(List<Long> shape) {
        if (shape.isEmpty() || shape.size() > MAX_RANK) {
            return -1;
        }
        long count = 1;
        for (long extent : shape) {
            if (extent < 1 || extent > MAX_ELEMENTS) {
                return -1;
            }
            count *= extent; // both factors are at most 2^31 - 1, so the product fits a long
            if (count > MAX_ELEMENTS) {
                return -1;
            }
        }
        return count;
    }

    private static boolean validPath(String path) {
        if (path.isEmpty() || path.length() > MAX_PATH_LENGTH) {
            return false;
        }
        String[] segments = path.split("/", -1);
        if (segments.length > MAX_PATH_SEGMENTS) {
            return false;
        }
        for (String segment : segments) {
            if (!matches(PATH_SEGMENT, segment)) {
                return false;
            }
        }
        return true;
    }

    private void identity(String value, String expected, String path) {
        if (!matches(UUID, value)) {
            add(ISSUE_INVALID_IDENTIFIER, path);
        } else if (expected != null && !value.equals(expected)) {
            add(ISSUE_IDENTITY_MISMATCH, path);
        }
    }

    private void matrix(ModelTraining mt) {
        ExecutionContract c = contract;
        boolean allKnown = known(Recipe::forNumber, c.getRecipeValue())
                && known(Strategy::forNumber, c.getStrategyValue())
                && known(Arm::forNumber, mt.getArmValue())
                && known(Task::forNumber, mt.getTaskValue())
                && known(Objective::forNumber, mt.getObjectiveValue())
                && known(UpdateProtocol::forNumber, mt.getUpdateProtocolValue());
        if (allKnown && !APPROVED_MATRIX.contains(List.of(c.getRecipeValue(), c.getStrategyValue(),
                mt.getArmValue(), mt.getTaskValue(), mt.getObjectiveValue(), mt.getUpdateProtocolValue()))) {
            add(ISSUE_UNSUPPORTED_COMBINATION, "");
        }
    }

    private void roundPolicy(RoundPolicy r) {
        bounded(r.getTimeoutMs(), 1, MAX_TIMEOUT_MS, "round.timeoutMs");
        check(r.getOneAcceptedUpdatePerRound(), ISSUE_OUT_OF_RANGE, "round.oneAcceptedUpdatePerRound");
        if (!r.hasMaxTransientRetries()) {
            add(ISSUE_MISSING_FIELD, "round.maxTransientRetries");
        } else {
            bounded(Integer.toUnsignedLong(r.getMaxTransientRetries()), 0, MAX_TRANSIENT_RETRIES,
                    "round.maxTransientRetries");
        }
        bounded(r.getRetryBackoffMs(), 1, MAX_RETRY_BACKOFF_MS, "round.retryBackoffMs");
    }

    private void security(SecurityPolicy s) {
        enumValue(Transport::forNumber, s.getTransportValue(), "security.transport");
        enumValue(ClientAuth::forNumber, s.getClientAuthValue(), "security.clientAuth");
        enumValue(SecureAggregation::forNumber, s.getSecureAggregationValue(), "security.secureAggregation");
        int strategy = contract.getStrategyValue();
        if (s.getSecureAggregationValue() == SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR_VALUE) {
            if (!s.hasSecureAggThreshold()) {
                add(ISSUE_MISSING_FIELD, "security.secureAggThreshold");
            } else {
                bounded(Integer.toUnsignedLong(s.getSecureAggThreshold()), 2,
                        Integer.toUnsignedLong(contract.getClientsPerRound()), "security.secureAggThreshold");
            }
            if (known(Strategy::forNumber, strategy) && strategy != Strategy.STRATEGY_DECOMFL_VALUE) {
                add(ISSUE_INVALID_SECURITY, "security.secureAggregation");
            }
        } else if (s.getSecureAggregationValue() == SecureAggregation.SECAGG_NONE_VALUE
                && s.hasSecureAggThreshold()) {
            add(ISSUE_INVALID_SECURITY, "security.secureAggThreshold");
        }
        if (s.hasCentralDp()) {
            CentralDp dp = s.getCentralDp();
            check(positiveFinite(dp.getTargetEpsilon()), ISSUE_OUT_OF_RANGE, "security.centralDp.targetEpsilon");
            check(openUnit(dp.getDelta()), ISSUE_OUT_OF_RANGE, "security.centralDp.delta");
            check(positiveFinite(dp.getClipNorm()), ISSUE_OUT_OF_RANGE, "security.centralDp.clipNorm");
        }
    }

    private void modelTraining(ModelTraining mt) {
        String p = "modelTraining";
        check(matches(MODEL_ID, mt.getModelId()), ISSUE_INVALID_IDENTIFIER, p + ".modelId");
        check(matches(REVISION, mt.getModelRevision()), ISSUE_INVALID_IDENTIFIER, p + ".modelRevision");
        enumValue(Arm::forNumber, mt.getArmValue(), p + ".arm");
        enumValue(Task::forNumber, mt.getTaskValue(), p + ".task");
        enumValue(Objective::forNumber, mt.getObjectiveValue(), p + ".objective");
        enumValue(UpdateProtocol::forNumber, mt.getUpdateProtocolValue(), p + ".updateProtocol");
        trainable(mt.getTrainableList());
        check(matches(SHA256, mt.getFrozenStateSha256()), ISSUE_INVALID_HASH, p + ".frozenStateSha256");
        check(matches(SHA256, mt.getInitialStateSha256()), ISSUE_INVALID_HASH, p + ".initialStateSha256");
        if (mt.hasLocalTraining()) {
            localTraining(mt.getLocalTraining());
            // DeComFL scalars come only from zeroth-order training, and zeroth-order training produces nothing else.
            LocalTraining.OptimizerCase optimizer = mt.getLocalTraining().getOptimizerCase();
            if (optimizer != LocalTraining.OptimizerCase.OPTIMIZER_NOT_SET
                    && known(UpdateProtocol::forNumber, mt.getUpdateProtocolValue()) &&
                    (optimizer == LocalTraining.OptimizerCase.ZEROTH_ORDER_SGD) !=
                            (mt.getUpdateProtocolValue() == UpdateProtocol.UPDATE_DECOMFL_SCALAR_VALUE)) {
                add(ISSUE_INVALID_STRATEGY_SETTINGS, p + ".updateProtocol");
            }
        } else {
            add(ISSUE_MISSING_FIELD, p + ".localTraining");
        }
        if (mt.hasData()) {
            data(mt.getData(), mt.getTaskValue());
        } else {
            add(ISSUE_MISSING_FIELD, p + ".data");
        }
        artifacts(mt.getArtifactsList());
        dropout(mt.getDropoutList());
        int strategy = contract.getStrategyValue();
        if (strategy == Strategy.STRATEGY_FEDPROX_VALUE) {
            if (!mt.hasFedproxMu()) {
                add(ISSUE_MISSING_FIELD, p + ".fedproxMu");
            } else if (!nonnegativeFinite(mt.getFedproxMu())) {
                add(ISSUE_OUT_OF_RANGE, p + ".fedproxMu");
            }
        } else if (known(Strategy::forNumber, strategy) && mt.hasFedproxMu()) {
            add(ISSUE_INVALID_STRATEGY_SETTINGS, p + ".fedproxMu");
        }
    }

    /** Each dropout layer names a module once, drops with a rate in [0, 1), and states how its masks are drawn. */
    private void dropout(List<com.fedlearn.contract.v1.DropoutLayer> layers) {
        String p = "modelTraining.dropout";
        if (layers.size() > MAX_TENSORS) {
            add(ISSUE_MALFORMED_LAYOUT, p);
            return;
        }
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < layers.size(); i++) {
            com.fedlearn.contract.v1.DropoutLayer layer = layers.get(i);
            String at = p + "[" + i + "]";
            boolean nameOk = layer.getModule().length() <= MAX_TENSOR_NAME_LENGTH
                    && matches(TENSOR_NAME, layer.getModule());
            check(nameOk && !seen.contains(layer.getModule()), ISSUE_MALFORMED_LAYOUT, at + ".module");
            seen.add(layer.getModule());
            check(Double.isFinite(layer.getRate()) && layer.getRate() >= 0 && layer.getRate() < 1, ISSUE_OUT_OF_RANGE,
                    at + ".rate");
            enumValue(com.fedlearn.contract.v1.DropoutMasks::forNumber, layer.getMasksValue(), at + ".masks");
        }
    }

    private void trainable(List<TensorSpec> tensors) {
        String p = "modelTraining.trainable";
        if (tensors.isEmpty() || tensors.size() > MAX_TENSORS) {
            add(ISSUE_MALFORMED_LAYOUT, p);
            return;
        }
        Set<String> seen = new HashSet<>();
        long total = 0;
        for (int i = 0; i < tensors.size(); i++) {
            TensorSpec tensor = tensors.get(i);
            String at = p + "[" + i + "]";
            boolean nameOk = tensor.getName().length() <= MAX_TENSOR_NAME_LENGTH
                    && matches(TENSOR_NAME, tensor.getName());
            check(nameOk && !seen.contains(tensor.getName()), ISSUE_MALFORMED_LAYOUT, at + ".name");
            seen.add(tensor.getName());
            long count = elementCount(tensor.getShapeList());
            if (count < 0) {
                add(ISSUE_MALFORMED_LAYOUT, at + ".shape");
            } else {
                total += count;
            }
            enumValue(DType::forNumber, tensor.getDtypeValue(), at + ".dtype");
        }
        if (total > MAX_ELEMENTS) {
            add(ISSUE_MALFORMED_LAYOUT, p);
        }
    }

    private void localTraining(LocalTraining lt) {
        String p = "modelTraining.localTraining";
        LocalTraining.OptimizerCase optimizer = lt.getOptimizerCase();
        if (optimizer == LocalTraining.OptimizerCase.ZEROTH_ORDER_SGD) {
            // Zeroth-order training counts its own steps; an epoch budget or a step cap would be a second one.
            check(lt.getLocalEpochs() == 0, ISSUE_INVALID_STRATEGY_SETTINGS, p + ".localEpochs");
            check(!lt.hasMaxLocalSteps(), ISSUE_INVALID_STRATEGY_SETTINGS, p + ".maxLocalSteps");
        } else {
            bounded(Integer.toUnsignedLong(lt.getLocalEpochs()), 1, MAX_LOCAL_EPOCHS, p + ".localEpochs");
            if (lt.hasMaxLocalSteps()) {
                bounded(Integer.toUnsignedLong(lt.getMaxLocalSteps()), 1, MAX_LOCAL_STEPS, p + ".maxLocalSteps");
            }
        }
        if (lt.hasGradientClipNorm()) {
            check(positiveFinite(lt.getGradientClipNorm()), ISSUE_OUT_OF_RANGE, p + ".gradientClipNorm");
        }
        switch (optimizer) {
            case SGD -> sgd(lt.getSgd(), p + ".sgd");
            case ADAM -> {
                Adam a = lt.getAdam();
                adam(a.getLearningRate(), a.getBeta1(), a.getBeta2(), a.getEpsilon(), a.hasWeightDecay(),
                        a.getWeightDecay(), a.hasAmsgrad(), p + ".adam");
            }
            case ADAMW -> {
                AdamW a = lt.getAdamw();
                adam(a.getLearningRate(), a.getBeta1(), a.getBeta2(), a.getEpsilon(), a.hasWeightDecay(),
                        a.getWeightDecay(), a.hasAmsgrad(), p + ".adamw");
            }
            case RMSPROP -> rmsprop(lt.getRmsprop(), p + ".rmsprop");
            case ZEROTH_ORDER_SGD -> zerothOrderSgd(lt.getZerothOrderSgd(), p + ".zerothOrderSgd");
            default -> add(ISSUE_MISSING_FIELD, p + ".optimizer");
        }
        present(lt.hasResetOptimizerEachRound(), p + ".resetOptimizerEachRound");
        bounded(Integer.toUnsignedLong(lt.getBatchSize()), 1, MAX_BATCH_SIZE, p + ".batchSize");
        present(lt.hasDropLast(), p + ".dropLast");
        enumValue(BatchOrder::forNumber, lt.getBatchOrderValue(), p + ".batchOrder");
    }

    private void zerothOrderSgd(ZerothOrderSgd zo, String p) {
        check(positiveFinite(zo.getLearningRate()), ISSUE_OUT_OF_RANGE, p + ".learningRate");
        check(positiveFinite(zo.getSmoothing()), ISSUE_OUT_OF_RANGE, p + ".smoothing");
        bounded(Integer.toUnsignedLong(zo.getNumLocalSteps()), 1, MAX_LOCAL_STEPS, p + ".numLocalSteps");
        bounded(Integer.toUnsignedLong(zo.getNumPerturbations()), 1, MAX_PERTURBATIONS, p + ".numPerturbations");
        enumValue(GradientEstimator::forNumber, zo.getEstimatorValue(), p + ".estimator");
        enumValue(PerturbationRng::forNumber, zo.getRngValue(), p + ".rng");
    }

    private boolean present(boolean has, String path) {
        check(has, ISSUE_MISSING_FIELD, path);
        return has;
    }

    /** Checks a required explicit-presence value; true when present and valid. */
    private boolean nonnegative(boolean has, double value, String path) {
        if (!present(has, path)) {
            return false;
        }
        boolean ok = nonnegativeFinite(value);
        check(ok, ISSUE_OUT_OF_RANGE, path);
        return ok;
    }

    private void sgd(Sgd sgd, String p) {
        check(positiveFinite(sgd.getLearningRate()), ISSUE_OUT_OF_RANGE, p + ".learningRate");
        boolean momentumOk = nonnegative(sgd.hasMomentum(), sgd.getMomentum(), p + ".momentum");
        boolean dampeningOk = nonnegative(sgd.hasDampening(), sgd.getDampening(), p + ".dampening");
        nonnegative(sgd.hasWeightDecay(), sgd.getWeightDecay(), p + ".weightDecay");
        if (present(sgd.hasNesterov(), p + ".nesterov") && sgd.getNesterov() && momentumOk && dampeningOk
                && !(sgd.getMomentum() > 0 && sgd.getDampening() == 0)) {
            add(ISSUE_INVALID_OPTIMIZER, p + ".nesterov");
        }
    }

    private void adam(double learningRate, double beta1, double beta2, double epsilon, boolean hasWeightDecay,
                      double weightDecay, boolean hasAmsgrad, String p) {
        check(positiveFinite(learningRate), ISSUE_OUT_OF_RANGE, p + ".learningRate");
        check(openUnit(beta1), ISSUE_OUT_OF_RANGE, p + ".beta1");
        check(openUnit(beta2), ISSUE_OUT_OF_RANGE, p + ".beta2");
        check(positiveFinite(epsilon), ISSUE_OUT_OF_RANGE, p + ".epsilon");
        nonnegative(hasWeightDecay, weightDecay, p + ".weightDecay");
        present(hasAmsgrad, p + ".amsgrad");
    }

    private void rmsprop(Rmsprop rms, String p) {
        check(positiveFinite(rms.getLearningRate()), ISSUE_OUT_OF_RANGE, p + ".learningRate");
        check(openUnit(rms.getAlpha()), ISSUE_OUT_OF_RANGE, p + ".alpha");
        check(positiveFinite(rms.getEpsilon()), ISSUE_OUT_OF_RANGE, p + ".epsilon");
        nonnegative(rms.hasWeightDecay(), rms.getWeightDecay(), p + ".weightDecay");
        nonnegative(rms.hasMomentum(), rms.getMomentum(), p + ".momentum");
        present(rms.hasCentered(), p + ".centered");
    }

    private void data(DataRequirement data, int trainingTask) {
        String p = "modelTraining.data";
        int task = data.getTaskValue();
        boolean taskKnown = known(Task::forNumber, task);
        if (!taskKnown) {
            add(ISSUE_UNKNOWN_ENUM, p + ".task");
        } else if (known(Task::forNumber, trainingTask) && trainingTask != task) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, p + ".task");
        }
        boolean shapeOk = elementCount(data.getInputShapeList()) >= 0;
        check(shapeOk, ISSUE_INVALID_DATA_REQUIREMENT, p + ".inputShape");
        enumValue(DType::forNumber, data.getInputDtypeValue(), p + ".inputDtype");
        enumValue(DataSource::forNumber, data.getSourceValue(), p + ".source");
        long classCount = Integer.toUnsignedLong(data.getClassCount());
        if (CLASSIFICATION_TASKS.contains(task)) {
            bounded(classCount, 2, MAX_CLASSES, p + ".classCount");
        } else if (task == Task.TASK_CAUSAL_LM_VALUE && classCount != 0) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, p + ".classCount");
        }
        check(matches(REVISION, data.getLabelSchemaId()), ISSUE_INVALID_IDENTIFIER, p + ".labelSchemaId");
        List<Transform> transforms = data.getTransformsList();
        if (transforms.isEmpty() || transforms.size() > MAX_TRANSFORMS) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, p + ".transforms");
        } else {
            for (int i = 0; i < transforms.size(); i++) {
                Transform transform = transforms.get(i);
                String at = p + ".transforms[" + i + "]";
                switch (transform.getOperationCase()) {
                    case OPERATION_NOT_SET -> add(ISSUE_MISSING_FIELD, at + ".operation");
                    case IDENTITY_VECTOR -> identityVector(transform.getIdentityVector(), data, taskKnown, shapeOk, at);
                    case IMAGE_TO_UNIT_TENSOR ->
                            imageToUnitTensor(transform.getImageToUnitTensor(), data, taskKnown, shapeOk, i, at);
                    case NORMALIZE_CHANNELS ->
                            normalizeChannels(transform.getNormalizeChannels(), data, taskKnown, i, at);
                }
            }
        }
        if (shapeOk && task == Task.TASK_IMAGE_CLASSIFICATION_VALUE && data.getInputShapeCount() != 3) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, p + ".inputShape");
        }
        boolean hasTokenizer = data.hasTokenizer();
        if (TEXT_TASKS.contains(task) && !hasTokenizer) {
            add(ISSUE_MISSING_FIELD, p + ".tokenizer");
        } else if (taskKnown && !TEXT_TASKS.contains(task) && hasTokenizer) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, p + ".tokenizer");
        }
        if (hasTokenizer) {
            artifactRef(data.getTokenizer(), p + ".tokenizer");
        }
    }

    private void identityVector(IdentityVector identity, DataRequirement data, boolean taskKnown, boolean shapeOk,
                                String at) {
        long width = Integer.toUnsignedLong(identity.getWidth());
        String path = at + ".identityVector.width";
        if (width < 1 || width > MAX_ELEMENTS) {
            add(ISSUE_OUT_OF_RANGE, path);
        } else if (taskKnown && (data.getTaskValue() != Task.TASK_VECTOR_CLASSIFICATION_VALUE
                || (shapeOk && !data.getInputShapeList().equals(List.of(width))))) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, path);
        }
    }

    /** First in an image task, its [channels, height, width] the input shape. */
    private void imageToUnitTensor(ImageToUnitTensor image, DataRequirement data, boolean taskKnown, boolean shapeOk,
                                   int index, String at) {
        String path = at + ".imageToUnitTensor";
        long height = Integer.toUnsignedLong(image.getHeight());
        long width = Integer.toUnsignedLong(image.getWidth());
        long channels = Integer.toUnsignedLong(image.getChannels());
        boolean inRange = true;
        if (height < 1 || height > MAX_IMAGE_SIDE) {
            add(ISSUE_OUT_OF_RANGE, path + ".height");
            inRange = false;
        }
        if (width < 1 || width > MAX_IMAGE_SIDE) {
            add(ISSUE_OUT_OF_RANGE, path + ".width");
            inRange = false;
        }
        if (channels != 1 && channels != 3) {
            add(ISSUE_OUT_OF_RANGE, path + ".channels");
            inRange = false;
        }
        if (taskKnown && (data.getTaskValue() != Task.TASK_IMAGE_CLASSIFICATION_VALUE || index != 0
                || (inRange && shapeOk && !data.getInputShapeList().equals(List.of(channels, height, width))))) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, path);
        }
    }

    /** Directly after the image conversion, one finite mean and one finite, positive std per channel. */
    private void normalizeChannels(NormalizeChannels norm, DataRequirement data, boolean taskKnown, int index,
                                   String at) {
        String path = at + ".normalizeChannels";
        for (int k = 0; k < norm.getMeanCount(); k++) {
            check(Float.isFinite(norm.getMean(k)), ISSUE_OUT_OF_RANGE, path + ".mean[" + k + "]");
        }
        for (int k = 0; k < norm.getStdCount(); k++) {
            float std = norm.getStd(k);
            check(Float.isFinite(std) && std > 0, ISSUE_OUT_OF_RANGE, path + ".std[" + k + "]");
        }
        Transform first = data.getTransforms(0);
        boolean followsImage = index == 1
                && first.getOperationCase() == Transform.OperationCase.IMAGE_TO_UNIT_TENSOR;
        if (taskKnown && (data.getTaskValue() != Task.TASK_IMAGE_CLASSIFICATION_VALUE || !followsImage
                || norm.getMeanCount() != norm.getStdCount()
                || norm.getMeanCount() != Integer.toUnsignedLong(first.getImageToUnitTensor().getChannels()))) {
            add(ISSUE_INVALID_DATA_REQUIREMENT, path);
        }
    }

    /** Checks one reference; true when its byte size is in range. */
    private boolean artifactRef(ArtifactRef ref, String p) {
        check(validPath(ref.getRelativePath()), ISSUE_INVALID_PATH, p + ".relativePath");
        check(matches(SHA256, ref.getSha256()), ISSUE_INVALID_HASH, p + ".sha256");
        boolean sizeOk = ref.getByteSize() >= 1 && ref.getByteSize() <= MAX_FILE_BYTES;
        check(sizeOk, ISSUE_OUT_OF_RANGE, p + ".byteSize");
        return sizeOk;
    }

    private void artifacts(List<ArtifactVariant> variants) {
        String p = "modelTraining.artifacts";
        if (variants.isEmpty()) {
            add(ISSUE_MISSING_ARTIFACT, p);
            return;
        }
        if (variants.size() > MAX_VARIANTS) {
            add(ISSUE_INVALID_ARTIFACT, p);
            return;
        }
        Set<String> seenIds = new HashSet<>();
        boolean hasCpu = false;
        for (int i = 0; i < variants.size(); i++) {
            ArtifactVariant variant = variants.get(i);
            String at = p + "[" + i + "]";
            if (!matches(VARIANT_ID, variant.getVariantId())) {
                add(ISSUE_INVALID_IDENTIFIER, at + ".variantId");
            } else if (seenIds.contains(variant.getVariantId())) {
                add(ISSUE_INVALID_ARTIFACT, at + ".variantId");
            }
            seenIds.add(variant.getVariantId());
            enumValue(ArtifactBackend::forNumber, variant.getBackendValue(), at + ".backend");
            hasCpu |= variant.getBackendValue() == ArtifactBackend.BACKEND_EXECUTORCH_CPU_VALUE;
            check(matches(ABI, variant.getAbi()), ISSUE_INVALID_IDENTIFIER, at + ".abi");
            boolean sizesOk = files(variant.getFilesList(), at + ".files");
            operators(variant.getRequiredOperatorsList(), at + ".requiredOperators");
            bounded(variant.getDeclaredPeakMemoryBytes(), 1, MAX_DECLARED_BYTES, at + ".declaredPeakMemoryBytes");
            long storage = variant.getDeclaredStorageBytes();
            boolean storageOk = storage >= 1 && storage <= MAX_DECLARED_BYTES;
            check(storageOk, ISSUE_OUT_OF_RANGE, at + ".declaredStorageBytes");
            bounded(variant.getDeclaredProbeMs(), 1, MAX_DECLARED_MS, at + ".declaredProbeMs");
            bounded(variant.getDeclaredTrainMs(), 1, MAX_DECLARED_MS, at + ".declaredTrainMs");
            if (storageOk && sizesOk) {
                long total = 0; // at most MAX_FILES files of at most 2^36 bytes each
                for (ArtifactRef ref : variant.getFilesList()) {
                    total += ref.getByteSize();
                }
                check(total <= storage, ISSUE_INVALID_ARTIFACT, at + ".declaredStorageBytes");
            }
        }
        check(hasCpu, ISSUE_MISSING_ARTIFACT, p);
    }

    /** Checks a variant's files; true when there are 1..MAX_FILES, all with sizes in range. */
    private boolean files(List<ArtifactRef> files, String p) {
        if (files.isEmpty()) {
            add(ISSUE_MISSING_ARTIFACT, p);
            return false;
        }
        if (files.size() > MAX_FILES) {
            add(ISSUE_INVALID_ARTIFACT, p);
            return false;
        }
        Set<String> seenPaths = new HashSet<>();
        boolean allSizesOk = true;
        for (int j = 0; j < files.size(); j++) {
            ArtifactRef ref = files.get(j);
            String at = p + "[" + j + "]";
            allSizesOk &= artifactRef(ref, at);
            if (validPath(ref.getRelativePath())) {
                String folded = ref.getRelativePath().toLowerCase(Locale.ROOT);
                check(seenPaths.add(folded), ISSUE_INVALID_PATH, at + ".relativePath");
            }
        }
        return allSizesOk;
    }

    private void operators(List<String> operators, String p) {
        if (operators.isEmpty() || operators.size() > MAX_OPERATORS) {
            add(ISSUE_INVALID_ARTIFACT, p);
            return;
        }
        Set<String> seen = new HashSet<>();
        for (int k = 0; k < operators.size(); k++) {
            String operator = operators.get(k);
            boolean valid = operator.length() <= MAX_OPERATOR_LENGTH && matches(OPERATOR, operator);
            check(seen.add(operator) && valid, ISSUE_INVALID_ARTIFACT, p + "[" + k + "]");
        }
    }
}
