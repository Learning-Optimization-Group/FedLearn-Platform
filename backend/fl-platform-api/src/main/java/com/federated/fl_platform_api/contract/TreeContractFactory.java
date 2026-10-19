package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.federated.fl_platform_api.model.TrainingArm;
import com.federated.fl_platform_api.model.TrainingDataSource;
import com.fedlearn.contract.v1.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;
import java.util.UUID;

/** Contract version 2 for fixed-cohort, numeric binary XGBoost training. */
public final class TreeContractFactory {
    private TreeContractFactory() {}

    public static List<String> featureNames(String modelName) {
        if ("xgboost_binary".equals(modelName)) return List.of("f0","f1","f2","f3");
        if (!"xgboost_breast_cancer".equals(modelName)) {
            throw new IllegalArgumentException("Unsupported XGBoost model preset");
        }
        List<String> names = new ArrayList<>();
        for (String prefix : List.of("mean", "error", "worst")) {
            for (String measure : List.of("radius", "texture", "perimeter", "area", "smoothness",
                    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")) {
                names.add(prefix + "_" + measure);
            }
        }
        return List.copyOf(names);
    }

    private static TreeTraining training(String initialDigest, List<String> names) {
        return TreeTraining.newBuilder().setXgboostVersion("3.4.1").setObjective("binary:logistic")
            .setBooster("gbtree").addAllFeatureNames(names)
            .addAllClasses(List.of("0","1")).setBaseScore(.5).setLearningRate(.3)
            .setMaxDepth(3).setLocalBoostRounds(1).setTreeMethod("hist")
            .setInitialModelSha256(initialDigest).setInputDtype("float32").setMissingPolicy("reject")
            .setContributionCodec("xgboost-json-v1").setDataSource(DataSource.DATA_SOURCE_LOCAL_SNAPSHOT).build();
    }

    public static ExecutionContract assemble(Run run, Path initialModel) throws IOException {
        RunIntent intent=run.getIntent().orElseThrow(()->new IllegalArgumentException("Tree run has no intent"));
        if (!"XGBOOST".equals(run.getRecipeKey()) || !"XgbBagging".equals(run.getStrategy())
            || intent.trainingArm()!=TrainingArm.FULL || intent.dpEnabled() || run.isSecureAggregation()
            || intent.dataSource()!=TrainingDataSource.LOCAL_SNAPSHOT
            || run.getMinClients()!=run.getClientsPerRound() || intent.roundTimeoutMs()==null) {
            throw new IllegalArgumentException("XGBoost requires FULL, LOCAL_SNAPSHOT and a fixed non-DP cohort");
        }
        long size=Files.size(initialModel);
        if (size<=0 || size>16L*1024*1024) throw new IllegalArgumentException("Invalid tree artifact size");
        ExecutionContract result=ExecutionContract.newBuilder().setContractVersion(2).setMinClientProtocolVersion(3)
            .setRunId(run.getId().toString()).setProjectId(run.getProjectId().toString())
            .setRecipe(Recipe.RECIPE_XGBOOST).setStrategy(Strategy.STRATEGY_XGB_BAGGING)
            .setNumRounds(run.getNumRounds()).setClientsPerRound(run.getClientsPerRound())
            .setPartitioning(Partitioning.PARTITIONING_LOCAL).setSeed(run.getSeed())
            .setRound(RoundPolicy.newBuilder().setTimeoutMs(intent.roundTimeoutMs())
                .setOneAcceptedUpdatePerRound(true).setMaxTransientRetries(3).setRetryBackoffMs(1000))
            .setSecurity(SecurityPolicy.newBuilder()
                .setTransport(intent.tlsRequired()?Transport.TRANSPORT_TLS_REQUIRED:Transport.TRANSPORT_PLAINTEXT_DEV)
                .setClientAuth(intent.clientAuthRequired()?ClientAuth.CLIENT_AUTH_CONNECTION_TOKEN:ClientAuth.CLIENT_AUTH_DISABLED_DEV)
                .setSecureAggregation(SecureAggregation.SECAGG_NONE))
            .setTreeTraining(training(ExecutionContractStore.sha256(Files.readAllBytes(initialModel)),
                featureNames(intent.modelName()))).build();
        if (!valid(result,run.getId().toString(),run.getProjectId().toString())) {
            throw new IllegalArgumentException("Invalid tree execution contract");
        }
        return result;
    }

    public static boolean valid(ExecutionContract c, String runId, String projectId) {
        try {
            return c.getContractVersion()==2 && c.getMinClientProtocolVersion()==3 && c.hasTreeTraining()
                && c.getRunId().equals(runId) && UUID.fromString(runId).toString().equals(runId)
                && c.getProjectId().equals(projectId) && UUID.fromString(projectId).toString().equals(projectId)
                && c.getRecipe()==Recipe.RECIPE_XGBOOST && c.getStrategy()==Strategy.STRATEGY_XGB_BAGGING
                && c.getPartitioning()==Partitioning.PARTITIONING_LOCAL && c.hasSeed()
                && c.getNumRounds()>=1 && c.getNumRounds()<=100 && c.getClientsPerRound()>=1 && c.getClientsPerRound()<=100
                && c.getRound().getTimeoutMs()>=1 && c.getRound().getTimeoutMs()<=86_400_000
                && c.getRound().getOneAcceptedUpdatePerRound() && c.getRound().hasMaxTransientRetries()
                && c.getRound().getMaxTransientRetries()<=3 && c.getRound().getRetryBackoffMs()>=1
                && c.getRound().getRetryBackoffMs()<=60_000
                && List.of(Transport.TRANSPORT_TLS_REQUIRED,Transport.TRANSPORT_PLAINTEXT_DEV).contains(c.getSecurity().getTransport())
                && List.of(ClientAuth.CLIENT_AUTH_CONNECTION_TOKEN,ClientAuth.CLIENT_AUTH_DISABLED_DEV).contains(c.getSecurity().getClientAuth())
                && c.getSecurity().getSecureAggregation()==SecureAggregation.SECAGG_NONE
                && !c.getSecurity().hasSecureAggThreshold() && !c.getSecurity().hasCentralDp()
                && c.getTreeTraining().getInitialModelSha256().matches("[a-f0-9]{64}")
                && (c.getTreeTraining().getFeatureNamesList().equals(featureNames("xgboost_binary"))
                    || c.getTreeTraining().getFeatureNamesList().equals(featureNames("xgboost_breast_cancer")))
                && c.getTreeTraining().equals(training(c.getTreeTraining().getInitialModelSha256(),
                    c.getTreeTraining().getFeatureNamesList()));
        } catch (IllegalArgumentException e) { return false; }
    }
}
