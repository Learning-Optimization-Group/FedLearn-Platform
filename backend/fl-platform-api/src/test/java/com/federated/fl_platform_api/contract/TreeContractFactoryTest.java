package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.*;
import com.fedlearn.contract.v1.ExecutionContract;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import static org.assertj.core.api.Assertions.*;

class TreeContractFactoryTest {
    @TempDir Path directory;

    private Run run() {
        Project p = new Project();
        p.setModelType("XGBOOST"); p.setModelName("xgboost_binary");
        p.setTrainingArm(TrainingArm.FULL);
        Run r = new Run(); r.setId(UUID.randomUUID()); r.setProjectId(UUID.randomUUID());
        r.setRecipeKey("XGBOOST"); r.setStrategy("XgbBagging");
        r.setNumRounds(3); r.setMinClients(2); r.setClientsPerRound(2);
        r.setSeed(42L); r.setPartitioningMode(PartitioningMode.LOCAL);
        r.setIntent(RunIntent.capture(p, false, true, 120_000L, TrainingDataSource.LOCAL_SNAPSHOT));
        return r;
    }

    @Test void treeContractPinsArtifactAndRequiresItsOwnVersionAndProtocol() throws Exception {
        Path model = directory.resolve("initial.ubj"); Files.write(model, new byte[]{1,2,3});
        Run run = run(); ExecutionContract contract = TreeContractFactory.assemble(run, model);
        assertThat(contract.getContractVersion()).isEqualTo(2);
        assertThat(contract.getMinClientProtocolVersion()).isEqualTo(3);
        assertThat(contract.hasTreeTraining()).isTrue();
        assertThat(contract.hasModelTraining()).isFalse();
        assertThat(contract.getTreeTraining().getInitialModelSha256())
            .isEqualTo(ExecutionContractStore.sha256(Files.readAllBytes(model)));
        assertThat(TreeContractFactory.valid(contract, run.getId().toString(), run.getProjectId().toString())).isTrue();
        assertThat(TreeContractFactory.valid(contract.toBuilder().setContractVersion(1).build(),
            run.getId().toString(), run.getProjectId().toString())).isFalse();
        assertThat(TreeContractFactory.valid(contract.toBuilder().setTreeTraining(contract.getTreeTraining().toBuilder()
            .setBaseScore(.8)).build(),run.getId().toString(),run.getProjectId().toString())).isFalse();
        assertThat(ExecutionContractValidator.validate(contract, 2, run.getId().toString(),
            run.getProjectId().toString())).isNotEmpty();
        assertThat(ExecutionContractValidator.validate(contract.toBuilder().setContractVersion(1).build(), 2,
            run.getId().toString(),run.getProjectId().toString())).contains(
                new ContractIssue(com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNKNOWN_ENUM,"recipe"),
                new ContractIssue(com.fedlearn.contract.v1.ContractIssueCode.ISSUE_UNKNOWN_ENUM,"strategy"));
    }

    @Test void breastCancerPresetPublishesThirtyOrderedFeatures() throws Exception {
        Path model=directory.resolve("initial.ubj"); Files.write(model,new byte[]{1});
        Run r=run();
        Project project=new Project(); project.setModelType("XGBOOST");
        project.setModelName("xgboost_breast_cancer"); project.setTrainingArm(TrainingArm.FULL);
        r.setIntent(RunIntent.capture(project,false,true,120_000L,TrainingDataSource.LOCAL_SNAPSHOT));
        ExecutionContract c=TreeContractFactory.assemble(r,model);
        assertThat(c.getTreeTraining().getFeatureNamesList()).hasSize(30)
            .startsWith("mean_radius","mean_texture","mean_perimeter")
            .endsWith("worst_concave_points","worst_symmetry","worst_fractal_dimension");
        assertThat(TreeContractFactory.valid(c,r.getId().toString(),r.getProjectId().toString())).isTrue();
        assertThat(TreeContractFactory.valid(c.toBuilder().setTreeTraining(c.getTreeTraining().toBuilder()
            .setFeatureNames(0,"unknown")).build(),r.getId().toString(),r.getProjectId().toString())).isFalse();
        project.setModelName("unknown");
        r.setIntent(RunIntent.capture(project,false,true,120_000L,TrainingDataSource.LOCAL_SNAPSHOT));
        assertThatThrownBy(()->TreeContractFactory.assemble(r,model)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test void partialCohortsAndFixtureTrainingAreRefused() throws Exception {
        Path model=directory.resolve("initial.ubj"); Files.write(model,new byte[]{1});
        Run r=run(); r.setMinClients(1);
        assertThatThrownBy(()->TreeContractFactory.assemble(r,model)).isInstanceOf(IllegalArgumentException.class);
        r.setMinClients(2);
        Project p=new Project(); p.setModelType("XGBOOST"); p.setModelName("xgboost_binary"); p.setTrainingArm(TrainingArm.FULL);
        r.setIntent(RunIntent.capture(p,false,false,120_000L,TrainingDataSource.FIXTURE));
        assertThatThrownBy(()->TreeContractFactory.assemble(r,model)).isInstanceOf(IllegalArgumentException.class);
    }
}
