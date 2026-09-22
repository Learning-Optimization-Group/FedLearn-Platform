package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.federated.fl_platform_api.model.TrainingArm;
import com.federated.fl_platform_api.repository.RunRepository;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * The FL server is spawned from the run's recorded intent, not from the project as it reads at spawn time, so the
 * server and the execution contract describe one snapshot.
 */
class FlServerManagerRunIntentTest {

    private static final String SCRIPT = "/x/run_fl_server.sh";

    private static Project project() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CIFAR_RESNET18");
        p.setModelName("project-model");
        p.setModelPath("/tmp/model.npz");
        p.setTrainingArm(TrainingArm.FULL);
        return p;
    }

    private static RunIntent recordedIntent() {
        return new RunIntent(TrainingArm.FROZEN_HEAD, "intent-model", null, true, 4.0, 1e-5, 1.0, true, true);
    }

    private static String after(List<String> argv, String flag) {
        int i = argv.indexOf(flag);
        assertThat(i).as("%s present in %s", flag, argv).isNotNegative();
        return argv.get(i + 1);
    }

    @Test
    void theCommandUsesTheRecordedIntentRatherThanTheProject() {
        List<String> argv = FlServerManager.buildServerCommand(project(), recordedIntent(), "FedAvg", 5, 2, 50000,
                SCRIPT, false, null, null, null, null);
        assertThat(after(argv, "--model-name")).isEqualTo("intent-model");
        assertThat(after(argv, "--training-arm")).isEqualTo("FROZEN_HEAD");
        assertThat(argv).contains("--dp-enabled");
        assertThat(after(argv, "--dp-target-epsilon")).isEqualTo("4.0");
    }

    @Test
    void thePreviousArityIsUnchangedForAProjectWithoutARecordedIntent() {
        Project p = project();
        List<String> previous = FlServerManager.buildServerCommand(p, "FedAvg", 5, 2, 50000, SCRIPT, false, null,
                null, null, null);
        List<String> explicit = FlServerManager.buildServerCommand(p, RunIntent.capture(p, false, false), "FedAvg",
                5, 2, 50000, SCRIPT, false, null, null, null, null);
        assertThat(explicit).isEqualTo(previous);
    }

    private static FlServerManager manager(RunRepository runs) {
        FlServerManager manager = new FlServerManager();
        ReflectionTestUtils.setField(manager, "runRepository", runs);
        ReflectionTestUtils.setField(manager, "requireTls", false);
        ReflectionTestUtils.setField(manager, "requireClientAuth", true);
        return manager;
    }

    @Test
    void theActiveRunsRecordedIntentIsUsed() {
        RunRepository runs = mock(RunRepository.class);
        Run run = new Run();
        run.setIntent(recordedIntent());
        Project p = project();
        p.setActiveRunId(UUID.randomUUID());
        when(runs.findById(p.getActiveRunId())).thenReturn(Optional.of(run));

        assertThat(manager(runs).intentFor(p)).isEqualTo(recordedIntent());
    }

    @Test
    void anActiveRunWithoutARecordedIntentIsRefused() {
        RunRepository runs = mock(RunRepository.class);
        Project p = project();
        p.setActiveRunId(UUID.randomUUID());
        when(runs.findById(p.getActiveRunId())).thenReturn(Optional.of(new Run()));

        assertThatThrownBy(() -> manager(runs).intentFor(p)).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void withoutAnActiveRunTheIntentIsCapturedAtSpawn() {
        Project p = project();
        FlServerManager manager = manager(mock(RunRepository.class));
        ReflectionTestUtils.setField(manager, "roundTimeoutSeconds", 120.0);
        assertThat(manager.intentFor(p)).isEqualTo(RunIntent.capture(p, false, true, 120_000L));
    }

    @Test
    void theServerIsGivenTheRecordedRoundTimeout() {
        Map<String, String> env = new HashMap<>(Map.of("FEDLEARN_ROUND_TIMEOUT_S", "7"));
        FlServerManager.applyRoundTimeout(env, RunIntent.capture(project(), false, false, 900_000L));
        assertThat(env).containsEntry("FEDLEARN_ROUND_TIMEOUT_S", "900");

        FlServerManager.applyRoundTimeout(env, RunIntent.capture(project(), false, false, 1_500L));
        assertThat(env).containsEntry("FEDLEARN_ROUND_TIMEOUT_S", "1.5");
    }

    @Test
    void aSnapshotWithoutARoundTimeoutLeavesTheInheritedSetting() {
        Map<String, String> env = new HashMap<>(Map.of("FEDLEARN_ROUND_TIMEOUT_S", "7"));
        FlServerManager.applyRoundTimeout(env, RunIntent.capture(project(), false, false));
        assertThat(env).containsEntry("FEDLEARN_ROUND_TIMEOUT_S", "7");
    }
}
