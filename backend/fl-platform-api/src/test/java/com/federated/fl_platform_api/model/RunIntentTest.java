package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** A run intent is captured once, from the project and the effective deployment settings, at start. */
class RunIntentTest {

    private static Project project() {
        Project p = new Project();
        p.setModelType("TINYNET_GOLDEN");
        p.setModelName("tinynet_golden");
        p.setTaskType("SEQ_CLASSIFICATION");
        p.setTrainingArm(TrainingArm.FROZEN_HEAD);
        return p;
    }

    @Test
    void captureCopiesTheTrainingSettingsAndDeploymentSettings() {
        RunIntent intent = RunIntent.capture(project(), true, false);
        assertThat(intent.trainingArm()).isEqualTo(TrainingArm.FROZEN_HEAD);
        assertThat(intent.modelName()).isEqualTo("tinynet_golden");
        assertThat(intent.taskType()).isEqualTo("SEQ_CLASSIFICATION");
        assertThat(intent.tlsRequired()).isTrue();
        assertThat(intent.clientAuthRequired()).isFalse();
        assertThat(intent.dpEnabled()).isFalse();
    }

    @Test
    void aProjectWithoutAnArmIsCapturedAsFull() {
        Project p = project();
        p.setTrainingArm(null);
        assertThat(RunIntent.capture(p, false, false).trainingArm()).isEqualTo(TrainingArm.FULL);
    }

    @Test
    void privacySettingsAreCapturedOnlyWhenPrivacyIsOn() {
        Project p = project();
        p.setDpTargetEpsilon(4.0);
        p.setDpDelta(1e-5);
        p.setDpClipNorm(1.0);
        RunIntent off = RunIntent.capture(p, false, false);
        assertThat(off.dpTargetEpsilon()).isNull();
        assertThat(off.dpDelta()).isNull();
        assertThat(off.dpClipNorm()).isNull();

        p.setDpEnabled(true);
        RunIntent on = RunIntent.capture(p, false, false);
        assertThat(on.dpEnabled()).isTrue();
        assertThat(on.dpTargetEpsilon()).isEqualTo(4.0);
        assertThat(on.dpDelta()).isEqualTo(1e-5);
        assertThat(on.dpClipNorm()).isEqualTo(1.0);
    }

    @Test
    void laterProjectEditsDoNotChangeACapturedIntent() {
        Project p = project();
        RunIntent intent = RunIntent.capture(p, false, false);
        p.setModelName("other");
        p.setTrainingArm(TrainingArm.FULL);
        p.setDpEnabled(true);
        assertThat(intent.modelName()).isEqualTo("tinynet_golden");
        assertThat(intent.trainingArm()).isEqualTo(TrainingArm.FROZEN_HEAD);
        assertThat(intent.dpEnabled()).isFalse();
    }

    @Test
    void aRunRoundTripsItsIntentAndALegacyRunHasNone() {
        Run run = new Run();
        assertThat(run.getIntent()).isEmpty();
        RunIntent intent = RunIntent.capture(project(), true, true);
        run.setIntent(intent);
        assertThat(run.getIntent()).contains(intent);
    }

    @Test
    void theRoundTimeoutIsCapturedAndRoundTrips() {
        RunIntent intent = RunIntent.capture(project(), true, false, 900_000L);
        assertThat(intent.roundTimeoutMs()).isEqualTo(900_000L);
        Run run = new Run();
        run.setIntent(intent);
        assertThat(run.getIntent()).contains(intent);
    }

    @Test
    void aSnapshotWithoutARoundTimeoutRoundTripsWithoutOne() {
        Run run = new Run();
        run.setIntent(RunIntent.capture(project(), true, false));
        assertThat(run.getIntent()).hasValueSatisfying(i -> assertThat(i.roundTimeoutMs()).isNull());
    }

    @Test
    void aRoundTimeoutMustBePositive() {
        assertThatThrownBy(() -> RunIntent.capture(project(), true, false, 0L))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void anIntentNeedsAnArmAndAModelName() {
        assertThatThrownBy(() -> new RunIntent(null, "m", null, false, null, null, null, false, false))
                .isInstanceOf(NullPointerException.class);
        assertThatThrownBy(() -> new RunIntent(TrainingArm.FULL, null, null, false, null, null, null, false, false))
                .isInstanceOf(NullPointerException.class);
    }
}
