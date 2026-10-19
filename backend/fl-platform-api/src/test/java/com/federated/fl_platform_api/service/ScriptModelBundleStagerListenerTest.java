package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeoutException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

/**
 * Every staging attempt ends in exactly one outcome that listeners hear -- staged, or failed with whether it timed
 * out -- so a run waiting on its bundle (and its execution contract) is never left pending by a silent path.
 */
class ScriptModelBundleStagerListenerTest {

    private final List<String> heard = new ArrayList<>();

    private final ModelBundleStagingListener recorder = new ModelBundleStagingListener() {
        @Override
        public void onStaged(UUID runId) {
            heard.add("staged");
        }

        @Override
        public void onStagingFailed(UUID runId, boolean timedOut, String detail) {
            heard.add((timedOut ? "timed-out: " : "failed: ") + detail);
        }
    };

    private ScriptModelBundleStager stager(Path dir, boolean enabled, ScriptModelBundleStager.ProcessInvoker invoker) {
        ScriptModelBundleStager s = new ScriptModelBundleStager();
        ReflectionTestUtils.setField(s, "enabled", enabled);
        ReflectionTestUtils.setField(s, "modelBundleDir", dir.toString());
        ReflectionTestUtils.setField(s, "exportScript", "scripts/export_model.py");
        ReflectionTestUtils.setField(s, "stageScript", "scripts/stage_model_bundle.py");
        ReflectionTestUtils.setField(s, "fixtureRecipesCsv", "TINYNET_GOLDEN");
        ReflectionTestUtils.setField(s, "pythonExecutable", "python3");
        ReflectionTestUtils.setField(s, "timeoutSeconds", 5L);
        s.setExecutor(Runnable::run);
        s.setInvoker(invoker);
        s.setListeners(List.of(recorder));
        return s;
    }

    @Test
    void aSuccessfulStagingIsReported(@TempDir Path dir) {
        stager(dir, true, (cmd, t) -> 0).stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).containsExactly("staged");
    }

    @Test
    void anAlreadyStagedRunIsReportedStaged(@TempDir Path dir) throws Exception {
        UUID runId = UUID.randomUUID();
        Files.createDirectories(dir.resolve(runId.toString()));
        Files.writeString(dir.resolve(runId.toString()).resolve("manifest.json"), "{}");
        stager(dir, true, (cmd, t) -> {
            throw new AssertionError("must not stage twice");
        }).stageForRun(runId, "TINYNET_GOLDEN");
        assertThat(heard).containsExactly("staged");
    }

    @Test
    void aFailingExitIsReportedAsAFailure(@TempDir Path dir) {
        stager(dir, true, (cmd, t) -> 3).stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).singleElement().asString().startsWith("failed: ").contains("exited 3");
    }

    @Test
    void aTimeoutIsReportedAsATimeout(@TempDir Path dir) {
        stager(dir, true, (cmd, t) -> {
            throw new TimeoutException("stage timed out after 5s");
        }).stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).singleElement().asString().startsWith("timed-out: ");
    }

    @Test
    void anErrorIsReportedAsAFailure(@TempDir Path dir) {
        stager(dir, true, (cmd, t) -> {
            throw new IllegalStateException("invoker blew up");
        }).stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).singleElement().asString().startsWith("failed: ");
    }

    @Test
    void disabledStagingIsReportedAsAFailure(@TempDir Path dir) {
        stager(dir, false, (cmd, t) -> 0).stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).singleElement().asString().startsWith("failed: ").contains("disabled");
    }

    @Test
    void aDroppedStagingTaskIsReportedAsAFailure(@TempDir Path dir) {
        ScriptModelBundleStager s = stager(dir, true, (cmd, t) -> 0);
        s.setExecutor(task -> {
            throw new RejectedExecutionException("backlog full");
        });
        s.stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN");
        assertThat(heard).singleElement().asString().startsWith("failed: ");
    }

    @Test
    void aThrowingListenerNeverFailsTheCaller(@TempDir Path dir) {
        ScriptModelBundleStager s = stager(dir, true, (cmd, t) -> 0);
        s.setListeners(List.of(new ModelBundleStagingListener() {
            @Override
            public void onStaged(UUID runId) {
                throw new IllegalStateException("listener failure");
            }

            @Override
            public void onStagingFailed(UUID runId, boolean timedOut, String detail) {
                throw new IllegalStateException("listener failure");
            }
        }));
        assertThatCode(() -> s.stageForRun(UUID.randomUUID(), "TINYNET_GOLDEN")).doesNotThrowAnyException();
    }
}
