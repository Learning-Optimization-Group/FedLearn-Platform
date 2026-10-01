package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.TrainingDataSource;
import com.fedlearn.contract.v1.ModelTraining;
import com.google.protobuf.util.JsonFormat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The backend's side of fl-runtime/execution_plan.py: it is invoked with a strict argv and its printed plan is
 * parsed strictly. The Python side's behaviour is tested in fl-runtime; here a stand-in wrapper prints fixed output.
 */
@DisabledOnOs(OS.WINDOWS)
class ScriptExecutionPlanResolverTest {

    private static ScriptExecutionPlanResolver resolver(Path wrapper) {
        ScriptExecutionPlanResolver resolver = new ScriptExecutionPlanResolver();
        ReflectionTestUtils.setField(resolver, "wrapperPath", wrapper.toString());
        ReflectionTestUtils.setField(resolver, "timeoutSeconds", 30L);
        return resolver;
    }

    /** A wrapper that records its arguments and prints {@code stdout}, then exits {@code exit}. */
    private static Path wrapper(Path dir, String stdout, int exit) throws IOException {
        Path script = dir.resolve("run_execution_plan.sh");
        Files.writeString(script, "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"" + dir.resolve("args.txt")
                + "\"\necho 'noise on stderr' >&2\ncat <<'JSON'\n" + stdout + "\nJSON\nexit " + exit + "\n");
        return script;
    }

    private static String golden() throws Exception {
        ModelTraining plan = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes())
                .getModelTraining();
        return "{\"representable\": true, \"modelTraining\": "
                + JsonFormat.printer().omittingInsignificantWhitespace().print(plan) + "}";
    }

    @Test
    void thePrintedPlanIsParsed(@TempDir Path dir) throws Exception {
        ModelTraining plan = resolver(wrapper(dir, golden(), 0))
                .resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, Path.of("/models/p.npz"));
        assertThat(plan).isEqualTo(ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes())
                .getModelTraining());
        assertThat(Files.readAllLines(dir.resolve("args.txt"))).containsExactly(
                "--recipe", "TINYNET_GOLDEN", "--strategy", "FedAvg", "--training-arm", "FULL",
                "--data-source", "FIXTURE", "--initial-state=/models/p.npz");
    }

    @Test
    void theRunsDataSourceIsPassedToTheScript(@TempDir Path dir) throws Exception {
        resolver(wrapper(dir, golden(), 0))
                .resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.LOCAL_SNAPSHOT, Path.of("/models/p.npz"));
        assertThat(Files.readAllLines(dir.resolve("args.txt"))).containsSubsequence("--data-source", "LOCAL_SNAPSHOT");
    }

    @Test
    void aMissingDataSourceIsRefusedBeforeSpawning(@TempDir Path dir) throws Exception {
        assertThatThrownBy(() -> resolver(wrapper(dir, golden(), 0))
                .resolve("TINYNET_GOLDEN", "FedAvg", "FULL", null, Path.of("/models/p.npz")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThat(Files.exists(dir.resolve("args.txt"))).isFalse();
    }

    @Test
    void anUnrepresentableRunIsReportedWithTheScriptsReason(@TempDir Path dir) {
        assertThatThrownBy(() -> resolver(wrapper(dir, "{\"representable\": false, \"reason\": \"no v1 plan\"}", 0))
                .resolve("CNN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, Path.of("/models/p.npz")))
                .isInstanceOf(NotRepresentableException.class).hasMessageContaining("no v1 plan");
    }

    @Test
    void aFailingScriptIsAnError(@TempDir Path dir) {
        assertThatThrownBy(() -> resolver(wrapper(dir, "", 2))
                .resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, Path.of("/models/p.npz")))
                .isInstanceOf(IOException.class).hasMessageContaining("exited 2");
    }

    @Test
    void unparseableOutputIsAnError(@TempDir Path dir) {
        assertThatThrownBy(() -> resolver(wrapper(dir, "{\"representable\": true, \"modelTraining\": {\"nope\": 1}}", 0))
                .resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, Path.of("/models/p.npz")))
                .isInstanceOf(IOException.class);
    }

    @Test
    void argumentsOutsideTheirVocabularyAreRefusedBeforeSpawning(@TempDir Path dir) throws Exception {
        ScriptExecutionPlanResolver r = resolver(wrapper(dir, golden(), 0));
        assertThatThrownBy(() -> r.resolve("--help", "FedAvg", "FULL", TrainingDataSource.FIXTURE, Path.of("/m.npz")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> r.resolve("TINYNET_GOLDEN", "Fed Avg", "FULL", TrainingDataSource.FIXTURE, Path.of("/m.npz")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> r.resolve("TINYNET_GOLDEN", "FedAvg", "full", TrainingDataSource.FIXTURE, Path.of("/m.npz")))
                .isInstanceOf(IllegalArgumentException.class);
        assertThat(Files.exists(dir.resolve("args.txt"))).isFalse();
    }

    // The resolver read stdout to EOF before waiting with a timeout, so a hung script (which holds stdout) blocked the
    // publisher forever and the timeout never fired. It must fire, and end what the script forked.
    @Test
    void aHungResolverTimesOutAndIsKilled(@TempDir Path dir) throws Exception {
        Path pidFile = dir.resolve("child.pid");
        Path script = dir.resolve("hang.sh");
        Files.writeString(script, "#!/bin/bash\nsleep 300 &\necho $! > " + pidFile + "\nwait\n");
        ScriptExecutionPlanResolver resolver = resolver(script);
        ReflectionTestUtils.setField(resolver, "timeoutSeconds", 1L);

        org.junit.jupiter.api.Assertions.assertTimeoutPreemptively(java.time.Duration.ofSeconds(15), () ->
                assertThatThrownBy(() -> resolver.resolve("TINYNET_GOLDEN", "FedAvg", "FULL", TrainingDataSource.FIXTURE, dir.resolve("m.npz")))
                        .isInstanceOf(IOException.class).hasMessageContaining("timed out"));

        ProcessHandle child = ProcessHandle.of(Long.parseLong(Files.readString(pidFile).trim())).orElse(null);
        org.awaitility.Awaitility.await().atMost(10, java.util.concurrent.TimeUnit.SECONDS)
                .untilAsserted(() -> assertThat(child == null || !child.isAlive()).isTrue());
    }
}
