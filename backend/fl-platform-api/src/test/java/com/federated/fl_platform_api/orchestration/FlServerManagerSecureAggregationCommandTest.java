package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Project;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Secure aggregation reaches fl_server.py as its own flags, only on DeComFL, and nothing else changes. */
class FlServerManagerSecureAggregationCommandTest {

    private static final String SCRIPT = "/x/run_fl_server.sh";

    private static Project project() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    private static List<String> cmd(String strategy, Integer threshold) {
        return FlServerManager.buildServerCommand(project(), strategy, 5, 3, 50000, SCRIPT, false, null, null, threshold);
    }

    @Test
    void withoutSecureAggregationTheCommandIsIdenticalToBefore() {
        Project p = project();
        List<String> withNull = FlServerManager.buildServerCommand(p, "DeComFL", 5, 3, 50000, SCRIPT, false, null, null, null);
        List<String> previousArity = FlServerManager.buildServerCommand(p, "DeComFL", 5, 3, 50000, SCRIPT, false, null, null);
        assertThat(withNull).isEqualTo(previousArity);
        assertThat(withNull).noneMatch(a -> a.startsWith("--secure-agg"));
    }

    @Test
    void aThresholdTurnsSecureAggregationOnWithThatThreshold() {
        List<String> c = cmd("DeComFL", 3);
        assertThat(c).contains("--secure-aggregation");
        int i = c.indexOf("--secure-agg-threshold");
        assertThat(i).as("--secure-agg-threshold present in %s", c).isNotNegative();
        assertThat(c.get(i + 1)).isEqualTo("3");
    }

    @ParameterizedTest
    @ValueSource(strings = {"FedAvg", "FedProx", "FedOpt", "Robust", "FoT"})
    void secureAggregationOnAnyOtherStrategyIsRefused(String strategy) {
        // Masking exists only on DeComFL's gradient-scalar channel; elsewhere the server flag would be inert and
        // the run would be labelled secure without being so.
        assertThrows(IllegalArgumentException.class, () -> cmd(strategy, 2));
    }
}
