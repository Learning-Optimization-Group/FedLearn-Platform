package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RobustAggregationSettings;
import com.federated.fl_platform_api.model.RobustMethod;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The Robust aggregation settings reach {@code fl_server.py} as its own flags, and nothing else changes.
 */
class FlServerManagerRobustCommandTest {

    private static final String SCRIPT = "/x/run_fl_server.sh";

    private static Project project() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    private static List<String> cmd(String strategy, RobustAggregationSettings robust) {
        return FlServerManager.buildServerCommand(project(), strategy, 5, 20, 50000, SCRIPT, false, null, robust);
    }

    private static String valueAfter(List<String> cmd, String flag) {
        int i = cmd.indexOf(flag);
        assertThat(i).as("%s present in %s", flag, cmd).isNotNegative();
        return cmd.get(i + 1);
    }

    @Test
    void withoutRobustSettingsTheCommandIsIdenticalToBefore() {
        // Every existing Robust start keeps its exact argv: the server then uses its own default (median).
        Project p = project();
        List<String> withNull = FlServerManager.buildServerCommand(p, "Robust", 5, 20, 50000, SCRIPT, false, null, null);
        List<String> previousArity = FlServerManager.buildServerCommand(p, "Robust", 5, 20, 50000, SCRIPT, false, null);
        assertThat(withNull).isEqualTo(previousArity);
        assertThat(withNull).noneMatch(a -> a.startsWith("--robust") || a.startsWith("--centered-clip"));
    }

    @Test
    void bulyanPassesTheServerMethodNameAndFraction() {
        List<String> c = cmd("Robust", new RobustAggregationSettings(RobustMethod.BULYAN, 0.2, null, null));
        assertThat(valueAfter(c, "--robust-method")).isEqualTo("bulyan");
        assertThat(valueAfter(c, "--robust-byzantine-fraction")).isEqualTo("0.2");
        assertThat(c).doesNotContain("--robust-trim-ratio", "--centered-clip-tau");
    }

    @Test
    void multiKrumUsesTheServerSpelling() {
        List<String> c = cmd("Robust", new RobustAggregationSettings(RobustMethod.MULTI_KRUM, 0.1, null, null));
        assertThat(valueAfter(c, "--robust-method")).isEqualTo("multi_krum");
    }

    @Test
    void aTrimRatioOfZeroIsPassedRatherThanDropped() {
        // Zero trims nothing (the plain mean). Omitting it would let the server use its default of 0.1.
        List<String> c = cmd("Robust", new RobustAggregationSettings(RobustMethod.TRIMMED_MEAN, null, 0.0, null));
        assertThat(valueAfter(c, "--robust-method")).isEqualTo("trimmed_mean");
        assertThat(valueAfter(c, "--robust-trim-ratio")).isEqualTo("0.0");
    }

    @Test
    void centeredClipPassesItsRadius() {
        List<String> c = cmd("Robust", new RobustAggregationSettings(RobustMethod.CENTERED_CLIP, null, null, 1.5));
        assertThat(valueAfter(c, "--centered-clip-tau")).isEqualTo("1.5");
    }

    @Test
    void aMethodWithNoParametersPassesOnlyTheMethod() {
        List<String> c = cmd("Robust", new RobustAggregationSettings(RobustMethod.MEDIAN, null, null, null));
        assertThat(valueAfter(c, "--robust-method")).isEqualTo("median");
        assertThat(c).doesNotContain("--robust-byzantine-fraction", "--robust-trim-ratio", "--centered-clip-tau");
    }

    @Test
    void robustSettingsOnAnyOtherStrategyAreRefused() {
        RobustAggregationSettings krum = new RobustAggregationSettings(RobustMethod.KRUM, 0.1, null, null);
        assertThrows(IllegalArgumentException.class, () -> cmd("FedAvg", krum));
        assertThrows(IllegalArgumentException.class, () -> cmd("FoT", krum));
    }
}
