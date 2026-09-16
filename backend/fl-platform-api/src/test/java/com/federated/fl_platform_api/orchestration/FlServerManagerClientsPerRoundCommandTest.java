package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Project;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The round size reaches fl_server.py as {@code --clients-per-round}, and only when a round waits for more clients
 * than minClients. Without it the two were always equal, and one client missing one round stopped the run.
 */
class FlServerManagerClientsPerRoundCommandTest {

    private static final String SCRIPT = "/x/run_fl_server.sh";

    private static Project project() {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setModelPath("/tmp/model.npz");
        return p;
    }

    private static List<String> cmd(Project p, String strategy, int minClients, Integer clientsPerRound) {
        return FlServerManager.buildServerCommand(p, strategy, 5, minClients, 50000, SCRIPT, false, null, null, null,
                clientsPerRound);
    }

    @Test
    void withoutARoundSizeTheCommandIsIdenticalToBefore() {
        Project p = project();
        List<String> previousArity = FlServerManager.buildServerCommand(p, "DeComFL", 5, 3, 50000, SCRIPT, false,
                null, null, null);
        assertThat(cmd(p, "DeComFL", 3, null)).isEqualTo(previousArity).doesNotContain("--clients-per-round");
    }

    @Test
    void aRoundSizeEqualToTheMinimumAddsNothing() {
        assertThat(cmd(project(), "FedAvg", 3, 3)).doesNotContain("--clients-per-round");
    }

    @Test
    void aLargerRoundSizeIsPassedAlongsideTheMinimum() {
        List<String> c = cmd(project(), "DeComFL", 3, 5);
        int i = c.indexOf("--clients-per-round");
        assertThat(i).as("--clients-per-round present in %s", c).isNotNegative();
        assertThat(c.get(i + 1)).isEqualTo("5");
        assertThat(c.get(c.indexOf("--min-clients") + 1)).isEqualTo("3");
    }

    @Test
    void aRoundSizeBelowTheMinimumIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> cmd(project(), "DeComFL", 3, 2));
    }

    @Test
    void textFederationHasNoRoundSizeToPass() {
        assertThrows(IllegalArgumentException.class, () -> cmd(project(), "FoT", 3, 5));
    }

    @Test
    void aDpProjectIsRefusedALargerRoundSize() {
        Project p = project();
        p.setDpEnabled(true);
        p.setDpTargetEpsilon(6.0);
        p.setDpDelta(1e-5);
        p.setDpClipNorm(1.5);
        assertThrows(IllegalArgumentException.class, () -> cmd(p, "FedAvg", 3, 5));
        assertThat(cmd(p, "FedAvg", 3, 3)).contains("--dp-enabled");
    }
}
