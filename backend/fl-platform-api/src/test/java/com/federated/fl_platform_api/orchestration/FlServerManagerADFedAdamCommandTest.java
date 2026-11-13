package com.federated.fl_platform_api.orchestration;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RunIntent;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlServerManagerADFedAdamCommandTest {

    private static Project project() {
        var p = new Project();
        p.setId(UUID.randomUUID());
        p.setModelType("CNN");
        p.setModelName("net");
        p.setModelPath("/tmp/m.npz");
        return p;
    }

    @Test
    void recordedCenteredClipTauReachesThePythonServer() {
        var p = project();
        List<String> command = FlServerManager.buildServerCommand(p, RunIntent.capture(p, false, false), "ADFedAdam", 5, 2,
                50000, "/x/run_fl_server.sh", false, null, null, null, 2, null, 0.5);
        assertThat(command).containsSubsequence("--strategy", "ADFedAdam", "--centered-clip-tau", "0.5");
    }

    @Test
    void adFedAdamWithoutTauEmitsNoCenteredClipFlag() {
        var p = project();
        List<String> command = FlServerManager.buildServerCommand(p, RunIntent.capture(p, false, false), "ADFedAdam", 5, 2,
                50000, "/x/run_fl_server.sh", false, null, null, null, 2, null, null);
        assertThat(command).contains("--strategy", "ADFedAdam");
        assertThat(command).doesNotContain("--centered-clip-tau");
    }

    @Test
    void adFedAdamTauOnAnotherStrategyIsRefused() {
        var p = project();
        assertThrows(IllegalArgumentException.class, () -> FlServerManager.buildServerCommand(p,
                RunIntent.capture(p, false, false), "FedAvg", 2, 2, 50000, "/x/run_fl_server.sh", false,
                null, null, null, 2, null, 0.5));
    }
}
