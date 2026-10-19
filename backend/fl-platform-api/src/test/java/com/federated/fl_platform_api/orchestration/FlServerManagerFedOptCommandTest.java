package com.federated.fl_platform_api.orchestration;
import com.federated.fl_platform_api.model.*;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
class FlServerManagerFedOptCommandTest {
    @Test void recordedSettingsReachThePythonServer() {
        var p = new Project(); p.setId(java.util.UUID.randomUUID()); p.setModelType("CNN"); p.setModelName("net"); p.setModelPath("/tmp/m.npz");
        var settings = new FedOptSettings("yogi", .01, .8, .95, .002);
        var command = FlServerManager.buildServerCommand(p, RunIntent.capture(p, false, false), "FedOpt", 2, 2,
                50000, "/x/run_fl_server.sh", false, null, null, null, 2, settings);
        assertThat(command).containsSubsequence("--fedopt-variant", "yogi", "--fedopt-server-learning-rate", "0.01",
                "--fedopt-beta1", "0.8", "--fedopt-beta2", "0.95", "--fedopt-tau", "0.002");
        assertThrows(IllegalArgumentException.class, () -> FlServerManager.buildServerCommand(p,
                RunIntent.capture(p, false, false), "FedAvg", 2, 2, 50000, "/x/run_fl_server.sh", false,
                null, null, null, 2, settings));
    }
}
