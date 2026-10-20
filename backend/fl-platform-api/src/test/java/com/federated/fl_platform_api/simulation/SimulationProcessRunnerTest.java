package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.UUID;

import static org.assertj.core.api.Assertions.*;

class SimulationProcessRunnerTest {
    @TempDir Path temp;

    SimulationProperties properties() throws Exception {
        Path source = temp.resolve("source/fedlearn"); Files.createDirectories(source);
        Files.writeString(source.resolve("__init__.py"), "");
        Files.writeString(source.resolve("lab_worker.py"), """
                import argparse,json,os,signal,subprocess,time
                from pathlib import Path
                p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--config');p.add_argument('--output');p.add_argument('--report-sha256');p.add_argument('--run-id');p.add_argument('--deadline-seconds');a=p.parse_args()
                out=Path(a.output);out.mkdir()
                (out/'environment.json').write_text(json.dumps(dict(os.environ)))
                signal.signal(signal.SIGTERM,signal.SIG_IGN)
                child=subprocess.Popen(['/bin/sleep','30'])
                (out/'child.pid').write_text(str(child.pid))
                print('x'*200000,flush=True)
                time.sleep(30)
                """);
        SimulationProperties properties = new SimulationProperties();
        properties.setEnabled(true); properties.setInterpreter(Path.of("/usr/bin/python3"));
        properties.setSourceRoot(source.getParent()); properties.setCache(temp.resolve("cache"));
        properties.setStorage(temp.resolve("jobs")); properties.setReportSha256("b".repeat(64));
        properties.setDeadlineSeconds(1);
        return properties;
    }

    @Test void externalDeadlineKillsAndDrainsActualChildTreeAndBoundsLogs() throws Exception {
        var properties = properties();
        try (var runner = new SimulationProcessRunner(properties)) {
            UUID id = UUID.randomUUID();
            var child = runner.launch(id, SimulationJson.MAPPER.createObjectNode());
            assertThat(child.awaitDrain(Duration.ofSeconds(7))).isTrue();
            assertThat(child.timedOut()).isTrue();
            assertThat(child.process().isAlive()).isFalse();
            Path output = runner.jobDirectory(id).resolve("execution");
            long descendant = Long.parseLong(Files.readString(output.resolve("child.pid")));
            assertThat(ProcessHandle.of(descendant).map(ProcessHandle::isAlive).orElse(false)).isFalse();
            assertThat(Files.size(runner.jobDirectory(id).resolve("process.log"))).isLessThanOrEqualTo(65536);
            JsonNode env = SimulationJson.stored(Files.readString(output.resolve("environment.json")));
            assertThat(env.has("HOME")).isFalse();
            assertThat(env.has("APP_JWT_SECRET")).isFalse();
            assertThat(env.path("OMP_NUM_THREADS").asText()).isEqualTo("4");
        }
    }

    @Test void explicitStopWaitsForUnresponsiveChildAndNeverReusesExistingDirectory() throws Exception {
        var properties = properties(); properties.setDeadlineSeconds(1);
        try (var runner = new SimulationProcessRunner(properties)) {
            UUID id = UUID.randomUUID();
            var child = runner.launch(id, SimulationJson.MAPPER.createObjectNode());
            try {
                Path marker = runner.jobDirectory(id).resolve("execution/child.pid");
                long until = System.nanoTime() + Duration.ofSeconds(5).toNanos();
                while (!Files.exists(marker) && System.nanoTime() < until) Thread.sleep(10);
                assertThat(marker).exists();
                assertThat(runner.matches(child.process().toHandle(), id, child.startIdentity(), child.executableIdentity()))
                        .as("identity command=%s arguments=%s configured=%s", child.process().info().command(),
                                java.util.Arrays.toString(child.process().info().arguments().orElse(new String[0])), properties.getInterpreter()).isTrue();
                assertThat(runner.matches(child.process().toHandle(), id, "1970-01-01T00:00:00Z", child.executableIdentity())).isFalse();
                assertThat(runner.matches(ProcessHandle.current(), id, ProcessHandle.current().info().startInstant().orElseThrow().toString(),
                        ProcessHandle.current().info().command().orElseThrow())).isFalse();
                child.stop();
                assertThat(child.awaitDrain(Duration.ofSeconds(6))).isTrue();
                assertThat(child.timedOut()).isFalse();
                assertThatThrownBy(() -> runner.launch(id, SimulationJson.MAPPER.createObjectNode())).isInstanceOf(Exception.class);
            } finally { child.stop(); child.awaitDrain(Duration.ofSeconds(6)); }
        }
    }
}
