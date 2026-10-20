package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;

import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashSet;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

/** Opt-in numerical-worker check; schema-valid synthetic features make no NIH quality claim. */
@EnabledIfEnvironmentVariable(named = "SIMULATION_REAL_WORKER_PYTHON", matches = ".+")
class SimulationActualWorkerTest {
    @TempDir Path temp;
    SimulationProperties properties;

    @BeforeEach void prepare() throws Exception {
        Path framework = Path.of("../../framework").toAbsolutePath().normalize();
        String script = """
                import json,runpy,sys
                from pathlib import Path
                fixture=runpy.run_path(str(Path(sys.argv[1])/'tests/test_simulation_lab_worker.py'))['lab_cache']
                cache,pin=fixture.__wrapped__(Path(sys.argv[2]))
                print(json.dumps({'cache':str(cache),'pin':pin}))
                """;
        var builder = new ProcessBuilder(System.getenv("SIMULATION_REAL_WORKER_PYTHON"), "-c", script, framework.toString(), temp.toString());
        builder.environment().put("PYTHONPATH", framework.resolve("src").toString());
        Process setup = builder.redirectError(temp.resolve("fixture-setup.log").toFile()).start();
        assertThat(setup.waitFor(20, TimeUnit.SECONDS)).isTrue();
        String output = new String(setup.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        assertThat(setup.exitValue()).as(output).isZero();
        JsonNode fixture = SimulationJson.stored(output);
        properties = new SimulationProperties(); properties.setEnabled(true);
        properties.setInterpreter(Path.of(System.getenv("SIMULATION_REAL_WORKER_PYTHON")));
        properties.setSourceRoot(framework.resolve("src")); properties.setCache(Path.of(fixture.path("cache").asText()));
        properties.setReportSha256(fixture.path("pin").asText()); properties.setStorage(temp.resolve("jobs"));
        properties.setDeadlineSeconds(15);
    }

    @Test void productionWorkerCompletesTwoActualRoundsThroughJavaAndExportsBoundEvidence() throws Exception {
        try (var runner = new SimulationProcessRunner(properties)) {
            assertThat(runner.inspect().path("dataset").asText()).isEqualTo("synthetic-schema-control");
            var resolved = runner.resolve(SimulationJson.stored("{\"rounds\":2}"));
            UUID id = UUID.randomUUID(); var child = runner.launch(id, resolved.config());
            try {
                assertThat(child.awaitDrain(Duration.ofSeconds(20))).isTrue();
                assertThat(child.process().exitValue()).isZero();
                SimulationRun run = new SimulationRun(); run.setId(id); run.setConfigHash(resolved.hash());
                run.setRequestedRounds(2); run.setRecipeJson(resolved.recipe().toString()); run.setWorkerPid(child.process().pid());
                var evidence = SimulationEvidence.read(run, runner.jobDirectory(id).resolve("execution"), 0);
                assertThat(evidence.qualified()).isTrue(); assertThat(evidence.completedRounds()).isEqualTo(2);
                assertThat(evidence.record().toString()).doesNotContain(temp.toString(), "features_path", "\"host\"");
                JsonNode raw = SimulationJson.stored(Files.readString(runner.jobDirectory(id).resolve("execution/record.json")));
                assertThat(evidence.record().path("resolvedConfig").path("seeds")).isEqualTo(raw.path("meta").path("config").path("seeds"));
                assertThat(evidence.record().path("resolvedConfig").path("federation").path("network"))
                        .isEqualTo(raw.path("meta").path("config").path("federation").path("network"));
                assertThat(evidence.record().path("execution").path("wallSeconds")).isEqualTo(raw.path("meta").path("wall_seconds"));
                assertThat(evidence.record().path("execution").path("peakRssBytes")).isEqualTo(raw.path("meta").path("peak_rss_bytes"));
                assertThat(evidence.record().path("source").path("runtime").path("torch")).isEqualTo(raw.path("meta").path("provenance").path("torch"));
                assertThat(evidence.record().path("source").path("workload").path("curated_cohort_sha256"))
                        .isEqualTo(raw.path("meta").path("workload").path("curated_cohort_sha256"));
                assertThat(evidence.record().path("rounds").get(0).path("participants")).isEqualTo(raw.path("rounds").get(0).path("participants"));
            } finally { child.stop(); child.awaitDrain(Duration.ofSeconds(6)); }
        }
    }

    @Test void killingActualJavaOwnerClosesParentPipeAndWorkerInterrupts() throws Exception {
        UUID id = UUID.randomUUID(); Path config = temp.resolve("prepared-config.json");
        try (var resolver = new SimulationProcessRunner(properties)) {
            Files.writeString(config, resolver.resolve(SimulationJson.stored("{\"rounds\":20,\"localEpochs\":7}")).config().toString());
        }
        properties.setDeadlineSeconds(180);
        Path marker = temp.resolve("child.pid");
        Process parent = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin/java").toString(), "-cp", classpath(),
                SimulationParentProcess.class.getName(), properties.getInterpreter().toString(), properties.getSourceRoot().toString(),
                properties.getCache().toString(), properties.getStorage().toString(), properties.getReportSha256(), id.toString(),
                config.toString(), marker.toString()).redirectErrorStream(true).redirectOutput(temp.resolve("owner.log").toFile()).start();
        Long pid = null;
        try {
            long until = System.nanoTime() + Duration.ofSeconds(8).toNanos();
            while (!Files.exists(marker) && parent.isAlive() && System.nanoTime() < until) Thread.sleep(10);
            assertThat(marker).as(Files.readString(temp.resolve("owner.log"))).exists();
            pid = Long.parseLong(Files.readString(marker));
            parent.destroyForcibly(); assertThat(parent.waitFor(4, TimeUnit.SECONDS)).isTrue();
            until = System.nanoTime() + Duration.ofSeconds(9).toNanos();
            while (ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false) && System.nanoTime() < until) Thread.sleep(20);
            assertThat(ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false)).isFalse();
            JsonNode progress = SimulationJson.stored(Files.readString(properties.getStorage().resolve(id.toString()).resolve("execution/progress.json")));
            assertThat(progress.path("state").asText()).isEqualTo("interrupted");
            assertThat(progress.path("completed_rounds").asInt()).isLessThan(20);
        } finally {
            if (parent.isAlive()) { parent.destroyForcibly(); parent.waitFor(); }
            if (pid != null) ProcessHandle.of(pid).filter(ProcessHandle::isAlive).ifPresent(ProcessHandle::destroyForcibly);
        }
    }

    private String classpath() throws Exception {
        LinkedHashSet<String> paths = new LinkedHashSet<>();
        for (ClassLoader loader = getClass().getClassLoader(); loader != null; loader = loader.getParent()) {
            if (loader instanceof URLClassLoader urls) for (var url : urls.getURLs()) if (url.getProtocol().equals("file")) paths.add(Path.of(url.toURI()).toString());
        }
        paths.addAll(java.util.Arrays.asList(System.getProperty("java.class.path").split(java.io.File.pathSeparator)));
        return String.join(java.io.File.pathSeparator, paths);
    }
}
