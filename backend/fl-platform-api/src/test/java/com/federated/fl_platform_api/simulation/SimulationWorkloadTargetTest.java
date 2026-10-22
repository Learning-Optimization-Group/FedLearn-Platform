package com.federated.fl_platform_api.simulation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

/** Each Lab workload resolves against its own cache and pin; an unconfigured workload is unavailable. */
class SimulationWorkloadTargetTest {
    @TempDir Path temp;
    static final String HISTORICAL = "b".repeat(64), EFFUSION = "c".repeat(64);

    SimulationProperties properties(boolean effusion, String echoedSha) throws Exception {
        Path source = temp.resolve("source/fedlearn"); Files.createDirectories(source);
        Files.writeString(source.resolve("__init__.py"), "");
        Files.writeString(source.resolve("lab_worker.py"), """
                import argparse,json
                p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--choices');p.add_argument('--cache');p.add_argument('--report-sha256');a=p.parse_args()
                choices=json.load(open(a.choices))
                echoed=%s
                print(json.dumps({'config_hash':'d'*64,'config':{'cache':a.cache},'display_recipe':{'workloadId':choices.get('workloadId','nih-qualified'),'featureReportSha256':echoed or a.report_sha256}}))
                """.formatted(echoedSha == null ? "None" : "'" + echoedSha + "'"));
        SimulationProperties properties = new SimulationProperties();
        properties.setEnabled(true); properties.setInterpreter(Path.of("/usr/bin/python3"));
        properties.setSourceRoot(source.getParent()); properties.setCache(temp.resolve("cache"));
        properties.setStorage(temp.resolve("jobs")); properties.setReportSha256(HISTORICAL); properties.setDeadlineSeconds(5);
        if (effusion) { properties.setEffusionBundle(temp.resolve("bundle")); properties.setEffusionReportSha256(EFFUSION); }
        return properties;
    }

    static com.fasterxml.jackson.databind.JsonNode choices(String workloadId) throws Exception {
        return SimulationJson.stored(workloadId == null ? "{}" : "{\"workloadId\":\"" + workloadId + "\"}");
    }

    @Test void historicalDefaultKeepsItsCacheAndPin() throws Exception {
        try (var runner = new SimulationProcessRunner(properties(true, null))) {
            var resolution = runner.resolve(choices(null));
            assertThat(resolution.workloadId()).isEqualTo("nih-qualified");
            assertThat(resolution.reportSha256()).isEqualTo(HISTORICAL);
            assertThat(resolution.config().path("cache").asText()).endsWith("cache");
        }
    }

    @Test void effusionResolvesAgainstItsOwnBundleAndPin() throws Exception {
        try (var runner = new SimulationProcessRunner(properties(true, null))) {
            var resolution = runner.resolve(choices("nih-effusion-r0"));
            assertThat(resolution.workloadId()).isEqualTo("nih-effusion-r0");
            assertThat(resolution.reportSha256()).isEqualTo(EFFUSION);
            assertThat(resolution.config().path("cache").asText()).endsWith("bundle");
        }
    }

    @Test void unconfiguredOrUnknownWorkloadIsUnavailable() throws Exception {
        try (var runner = new SimulationProcessRunner(properties(false, null))) {
            assertThatThrownBy(() -> runner.resolve(choices("nih-effusion-r0"))).isInstanceOf(SimulationException.class);
            assertThatThrownBy(() -> runner.resolve(choices("something-else"))).isInstanceOf(SimulationException.class);
        }
    }

    @Test void aResolverReportingAnotherWorkloadsPinIsRefused() throws Exception {
        try (var runner = new SimulationProcessRunner(properties(true, HISTORICAL))) {
            assertThatThrownBy(() -> runner.resolve(choices("nih-effusion-r0"))).isInstanceOf(SimulationException.class);
        }
    }
}
