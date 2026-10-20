package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.UUID;

import static org.assertj.core.api.Assertions.*;

class SimulationEvidenceTest {
    @TempDir Path root;
    Path directory;
    SimulationRun run;

    @BeforeEach void job() throws Exception {
        directory = Files.createDirectory(root.resolve("execution"));
        run = new SimulationRun(); run.setId(UUID.randomUUID()); run.setConfigHash("a".repeat(64));
        run.setRequestedRounds(2); run.setCreatedAt(Instant.now()); run.setWorkerPid(123L);
        run.setRecipeJson("{\"featureReportSha256\":\"" + "b".repeat(64) + "\",\"numClients\":10,\"privacy\":\"none\"}");
    }

    JsonNode progress(int count, String state) throws Exception {
        var progress = SimulationJson.MAPPER.createObjectNode();
        progress.put("version", 1).put("run_id", run.getId().toString()).put("config_hash", run.getConfigHash())
                .put("state", state).put("completed_rounds", count).put("requested_rounds", 2)
                .putNull("record_encoding").putNull("record_file");
        var rounds = progress.putArray("rounds");
        for (int index = 1; index <= count; index++) rounds.addObject().put("round", index).put("loss", .6).put("auc", .7);
        write("progress.json", progress); return progress;
    }

    void write(String name, JsonNode value) throws Exception { Files.writeString(directory.resolve(name), value.toString()); }

    void completed() throws Exception {
        JsonNode p = progress(2, "completed");
        ((com.fasterxml.jackson.databind.node.ObjectNode) p).put("record_encoding", "json").put("record_file", "record.json");
        write("progress.json", p);
        var execution = SimulationJson.MAPPER.createObjectNode();
        execution.put("version", 1).put("run_id", run.getId().toString()).put("config_hash", run.getConfigHash())
                .put("state", "completed").put("exit_code", 0).put("worker_pid", 123).put("cpu_threads", 4);
        write("execution.json", execution);
        var record = SimulationJson.MAPPER.createObjectNode(); record.set("rounds", p.path("rounds"));
        var meta = record.putObject("meta"); meta.put("config_hash", run.getConfigHash()).put("peak_rss_bytes", 100000000);
        var config = meta.putObject("config"); config.putObject("workload").put("features_report_sha256", "b".repeat(64))
                .put("features_path", "/private/never-export/cache").put("evaluation_split", "validation");
        Files.writeString(root.resolve("config.json"), config.toString());
        meta.putObject("provenance").put("host", "private-host");
        meta.putObject("core_qualification").put("version", 1).put("scope", "simulation_execution_only")
                .put("evaluation_split", "validation").put("accepted_clients", 10).put("privacy_mode", "none")
                .put("research_record_dp_safe", false);
        write("record.json", record);
    }

    @Test void onlyActualExitAndConsistentQualificationCanComplete() throws Exception {
        completed();
        assertThat(SimulationEvidence.read(run, directory, null).qualified()).isFalse();
        assertThat(SimulationEvidence.read(run, directory, -9).qualified()).isFalse();
        var evidence = SimulationEvidence.read(run, directory, 0);
        assertThat(evidence.qualified()).isTrue();
        assertThat(evidence.record().toString()).doesNotContain("never-export", "private-host", "features_path");
        assertThat(evidence.record().path("runId").asText()).isEqualTo(run.getId().toString());
        assertThat(evidence.record().path("researchRecordDpSafe").asBoolean(true)).isFalse();
        assertThat(evidence.record().path("finalSummary").path("auc").asDouble()).isEqualTo(.7);
        var record = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("record.json")));
        ((com.fasterxml.jackson.databind.node.ObjectNode) record.path("meta")).remove("core_qualification");
        write("record.json", record);
        assertThat(SimulationEvidence.read(run, directory, 0).qualified()).isFalse();
    }

    @Test void foreignBackwardsOrRewrittenRoundSnapshotsAreRejected() throws Exception {
        progress(1, "running");
        var first = SimulationEvidence.read(run, directory, null);
        run.setProgressJson(first.progress().toString()); run.setCompletedRounds(1);
        assertThat(SimulationEvidence.read(run, directory, null).completedRounds()).isEqualTo(1);
        progress(0, "running");
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, null)).isInstanceOf(Exception.class);
        var foreign = (com.fasterxml.jackson.databind.node.ObjectNode) progress(1, "running");
        foreign.put("run_id", UUID.randomUUID().toString()); write("progress.json", foreign);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, null)).isInstanceOf(Exception.class);
        var rewritten = progress(1, "running");
        ((com.fasterxml.jackson.databind.node.ObjectNode) rewritten.path("rounds").path(0)).put("loss", .2);
        write("progress.json", rewritten);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, null)).isInstanceOf(Exception.class);
    }

    @Test void recordCannotRedirectReadOrClaimAnotherConfig() throws Exception {
        completed();
        var p = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("progress.json")));
        p.put("record_file", "../../secrets.json"); write("progress.json", p);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, 0)).isInstanceOf(Exception.class);
        completed();
        var record = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("record.json")));
        ((com.fasterxml.jackson.databind.node.ObjectNode) record.path("meta")).put("config_hash", "c".repeat(64));
        write("record.json", record);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, 0)).isInstanceOf(Exception.class);
    }

    @Test void runningStateCannotRegressToStartupAtTheSameRoundCount() throws Exception {
        progress(1, "running");
        var first = SimulationEvidence.read(run, directory, null);
        run.setProgressJson(first.progress().toString()); run.setCompletedRounds(1);
        progress(1, "starting");
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, null)).isInstanceOf(Exception.class);
    }

    @Test void fullPrivateRecipeMustMatchTheConfigurationWrittenBeforeLaunch() throws Exception {
        completed();
        var record = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("record.json")));
        ((com.fasterxml.jackson.databind.node.ObjectNode) record.path("meta").path("config")).putObject("client").put("local_epochs", 99);
        write("record.json", record);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, 0)).isInstanceOf(Exception.class);
    }

    @Test void nonfinitePartialValuesStayTaggedAndCannotBecomeQualified() throws Exception {
        completed();
        var record = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("record.json")));
        var progress = (com.fasterxml.jackson.databind.node.ObjectNode) SimulationJson.stored(Files.readString(directory.resolve("progress.json")));
        var tag = SimulationJson.MAPPER.createObjectNode().put("__nonfinite_float__", "inf");
        ((com.fasterxml.jackson.databind.node.ObjectNode) record.path("rounds").get(1)).set("loss", tag);
        ((com.fasterxml.jackson.databind.node.ObjectNode) progress.path("rounds").get(1)).set("loss", tag);
        progress.put("record_encoding", "nonfinite_tagged_v1").put("state", "failed"); write("progress.json", progress);
        var envelope = SimulationJson.MAPPER.createObjectNode().put("encoding", "nonfinite_tagged_v1"); envelope.set("record", record);
        write("record.json", envelope);
        var evidence = SimulationEvidence.read(run, directory, 2);
        assertThat(evidence.qualified()).isFalse();
        assertThat(evidence.record().path("finalSummary").path("loss")).isEqualTo(tag);
        progress.put("record_encoding", "json").put("state", "completed"); write("progress.json", progress); write("record.json", record);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, 0)).isInstanceOf(Exception.class);
    }

    @Test void aNullIdentityCannotReplacePreviouslyResolvedEmptyProgress() throws Exception {
        progress(0, "starting");
        var resolved = SimulationEvidence.read(run, directory, null);
        run.setProgressJson(resolved.progress().toString()); run.setCompletedRounds(0);
        var unbound = (com.fasterxml.jackson.databind.node.ObjectNode) progress(0, "stopped");
        unbound.putNull("config_hash").put("requested_rounds", 0); write("progress.json", unbound);
        assertThatThrownBy(() -> SimulationEvidence.read(run, directory, 3)).isInstanceOf(Exception.class);
    }
}
