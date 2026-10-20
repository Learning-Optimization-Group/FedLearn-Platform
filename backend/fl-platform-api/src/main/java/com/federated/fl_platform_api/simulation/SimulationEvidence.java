package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.*;
import java.util.Set;

/** Reject unbound evidence and export only explicit, path-free research projections. */
final class SimulationEvidence {
    record Observation(JsonNode progress, JsonNode record, int completedRounds, boolean qualified) { }
    private static final Set<String> STATES = Set.of("starting", "running", "completed", "stopped", "interrupted", "failed", "timed_out");

    private SimulationEvidence() { }

    static Observation read(SimulationRun run, Path directory, Integer actualExit) throws IOException {
        JsonNode raw = file(directory.resolve("progress.json"));
        if (!raw.path("version").isIntegralNumber() || raw.path("version").asInt() != 1
                || !run.getId().toString().equals(raw.path("run_id").asText()) || !STATES.contains(raw.path("state").asText())) invalid();
        int count = integer(raw.path("completed_rounds"));
        int requested = integer(raw.path("requested_rounds"));
        boolean unresolved = !raw.path("state").asText().equals("completed") && raw.path("config_hash").isNull()
                && count == 0 && requested == 0 && raw.path("rounds").isArray() && raw.path("rounds").isEmpty()
                && raw.path("record_encoding").isNull() && raw.path("record_file").isNull();
        if (!unresolved && (!run.getConfigHash().equals(raw.path("config_hash").asText()) || requested != run.getRequestedRounds())) invalid();
        if (count < 0 || count > run.getRequestedRounds() || !raw.path("rounds").isArray() || raw.path("rounds").size() != count
                || (run.getCompletedRounds() != null && count < run.getCompletedRounds())) invalid();
        JsonNode retained = SimulationJson.stored(run.getProgressJson());
        String previousState = retained.path("state").asText();
        String nextState = raw.path("state").asText();
        if ((previousState.equals("running") && nextState.equals("starting"))
                || (STATES.contains(previousState) && !Set.of("starting", "running").contains(previousState)
                    && (!nextState.equals(previousState) || count != retained.path("completed_rounds").asInt()))
                || (unresolved && retained.path("config_hash").isTextual())) invalid();
        ObjectNode progress = SimulationJson.MAPPER.createObjectNode();
        for (String key : new String[]{"version", "run_id", "config_hash", "state", "completed_rounds", "requested_rounds"}) progress.set(key, raw.path(key));
        var rounds = progress.putArray("rounds");
        for (int index = 0; index < count; index++) {
            JsonNode round = raw.path("rounds").get(index);
            if (integer(round.path("round")) != index + 1) invalid();
            rounds.add(projectRound(round));
        }
        JsonNode previous = SimulationJson.stored(run.getProgressJson()).path("rounds");
        for (int index = 0; index < previous.size(); index++) if (!previous.get(index).equals(rounds.get(index))) invalid();
        JsonNode encoding = raw.path("record_encoding");
        if (!encoding.isNull() && !(encoding.isTextual() && Set.of("json", "nonfinite_tagged_v1").contains(encoding.asText()))) invalid();
        progress.set("record_encoding", encoding);
        JsonNode record = SimulationJson.MAPPER.nullNode(); boolean qualified = false;
        if (!raw.path("record_file").isNull() && !raw.path("record_file").isMissingNode()) {
            if (!raw.path("record_file").asText().equals("record.json") || !encoding.isTextual()) invalid();
            JsonNode envelope = file(directory.resolve("record.json"));
            JsonNode content = encoding.asText().equals("nonfinite_tagged_v1") ? envelope.path("record") : envelope;
            if (encoding.asText().equals("nonfinite_tagged_v1") && !envelope.path("encoding").asText().equals("nonfinite_tagged_v1")) invalid();
            JsonNode meta = content.path("meta"); JsonNode recipe = SimulationJson.stored(run.getRecipeJson());
            if (!run.getConfigHash().equals(meta.path("config_hash").asText())
                    || !meta.path("config").equals(file(directory.getParent().resolve("config.json")))
                    || !recipe.path("featureReportSha256").equals(meta.path("config").path("workload").path("features_report_sha256"))
                    || !meta.path("config").path("workload").path("evaluation_split").asText().equals("validation")
                    || !content.path("rounds").isArray() || content.path("rounds").size() != count) invalid();
            if (encoding.asText().equals("json") && containsTag(content)) invalid();
            for (int index = 0; index < count; index++) if (!rounds.get(index).equals(projectRound(content.path("rounds").get(index)))) invalid();
            JsonNode qualification = meta.path("core_qualification");
            boolean qualifiedMarker = qualification.path("version").asInt() == 1
                    && qualification.path("scope").asText().equals("simulation_execution_only")
                    && qualification.path("evaluation_split").asText().equals("validation")
                    && qualification.path("accepted_clients").equals(recipe.path("numClients"))
                    && qualification.path("privacy_mode").equals(recipe.path("privacy"))
                    && qualification.path("research_record_dp_safe").isBoolean() && !qualification.path("research_record_dp_safe").asBoolean()
                    && meta.path("peak_rss_bytes").isIntegralNumber() && meta.path("peak_rss_bytes").asLong() > 0
                    && meta.path("peak_rss_bytes").asLong() <= 2147483648L;
            if (actualExit != null && actualExit == 0 && count == run.getRequestedRounds() && qualifiedMarker
                    && raw.path("state").asText().equals("completed") && encoding.asText().equals("json")) {
                JsonNode execution = file(directory.resolve("execution.json"));
                qualified = execution.path("version").asInt() == 1 && run.getId().toString().equals(execution.path("run_id").asText())
                        && run.getConfigHash().equals(execution.path("config_hash").asText()) && execution.path("exit_code").isIntegralNumber()
                        && execution.path("exit_code").asInt() == 0 && execution.path("state").asText().equals("completed")
                        && execution.path("worker_pid").isIntegralNumber() && run.getWorkerPid() != null
                        && execution.path("worker_pid").asLong() == run.getWorkerPid() && execution.path("cpu_threads").asInt() == 4;
            }
            ObjectNode exported = SimulationJson.MAPPER.createObjectNode();
            exported.put("projectionVersion", 1).put("runId", run.getId().toString()).put("configHash", run.getConfigHash())
                    .put("recordEncoding", encoding.asText()).put("researchRecordDpSafe", false).put("scope", SimulationDtos.SCOPE);
            exported.set("recipe", recipe); exported.set("rounds", rounds);
            exported.set("resolvedConfig", configProjection(meta.path("config")));
            exported.set("source", sourceProjection(meta));
            var observed = exported.putObject("execution");
            observed.set("wallSeconds", scalar(meta.path("wall_seconds")));
            observed.set("peakRssBytes", scalar(meta.path("peak_rss_bytes")));
            observed.set("peakRssScope", meta.path("peak_rss_scope").isTextual()
                    && Set.of("run", "process").contains(meta.path("peak_rss_scope").asText())
                    ? meta.path("peak_rss_scope") : SimulationJson.MAPPER.nullNode());
            observed.put("scope", "Local measured wall time and observed process high-water RSS; not physical-device cost or a hard memory cap.");
            exported.set("totals", selectedNumbers(meta.path("totals"), "up_bytes", "down_bytes", "cpu_seconds",
                    "simulated_seconds", "client_flops", "client_flops_analytic"));
            var terminal = exported.putObject("terminal");
            terminal.put("workerState", raw.path("state").asText());
            String workerCode = raw.path("error_code").asText();
            if (Set.of("invalid_recipe", "invalid_cache", "qualification_failed", "execution_failed", "memory_limit",
                    "artifact_failed", "stop_requested", "parent_disconnected", "deadline_exceeded").contains(workerCode)) terminal.put("workerErrorCode", workerCode);
            else terminal.putNull("workerErrorCode");
            if (actualExit == null) terminal.putNull("exitCode"); else terminal.put("exitCode", actualExit);
            exported.set("finalSummary", count == 0 ? SimulationJson.MAPPER.nullNode() : rounds.get(count - 1));
            exported.set("qualification", qualified ? qualificationProjection(qualification) : SimulationJson.MAPPER.nullNode());
            record = exported;
        }
        return new Observation(progress, record, count, qualified);
    }

    private static JsonNode qualificationProjection(JsonNode node) {
        ObjectNode result = SimulationJson.MAPPER.createObjectNode();
        for (String key : new String[]{"version", "scope", "evaluation_split", "accepted_clients", "privacy_mode", "research_record_dp_safe"}) result.set(key, node.path(key));
        return result;
    }

    private static JsonNode projectRound(JsonNode round) throws IOException {
        if (!round.isObject()) invalid();
        ObjectNode projected = SimulationJson.MAPPER.createObjectNode();
        for (String key : new String[]{"round", "loss", "auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ece"}) {
            projected.set(key, scalar(round.path(key)));
        }
        for (String group : new String[]{"comm", "privacy"}) {
            ObjectNode target = projected.putObject(group);
            String[] keys = group.equals("comm") ? new String[]{"up_bytes", "down_bytes"}
                    : new String[]{"epsilon_spent", "clip_rate", "noise_std", "sensitivity"};
            for (String key : keys) target.set(key, scalar(round.path(group).path(key)));
        }
        projected.set("simulatedRoundSeconds", scalar(round.path("time").path("simulated_round_s")));
        projected.set("cpuSeconds", scalar(round.path("compute").path("cpu_s")));
        projected.set("participants", selectedNumbers(round.path("participants"), "sampled", "submitted", "admitted",
                "accepted", "aggregated", "dropped", "straggled"));
        projected.set("compute", selectedNumbers(round.path("compute"), "client_flops", "client_flops_analytic", "cpu_s", "server_flops_est"));
        var time = projected.putObject("time");
        time.set("simulated_round_s", scalar(round.path("time").path("simulated_round_s")));
        for (String clock : new String[]{"wall_s", "cpu_s"}) time.set(clock, selectedNumbers(round.path("time").path(clock),
                "train", "attack", "local_dp", "clip", "noise", "rule", "optimizer", "eval"));
        return projected;
    }

    private static ObjectNode selectedNumbers(JsonNode source, String... fields) throws IOException {
        ObjectNode result = SimulationJson.MAPPER.createObjectNode();
        for (String field : fields) result.set(field, scalar(source.path(field)));
        return result;
    }

    /** Only known leaf settings are copied; future private config fields stay private. */
    private static ObjectNode leaves(JsonNode source, String... fields) throws IOException {
        ObjectNode result = SimulationJson.MAPPER.createObjectNode();
        for (String field : fields) {
            JsonNode value = source.path(field);
            if (value.isContainerNode()) invalid();
            result.set(field, value.isMissingNode() ? SimulationJson.MAPPER.nullNode() : value);
        }
        return result;
    }

    private static JsonNode configProjection(JsonNode config) throws IOException {
        ObjectNode result = leaves(config, "schema_version", "name", "seed");
        result.set("seeds", selectedNumbers(config.path("seeds"), "init", "partition", "client", "attack", "noise"));
        ObjectNode workload = leaves(config.path("workload"), "dataset", "features_report_sha256", "evaluation_split",
                "allow_random_features", "balance", "auc_score");
        workload.set("head", leaves(config.path("workload").path("head"), "type", "hidden")); result.set("workload", workload);
        ObjectNode federation = leaves(config.path("federation"), "num_clients", "clients_per_round", "rounds");
        federation.set("partition", leaves(config.path("federation").path("partition"), "type", "alpha"));
        federation.set("network", leaves(config.path("federation").path("network"), "device_gflops", "downlink_mbps", "uplink_mbps", "latency_ms", "round_deadline_s"));
        result.set("federation", federation);
        result.set("client", leaves(config.path("client"), "algorithm", "batch_size", "local_epochs", "lr", "proximal_mu"));
        ObjectNode server = leaves(config.path("server"), "rule", "clip_norm");
        server.set("optimizer", leaves(config.path("server").path("optimizer"), "type", "lr"));
        server.set("rule_params", leaves(config.path("server").path("rule_params"), "byzantine_fraction", "iterations", "multi_krum_m", "tau", "trim_ratio"));
        result.set("server", server);
        result.set("privacy", leaves(config.path("privacy"), "mode", "epsilon", "delta", "adjacency"));
        result.set("eval", leaves(config.path("eval"), "per_client_every"));
        var behaviors = result.putArray("behaviors");
        for (JsonNode behavior : config.path("behaviors")) behaviors.add(leaves(behavior, "type", "fraction", "scale"));
        return result;
    }

    private static JsonNode sourceProjection(JsonNode meta) throws IOException {
        ObjectNode source = SimulationJson.MAPPER.createObjectNode();
        source.set("runtime", leaves(meta.path("provenance"), "git_commit", "python", "torch", "numpy"));
        ObjectNode workload = leaves(meta.path("workload"), "allow_random_features", "assignment_sha256", "assignment_version",
                "auc_score", "auc_score_definition", "backbone", "curated_cohort_sha256", "dataset",
                "evaluation_manifest_sha256", "evaluation_split", "feat_dim", "features_report_sha256", "model_state_sha256",
                "n_classes", "n_test", "n_train", "parameters", "partition_algorithm", "pretrained", "source_dataset",
                "torch_num_threads", "train_manifest_sha256", "training_device");
        workload.set("source_hashes", leaves(meta.path("workload").path("source_hashes"), "Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"));
        var splits = workload.putObject("split_counts");
        for (String split : new String[]{"train", "validation", "test"}) splits.set(split,
                selectedNumbers(meta.path("workload").path("split_counts").path(split), "images", "patients", "positive_images", "negative_images"));
        source.set("workload", workload);
        return source;
    }

    private static JsonNode scalar(JsonNode node) throws IOException {
        if (node.isMissingNode() || node.isNull()) return SimulationJson.MAPPER.nullNode();
        if (node.isNumber() && Double.isFinite(node.asDouble())) return node;
        if (node.isObject() && node.size() == 1 && Set.of("nan", "inf", "-inf").contains(node.path("__nonfinite_float__").asText())) return node;
        invalid(); return null;
    }

    private static boolean containsTag(JsonNode node) {
        if (node.isObject() && node.has("__nonfinite_float__")) return true;
        if (node.isContainerNode()) for (JsonNode child : node) if (containsTag(child)) return true;
        return false;
    }

    private static int integer(JsonNode value) throws IOException {
        if (!value.isIntegralNumber() || !value.canConvertToInt()) invalid();
        return value.asInt();
    }

    private static JsonNode file(Path path) throws IOException {
        if (Files.isSymbolicLink(path) || !Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) || Files.size(path) > 8 * 1024 * 1024) invalid();
        return SimulationJson.parse(Files.readAllBytes(path), 8 * 1024 * 1024);
    }

    private static void invalid() throws IOException { throw new IOException("Invalid simulation evidence"); }
}
