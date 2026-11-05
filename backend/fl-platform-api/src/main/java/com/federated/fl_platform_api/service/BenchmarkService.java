package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.*;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.BenchmarkRound;
import com.federated.fl_platform_api.model.BenchmarkRun;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.InternalCallbackAuthorization;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import java.util.Objects;
import com.federated.fl_platform_api.repository.BenchmarkRoundRepository;
import com.federated.fl_platform_api.repository.BenchmarkRunRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Ingest + read model for the benchmarking suite. {@link #recordRound} preserves an
 * immutable run/round metric vector and recomputes the denormalized run rollup (including
 * time-to-target-accuracy); the read methods power the admin dashboard.
 */
@Service
public class BenchmarkService {

    private static final Logger log = LoggerFactory.getLogger(BenchmarkService.class);

    private final BenchmarkRoundRepository roundRepo;
    private final BenchmarkRunRepository runRepo;
    private final ProjectRepository projectRepository;
    private final ObjectMapper objectMapper;
    private final RunRepository trainingRuns;
    private final InternalCallbackAuthorization authorization;

    public BenchmarkService(BenchmarkRoundRepository roundRepo,
                            BenchmarkRunRepository runRepo,
                            ProjectRepository projectRepository,
                            ObjectMapper objectMapper,
                            RunRepository trainingRuns,
                            InternalCallbackAuthorization authorization) {
        this.roundRepo = roundRepo;
        this.runRepo = runRepo;
        this.projectRepository = projectRepository;
        this.objectMapper = objectMapper;
        this.trainingRuns = trainingRuns;
        this.authorization = authorization;
    }

    // ── Ingest ────────────────────────────────────────────────────────────────
    @Transactional
    public void recordRound(UUID projectId, InternalCallbackContext context, BenchmarkRoundDto dto) {
        var bound = authorization.requireActiveRun(projectId, context);
        Project project = bound.project();
        Run trainingRun = bound.run();
        if (dto == null || dto.getServerRound() == null || dto.getServerRound() < 1
                || dto.getServerRound() > trainingRun.getNumRounds()) {
            throw new IllegalArgumentException("Round must be within the run's round budget");
        }
        var intent = trainingRun.getIntent().orElseThrow(() ->
                new IllegalStateException("Run has no recorded benchmark intent"));
        if (dto.getModelType() != null && !dto.getModelType().equals(trainingRun.getRecipeKey())) {
            throw new IllegalArgumentException("Benchmark model must match the recorded run");
        }
        String expectedTask = taskType(intent.taskType());
        if (dto.getTaskType() != null && !taskType(dto.getTaskType()).equals(expectedTask)) {
            throw new IllegalArgumentException("Benchmark task must match the recorded run");
        }
        dto.setModelType(trainingRun.getRecipeKey());
        dto.setTaskType(expectedTask);
        String fingerprint = BenchmarkPayloadFingerprint.of(dto, objectMapper);
        var existing = roundRepo.findByRunIdAndServerRound(trainingRun.getId(), dto.getServerRound());
        if (existing.isPresent()) {
            if (!fingerprint.equals(existing.get().getPayloadSha256())) {
                throw new IllegalStateException("Benchmark replay conflicts with recorded or unverified legacy evidence");
            }
            return;
        }
        BenchmarkRound row = new BenchmarkRound();
        row.setProjectId(projectId);
        row.setRunId(trainingRun.getId());
        row.setPayloadSha256(fingerprint);
        row.setServerRound(dto.getServerRound());
        row.setModelType(dto.getModelType());
        row.setTaskType(dto.getTaskType());

        row.setLoss(dto.getLoss());
        row.setAccuracy(dto.getAccuracy());
        row.setBalancedAccuracy(dto.getBalancedAccuracy());
        row.setPrecisionMacro(dto.getPrecisionMacro());
        row.setRecallMacro(dto.getRecallMacro());
        row.setF1Macro(dto.getF1Macro());
        row.setPrecisionMicro(dto.getPrecisionMicro());
        row.setRecallMicro(dto.getRecallMicro());
        row.setF1Micro(dto.getF1Micro());
        row.setPrecisionWeighted(dto.getPrecisionWeighted());
        row.setRecallWeighted(dto.getRecallWeighted());
        row.setF1Weighted(dto.getF1Weighted());
        row.setMcc(dto.getMcc());
        row.setCohenKappa(dto.getCohenKappa());
        row.setRocAuc(dto.getRocAuc());
        row.setLogLoss(dto.getLogLoss());
        row.setPerplexity(dto.getPerplexity());
        row.setTokenAccuracy(dto.getTokenAccuracy());
        row.setEce(dto.getEce());
        row.setBrier(dto.getBrier());
        row.setTargetAccuracy(dto.getTargetAccuracy());

        row.setRoundDurationMs(dto.getRoundDurationMs());
        row.setEvalDurationMs(dto.getEvalDurationMs());
        row.setModelSizeMb(dto.getModelSizeMb());
        row.setParamCount(dto.getParamCount());
        row.setClientCount(dto.getClientCount());
        row.setSamplesEvaluated(dto.getSamplesEvaluated());
        row.setNumClasses(dto.getNumClasses());

        row.setPerClassJson(toJson(dto.getPerClass()));
        row.setConfusionMatrixJson(toJson(dto.getConfusionMatrix()));
        row.setClassLabelsJson(toJson(dto.getClassLabels()));
        row.setExtraMetricsJson(toJson(dto.getExtraMetrics()));
        row.setRecordedAt(Instant.now());

        roundRepo.save(row);
        List<BenchmarkRound> selected = roundRepo.findByProjectIdAndRunIdOrderByServerRoundAsc(projectId, trainingRun.getId());
        BenchmarkRun summary = buildSummary(project, trainingRun.getId(), selected);
        runRepo.findByProjectId(projectId).ifPresent(previous -> summary.setId(previous.getId()));
        runRepo.save(summary);
    }

    private BenchmarkRun buildSummary(Project project, UUID runId, List<BenchmarkRound> rounds) {
        BenchmarkRun run = new BenchmarkRun();
        BenchmarkRound last = rounds.get(rounds.size() - 1);

        run.setProjectId(project.getId());
        run.setRunId(runId);
        run.setRunIdentityVerified(runId != null && rounds.stream().allMatch(r -> r.getPayloadSha256() != null));
        run.setProjectName(project.getName());
        run.setModelType(last.getModelType());
        run.setTaskType(last.getTaskType());
        run.setRoundsCompleted(rounds.size());
        run.setFinalLoss(last.getLoss());
        run.setFinalAccuracy(last.getAccuracy());
        run.setFinalF1Macro(last.getF1Macro());
        run.setFinalPerplexity(last.getPerplexity());
        run.setFinalEce(last.getEce());
        run.setModelSizeMb(last.getModelSizeMb());
        run.setParamCount(last.getParamCount());
        run.setClientCount(last.getClientCount());

        Double bestAcc = null, bestPpl = null, target = null;
        Integer bestRound = null;
        long totalMs = 0;
        int msCount = 0;
        for (BenchmarkRound r : rounds) {
            if (r.getAccuracy() != null && (bestAcc == null || r.getAccuracy() > bestAcc)) {
                bestAcc = r.getAccuracy();
                bestRound = r.getServerRound();
            }
            if (r.getPerplexity() != null && (bestPpl == null || r.getPerplexity() < bestPpl)) {
                bestPpl = r.getPerplexity();
            }
            if (r.getTargetAccuracy() != null) target = r.getTargetAccuracy();
            if (r.getRoundDurationMs() != null) {
                totalMs = addDuration(totalMs, r.getRoundDurationMs());
                msCount++;
            }
        }
        run.setBestAccuracy(bestAcc);
        run.setBestRound(bestRound);
        run.setBestPerplexity(bestPpl);
        run.setTotalRoundMs(totalMs);
        run.setAvgRoundMs(msCount > 0 ? totalMs / msCount : null);

        // Time-to-target-accuracy: first round to reach the target, with the
        // cumulative wall-clock up to (and including) that round.
        run.setTargetAccuracy(target);
        run.setRoundsToTarget(null);
        run.setMsToTarget(null);
        if (target != null && run.isRunIdentityVerified()) {
            long cum = 0;
            int expectedRound = 1;
            boolean completeAccuracy = true, completeDuration = true;
            for (BenchmarkRound r : rounds) {
                if (r.getServerRound() != expectedRound || r.getAccuracy() == null) completeAccuracy = false;
                expectedRound = r.getServerRound() + 1;
                if (r.getRoundDurationMs() != null) cum = addDuration(cum, r.getRoundDurationMs());
                else completeDuration = false;
                if (r.getAccuracy() != null && r.getAccuracy() >= target) {
                    if (completeAccuracy) {
                        run.setRoundsToTarget(r.getServerRound());
                        if (completeDuration) run.setMsToTarget(cum);
                    }
                    break;
                }
            }
        }

        run.setFirstRecordedAt(rounds.stream().map(BenchmarkRound::getRecordedAt).min(Comparator.naturalOrder()).orElseThrow());
        run.setLastRecordedAt(rounds.stream().map(BenchmarkRound::getRecordedAt).max(Comparator.naturalOrder()).orElseThrow());
        return run;
    }

    // ── Reads ───────────────────────────────────────────────────────────────
    @Transactional(readOnly = true)
    public BenchmarkOverviewDto getOverview() {
        List<BenchmarkRun> runs = runRepo.findAllByOrderByLastRecordedAtDesc();
        List<BenchmarkRun> verified = runs.stream().filter(BenchmarkRun::isRunIdentityVerified).toList();
        BenchmarkOverviewDto o = new BenchmarkOverviewDto();
        o.setBenchmarkedProjects(runs.size());
        o.setTotalRoundsRecorded(verified.stream()
                .mapToLong(r -> r.getRoundsCompleted() == null ? 0 : r.getRoundsCompleted()).sum());

        List<BenchmarkRun> classification = verified.stream()
                .filter(r -> !isGenerative(r.getTaskType())).collect(Collectors.toList());
        o.setClassificationRuns(classification.size());
        o.setGenerativeRuns(verified.size() - classification.size());

        o.setAvgFinalAccuracy(avg(classification, BenchmarkRun::getFinalAccuracy));
        o.setAvgFinalF1Macro(avg(classification, BenchmarkRun::getFinalF1Macro));
        o.setAvgRoundDurationMs(avg(verified, r -> r.getAvgRoundMs() == null ? null : r.getAvgRoundMs().doubleValue()));
        o.setAvgModelSizeMb(avg(verified, BenchmarkRun::getModelSizeMb));

        verified.stream()
                .filter(r -> r.getBestAccuracy() != null)
                .max(Comparator.comparingDouble(BenchmarkRun::getBestAccuracy))
                .ifPresent(best -> {
                    o.setBestAccuracy(best.getBestAccuracy());
                    o.setBestAccuracyProject(best.getProjectName());
                });

        o.setRuns(runs.stream().map(this::toRunDto).collect(Collectors.toList()));
        return o;
    }

    @Transactional(readOnly = true)
    public ProjectBenchmarkDto getProjectBenchmark(UUID projectId) {
        return getProjectBenchmark(projectId, null);
    }

    @Transactional(readOnly = true)
    public ProjectBenchmarkDto getProjectBenchmark(UUID projectId, UUID requestedRunId) {
        return getProjectBenchmark(projectId, requestedRunId, false);
    }

    @Transactional(readOnly = true)
    public ProjectBenchmarkDto getProjectBenchmark(UUID projectId, UUID requestedRunId, boolean legacy) {
        if (legacy && requestedRunId != null) throw new IllegalArgumentException("Select a run or legacy records, not both");
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        UUID runId = legacy ? null : requestedRunId != null ? requestedRunId : project.getActiveRunId();
        if (!legacy && runId == null) runId = runRepo.findByProjectId(projectId).map(BenchmarkRun::getRunId).orElse(null);
        List<BenchmarkRound> rounds = roundRepo.findByProjectIdAndRunIdOrderByServerRoundAsc(projectId, runId);
        if (runId != null) {
            var selected = trainingRuns.findById(runId);
            if ((selected.isPresent() && !projectId.equals(selected.get().getProjectId()))
                    || (selected.isEmpty() && rounds.isEmpty())) {
                throw new ResourceNotFoundException("Benchmark run not found");
            }
        }
        ProjectBenchmarkDto d = new ProjectBenchmarkDto();
        d.setRunId(runId == null ? null : runId.toString());
        if (!rounds.isEmpty()) d.setSummary(toRunDto(buildSummary(project, runId, rounds)));
        d.setRounds(rounds.stream().map(this::toPointDto).collect(Collectors.toList()));

        if (!rounds.isEmpty()) {
            // Use the most recent round that actually carries structured metrics
            // (generative rounds won't have a confusion matrix / per-class table).
            BenchmarkRound src = rounds.get(rounds.size() - 1);
            for (int i = rounds.size() - 1; i >= 0; i--) {
                if (rounds.get(i).getPerClassJson() != null) {
                    src = rounds.get(i);
                    break;
                }
            }
            d.setTaskType(src.getTaskType());
            d.setClassLabels(fromJson(src.getClassLabelsJson(), new TypeReference<List<String>>() {}));
            d.setLatestPerClass(fromJson(src.getPerClassJson(), new TypeReference<List<PerClassMetricDto>>() {}));
            d.setLatestConfusionMatrix(fromJson(src.getConfusionMatrixJson(), new TypeReference<List<List<Integer>>>() {}));
        }
        return d;
    }

    // ── Mapping ───────────────────────────────────────────────────────────────
    private BenchmarkRunDto toRunDto(BenchmarkRun r) {
        BenchmarkRunDto d = new BenchmarkRunDto();
        d.setProjectId(r.getProjectId() != null ? r.getProjectId().toString() : null);
        d.setRunId(r.getRunId() == null ? null : r.getRunId().toString());
        d.setRunIdentityVerified(r.isRunIdentityVerified());
        d.setProjectName(r.getProjectName());
        d.setModelType(r.getModelType());
        d.setTaskType(r.getTaskType());
        d.setRoundsCompleted(r.getRoundsCompleted());
        d.setFinalLoss(r.getFinalLoss());
        d.setFinalAccuracy(r.getFinalAccuracy());
        d.setBestAccuracy(r.getBestAccuracy());
        d.setBestRound(r.getBestRound());
        d.setFinalF1Macro(r.getFinalF1Macro());
        d.setFinalPerplexity(r.getFinalPerplexity());
        d.setBestPerplexity(r.getBestPerplexity());
        d.setFinalEce(r.getFinalEce());
        d.setTargetAccuracy(r.getTargetAccuracy());
        d.setRoundsToTarget(r.isRunIdentityVerified() ? r.getRoundsToTarget() : null);
        d.setMsToTarget(r.isRunIdentityVerified() ? r.getMsToTarget() : null);
        d.setTotalRoundMs(r.getTotalRoundMs());
        d.setAvgRoundMs(r.getAvgRoundMs());
        d.setModelSizeMb(r.getModelSizeMb());
        d.setParamCount(r.getParamCount());
        d.setClientCount(r.getClientCount());
        d.setFirstRecordedAt(r.getFirstRecordedAt() != null ? r.getFirstRecordedAt().toString() : null);
        d.setLastRecordedAt(r.getLastRecordedAt() != null ? r.getLastRecordedAt().toString() : null);
        if (isGenerative(r.getTaskType())) {
            d.setPrimaryMetricName("perplexity");
            d.setPrimaryMetricValue(r.getFinalPerplexity());
        } else {
            d.setPrimaryMetricName("accuracy");
            d.setPrimaryMetricValue(r.getFinalAccuracy());
        }
        return d;
    }

    private BenchmarkRoundPointDto toPointDto(BenchmarkRound r) {
        BenchmarkRoundPointDto d = new BenchmarkRoundPointDto();
        d.setServerRound(r.getServerRound());
        d.setLoss(r.getLoss());
        d.setAccuracy(r.getAccuracy());
        d.setBalancedAccuracy(r.getBalancedAccuracy());
        d.setPrecisionMacro(r.getPrecisionMacro());
        d.setRecallMacro(r.getRecallMacro());
        d.setF1Macro(r.getF1Macro());
        d.setF1Micro(r.getF1Micro());
        d.setF1Weighted(r.getF1Weighted());
        d.setMcc(r.getMcc());
        d.setCohenKappa(r.getCohenKappa());
        d.setRocAuc(r.getRocAuc());
        d.setLogLoss(r.getLogLoss());
        d.setEce(r.getEce());
        d.setBrier(r.getBrier());
        d.setPerplexity(r.getPerplexity());
        d.setTokenAccuracy(r.getTokenAccuracy());
        d.setRoundDurationMs(r.getRoundDurationMs());
        d.setEvalDurationMs(r.getEvalDurationMs());
        d.setModelSizeMb(r.getModelSizeMb());
        d.setParamCount(r.getParamCount());
        d.setClientCount(r.getClientCount());
        d.setSamplesEvaluated(r.getSamplesEvaluated());
        return d;
    }

    // ── Helpers ────────────────────────────────────────────────────────────────
    private static String taskType(String value) {
        if (value == null || value.isBlank() || "CLASSIFICATION".equalsIgnoreCase(value)
                || "SEQ_CLASSIFICATION".equalsIgnoreCase(value)) return "SEQ_CLASSIFICATION";
        if ("CAUSAL_LM".equalsIgnoreCase(value)) return "CAUSAL_LM";
        throw new IllegalArgumentException("Unknown benchmark task");
    }

    private static long addDuration(long first, long second) {
        try { return Math.addExact(first, second); }
        catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("Benchmark duration total exceeds the supported range", overflow);
        }
    }

    private static boolean isGenerative(String taskType) {
        return "CAUSAL_LM".equalsIgnoreCase(taskType);
    }

    private static Double avg(List<BenchmarkRun> runs, Function<BenchmarkRun, Double> f) {
        List<Double> vals = new ArrayList<>();
        for (BenchmarkRun r : runs) {
            Double v = f.apply(r);
            if (v != null) vals.add(v);
        }
        if (vals.isEmpty()) return null;
        double s = 0;
        for (Double v : vals) s += v;
        return s / vals.size();
    }

    private String toJson(Object o) {
        if (o == null) return null;
        try {
            return objectMapper.writeValueAsString(o);
        } catch (Exception e) {
            log.warn("Failed to serialize benchmark JSON field: {}", e.getMessage());
            return null;
        }
    }

    private <T> T fromJson(String json, TypeReference<T> type) {
        if (json == null || json.isBlank()) return null;
        try {
            return objectMapper.readValue(json, type);
        } catch (Exception e) {
            log.warn("Failed to parse benchmark JSON field: {}", e.getMessage());
            return null;
        }
    }
}
