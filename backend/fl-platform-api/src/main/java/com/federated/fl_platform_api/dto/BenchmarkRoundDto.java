package com.federated.fl_platform_api.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * Ingest payload POSTed once per round by scripts/benchmarks.py (via fl_server.py)
 * to /api/internal/benchmarks/{projectId}. Field names match the camelCase keys
 * emitted by benchmarks.build_round_record() 1:1, so Jackson binds with no
 * renaming. Every metric is nullable — a field is simply absent when the task
 * (classification vs generative) or available data doesn't produce it.
 */
@Data
public class BenchmarkRoundDto {

    private Integer serverRound;
    private String modelType;
    private String taskType;

    // Model quality (macro scalars)
    private Double loss;
    private Double accuracy;
    private Double balancedAccuracy;
    private Double precisionMacro;
    private Double recallMacro;
    private Double f1Macro;
    private Double precisionMicro;
    private Double recallMicro;
    private Double f1Micro;
    private Double precisionWeighted;
    private Double recallWeighted;
    private Double f1Weighted;
    private Double mcc;
    private Double cohenKappa;
    private Double rocAuc;
    private Double logLoss;
    private Double perplexity;
    private Double tokenAccuracy;
    private Double ece;
    private Double brier;
    private Double targetAccuracy;

    // System / efficiency
    private Long roundDurationMs;
    private Long evalDurationMs;
    private Double modelSizeMb;
    private Long paramCount;
    private Integer clientCount;
    private Integer samplesEvaluated;
    private Integer numClasses;

    // Micro / structured
    private List<PerClassMetricDto> perClass;
    private List<List<Integer>> confusionMatrix;
    private List<String> classLabels;
    private Map<String, Double> extraMetrics;
}
