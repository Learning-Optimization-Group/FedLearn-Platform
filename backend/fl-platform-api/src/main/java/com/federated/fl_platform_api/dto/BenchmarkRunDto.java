package com.federated.fl_platform_api.dto;

import lombok.Data;

/** Per-project benchmark rollup row — the dashboard's runs table + drilldown header. */
@Data
public class BenchmarkRunDto {
    private String projectId;
    private String projectName;
    private String modelType;
    private String taskType;

    private Integer roundsCompleted;
    private Double finalLoss;
    private Double finalAccuracy;
    private Double bestAccuracy;
    private Integer bestRound;
    private Double finalF1Macro;
    private Double finalPerplexity;
    private Double bestPerplexity;
    private Double finalEce;

    // Time-to-target-accuracy
    private Double targetAccuracy;
    private Integer roundsToTarget;
    private Long msToTarget;

    private Long totalRoundMs;
    private Long avgRoundMs;
    private Double modelSizeMb;
    private Long paramCount;
    private Integer clientCount;

    private String firstRecordedAt;
    private String lastRecordedAt;

    /** Convenience headline keyed off task_type (accuracy↑ for classification, perplexity↓ for generative). */
    private String primaryMetricName;
    private Double primaryMetricValue;
}
