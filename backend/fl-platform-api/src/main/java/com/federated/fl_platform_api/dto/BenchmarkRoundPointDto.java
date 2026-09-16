package com.federated.fl_platform_api.dto;

import lombok.Data;

/** One round's scalar metrics — the unit of the per-run time series the charts plot. */
@Data
public class BenchmarkRoundPointDto {
    private Integer serverRound;

    // Quality
    private Double loss;
    private Double accuracy;
    private Double balancedAccuracy;
    private Double precisionMacro;
    private Double recallMacro;
    private Double f1Macro;
    private Double f1Micro;
    private Double f1Weighted;
    private Double mcc;
    private Double cohenKappa;
    private Double rocAuc;
    private Double logLoss;
    private Double ece;
    private Double brier;
    private Double perplexity;
    private Double tokenAccuracy;

    // System / efficiency
    private Long roundDurationMs;
    private Long evalDurationMs;
    private Double modelSizeMb;
    private Long paramCount;
    private Integer clientCount;
    private Integer samplesEvaluated;
}
