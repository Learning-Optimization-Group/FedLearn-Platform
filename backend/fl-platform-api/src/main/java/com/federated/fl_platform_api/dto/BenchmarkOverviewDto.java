package com.federated.fl_platform_api.dto;

import lombok.Data;

import java.util.List;

/**
 * Admin benchmark landing payload: platform-wide aggregates plus the full runs
 * table (one fetch powers the dashboard's overview). Drilldown is a separate call.
 */
@Data
public class BenchmarkOverviewDto {
    private long benchmarkedProjects;
    private long totalRoundsRecorded;
    private long classificationRuns;
    private long generativeRuns;

    private Double avgFinalAccuracy;     // over classification runs
    private Double avgFinalF1Macro;
    private Double bestAccuracy;
    private String bestAccuracyProject;
    private Double avgRoundDurationMs;
    private Double avgModelSizeMb;

    private List<BenchmarkRunDto> runs;
}
