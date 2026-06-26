package com.federated.fl_platform_api.dto;

import lombok.Data;

import java.util.List;

/**
 * Full per-project benchmark drilldown: the run rollup, the per-round time series,
 * and the latest round's structured (micro) metrics — per-class table + confusion
 * matrix — for the quality detail panels.
 */
@Data
public class ProjectBenchmarkDto {
    private BenchmarkRunDto summary;
    private List<BenchmarkRoundPointDto> rounds;
    private String taskType;
    private List<String> classLabels;
    private List<PerClassMetricDto> latestPerClass;
    private List<List<Integer>> latestConfusionMatrix;
}
