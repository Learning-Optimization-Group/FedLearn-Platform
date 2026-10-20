package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class SimulationDtos {
    private SimulationDtos() { }

    public static final String SCOPE = "Exploratory validation simulation. Research records are not DP-safe. "
            + "Communication and deployment time are modeled, not measured device costs. No patient privacy or clinical guarantee.";
    public record Limits(int deadlineSeconds, int cpuThreads, long observedRssQualificationBytes, int activeWorkers, String scope) { }
    public record Workload(String id, boolean ready, String reason, String label, String representation,
                           String reportSha256, JsonNode splitSummary, Map<String, Object> supportedChoices, Limits limits) { }
    public record Workloads(List<Workload> workloads, String scope) { }
    public record Resolved(String workloadId, String configHash, JsonNode recipe, Limits limits, String scope) { }
    public record Accepted(UUID id, SimulationStatus status, String configHash, JsonNode recipe, Instant createdAt, String scope) { }
    public record Detail(UUID id, SimulationStatus status, String configHash, JsonNode recipe,
                         Instant createdAt, Instant startedAt, Instant endedAt, Integer completedRounds, Integer requestedRounds,
                         JsonNode rounds, JsonNode finalSummary, JsonNode qualification, boolean recordAvailable,
                         String recordEncoding, String errorCode, String errorMessage, String scope) { }
    public record Page(List<Detail> content, long totalElements, int page, int size, String scope) { }
}
