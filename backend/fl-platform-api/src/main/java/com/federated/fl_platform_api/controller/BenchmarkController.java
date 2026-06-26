package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.BenchmarkOverviewDto;
import com.federated.fl_platform_api.dto.ProjectBenchmarkDto;
import com.federated.fl_platform_api.service.BenchmarkService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

/**
 * Admin-only read API for the benchmarking & observability dashboard. Mirrors
 * {@code AdminController}'s authorization (class-level PLATFORM_ADMIN).
 */
@RestController
@RequestMapping("/api/admin/benchmarks")
@PreAuthorize("hasRole('PLATFORM_ADMIN')")
public class BenchmarkController {

    private final BenchmarkService benchmarkService;

    public BenchmarkController(BenchmarkService benchmarkService) {
        this.benchmarkService = benchmarkService;
    }

    /** Platform-wide aggregates + the full per-project runs table. */
    @GetMapping("/overview")
    public ResponseEntity<BenchmarkOverviewDto> overview() {
        return ResponseEntity.ok(benchmarkService.getOverview());
    }

    /** Per-project drilldown: time series + latest per-class table + confusion matrix. */
    @GetMapping("/projects/{projectId}")
    public ResponseEntity<ProjectBenchmarkDto> project(@PathVariable UUID projectId) {
        return ResponseEntity.ok(benchmarkService.getProjectBenchmark(projectId));
    }
}
