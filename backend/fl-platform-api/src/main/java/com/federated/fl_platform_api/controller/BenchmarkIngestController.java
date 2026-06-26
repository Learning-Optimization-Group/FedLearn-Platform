package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.BenchmarkRoundDto;
import com.federated.fl_platform_api.service.BenchmarkService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

/**
 * Internal ingest for per-round benchmark records POSTed by scripts/benchmarks.py
 * (via fl_server.py). Lives under /api/internal/** so it is gated by
 * {@code InternalApiKeyFilter} (X-Internal-Key) — no user JWT, same as the
 * round-result callback.
 */
@RestController
@RequestMapping("/api/internal/benchmarks")
public class BenchmarkIngestController {

    private final BenchmarkService benchmarkService;

    @Value("${feature.benchmark-reporting.enabled:true}")
    private boolean enabled;

    public BenchmarkIngestController(BenchmarkService benchmarkService) {
        this.benchmarkService = benchmarkService;
    }

    @PostMapping("/{projectId}")
    public ResponseEntity<Void> recordRound(@PathVariable @NonNull UUID projectId,
                                            @RequestBody BenchmarkRoundDto dto) {
        if (!enabled || dto == null || dto.getServerRound() == null) {
            return ResponseEntity.ok().build();
        }
        benchmarkService.recordRound(projectId, dto);
        return ResponseEntity.ok().build();
    }
}
