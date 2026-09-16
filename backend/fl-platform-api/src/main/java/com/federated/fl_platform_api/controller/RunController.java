package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.EnrollmentDto;
import com.federated.fl_platform_api.dto.ModelBundleDto;
import com.federated.fl_platform_api.dto.RunManifestDto;
import com.federated.fl_platform_api.dto.RunStatusDto;
import com.federated.fl_platform_api.service.RunService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/runs")
public class RunController {

    @Autowired private RunService runService;

    @GetMapping("/{runId}/status")
    public ResponseEntity<RunStatusDto> status(@PathVariable UUID runId) {
        return ResponseEntity.ok(runService.getStatus(runId));
    }

    @GetMapping("/{runId}/manifest")
    public ResponseEntity<RunManifestDto> manifest(@PathVariable UUID runId) {
        return ResponseEntity.ok(runService.getManifest(runId));
    }

    @PostMapping("/{runId}/enroll")
    public ResponseEntity<EnrollmentDto> enroll(@PathVariable UUID runId) {
        return ResponseEntity.ok(runService.enroll(runId));
    }

    /** On-device training bundle metadata (paramLayout, shas, file URLs). */
    @GetMapping("/{runId}/model-bundle")
    public ResponseEntity<ModelBundleDto> modelBundle(@PathVariable UUID runId) {
        return ResponseEntity.ok(runService.getModelBundle(runId));
    }

    /** Stream one whitelisted bundle binary (loss.pte / infer.pte / inputs.f32 / targets.i64). */
    @GetMapping("/{runId}/files/{filename}")
    public ResponseEntity<Resource> bundleFile(@PathVariable UUID runId, @PathVariable String filename) {
        Resource file = runService.getModelFile(runId, filename);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .body(file);
    }
}
