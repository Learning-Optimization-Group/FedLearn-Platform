package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.EnrollmentDto;
import com.federated.fl_platform_api.dto.ModelBundleDto;
import com.federated.fl_platform_api.dto.RunManifestDto;
import com.federated.fl_platform_api.dto.RunStatusDto;
import com.federated.fl_platform_api.service.RunService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.nio.file.Files;

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

    /**
     * Stream one artifact the run's execution contract lists, by its SHA-256 (Stage 3 slice A2). The strong ETag is the
     * hash. A single byte range ({@code bytes=N-} or {@code bytes=N-M}) is honoured when there is no If-Range or it
     * matches the ETag, so a phone can resume a large download; a stale If-Range, or a range this does not serve, gets
     * the whole file. An unsatisfiable range gets 416. Ranges are handled here rather than by Spring's Resource
     * support, which does not evaluate If-Range.
     */
    @GetMapping("/{runId}/artifacts/{sha256}")
    public ResponseEntity<StreamingResponseBody> contractArtifact(
            @PathVariable UUID runId, @PathVariable String sha256,
            @RequestHeader(value = HttpHeaders.RANGE, required = false) String range,
            @RequestHeader(value = HttpHeaders.IF_RANGE, required = false) String ifRange) {
        RunService.ContractArtifact artifact = runService.getContractArtifact(runId, sha256);
        String etag = "\"" + artifact.sha256() + "\"";
        long size = artifact.byteSize();
        HttpHeaders headers = new HttpHeaders();
        headers.setETag(etag);
        headers.set(HttpHeaders.ACCEPT_RANGES, "bytes");
        headers.setCacheControl(CacheControl.noCache().cachePrivate());
        headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);

        long[] span = (range != null && (ifRange == null || ifRange.equals(etag))) ? singleRange(range, size) : null;
        if (span == UNSATISFIABLE) {
            headers.set(HttpHeaders.CONTENT_RANGE, "bytes */" + size);
            return new ResponseEntity<>(headers, HttpStatus.REQUESTED_RANGE_NOT_SATISFIABLE);
        }
        long from = span == null ? 0 : span[0];
        long to = span == null ? size - 1 : span[1];
        headers.setContentLength(to - from + 1);
        if (span != null) {
            headers.set(HttpHeaders.CONTENT_RANGE, "bytes " + from + "-" + to + "/" + size);
        }
        StreamingResponseBody body = out -> {
            try (var in = Files.newInputStream(artifact.file())) {
                in.skipNBytes(from);
                long remaining = to - from + 1;
                byte[] buffer = new byte[64 * 1024];
                while (remaining > 0) {
                    int n = in.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                    if (n < 0) break;
                    out.write(buffer, 0, n);
                    remaining -= n;
                }
            }
        };
        return new ResponseEntity<>(body, headers, span == null ? HttpStatus.OK : HttpStatus.PARTIAL_CONTENT);
    }

    private static final long[] UNSATISFIABLE = new long[0];
    private static final java.util.regex.Pattern SINGLE_RANGE = java.util.regex.Pattern.compile("bytes=(\\d+)-(\\d*)");

    /** [from, to] for one satisfiable range, UNSATISFIABLE when it starts past the end, null when not served. */
    private static long[] singleRange(String header, long size) {
        java.util.regex.Matcher m = SINGLE_RANGE.matcher(header.trim());
        if (!m.matches()) {
            return null;   // suffix or multiple ranges: serve the whole file, which a server may always do
        }
        long from = Long.parseLong(m.group(1));
        if (from >= size) {
            return UNSATISFIABLE;
        }
        long to = m.group(2).isEmpty() ? size - 1 : Math.min(Long.parseLong(m.group(2)), size - 1);
        return to < from ? null : new long[]{from, to};
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
