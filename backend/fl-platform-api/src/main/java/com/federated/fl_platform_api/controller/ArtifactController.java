package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ArtifactDto;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactBlobStore;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

/**
 * BA-11: the read side of the content-addressed model-artifact registry. Until now the registry was
 * write-only (register on run completion) with no way to read a blob back — so it merely duplicated the
 * overwritable {@code .npz}. This exposes the registry as the readable source of truth:
 *
 * <ul>
 *   <li>{@code GET /api/artifacts/{id}} — artifact metadata (incl. the {@code blobSha256} content address);</li>
 *   <li>{@code GET /api/artifacts/{id}/blob} — the immutable bytes, integrity-checked on read (the store
 *       recomputes the sha256 and refuses to serve a corrupted blob);</li>
 *   <li>{@code GET /api/artifacts/latest?projectId=&kind=} — the project's current head artifact.</li>
 * </ul>
 *
 * <p>Session-authenticated ({@code /api/**}) and org-scoped: a caller may only read artifacts in an org it
 * can see — a cross-org id returns 404, never 403, so existence doesn't leak (mirrors
 * {@link ArtifactLineageController}).</p>
 */
@RestController
@RequestMapping("/api/artifacts")
public class ArtifactController {

    private final ModelArtifactRepository artifacts;
    private final ArtifactBlobStore blobStore;
    private final OrgScope orgScope;

    public ArtifactController(ModelArtifactRepository artifacts, ArtifactBlobStore blobStore, OrgScope orgScope) {
        this.artifacts = artifacts;
        this.blobStore = blobStore;
        this.orgScope = orgScope;
    }

    /** Artifact metadata, or 404 if it does not exist OR is outside the caller's orgs (no existence leak). */
    @GetMapping("/{id}")
    public ResponseEntity<ArtifactDto> get(@PathVariable UUID id) {
        ModelArtifact a = visibleOr404(id);
        return a == null ? ResponseEntity.notFound().build() : ResponseEntity.ok(ArtifactDto.from(a));
    }

    /** Download the artifact's immutable, content-addressed bytes. The blob store verifies the on-disk
     *  content hashes to the stored key before returning, so a corrupted blob 500s rather than serving
     *  wrong weights. The content hash is echoed as a (strong) ETag. */
    @GetMapping("/{id}/blob")
    public ResponseEntity<byte[]> blob(@PathVariable UUID id) {
        ModelArtifact a = visibleOr404(id);
        if (a == null) {
            return ResponseEntity.notFound().build();
        }
        byte[] bytes = blobStore.get(a.getBlobSha256()); // integrity-checked on read (BA-11)
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .eTag("\"" + a.getBlobSha256() + "\"")
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + id + ".bin\"")
                .body(bytes);
    }

    /** The project's most recent artifact of {@code kind} (default FULL_CHECKPOINT) — its current head. */
    @GetMapping("/latest")
    public ResponseEntity<ArtifactDto> latest(
            @RequestParam UUID projectId,
            @RequestParam(defaultValue = "FULL_CHECKPOINT") ArtifactKind kind) {
        ModelArtifact a = artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(projectId, kind).orElse(null);
        if (a == null || !orgScope.allows(a.getOrgId())) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(ArtifactDto.from(a));
    }

    private ModelArtifact visibleOr404(UUID id) {
        ModelArtifact a = artifacts.findById(id).orElse(null);
        return (a != null && orgScope.allows(a.getOrgId())) ? a : null;
    }
}
