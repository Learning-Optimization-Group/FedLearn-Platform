package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ArtifactDto;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactBlobStore;
import com.federated.fl_platform_api.service.AuthorizationService;

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
    private final AuthorizationService authz;
    private final ProjectRepository projects;

    public ArtifactController(ModelArtifactRepository artifacts, ArtifactBlobStore blobStore, OrgScope orgScope,
                              AuthorizationService authz, ProjectRepository projects) {
        this.artifacts = artifacts;
        this.blobStore = blobStore;
        this.orgScope = orgScope;
        this.authz = authz;
        this.projects = projects;
    }

    /**
     * The project's artifacts the caller may see, newest first — the catalog the registry surface (FE-11)
     * lists. Cross-org rows are filtered out (an empty list, never a 403/leak); a projectId in no visible
     * org simply yields {@code []}. Org-scoped exactly like {@link #get}, just widened to the whole project.
     */
    @GetMapping
    public java.util.List<ArtifactDto> list(@RequestParam UUID projectId) {
        return artifacts.findByProjectId(projectId).stream()
                // SE-16: org-visible AND (published OR participant). A non-participant sees only the
                // project's PUBLISHED rows (the marketplace items), never its private weights.
                .filter(this::mayRead)
                .sorted(java.util.Comparator.comparing(ModelArtifact::getCreatedAt,
                        java.util.Comparator.nullsLast(java.util.Comparator.naturalOrder())).reversed())
                .map(ArtifactDto::from)
                .toList();
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
        if (a == null || !mayRead(a)) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(ArtifactDto.from(a));
    }

    private ModelArtifact visibleOr404(UUID id) {
        ModelArtifact a = artifacts.findById(id).orElse(null);
        return (a != null && mayRead(a)) ? a : null;
    }

    /**
     * SE-16 read gate. An artifact is readable iff it is in the caller's org scope AND one of:
     * it is org-shared with no owning project ({@code projectId == null}, e.g. a {@code BASE_REF});
     * it has been explicitly PUBLISHED to the org marketplace (FE-12); or the caller is a participant
     * (owner/member/client/admin) of its project. Otherwise a non-participant reads it as absent, so a
     * project's private model weights — and the mere existence of the project — never leak, while the
     * intentional publish-to-share flow keeps working. Mirrors the gate the rest of the project read
     * surface applies (results, logs, STOMP).
     */
    private boolean mayRead(ModelArtifact a) {
        if (!orgScope.allows(a.getOrgId())) {
            return false; // tenant isolation
        }
        if (a.getProjectId() == null || a.isPublished()) {
            return true; // org-shared base, or explicitly published to the marketplace
        }
        Project p = projects.findById(a.getProjectId()).orElse(null);
        return p != null && authz.isParticipant(p);
    }
}
