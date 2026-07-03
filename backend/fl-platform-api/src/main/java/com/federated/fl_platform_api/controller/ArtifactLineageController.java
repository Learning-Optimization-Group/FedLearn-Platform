package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Reads the provenance chain (base → … → leaf) for a model artifact. Authenticated (all of
 * {@code /api/**} requires a session) and org-scoped: a caller may only read lineage for an artifact
 * in an org it can see — anything else returns 404 so cross-org existence is never leaked.
 */
@RestController
@RequestMapping("/api/artifacts")
public class ArtifactLineageController {

    private final ArtifactRegistryService registry;
    private final ModelArtifactRepository artifacts;
    private final OrgScope orgScope;

    public ArtifactLineageController(ArtifactRegistryService registry,
                                     ModelArtifactRepository artifacts,
                                     OrgScope orgScope) {
        this.registry = registry;
        this.artifacts = artifacts;
        this.orgScope = orgScope;
    }

    @GetMapping("/{id}/lineage")
    public ResponseEntity<List<Map<String, Object>>> lineage(@PathVariable UUID id) {
        ModelArtifact target = artifacts.findById(id).orElse(null);
        if (target == null || !orgScope.allows(target.getOrgId())) {
            return ResponseEntity.notFound().build(); // 404 for both — no cross-org existence leak
        }
        List<Map<String, Object>> chain = registry.getLineageChain(id).stream()
                .map(ArtifactLineageController::toDto)
                .toList();
        return ResponseEntity.ok(chain);
    }

    private static Map<String, Object> toDto(ModelArtifact a) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", a.getId().toString());
        m.put("kind", a.getKind().name());
        m.put("sha256", a.getBlobSha256());
        m.put("baseModelRef", a.getBaseModelRef());
        m.put("licenseTag", a.getLicenseTag());
        m.put("createdAt", a.getCreatedAt() == null ? null : a.getCreatedAt().toString());
        return m;
    }
}
