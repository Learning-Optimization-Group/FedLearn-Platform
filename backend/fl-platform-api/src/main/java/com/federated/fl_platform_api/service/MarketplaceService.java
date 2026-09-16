package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ArtifactDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.security.OrgScope;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

/**
 * FE-12: the org-internal adapter marketplace. A {@link ArtifactKind#LORA_ADAPTER} artifact can be
 * <em>published</em> by its project's owner (or a platform admin) to make it discoverable across the
 * other projects in the SAME org; discovery never crosses the org boundary.
 *
 * <p>Security model (unchanged from the rest of the artifact API):</p>
 * <ul>
 *   <li>A foreign-org artifact id is a 404, never a 403 — publishing does not leak existence.</li>
 *   <li>Publish/unpublish require project owner or platform admin ({@code requireOwnerOrAdmin}).</li>
 *   <li>Only {@code LORA_ADAPTER} artifacts (the tradable unit) are publishable, and only if they
 *       have an owning project to check ownership against.</li>
 *   <li>Discovery is filtered to the caller's visible orgs at the DB level (leak-proof), or all orgs
 *       for a platform admin (unrestricted scope). A cross-org / PUBLIC marketplace is intentionally
 *       out of scope — that is a separate, threat-model-sensitive design.</li>
 * </ul>
 */
@Service
public class MarketplaceService {

    private static final ArtifactKind MARKETPLACE_KIND = ArtifactKind.LORA_ADAPTER;

    private final ModelArtifactRepository artifacts;
    private final ProjectRepository projects;
    private final AuthorizationService authz;
    private final OrgScope orgScope;

    public MarketplaceService(ModelArtifactRepository artifacts, ProjectRepository projects,
                              AuthorizationService authz, OrgScope orgScope) {
        this.artifacts = artifacts;
        this.projects = projects;
        this.authz = authz;
        this.orgScope = orgScope;
    }

    /** Publish a LORA_ADAPTER to the org marketplace. Owner-or-admin, org-scoped, idempotent. */
    @Transactional
    public ArtifactDto publish(UUID artifactId) {
        ModelArtifact a = requirePublishable(artifactId);
        if (!a.isPublished()) {
            a.setPublished(true);
            a.setPublishedAt(Instant.now());
            a = artifacts.save(a);
        }
        return ArtifactDto.from(a);
    }

    /** Withdraw a published adapter from the marketplace. Same authz as {@link #publish}. */
    @Transactional
    public ArtifactDto unpublish(UUID artifactId) {
        ModelArtifact a = requirePublishable(artifactId);
        if (a.isPublished()) {
            a.setPublished(false);
            a.setPublishedAt(null);
            a = artifacts.save(a);
        }
        return ArtifactDto.from(a);
    }

    /** The marketplace feed: published adapters the caller may see, newest-published first. */
    @Transactional(readOnly = true)
    public List<ArtifactDto> listPublishedAdapters() {
        List<ModelArtifact> rows;
        if (orgScope.isUnrestricted()) {
            rows = artifacts.findByKindAndPublishedIsTrueOrderByPublishedAtDesc(MARKETPLACE_KIND);
        } else if (orgScope.visibleOrgIds().isEmpty()) {
            rows = List.of();
        } else {
            rows = artifacts.findByOrgIdInAndKindAndPublishedIsTrueOrderByPublishedAtDesc(
                    orgScope.visibleOrgIds(), MARKETPLACE_KIND);
        }
        return rows.stream().map(ArtifactDto::from).toList();
    }

    /**
     * Load an artifact and assert the caller may publish it: it exists AND is in the caller's org
     * (else 404 — no existence leak), is a LORA_ADAPTER with an owning project (else 409), and the
     * caller owns that project or is a platform admin (else 403).
     */
    private ModelArtifact requirePublishable(UUID artifactId) {
        ModelArtifact a = artifacts.findById(artifactId).orElse(null);
        if (a == null || !orgScope.allows(a.getOrgId())) {
            throw new ResourceNotFoundException("Artifact not found: " + artifactId);
        }
        if (a.getKind() != MARKETPLACE_KIND) {
            throw new ProjectStateException(
                    "Only LORA_ADAPTER artifacts can be published to the marketplace (was " + a.getKind() + ")");
        }
        if (a.getProjectId() == null) {
            throw new ProjectStateException("Artifact has no owning project to authorize publication against");
        }
        Project project = projects.findById(a.getProjectId())
                .orElseThrow(() -> ResourceNotFoundException.project(a.getProjectId()));
        authz.requireOwnerOrAdmin(project); // 403 for an in-org non-owner
        return a;
    }
}
