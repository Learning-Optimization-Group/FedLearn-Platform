package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.ModelArtifact;

import java.time.Instant;
import java.util.UUID;

/**
 * BA-11: read-side view of a content-addressed model artifact. {@code blobSha256} is the content address
 * — download the immutable bytes at {@code GET /api/artifacts/{id}/blob} and verify against it.
 *
 * <p>{@code baseModelRef}/{@code licenseTag}/{@code createdBy} carry provenance and {@code evalCardJson}
 * is the raw eval-card JSON (freeform; the registry-surface UI parses it defensively) — the fields the
 * catalog/lineage view (FE-11) renders. {@code evalCardJson} is only populated for a DP-labelled artifact
 * once a committed accountant trace exists (SE-11).</p>
 */
public record ArtifactDto(
        UUID id,
        UUID orgId,
        UUID projectId,
        UUID runId,
        String kind,
        String blobSha256,
        String recipeKey,
        String baseModelRef,
        String licenseTag,
        String evalCardJson,
        Long createdBy,
        Instant createdAt,
        boolean published,
        Instant publishedAt) {

    public static ArtifactDto from(ModelArtifact a) {
        return new ArtifactDto(
                a.getId(), a.getOrgId(), a.getProjectId(), a.getRunId(),
                a.getKind().name(), a.getBlobSha256(), a.getRecipeKey(),
                a.getBaseModelRef(), a.getLicenseTag(), a.getEvalCardJson(), a.getCreatedBy(),
                a.getCreatedAt(), a.isPublished(), a.getPublishedAt());
    }
}
