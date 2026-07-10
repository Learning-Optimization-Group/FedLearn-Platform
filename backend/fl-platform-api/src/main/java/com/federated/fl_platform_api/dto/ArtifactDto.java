package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.ModelArtifact;

import java.time.Instant;
import java.util.UUID;

/**
 * BA-11: read-side view of a content-addressed model artifact. {@code blobSha256} is the content address
 * — download the immutable bytes at {@code GET /api/artifacts/{id}/blob} and verify against it.
 */
public record ArtifactDto(
        UUID id,
        UUID orgId,
        UUID projectId,
        UUID runId,
        String kind,
        String blobSha256,
        String recipeKey,
        Instant createdAt) {

    public static ArtifactDto from(ModelArtifact a) {
        return new ArtifactDto(
                a.getId(), a.getOrgId(), a.getProjectId(), a.getRunId(),
                a.getKind().name(), a.getBlobSha256(), a.getRecipeKey(), a.getCreatedAt());
    }
}
