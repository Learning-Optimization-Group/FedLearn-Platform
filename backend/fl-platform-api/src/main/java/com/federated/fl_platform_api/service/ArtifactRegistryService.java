package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.ArtifactBlob;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ArtifactBlobRepository;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * Registers a run's final model as a versioned, content-addressed artifact — the write-new part of
 * "write-new-not-overwrite" (DA-2). Storage dedups on content (the blob is written once per unique
 * sha256); provenance never does (each registration is a new {@link ModelArtifact} row). Replaces
 * the old behaviour where a run overwrote the single projects.model_path .npz.
 */
@Service
public class ArtifactRegistryService {

    private final ArtifactBlobStore blobStore;
    private final ArtifactBlobRepository blobs;
    private final ModelArtifactRepository artifacts;
    private final ProjectRepository projects;

    public ArtifactRegistryService(ArtifactBlobStore blobStore,
                                   ArtifactBlobRepository blobs,
                                   ModelArtifactRepository artifacts,
                                   ProjectRepository projects) {
        this.blobStore = blobStore;
        this.blobs = blobs;
        this.artifacts = artifacts;
        this.projects = projects;
    }

    /**
     * Store {@code content} content-addressed and record a new artifact row. The blob is written
     * once per unique sha256 (idempotent); a new provenance row is inserted every call.
     */
    @Transactional
    public ModelArtifact register(UUID orgId, UUID projectId, UUID runId, byte[] content,
                                  ArtifactKind kind, String recipeKey, String baseModelRef,
                                  String licenseTag, String evalCardJson) {
        Objects.requireNonNull(orgId, "orgId");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(content, "content");

        String sha256 = blobStore.put(content); // content-addressed, write-once, dedup
        if (!blobs.existsById(sha256)) {
            blobs.save(new ArtifactBlob(sha256, content.length, blobStore.backendId(), Instant.now()));
        }

        ModelArtifact artifact = new ModelArtifact();
        artifact.setOrgId(orgId);
        artifact.setBlobSha256(sha256);
        artifact.setKind(kind);
        artifact.setProjectId(projectId);
        artifact.setRunId(runId);
        artifact.setRecipeKey(recipeKey);
        artifact.setBaseModelRef(baseModelRef);
        artifact.setLicenseTag(licenseTag);
        artifact.setEvalCardJson(evalCardJson);
        artifact.setCreatedAt(Instant.now());
        return artifacts.save(artifact);
    }

    /**
     * Resolve the owning org and active run from the project, then register. Used by the internal
     * callback that {@code fl_server.py} posts to (which knows its projectId).
     */
    @Transactional
    public ModelArtifact registerForProject(UUID projectId, byte[] content, ArtifactKind kind,
                                            String recipeKey, String baseModelRef, String licenseTag,
                                            String evalCardJson) {
        Project project = projects.findById(projectId)
                .orElseThrow(() -> new IllegalArgumentException("unknown project " + projectId));
        return register(project.getOrgId(), projectId, project.getActiveRunId(), content, kind,
                recipeKey, baseModelRef, licenseTag, evalCardJson);
    }
}
