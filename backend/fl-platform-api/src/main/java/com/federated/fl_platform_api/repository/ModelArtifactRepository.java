package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ModelArtifact;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

/** Per-org provenance rows. Queries are org-scoped by the caller (OrgScopeFilter). */
public interface ModelArtifactRepository extends JpaRepository<ModelArtifact, UUID> {
    List<ModelArtifact> findByOrgId(UUID orgId);
    List<ModelArtifact> findByProjectId(UUID projectId);
    List<ModelArtifact> findByRunId(UUID runId);
    List<ModelArtifact> findByBlobSha256(String blobSha256);
}
