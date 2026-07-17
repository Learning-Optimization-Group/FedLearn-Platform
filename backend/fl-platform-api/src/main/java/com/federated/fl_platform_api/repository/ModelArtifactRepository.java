package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/** Per-org provenance rows. Queries are org-scoped by the caller (OrgScopeFilter). */
public interface ModelArtifactRepository extends JpaRepository<ModelArtifact, UUID> {
    List<ModelArtifact> findByOrgId(UUID orgId);
    List<ModelArtifact> findByProjectId(UUID projectId);
    List<ModelArtifact> findByRunId(UUID runId);
    List<ModelArtifact> findByBlobSha256(String blobSha256);

    /** The org's shared BASE_REF row for a given base model, if it already exists (find-or-create). */
    Optional<ModelArtifact> findFirstByOrgIdAndBaseModelRefAndKind(UUID orgId, String baseModelRef, ArtifactKind kind);

    /** The artifact a run already produced for a kind, if any — the idempotency probe for a retried
     *  completion callback (a run produces at most one artifact per kind; UNIQUE(run_id, kind), V12). */
    Optional<ModelArtifact> findByRunIdAndKind(UUID runId, ArtifactKind kind);

    /**
     * DA-3: atomically insert the org's BASE_REF for {@code baseModelRef} ONLY if absent, backed by the
     * partial unique index {@code uq_base_ref_org_model} (V21). {@code ON CONFLICT DO NOTHING} makes the
     * find-or-create race-safe: a concurrent second creator is a silent no-op (no exception, no doomed
     * transaction), and the caller re-reads the single surviving row. {@code flushAutomatically} writes
     * any pending blob insert (the {@code blob_sha256} FK target) before this executes. {@code published}
     * is set explicitly to {@code false} — the DB {@code DEFAULT FALSE} exists only in the Flyway schema
     * (V18), not the Hibernate create-drop test schema, so relying on it would NOT-NULL-fail under test.
     * Returns rows-affected (1 = inserted, 0 = a concurrent creator already had it).
     */
    @Modifying(flushAutomatically = true)
    @Query(value = "INSERT INTO model_artifacts "
            + "(id, org_id, blob_sha256, kind, base_model_ref, license_tag, created_at, published) "
            + "VALUES (:id, :orgId, :sha, 'BASE_REF', :baseModelRef, :licenseTag, :createdAt, false) "
            + "ON CONFLICT DO NOTHING", nativeQuery = true)
    int insertBaseRefIfAbsent(@Param("id") UUID id, @Param("orgId") UUID orgId, @Param("sha") String sha,
                              @Param("baseModelRef") String baseModelRef, @Param("licenseTag") String licenseTag,
                              @Param("createdAt") Instant createdAt);

    /** The project's current head artifact of a kind (the CONTINUED_FROM parent for the next run). */
    Optional<ModelArtifact> findFirstByProjectIdAndKindOrderByCreatedAtDesc(UUID projectId, ArtifactKind kind);

    /** FE-12 marketplace discovery, scoped to a caller's visible orgs (leak-proof at the DB level). */
    List<ModelArtifact> findByOrgIdInAndKindAndPublishedIsTrueOrderByPublishedAtDesc(
            java.util.Collection<UUID> orgIds, ArtifactKind kind);

    /** FE-12 marketplace discovery for a platform admin (unrestricted org scope). */
    List<ModelArtifact> findByKindAndPublishedIsTrueOrderByPublishedAtDesc(ArtifactKind kind);
}
