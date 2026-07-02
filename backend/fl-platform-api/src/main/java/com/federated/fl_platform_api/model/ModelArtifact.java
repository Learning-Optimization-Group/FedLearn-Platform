package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

/**
 * A versioned, org-scoped provenance record for a specialized model — the "specialized model" as a
 * real, listable noun (replacing the single overwritable projects.model_path). It points at an
 * immutable {@link ArtifactBlob} by content hash; blob_sha256 is deliberately NOT unique, so two
 * orgs/runs may record identical bytes as distinct provenance rows over one deduplicated blob.
 *
 * <p>Append-only: run_id / project_id are FK ON DELETE SET NULL, so an artifact outlives its
 * producer. Never UPDATE a row's blob — a new model is a new row (write-new-not-overwrite).
 */
@Entity
@Table(name = "model_artifacts")
public class ModelArtifact {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    /** Owning organization — denormalized so OrgScopeFilter isolates without a join (mirrors projects.org_id). */
    @Column(name = "org_id", nullable = false)
    private UUID orgId;

    /** Content hash of the bytes in {@link ArtifactBlob}. */
    @Column(name = "blob_sha256", nullable = false, length = 64)
    private String blobSha256;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private ArtifactKind kind;

    /** Nullable: BASE_REF / imported artifacts have no owning project. */
    @Column(name = "project_id")
    private UUID projectId;

    /** Nullable: BASE_REF / imported artifacts have no producing run; SET NULL if the run is deleted. */
    @Column(name = "run_id")
    private UUID runId;

    @Column(name = "recipe_key", length = 64)
    private String recipeKey;

    /** For an adapter, the frozen base it was trained over (by value, e.g. "Qwen/Qwen2.5-0.5B"). */
    @Column(name = "base_model_ref", length = 255)
    private String baseModelRef;

    /** Effective license of the artifact (marketplace-load-bearing). */
    @Column(name = "license_tag", length = 64)
    private String licenseTag;

    /** JSON eval card (metrics/lineage summary) as text — matches V11's TEXT-for-JSON convention. */
    @Column(name = "eval_card_json", columnDefinition = "TEXT")
    private String evalCardJson;

    @Column(name = "created_by")
    private Long createdBy;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public UUID getOrgId() { return orgId; }
    public void setOrgId(UUID orgId) { this.orgId = orgId; }
    public String getBlobSha256() { return blobSha256; }
    public void setBlobSha256(String blobSha256) { this.blobSha256 = blobSha256; }
    public ArtifactKind getKind() { return kind; }
    public void setKind(ArtifactKind kind) { this.kind = kind; }
    public UUID getProjectId() { return projectId; }
    public void setProjectId(UUID projectId) { this.projectId = projectId; }
    public UUID getRunId() { return runId; }
    public void setRunId(UUID runId) { this.runId = runId; }
    public String getRecipeKey() { return recipeKey; }
    public void setRecipeKey(String recipeKey) { this.recipeKey = recipeKey; }
    public String getBaseModelRef() { return baseModelRef; }
    public void setBaseModelRef(String baseModelRef) { this.baseModelRef = baseModelRef; }
    public String getLicenseTag() { return licenseTag; }
    public void setLicenseTag(String licenseTag) { this.licenseTag = licenseTag; }
    public String getEvalCardJson() { return evalCardJson; }
    public void setEvalCardJson(String evalCardJson) { this.evalCardJson = evalCardJson; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
