package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.model.ArtifactBlob;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ArtifactLineage;
import com.federated.fl_platform_api.model.LineageRelationship;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ArtifactBlobRepository;
import com.federated.fl_platform_api.repository.ArtifactLineageRepository;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

/**
 * Registers a run's final model as a versioned, content-addressed artifact (write-new-not-overwrite,
 * DA-2) and records its provenance in the lineage DAG (DA-3): a LORA_ADAPTER is linked ADAPTER_OF to
 * a deduped, license-tagged BASE_REF, and a re-training run is linked CONTINUED_FROM the project's
 * prior head. Storage dedups on content; provenance and lineage never do.
 */
@Service
public class ArtifactRegistryService {

    private final ArtifactBlobStore blobStore;
    private final ArtifactBlobRepository blobs;
    private final ModelArtifactRepository artifacts;
    private final ArtifactLineageRepository lineage;
    private final ProjectRepository projects;
    private final ObjectMapper objectMapper;

    public ArtifactRegistryService(ArtifactBlobStore blobStore,
                                   ArtifactBlobRepository blobs,
                                   ModelArtifactRepository artifacts,
                                   ArtifactLineageRepository lineage,
                                   ProjectRepository projects,
                                   ObjectMapper objectMapper) {
        this.blobStore = blobStore;
        this.blobs = blobs;
        this.artifacts = artifacts;
        this.lineage = lineage;
        this.projects = projects;
        this.objectMapper = objectMapper;
    }

    /**
     * Store {@code content} content-addressed, record a new artifact row, and wire its lineage. A
     * LORA_ADAPTER must name its base ({@code baseModelRef}); it is linked ADAPTER_OF to that base's
     * (deduped) BASE_REF. If the project already has a prior head of the same kind, the new artifact
     * is linked CONTINUED_FROM it, capturing federated model evolution.
     */
    @Transactional
    public ModelArtifact register(UUID orgId, UUID projectId, UUID runId, byte[] content,
                                  ArtifactKind kind, String recipeKey, String baseModelRef,
                                  String licenseTag, String evalCardJson) {
        Objects.requireNonNull(orgId, "orgId");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(content, "content");
        if (kind == ArtifactKind.LORA_ADAPTER && (baseModelRef == null || baseModelRef.isBlank())) {
            throw new IllegalArgumentException("a LORA_ADAPTER must reference exactly one base model (baseModelRef)");
        }
        requireAccountantTraceForDpClaim(evalCardJson);   // SE-11: no DP label without a committed trace

        // Idempotency: a retried completion callback (a network timeout AFTER the first POST committed)
        // must not register a second artifact for the same (run, kind) — return the one already recorded.
        // A run produces at most one artifact per kind (UNIQUE(run_id, kind), V12), so without this a
        // retry 500s on that constraint in the Flyway schema (or silently duplicates where it is absent).
        if (runId != null) {
            Optional<ModelArtifact> already = artifacts.findByRunIdAndKind(runId, kind);
            if (already.isPresent()) {
                return already.get();
            }
        }

        // Find the project's current head BEFORE inserting the new row, so CONTINUED_FROM points at
        // the prior artifact rather than the one we are about to create.
        ModelArtifact priorHead = projectId == null ? null
                : artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(projectId, kind).orElse(null);

        String sha256 = putBlob(content);

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
        artifact = artifacts.save(artifact);

        if (kind == ArtifactKind.LORA_ADAPTER) {
            ModelArtifact base = findOrCreateBaseRef(orgId, baseModelRef, licenseTag);
            lineage.save(new ArtifactLineage(artifact.getId(), base.getId(), LineageRelationship.ADAPTER_OF, Instant.now()));
        }
        if (priorHead != null) {
            lineage.save(new ArtifactLineage(artifact.getId(), priorHead.getId(), LineageRelationship.CONTINUED_FROM, Instant.now()));
        }
        return artifact;
    }

    /**
     * Resolve the owning org and active run from the project, then register. Used by the internal
     * callback {@code fl_server.py} posts to (which knows its projectId).
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

    /**
     * The provenance chain for an artifact, ordered roots-first (base -&gt; ... -&gt; the artifact).
     * Walks parent edges; cycle-safe.
     */
    @Transactional(readOnly = true)
    public List<ModelArtifact> getLineageChain(UUID artifactId) {
        LinkedHashMap<UUID, ModelArtifact> ordered = new LinkedHashMap<>();
        walkParentsFirst(artifactId, ordered, new HashSet<>());
        return new ArrayList<>(ordered.values());
    }

    private void walkParentsFirst(UUID id, LinkedHashMap<UUID, ModelArtifact> out, Set<UUID> seen) {
        if (!seen.add(id)) {
            return; // already entered — dedup and cycle guard
        }
        for (ArtifactLineage edge : lineage.findByChildId(id)) {
            walkParentsFirst(edge.getParentId(), out, seen);
        }
        artifacts.findById(id).ifPresent(a -> out.put(id, a)); // post-order => parents before children
    }

    /**
     * The org's shared BASE_REF for {@code baseModelRef}, created (with a reference-manifest blob) if
     * absent. Race-safe (DA-3): the create path is an atomic {@code INSERT ... ON CONFLICT DO NOTHING}
     * ({@link ModelArtifactRepository#insertBaseRefIfAbsent}) backed by the partial unique index
     * {@code uq_base_ref_org_model} (V21), then a re-read of the single surviving row — so two concurrent
     * adapter registrations over the same base can no longer each insert a duplicate BASE_REF (the old
     * check-then-{@code save} did). The read-first fast path avoids the blob write on the common hit.
     */
    private ModelArtifact findOrCreateBaseRef(UUID orgId, String baseModelRef, String licenseTag) {
        var existing = artifacts.findFirstByOrgIdAndBaseModelRefAndKind(orgId, baseModelRef, ArtifactKind.BASE_REF);
        if (existing.isPresent()) {
            return existing.get();
        }
        // A BASE_REF's "content" is a small reference manifest (not the base weights, which live
        // upstream) — content-addressed, so the same base dedups across orgs at the blob.
        byte[] manifest = ("{\"base_model_ref\":" + jsonString(baseModelRef)
                + ",\"license\":" + jsonString(licenseTag) + "}").getBytes(StandardCharsets.UTF_8);
        String sha256 = putBlob(manifest);
        // Atomic insert-if-absent: whoever wins the (org, base_model_ref) race inserts the one BASE_REF;
        // a loser is a silent no-op. Both then re-read the same surviving row below.
        artifacts.insertBaseRefIfAbsent(UUID.randomUUID(), orgId, sha256, baseModelRef, licenseTag, Instant.now());
        return artifacts.findFirstByOrgIdAndBaseModelRefAndKind(orgId, baseModelRef, ArtifactKind.BASE_REF)
                .orElseThrow(() -> new IllegalStateException(
                        "BASE_REF for '" + baseModelRef + "' missing immediately after insert-if-absent"));
    }

    /**
     * SE-11: "no DP label without a committed accountant trace". If the submitted eval card claims
     * DP ({@code dp.enabled == true}), it must carry the accountant's committed trace: a numeric
     * {@code accounted_epsilon > 0} and a numeric {@code delta} in (0,1) exclusive. Cards without a
     * {@code dp} section, or with {@code dp.enabled != true}, are unaffected. A card that is not
     * parseable JSON cannot carry a machine-readable DP claim, so it passes through unchanged
     * (pre-existing contract: the card is stored opaque).
     */
    private void requireAccountantTraceForDpClaim(String evalCardJson) {
        if (evalCardJson == null || evalCardJson.isBlank()) {
            return;
        }
        JsonNode root;
        try {
            root = objectMapper.readTree(evalCardJson);
        } catch (JsonProcessingException e) {
            return; // unparseable card: no machine-readable DP claim to police
        }
        if (root == null || !root.isObject()) {
            return;
        }
        JsonNode dp = root.get("dp");
        if (dp == null || !dp.isObject() || !dp.path("enabled").asBoolean(false)) {
            return;
        }
        JsonNode epsilon = dp.get("accounted_epsilon");
        JsonNode delta = dp.get("delta");
        boolean committed = epsilon != null && epsilon.isNumber() && epsilon.asDouble() > 0
                && delta != null && delta.isNumber()
                && delta.asDouble() > 0 && delta.asDouble() < 1;
        if (!committed) {
            throw new IllegalArgumentException(
                    "an artifact may not claim DP without a committed accountant trace: "
                            + "dp.enabled=true requires a numeric dp.accounted_epsilon > 0 "
                            + "and a numeric dp.delta in (0,1)");
        }
    }

    private String putBlob(byte[] content) {
        String sha256 = blobStore.put(content);
        if (!blobs.existsById(sha256)) {
            blobs.save(new ArtifactBlob(sha256, content.length, blobStore.backendId(), Instant.now()));
        }
        return sha256;
    }

    private static String jsonString(String s) {
        if (s == null) {
            return "null";
        }
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
