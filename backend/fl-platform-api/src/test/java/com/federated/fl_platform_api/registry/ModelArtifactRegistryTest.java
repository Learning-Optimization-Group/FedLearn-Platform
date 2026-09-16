package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.model.ArtifactBlob;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ArtifactLineage;
import com.federated.fl_platform_api.model.LineageRelationship;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ArtifactBlobRepository;
import com.federated.fl_platform_api.repository.ArtifactLineageRepository;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Behaviour of the registry: storage dedups on content while provenance stays per-run, and lineage
 * links an adapter to its base. Runs against the {@code test} profile (Testcontainers Postgres,
 * Hibernate create-drop).
 */
@SpringBootTest
@ActiveProfiles("test")
class ModelArtifactRegistryTest {

    @Autowired ArtifactBlobRepository blobs;
    @Autowired ModelArtifactRepository artifacts;
    @Autowired ArtifactLineageRepository lineage;

    private ModelArtifact artifact(UUID org, String sha, ArtifactKind kind, UUID runId) {
        ModelArtifact a = new ModelArtifact();
        a.setOrgId(org);
        a.setBlobSha256(sha);
        a.setKind(kind);
        a.setRunId(runId);
        a.setCreatedAt(Instant.now());
        return a;
    }

    @Test
    void identical_bytes_dedup_at_the_blob_but_stay_distinct_provenance_rows() {
        UUID org = UUID.randomUUID();
        String sha = "a".repeat(64);
        blobs.save(new ArtifactBlob(sha, 123L, "LOCAL_FS", Instant.now()));

        // Two different runs produce byte-identical adapters — one blob, two provenance rows.
        ModelArtifact a1 = artifacts.save(artifact(org, sha, ArtifactKind.LORA_ADAPTER, UUID.randomUUID()));
        ModelArtifact a2 = artifacts.save(artifact(org, sha, ArtifactKind.LORA_ADAPTER, UUID.randomUUID()));

        assertThat(blobs.findById(sha)).isPresent();
        assertThat(artifacts.findByBlobSha256(sha))
                .extracting(ModelArtifact::getId)
                .containsExactlyInAnyOrder(a1.getId(), a2.getId());
    }

    @Test
    void lineage_edge_links_adapter_to_its_base() {
        UUID org = UUID.randomUUID();
        blobs.save(new ArtifactBlob("b".repeat(64), 10L, "LOCAL_FS", Instant.now()));
        blobs.save(new ArtifactBlob("c".repeat(64), 20L, "LOCAL_FS", Instant.now()));

        ModelArtifact base = artifacts.save(artifact(org, "b".repeat(64), ArtifactKind.BASE_REF, null));
        ModelArtifact adapter = artifacts.save(artifact(org, "c".repeat(64), ArtifactKind.LORA_ADAPTER, UUID.randomUUID()));

        lineage.save(new ArtifactLineage(adapter.getId(), base.getId(), LineageRelationship.ADAPTER_OF, Instant.now()));

        List<ArtifactLineage> edges = lineage.findByChildId(adapter.getId());
        assertThat(edges).hasSize(1);
        assertThat(edges.get(0).getParentId()).isEqualTo(base.getId());
        assertThat(edges.get(0).getRelationship()).isEqualTo(LineageRelationship.ADAPTER_OF);
    }
}
