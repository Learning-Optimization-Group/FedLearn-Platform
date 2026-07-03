package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ArtifactBlobRepository;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import com.federated.fl_platform_api.service.LocalFsArtifactBlobStore;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.nio.charset.StandardCharsets;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The write-new-not-overwrite core (DA-2): registering a run's final model stores its bytes
 * content-addressed and inserts a new provenance row — never overwriting a prior run's artifact.
 *
 * <p>Shares the common {@code @SpringBootTest("test")} context (artifact-store root is set in
 * application-test.properties, not a per-class @TestPropertySource) so it adds no extra Hibernate
 * create-drop lifecycle to the shared Testcontainers database.
 */
@SpringBootTest
@ActiveProfiles("test")
class ArtifactRegistryServiceTest {

    @Autowired ArtifactRegistryService registry;
    @Autowired ArtifactBlobRepository blobs;
    @Autowired ModelArtifactRepository artifacts;
    @Autowired LocalFsArtifactBlobStore blobStore;

    @Test
    void register_writes_a_content_addressed_artifact_and_blob() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID(), run = UUID.randomUUID();
        byte[] content = "adapter-weights-v1".getBytes(StandardCharsets.UTF_8);

        ModelArtifact a = registry.register(org, project, run, content, ArtifactKind.LORA_ADAPTER,
                "LLM_LORA", "Qwen/Qwen2.5-0.5B", "Apache-2.0", "{\"accuracy\":0.9}");

        assertThat(a.getId()).isNotNull();
        assertThat(a.getOrgId()).isEqualTo(org);
        assertThat(a.getRunId()).isEqualTo(run);
        assertThat(a.getKind()).isEqualTo(ArtifactKind.LORA_ADAPTER);
        assertThat(a.getBaseModelRef()).isEqualTo("Qwen/Qwen2.5-0.5B");
        assertThat(blobs.findById(a.getBlobSha256())).isPresent();
        assertThat(blobStore.get(a.getBlobSha256())).isEqualTo(content); // bytes round-trip via the store
    }

    @Test
    void two_runs_produce_two_distinct_artifacts_and_the_prior_blob_survives() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID();
        ModelArtifact a1 = registry.register(org, project, UUID.randomUUID(),
                "model-run-1".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT, "CNN", null, null, null);
        ModelArtifact a2 = registry.register(org, project, UUID.randomUUID(),
                "model-run-2".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT, "CNN", null, null, null);

        assertThat(a2.getId()).isNotEqualTo(a1.getId());
        assertThat(a2.getBlobSha256()).isNotEqualTo(a1.getBlobSha256());
        // write-new-not-overwrite: run 1's blob is still retrievable after run 2 registered.
        assertThat(blobStore.exists(a1.getBlobSha256())).isTrue();
        assertThat(blobStore.exists(a2.getBlobSha256())).isTrue();
    }

    @Test
    void identical_bytes_from_two_runs_dedup_the_blob_but_not_the_provenance() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID();
        byte[] same = "identical-model-bytes".getBytes(StandardCharsets.UTF_8);
        // adapters now require a base (DA-3); the dedup being tested is at the adapter blob, not the base.
        ModelArtifact a1 = registry.register(org, project, UUID.randomUUID(), same, ArtifactKind.LORA_ADAPTER, "LLM_LORA", "qwen2.5-0.5b", "Apache-2.0", null);
        ModelArtifact a2 = registry.register(org, project, UUID.randomUUID(), same, ArtifactKind.LORA_ADAPTER, "LLM_LORA", "qwen2.5-0.5b", "Apache-2.0", null);

        assertThat(a1.getBlobSha256()).isEqualTo(a2.getBlobSha256());   // storage dedups
        assertThat(a1.getId()).isNotEqualTo(a2.getId());                 // provenance does not
        assertThat(artifacts.findByBlobSha256(a1.getBlobSha256())).hasSize(2);
    }
}
