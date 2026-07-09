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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
    void registeredBlobSha256IsStandardLowercaseHexSha256() throws Exception {
        // DA-9: the manifest's artifact_sha256 (Python's hashlib.sha256(...).hexdigest()) must equal
        // this row's blobSha256 for the same bytes -- proves the registry's content-address is
        // standard lowercase-hex SHA-256, not some other digest/encoding.
        byte[] content = "fedlearn-da9-adapter-bytes".getBytes(StandardCharsets.UTF_8);
        var md = java.security.MessageDigest.getInstance("SHA-256");
        byte[] d = md.digest(content);
        StringBuilder hex = new StringBuilder();
        for (byte b : d) hex.append(String.format("%02x", b));

        ModelArtifact a = registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                content, ArtifactKind.LORA_ADAPTER, "LLM_LORA", "qwen2.5-0.5b", "Apache-2.0", null);

        assertThat(a.getBlobSha256()).isEqualTo(hex.toString());
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

    // ─── SE-11: "no DP label without a committed accountant trace" ──────────────────────────────
    // An eval card claiming dp.enabled=true must carry the accountant's committed trace: numeric
    // accounted_epsilon > 0 and delta in (0,1). Cards without a dp section (or dp.enabled != true)
    // are unaffected.

    @Test
    void dp_claim_without_accountant_trace_is_rejected_and_nothing_is_persisted() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID();
        assertThatThrownBy(() -> registry.register(org, project, UUID.randomUUID(),
                "dp-model-no-trace".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null, "{\"dp\":{\"enabled\":true}}"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("committed accountant trace");
        assertThat(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(
                project, ArtifactKind.FULL_CHECKPOINT)).isEmpty();
    }

    @Test
    void dp_claim_with_non_numeric_epsilon_is_rejected() {
        // A stringly-typed epsilon is not a committed numeric trace.
        assertThatThrownBy(() -> registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                "dp-model-string-eps".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null,
                "{\"dp\":{\"enabled\":true,\"accounted_epsilon\":\"4.2\",\"delta\":1.0E-5}}"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("committed accountant trace");
    }

    @Test
    void dp_claim_with_delta_outside_zero_one_is_rejected() {
        assertThatThrownBy(() -> registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                "dp-model-bad-delta".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null,
                "{\"dp\":{\"enabled\":true,\"accounted_epsilon\":4.2,\"delta\":1.0}}"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("committed accountant trace");
    }

    @Test
    void dp_claim_with_committed_trace_is_accepted_and_the_stored_card_carries_it() {
        String card = "{\"accuracy\":0.91,\"dp\":{\"enabled\":true,\"accounted_epsilon\":5.2,"
                + "\"delta\":1.0E-5,\"noise_multiplier\":0.87}}";
        ModelArtifact a = registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                "dp-model-with-trace".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null, card);

        ModelArtifact reloaded = artifacts.findById(a.getId()).orElseThrow();
        // The committed (epsilon, delta) trace rides the persisted eval card verbatim.
        assertThat(reloaded.getEvalCardJson()).isEqualTo(card);
        assertThat(reloaded.getEvalCardJson()).contains("\"accounted_epsilon\":5.2");
        assertThat(reloaded.getEvalCardJson()).contains("\"delta\":1.0E-5");
    }

    @Test
    void cards_without_a_dp_section_or_with_dp_disabled_are_unaffected() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID();
        ModelArtifact noDp = registry.register(org, project, UUID.randomUUID(),
                "plain-model".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null, "{\"accuracy\":0.9}");
        ModelArtifact dpOff = registry.register(org, project, UUID.randomUUID(),
                "dp-off-model".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null, "{\"dp\":{\"enabled\":false}}");
        ModelArtifact noCard = registry.register(org, project, UUID.randomUUID(),
                "cardless-model".getBytes(StandardCharsets.UTF_8), ArtifactKind.FULL_CHECKPOINT,
                "CNN", null, null, null);

        assertThat(noDp.getId()).isNotNull();
        assertThat(dpOff.getId()).isNotNull();
        assertThat(noCard.getId()).isNotNull();
    }
}
