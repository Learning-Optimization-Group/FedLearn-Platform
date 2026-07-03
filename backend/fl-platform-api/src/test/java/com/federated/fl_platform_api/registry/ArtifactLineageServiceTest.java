package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ArtifactLineage;
import com.federated.fl_platform_api.model.LineageRelationship;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ArtifactLineageRepository;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * DA-3 lineage: registering an adapter links it to a (deduped, license-tagged) BASE_REF; continued
 * training links CONTINUED_FROM to the project's prior head; the lineage chain reads base->...->leaf.
 * Shares the common test context (no per-class @TestPropertySource).
 */
@SpringBootTest
@ActiveProfiles("test")
class ArtifactLineageServiceTest {

    @Autowired ArtifactRegistryService registry;
    @Autowired ArtifactLineageRepository lineage;
    @Autowired ModelArtifactRepository artifacts;

    private byte[] bytes(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    private List<ArtifactLineage> parentEdges(UUID childId, LineageRelationship rel) {
        return lineage.findByChildId(childId).stream().filter(e -> e.getRelationship() == rel).toList();
    }

    @Test
    void registering_an_adapter_links_it_to_a_base_ref_with_license() {
        UUID org = UUID.randomUUID();
        ModelArtifact adapter = registry.register(org, UUID.randomUUID(), UUID.randomUUID(),
                bytes("adapter-a"), ArtifactKind.LORA_ADAPTER, "LLM_LORA", "qwen2.5-0.5b", "Apache-2.0", null);

        List<ArtifactLineage> adapterOf = parentEdges(adapter.getId(), LineageRelationship.ADAPTER_OF);
        assertThat(adapterOf).hasSize(1); // exactly one base parent
        ModelArtifact base = artifacts.findById(adapterOf.get(0).getParentId()).orElseThrow();
        assertThat(base.getKind()).isEqualTo(ArtifactKind.BASE_REF);
        assertThat(base.getBaseModelRef()).isEqualTo("qwen2.5-0.5b");
        assertThat(base.getLicenseTag()).isEqualTo("Apache-2.0");
    }

    @Test
    void base_ref_is_deduped_across_adapters_of_the_same_org() {
        UUID org = UUID.randomUUID();
        ModelArtifact a1 = registry.register(org, UUID.randomUUID(), UUID.randomUUID(),
                bytes("x"), ArtifactKind.LORA_ADAPTER, null, "phi-4", "MIT", null);
        ModelArtifact a2 = registry.register(org, UUID.randomUUID(), UUID.randomUUID(),
                bytes("y"), ArtifactKind.LORA_ADAPTER, null, "phi-4", "MIT", null);

        UUID base1 = parentEdges(a1.getId(), LineageRelationship.ADAPTER_OF).get(0).getParentId();
        UUID base2 = parentEdges(a2.getId(), LineageRelationship.ADAPTER_OF).get(0).getParentId();
        assertThat(base1).isEqualTo(base2); // one BASE_REF per (org, base) shared by both adapters
    }

    @Test
    void continued_training_produces_a_continued_from_edge_and_ordered_chain() {
        UUID org = UUID.randomUUID(), project = UUID.randomUUID();
        ModelArtifact v1 = registry.register(org, project, UUID.randomUUID(),
                bytes("v1"), ArtifactKind.LORA_ADAPTER, null, "qwen2.5-0.5b", "Apache-2.0", null);
        ModelArtifact v2 = registry.register(org, project, UUID.randomUUID(),
                bytes("v2"), ArtifactKind.LORA_ADAPTER, null, "qwen2.5-0.5b", "Apache-2.0", null);

        assertThat(parentEdges(v2.getId(), LineageRelationship.CONTINUED_FROM))
                .extracting(ArtifactLineage::getParentId).containsExactly(v1.getId());

        List<ModelArtifact> chain = registry.getLineageChain(v2.getId());
        assertThat(chain.get(0).getKind()).isEqualTo(ArtifactKind.BASE_REF);     // root is the base
        assertThat(chain).extracting(ModelArtifact::getId).endsWith(v1.getId(), v2.getId()); // ...-> v1 -> v2
    }

    @Test
    void a_lora_adapter_without_a_base_is_rejected() {
        assertThatThrownBy(() -> registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                bytes("no-base"), ArtifactKind.LORA_ADAPTER, null, null, null, null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void eval_card_is_stored_on_the_artifact() {
        ModelArtifact a = registry.register(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
                bytes("m"), ArtifactKind.FULL_CHECKPOINT, "CNN", null, null, "{\"accuracy\":0.947,\"seed\":42}");
        assertThat(artifacts.findById(a.getId()).orElseThrow().getEvalCardJson()).contains("0.947");
    }
}
