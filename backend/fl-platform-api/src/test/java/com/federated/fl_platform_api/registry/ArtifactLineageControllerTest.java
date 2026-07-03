package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.controller.ArtifactLineageController;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Standalone MockMvc (no Spring context / DB): the lineage read returns the chain for an artifact in
 * a visible org and 404 for a foreign or missing one (no cross-org existence leak).
 */
class ArtifactLineageControllerTest {

    private final ArtifactRegistryService registry = mock(ArtifactRegistryService.class);
    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);

    private MockMvc mvc(OrgScope scope) {
        return MockMvcBuilders.standaloneSetup(new ArtifactLineageController(registry, artifacts, scope)).build();
    }

    private ModelArtifact artifact(UUID id, UUID org, ArtifactKind kind) {
        ModelArtifact a = new ModelArtifact();
        a.setId(id);
        a.setOrgId(org);
        a.setKind(kind);
        a.setBlobSha256("a".repeat(64));
        a.setCreatedAt(Instant.now());
        return a;
    }

    @Test
    void returns_the_chain_for_an_artifact_in_a_visible_org() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        ModelArtifact target = artifact(id, org, ArtifactKind.LORA_ADAPTER);
        when(artifacts.findById(id)).thenReturn(Optional.of(target));
        when(registry.getLineageChain(id)).thenReturn(List.of(
                artifact(UUID.randomUUID(), org, ArtifactKind.BASE_REF), target));

        OrgScope scope = new OrgScope();
        scope.set(Set.of(org), false);

        mvc(scope).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].kind").value("BASE_REF"))   // root is the base
                .andExpect(jsonPath("$[1].id").value(id.toString())); // ...then the leaf
    }

    @Test
    void returns_404_for_an_artifact_in_a_foreign_org() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org, ArtifactKind.LORA_ADAPTER)));

        OrgScope scope = new OrgScope();
        scope.set(Set.of(UUID.randomUUID()), false); // caller sees a DIFFERENT org

        mvc(scope).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isNotFound());
    }

    @Test
    void returns_404_for_a_missing_artifact() throws Exception {
        UUID id = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.empty());

        mvc(new OrgScope()).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isNotFound());
    }
}
