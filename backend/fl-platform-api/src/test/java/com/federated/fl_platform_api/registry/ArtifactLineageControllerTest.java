package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.controller.ArtifactLineageController;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Standalone MockMvc (no Spring context / DB): the lineage read returns the chain for an artifact in
 * a visible org and 404 for a foreign or missing one (no cross-org existence leak). SE-16: it is also
 * participant-scoped — a non-participant of the artifact's project gets 404, mirroring the metadata/blob
 * read path.
 */
class ArtifactLineageControllerTest {

    private final ArtifactRegistryService registry = mock(ArtifactRegistryService.class);
    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);
    private final AuthorizationService authz = mock(AuthorizationService.class);
    private final ProjectRepository projects = mock(ProjectRepository.class);

    @BeforeEach
    void participantByDefault() {
        when(projects.findById(any())).thenReturn(Optional.of(mock(Project.class)));
        when(authz.isParticipant(any())).thenReturn(true);
    }

    private MockMvc mvc(OrgScope scope) {
        return MockMvcBuilders.standaloneSetup(
                new ArtifactLineageController(registry, artifacts, scope, authz, projects)).build();
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

    private ModelArtifact projectArtifact(UUID id, UUID org, ArtifactKind kind, UUID projectId) {
        ModelArtifact a = artifact(id, org, kind);
        a.setProjectId(projectId);
        return a;
    }

    @Test
    void returns_the_chain_for_an_artifact_in_a_visible_org() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        ModelArtifact target = artifact(id, org, ArtifactKind.LORA_ADAPTER);
        when(artifacts.findById(id)).thenReturn(Optional.of(target));
        when(registry.getLineageChain(id)).thenReturn(List.of(
                artifact(UUID.randomUUID(), org, ArtifactKind.BASE_REF), target));

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].kind").value("BASE_REF"))   // root is the base
                .andExpect(jsonPath("$[1].id").value(id.toString())); // ...then the leaf
    }

    @Test
    void returns_404_for_an_artifact_in_a_foreign_org() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org, ArtifactKind.LORA_ADAPTER)));

        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts/{id}/lineage", id)) // caller sees a DIFFERENT org
                .andExpect(status().isNotFound());
    }

    @Test
    void returns_404_for_a_missing_artifact() throws Exception {
        UUID id = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.empty());

        mvc(new OrgScope()).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isNotFound());
    }

    // ---- SE-16: participant gate ----

    @Test
    void lineage_is_404_for_a_same_org_non_participant() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(
                projectArtifact(id, org, ArtifactKind.LORA_ADAPTER, pid)));
        when(authz.isParticipant(any())).thenReturn(false); // in-org, but not a project participant

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isNotFound());
    }

    @Test
    void lineage_ok_for_a_participant() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        ModelArtifact target = projectArtifact(id, org, ArtifactKind.LORA_ADAPTER, pid);
        when(artifacts.findById(id)).thenReturn(Optional.of(target));
        when(registry.getLineageChain(id)).thenReturn(List.of(target));
        when(authz.isParticipant(any())).thenReturn(true);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(id.toString()));
    }

    @Test
    void lineage_ok_for_a_published_artifact_to_a_non_participant() throws Exception {
        // A published marketplace adapter's provenance is readable despite non-participation (FE-12).
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        ModelArtifact target = projectArtifact(id, org, ArtifactKind.LORA_ADAPTER, pid);
        target.setPublished(true);
        when(artifacts.findById(id)).thenReturn(Optional.of(target));
        when(registry.getLineageChain(id)).thenReturn(List.of(target));
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(id.toString()));
    }

    @Test
    void lineage_hides_a_private_ancestor_from_a_non_participant_of_a_published_leaf() throws Exception {
        // The read gate was applied only to the TARGET; getLineageChain returned every ancestor. A
        // non-participant reading a PUBLISHED adapter must NOT see its PRIVATE ancestors' provenance
        // (sha256 / base / license) — those private intermediates must be filtered out (SE-16).
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        UUID privateAncestorId = UUID.randomUUID();
        ModelArtifact base = artifact(UUID.randomUUID(), org, ArtifactKind.BASE_REF);          // no project -> readable
        ModelArtifact privateAncestor =
                projectArtifact(privateAncestorId, org, ArtifactKind.LORA_ADAPTER, pid);        // unpublished, private
        ModelArtifact target = projectArtifact(id, org, ArtifactKind.LORA_ADAPTER, pid);
        target.setPublished(true);

        when(artifacts.findById(id)).thenReturn(Optional.of(target));
        when(registry.getLineageChain(id)).thenReturn(List.of(base, privateAncestor, target));
        when(authz.isParticipant(any())).thenReturn(false);                                     // outsider

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/lineage", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))                                     // base + published leaf only
                .andExpect(jsonPath("$[0].kind").value("BASE_REF"))
                .andExpect(jsonPath("$[1].id").value(id.toString()))
                .andExpect(jsonPath("$[?(@.id=='" + privateAncestorId + "')]").doesNotExist());  // private ancestor gone
    }

    private OrgScope scopeOf(UUID... orgs) {
        OrgScope s = new OrgScope();
        s.set(Set.of(orgs), false);
        return s;
    }
}
