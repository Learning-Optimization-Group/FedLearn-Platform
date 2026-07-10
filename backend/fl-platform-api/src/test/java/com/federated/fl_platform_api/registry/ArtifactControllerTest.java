package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.controller.ArtifactController;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactBlobStore;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * BA-11: the registry READ endpoints — get-by-id, content-addressed blob download, and project head —
 * are org-scoped (404 for a foreign/missing id, never 403: no cross-org existence leak) and the download
 * returns the immutable bytes with the content hash as the ETag. Standalone MockMvc; mocked repo + store.
 */
class ArtifactControllerTest {

    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);
    private final ArtifactBlobStore blobStore = mock(ArtifactBlobStore.class);

    private MockMvc mvc(OrgScope scope) {
        return MockMvcBuilders.standaloneSetup(new ArtifactController(artifacts, blobStore, scope)).build();
    }

    private ModelArtifact artifact(UUID id, UUID org) {
        ModelArtifact a = new ModelArtifact();
        a.setId(id);
        a.setOrgId(org);
        a.setKind(ArtifactKind.FULL_CHECKPOINT);
        a.setBlobSha256("a".repeat(64));
        a.setCreatedAt(Instant.now());
        return a;
    }

    private OrgScope scopeOf(UUID... orgs) {
        OrgScope s = new OrgScope();
        s.set(Set.of(orgs), false);
        return s;
    }

    @Test
    void get_returns_metadata_for_a_visible_artifact() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org)));

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id.toString()))
                .andExpect(jsonPath("$.kind").value("FULL_CHECKPOINT"))
                .andExpect(jsonPath("$.blobSha256").value("a".repeat(64)));
    }

    @Test
    void get_returns_404_for_a_foreign_org() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org)));

        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts/{id}", id))
                .andExpect(status().isNotFound());
    }

    @Test
    void get_returns_404_for_a_missing_artifact() throws Exception {
        UUID id = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.empty());

        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts/{id}", id))
                .andExpect(status().isNotFound());
    }

    @Test
    void blob_download_returns_the_bytes_with_the_content_hash_etag() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        byte[] bytes = "model weights".getBytes(StandardCharsets.UTF_8);
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org)));
        when(blobStore.get("a".repeat(64))).thenReturn(bytes);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/blob", id))
                .andExpect(status().isOk())
                .andExpect(content().contentType("application/octet-stream"))
                .andExpect(header().string("ETag", "\"" + "a".repeat(64) + "\""))
                .andExpect(content().bytes(bytes));
    }

    @Test
    void blob_download_is_404_for_a_foreign_org_and_never_touches_the_store() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org)));

        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts/{id}/blob", id))
                .andExpect(status().isNotFound());
        verify(blobStore, never()).get(anyString()); // no blob read for an out-of-org caller
    }

    @Test
    void latest_returns_the_project_head_and_404_cross_org() throws Exception {
        UUID pid = UUID.randomUUID(), org = UUID.randomUUID(), id = UUID.randomUUID();
        ModelArtifact head = artifact(id, org);
        head.setProjectId(pid);
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(pid, ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(head));

        mvc(scopeOf(org)).perform(get("/api/artifacts/latest").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id.toString()));

        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts/latest").param("projectId", pid.toString()))
                .andExpect(status().isNotFound());
    }
}
