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

    @Test
    void list_returns_the_projects_visible_artifacts_newest_first_with_provenance() throws Exception {
        UUID pid = UUID.randomUUID(), org = UUID.randomUUID();
        ModelArtifact older = artifact(UUID.randomUUID(), org);
        older.setProjectId(pid);
        older.setCreatedAt(Instant.parse("2026-01-01T00:00:00Z"));
        ModelArtifact newer = artifact(UUID.randomUUID(), org);
        newer.setProjectId(pid);
        newer.setCreatedAt(Instant.parse("2026-06-01T00:00:00Z"));
        newer.setBaseModelRef("bert-base");
        newer.setLicenseTag("apache-2.0");
        newer.setEvalCardJson("{\"accuracy\":0.91}");
        // repo returns them oldest-first; the endpoint must sort newest-first regardless.
        when(artifacts.findByProjectId(pid)).thenReturn(java.util.List.of(older, newer));

        mvc(scopeOf(org)).perform(get("/api/artifacts").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].id").value(newer.getId().toString()))   // newest first
                .andExpect(jsonPath("$[0].baseModelRef").value("bert-base"))
                .andExpect(jsonPath("$[0].licenseTag").value("apache-2.0"))
                .andExpect(jsonPath("$[0].evalCardJson").value("{\"accuracy\":0.91}"))
                .andExpect(jsonPath("$[1].id").value(older.getId().toString()));
    }

    @Test
    void list_filters_out_cross_org_rows_and_never_leaks() throws Exception {
        UUID pid = UUID.randomUUID(), myOrg = UUID.randomUUID(), foreignOrg = UUID.randomUUID();
        ModelArtifact mine = artifact(UUID.randomUUID(), myOrg);
        mine.setProjectId(pid);
        ModelArtifact foreign = artifact(UUID.randomUUID(), foreignOrg);
        foreign.setProjectId(pid);
        when(artifacts.findByProjectId(pid)).thenReturn(java.util.List.of(mine, foreign));

        mvc(scopeOf(myOrg)).perform(get("/api/artifacts").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(1))
                .andExpect(jsonPath("$[0].id").value(mine.getId().toString()));

        // a caller in NEITHER org sees nothing — an empty list, not the foreign row.
        mvc(scopeOf(UUID.randomUUID())).perform(get("/api/artifacts").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(0));
    }
}
