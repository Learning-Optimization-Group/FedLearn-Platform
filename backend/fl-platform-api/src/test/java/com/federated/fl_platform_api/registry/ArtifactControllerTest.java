package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.controller.ArtifactController;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.ArtifactBlobStore;
import com.federated.fl_platform_api.service.AuthorizationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
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
 * returns the immutable bytes with the content hash as the ETag.
 *
 * <p>SE-16: they are ALSO participant-scoped. Org scope alone is too coarse for the most sensitive object
 * in the platform (the trained model bytes) — and collapses to nothing in the single-org fallback — so a
 * non-participant of the artifact's project is refused (404, no existence leak) exactly like the rest of
 * the project read surface. BASE_REF-style rows with no project stay org-scoped.</p>
 *
 * <p>Standalone MockMvc; mocked repo + store + authz + projects.</p>
 */
class ArtifactControllerTest {

    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);
    private final ArtifactBlobStore blobStore = mock(ArtifactBlobStore.class);
    private final AuthorizationService authz = mock(AuthorizationService.class);
    private final ProjectRepository projects = mock(ProjectRepository.class);

    @BeforeEach
    void participantByDefault() {
        // Default: the project loads and the caller IS a participant, so the pre-existing tests exercise
        // exactly the org boundary. The SE-16 tests override isParticipant to false to hit the new gate.
        when(projects.findById(any())).thenReturn(Optional.of(mock(Project.class)));
        when(authz.isParticipant(any())).thenReturn(true);
    }

    private MockMvc mvc(OrgScope scope) {
        return MockMvcBuilders.standaloneSetup(
                new ArtifactController(artifacts, blobStore, scope, authz, projects)).build();
    }

    /** A BASE_REF-style artifact: org-shared, no owning project (projectId == null). */
    private ModelArtifact artifact(UUID id, UUID org) {
        ModelArtifact a = new ModelArtifact();
        a.setId(id);
        a.setOrgId(org);
        a.setKind(ArtifactKind.FULL_CHECKPOINT);
        a.setBlobSha256("a".repeat(64));
        a.setCreatedAt(Instant.now());
        return a;
    }

    /** A project-owned artifact (the sensitive case the SE-16 participant gate protects). */
    private ModelArtifact projectArtifact(UUID id, UUID org, UUID projectId) {
        ModelArtifact a = artifact(id, org);
        a.setProjectId(projectId);
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
        ModelArtifact head = projectArtifact(id, org, pid);
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
        ModelArtifact older = projectArtifact(UUID.randomUUID(), org, pid);
        older.setCreatedAt(Instant.parse("2026-01-01T00:00:00Z"));
        ModelArtifact newer = projectArtifact(UUID.randomUUID(), org, pid);
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
        ModelArtifact mine = projectArtifact(UUID.randomUUID(), myOrg, pid);
        ModelArtifact foreign = projectArtifact(UUID.randomUUID(), foreignOrg, pid);
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

    // ---- SE-16: participant gate (the fix for the artifact-BOLA finding) ----

    @Test
    void blob_is_404_for_a_same_org_non_participant_and_never_touches_the_store() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(projectArtifact(id, org, pid)));
        when(authz.isParticipant(any())).thenReturn(false); // in-org, but NOT a participant of the project

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/blob", id))
                .andExpect(status().isNotFound());
        verify(blobStore, never()).get(anyString()); // the weights are never read for a non-participant
    }

    @Test
    void get_is_404_for_a_same_org_non_participant() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(projectArtifact(id, org, pid)));
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}", id))
                .andExpect(status().isNotFound());
    }

    @Test
    void blob_is_ok_for_a_participant() throws Exception {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        byte[] bytes = "weights".getBytes(StandardCharsets.UTF_8);
        when(artifacts.findById(id)).thenReturn(Optional.of(projectArtifact(id, org, pid)));
        when(blobStore.get("a".repeat(64))).thenReturn(bytes);
        when(authz.isParticipant(any())).thenReturn(true);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/blob", id))
                .andExpect(status().isOk())
                .andExpect(content().bytes(bytes));
    }

    @Test
    void latest_is_404_for_a_same_org_non_participant() throws Exception {
        UUID pid = UUID.randomUUID(), org = UUID.randomUUID(), id = UUID.randomUUID();
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(pid, ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(projectArtifact(id, org, pid)));
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts/latest").param("projectId", pid.toString()))
                .andExpect(status().isNotFound());
    }

    @Test
    void list_is_empty_for_a_same_org_non_participant() throws Exception {
        UUID pid = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findByProjectId(pid))
                .thenReturn(java.util.List.of(projectArtifact(UUID.randomUUID(), org, pid)));
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(0));
    }

    @Test
    void base_ref_with_null_project_stays_org_scoped_readable() throws Exception {
        // A BASE_REF-style artifact has no project to participate in; the org gate is the whole gate,
        // so a non-participant flag is irrelevant and the org-visible caller still reads it.
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(artifact(id, org))); // projectId == null
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(id.toString()));
    }

    @Test
    void published_artifact_is_downloadable_by_a_non_participant_marketplace_flow() throws Exception {
        // FE-12: an owner PUBLISHES an adapter to the org marketplace; a non-participant org member
        // must still be able to download it. The SE-16 gate must not break this intentional sharing.
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        byte[] bytes = "published adapter".getBytes(StandardCharsets.UTF_8);
        ModelArtifact pub = projectArtifact(id, org, pid);
        pub.setPublished(true);
        when(artifacts.findById(id)).thenReturn(Optional.of(pub));
        when(blobStore.get("a".repeat(64))).thenReturn(bytes);
        when(authz.isParticipant(any())).thenReturn(false); // NOT a participant — but it's published

        mvc(scopeOf(org)).perform(get("/api/artifacts/{id}/blob", id))
                .andExpect(status().isOk())
                .andExpect(content().bytes(bytes));
    }

    @Test
    void list_shows_published_rows_but_hides_private_ones_from_a_non_participant() throws Exception {
        UUID pid = UUID.randomUUID(), org = UUID.randomUUID();
        ModelArtifact publicRow = projectArtifact(UUID.randomUUID(), org, pid);
        publicRow.setPublished(true);
        ModelArtifact privateRow = projectArtifact(UUID.randomUUID(), org, pid); // unpublished
        when(artifacts.findByProjectId(pid)).thenReturn(java.util.List.of(publicRow, privateRow));
        when(authz.isParticipant(any())).thenReturn(false);

        mvc(scopeOf(org)).perform(get("/api/artifacts").param("projectId", pid.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(1))                       // only the published row
                .andExpect(jsonPath("$[0].id").value(publicRow.getId().toString()));
    }
}
