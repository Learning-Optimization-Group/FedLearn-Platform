package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.dto.ArtifactDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.service.MarketplaceService;
import org.junit.jupiter.api.Test;
import org.springframework.security.access.AccessDeniedException;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * FE-12: the marketplace authz core. Publishing is org-scoped + owner-or-admin, only LORA_ADAPTERs
 * with an owning project are publishable, a foreign-org id is a 404 (no existence leak), and
 * discovery is filtered to the caller's visible orgs at the DB level.
 */
class MarketplaceServiceTest {

    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);
    private final ProjectRepository projects = mock(ProjectRepository.class);
    private final AuthorizationService authz = mock(AuthorizationService.class);

    private MarketplaceService service(OrgScope scope) {
        return new MarketplaceService(artifacts, projects, authz, scope);
    }

    private OrgScope scope(UUID... orgs) {
        OrgScope s = new OrgScope();
        s.set(Set.of(orgs), false);
        return s;
    }

    private OrgScope unrestricted() {
        OrgScope s = new OrgScope();
        s.set(Set.of(), true);
        return s;
    }

    private ModelArtifact adapter(UUID id, UUID org, UUID projectId) {
        ModelArtifact a = new ModelArtifact();
        a.setId(id);
        a.setOrgId(org);
        a.setKind(ArtifactKind.LORA_ADAPTER);
        a.setProjectId(projectId);
        a.setBlobSha256("a".repeat(64));
        a.setCreatedAt(Instant.now());
        return a;
    }

    @Test
    void publish_marks_the_adapter_published_when_owner_and_in_org() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        ModelArtifact a = adapter(id, org, pid);
        Project p = new Project();
        p.setId(pid);
        when(artifacts.findById(id)).thenReturn(Optional.of(a));
        when(projects.findById(pid)).thenReturn(Optional.of(p));
        when(artifacts.save(any(ModelArtifact.class))).thenAnswer(i -> i.getArgument(0));

        ArtifactDto dto = service(scope(org)).publish(id);

        assertThat(dto.published()).isTrue();
        assertThat(dto.publishedAt()).isNotNull();
        verify(authz).requireOwnerOrAdmin(p); // ownership was enforced
        verify(artifacts).save(a);
    }

    @Test
    void publish_is_404_for_a_foreign_org_and_never_touches_the_project_or_save() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(adapter(id, org, UUID.randomUUID())));

        assertThatThrownBy(() -> service(scope(UUID.randomUUID())).publish(id))
                .isInstanceOf(ResourceNotFoundException.class);
        verify(projects, never()).findById(any());
        verify(artifacts, never()).save(any());
    }

    @Test
    void publish_is_404_for_a_missing_artifact() {
        UUID id = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.empty());
        assertThatThrownBy(() -> service(scope(UUID.randomUUID())).publish(id))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void publish_is_409_for_a_non_adapter_kind() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        ModelArtifact ckpt = adapter(id, org, UUID.randomUUID());
        ckpt.setKind(ArtifactKind.FULL_CHECKPOINT);
        when(artifacts.findById(id)).thenReturn(Optional.of(ckpt));

        assertThatThrownBy(() -> service(scope(org)).publish(id))
                .isInstanceOf(ProjectStateException.class);
        verify(artifacts, never()).save(any());
    }

    @Test
    void publish_is_409_when_the_adapter_has_no_owning_project() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID();
        when(artifacts.findById(id)).thenReturn(Optional.of(adapter(id, org, null)));

        assertThatThrownBy(() -> service(scope(org)).publish(id))
                .isInstanceOf(ProjectStateException.class);
        verify(projects, never()).findById(any());
    }

    @Test
    void publish_propagates_403_for_an_in_org_non_owner() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        Project p = new Project();
        p.setId(pid);
        when(artifacts.findById(id)).thenReturn(Optional.of(adapter(id, org, pid)));
        when(projects.findById(pid)).thenReturn(Optional.of(p));
        doThrow(new AccessDeniedException("not owner")).when(authz).requireOwnerOrAdmin(p);

        assertThatThrownBy(() -> service(scope(org)).publish(id))
                .isInstanceOf(AccessDeniedException.class);
        verify(artifacts, never()).save(any());
    }

    @Test
    void unpublish_withdraws_a_published_adapter() {
        UUID id = UUID.randomUUID(), org = UUID.randomUUID(), pid = UUID.randomUUID();
        ModelArtifact a = adapter(id, org, pid);
        a.setPublished(true);
        a.setPublishedAt(Instant.now());
        Project p = new Project();
        p.setId(pid);
        when(artifacts.findById(id)).thenReturn(Optional.of(a));
        when(projects.findById(pid)).thenReturn(Optional.of(p));
        when(artifacts.save(any(ModelArtifact.class))).thenAnswer(i -> i.getArgument(0));

        ArtifactDto dto = service(scope(org)).unpublish(id);

        assertThat(dto.published()).isFalse();
        assertThat(dto.publishedAt()).isNull();
    }

    @Test
    void list_for_a_scoped_caller_queries_only_the_visible_orgs() {
        UUID org = UUID.randomUUID();
        ModelArtifact a = adapter(UUID.randomUUID(), org, UUID.randomUUID());
        a.setPublished(true);
        when(artifacts.findByOrgIdInAndKindAndPublishedIsTrueOrderByPublishedAtDesc(
                Set.of(org), ArtifactKind.LORA_ADAPTER)).thenReturn(List.of(a));

        List<ArtifactDto> feed = service(scope(org)).listPublishedAdapters();

        assertThat(feed).hasSize(1);
        verify(artifacts, never()).findByKindAndPublishedIsTrueOrderByPublishedAtDesc(any());
    }

    @Test
    void list_for_a_platform_admin_queries_all_orgs() {
        when(artifacts.findByKindAndPublishedIsTrueOrderByPublishedAtDesc(ArtifactKind.LORA_ADAPTER))
                .thenReturn(List.of(adapter(UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID())));

        assertThat(service(unrestricted()).listPublishedAdapters()).hasSize(1);
        verify(artifacts, never())
                .findByOrgIdInAndKindAndPublishedIsTrueOrderByPublishedAtDesc(any(), any());
    }

    @Test
    void list_for_an_empty_scope_returns_empty_without_querying() {
        OrgScope empty = new OrgScope();
        empty.set(Set.of(), false);

        assertThat(service(empty).listPublishedAdapters()).isEmpty();
        verify(artifacts, never()).findByKindAndPublishedIsTrueOrderByPublishedAtDesc(any());
        verify(artifacts, never())
                .findByOrgIdInAndKindAndPublishedIsTrueOrderByPublishedAtDesc(any(), any());
    }
}
